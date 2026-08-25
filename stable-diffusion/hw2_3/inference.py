#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_sd_checkpoint(model, ckpt_path):
    checkpoint = torch_load_compat(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_control_missing = [
        key for key in missing
        if not key.startswith("control_model.")
    ]

    print("Loaded SD checkpoint:", ckpt_path)
    print(
        "  missing=%d (non-ControlNet=%d), unexpected=%d"
        % (len(missing), len(non_control_missing), len(unexpected))
    )

    if non_control_missing:
        raise RuntimeError(
            "SD checkpoint is missing non-ControlNet keys: %s"
            % non_control_missing[:10]
        )


def load_control_checkpoint(model, ckpt_path):
    checkpoint = torch_load_compat(ckpt_path, map_location="cpu")

    if "control_model" in checkpoint:
        state_dict = checkpoint["control_model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.control_model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Loaded ControlNet checkpoint:", ckpt_path)
    print("  missing=%d, unexpected=%d" % (len(missing), len(unexpected)))

    if missing or unexpected:
        raise RuntimeError(
            "ControlNet checkpoint/model mismatch."
        )


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            for key in ("source", "target", "prompt"):
                if key not in item:
                    raise KeyError(
                        "%s:%d missing key '%s'"
                        % (path, line_no, key)
                    )
            records.append(item)

    if not records:
        raise RuntimeError("No records found in %s" % path)

    return records


def resolve_source(source_folder, source_value):
    """
    Supports both JSON styles:
      "source": "0.png"
      "source": "source/0.png"

    The TA passes the actual source folder separately, so basename fallback
    is important for the private set.
    """
    source_folder = Path(source_folder)

    candidate = source_folder / source_value
    if candidate.is_file():
        return candidate

    candidate = source_folder / Path(source_value).name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "Could not resolve source '%s' under '%s'"
        % (source_value, source_folder)
    )


def load_hint(path, image_size, device):
    image = Image.open(path).convert("RGB")

    if image.size != (image_size, image_size):
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.BILINEAR
        else:
            resample = Image.BILINEAR
        image = image.resize(
            (image_size, image_size),
            resample=resample,
        )

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0).to(device)


def save_latent_image(model, latent, output_path):
    image = model.decode_first_stage(latent)
    image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

    image = image[0].detach().float().cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0 + 0.5).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def make_timesteps(num_train_steps, num_ddim_steps, device):
    values = np.linspace(
        0,
        num_train_steps - 1,
        num_ddim_steps,
        dtype=np.int64,
    )
    values = np.unique(values)

    if len(values) != num_ddim_steps:
        raise RuntimeError(
            "Could only create %d unique DDIM steps (requested %d)"
            % (len(values), num_ddim_steps)
        )

    return torch.from_numpy(values).long().to(device)


@torch.no_grad()
def ddim_sample(
    model,
    cond,
    uncond,
    shape,
    num_steps,
    guidance_scale,
    eta,
    seed,
    device,
):
    # Deterministic per output.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(shape, device=device, dtype=torch.float32)

    timesteps = make_timesteps(
        model.num_timesteps,
        num_steps,
        device,
    )

    alphas = model.alphas_cumprod.to(
        device=device,
        dtype=torch.float32,
    )

    amp_enabled = device.type == "cuda"

    for index in range(len(timesteps) - 1, -1, -1):
        t_value = int(timesteps[index].item())
        t = torch.full(
            (shape[0],),
            t_value,
            device=device,
            dtype=torch.long,
        )

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            eps_cond = model.apply_model(x, t, cond)

            if guidance_scale == 1.0:
                eps = eps_cond
            else:
                eps_uncond = model.apply_model(x, t, uncond)
                eps = eps_uncond + guidance_scale * (
                    eps_cond - eps_uncond
                )

        eps = eps.float()
        alpha_t = alphas[t_value]

        if index > 0:
            previous_t = int(timesteps[index - 1].item())
            alpha_prev = alphas[previous_t]
        else:
            alpha_prev = torch.tensor(
                1.0,
                device=device,
                dtype=torch.float32,
            )

        pred_x0 = (
            x - torch.sqrt(1.0 - alpha_t) * eps
        ) / torch.sqrt(alpha_t)

        if eta > 0.0 and index > 0:
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_prev) / (1.0 - alpha_t))
                * (1.0 - alpha_t / alpha_prev)
            )
        else:
            sigma = torch.tensor(
                0.0,
                device=device,
                dtype=torch.float32,
            )

        direction = torch.sqrt(
            torch.clamp(
                1.0 - alpha_prev - sigma * sigma,
                min=0.0,
            )
        ) * eps

        if eta > 0.0 and index > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            torch.sqrt(alpha_prev) * pred_x0
            + direction
            + sigma * noise
        )

    return x


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for DLCV HW2-3 ControlNet."
    )

    # These four positional args directly mirror the TA shell interface.
    parser.add_argument("json_path")
    parser.add_argument("source_folder")
    parser.add_argument("output_folder")
    parser.add_argument("sd_ckpt")

    parser.add_argument(
        "--config",
        default="configs/controlnet/fill50k.yaml",
    )
    parser.add_argument(
        "--control-ckpt",
        required=True,
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--control-scale", type=float, default=1.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    # Ordering matters: load base SD first, then trained ControlNet.
    load_sd_checkpoint(model, args.sd_ckpt)
    load_control_checkpoint(model, args.control_ckpt)

    model = model.to(device)
    model.eval()
    model.control_model.eval()

    num_controls = len(model.model.diffusion_model.input_blocks) + 1
    model.control_scales = [args.control_scale] * num_controls

    records = read_jsonl(args.json_path)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    latent_size = args.image_size // 8
    shape = (1, 4, latent_size, latent_size)

    with torch.no_grad():
        uncond_context = model.get_learned_conditioning([""])

    print("Number of conditions:", len(records))
    print("DDIM steps:", args.steps)
    print("CFG scale:", args.guidance_scale)
    print("Control scale:", args.control_scale)
    print("Base seed:", args.seed)

    for index, item in enumerate(records):
        source_path = resolve_source(
            args.source_folder,
            item["source"],
        )

        # Assignment requires output filename to come from "target".
        target_name = Path(item["target"]).name
        output_path = output_folder / target_name

        hint = load_hint(
            source_path,
            args.image_size,
            device,
        )

        with torch.no_grad():
            cond_context = model.get_learned_conditioning(
                [item["prompt"]]
            )

        cond = {
            "c_crossattn": [cond_context],
            "c_control": [hint],
        }
        uncond = {
            "c_crossattn": [uncond_context],
            "c_control": [hint],
        }

        print(
            "[%02d/%02d] %s -> %s | %s"
            % (
                index + 1,
                len(records),
                source_path.name,
                target_name,
                item["prompt"],
            )
        )

        latent = ddim_sample(
            model=model,
            cond=cond,
            uncond=uncond,
            shape=shape,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            seed=args.seed + index,
            device=device,
        )

        save_latent_image(
            model,
            latent,
            output_path,
        )

    print("Finished. Output:", output_folder)


if __name__ == "__main__":
    main()


