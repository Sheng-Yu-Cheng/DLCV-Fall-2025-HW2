#!/usr/bin/env python3
import argparse
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
        key for key in missing if not key.startswith("control_model.")
    ]

    print(f"Loaded SD checkpoint: {ckpt_path}")
    print(
        f"  missing={len(missing)} "
        f"(non-ControlNet={len(non_control_missing)}), "
        f"unexpected={len(unexpected)}"
    )

    if non_control_missing:
        raise RuntimeError(
            "The SD checkpoint is missing non-ControlNet weights. "
            f"First missing keys: {non_control_missing[:10]}"
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

    print(f"Loaded ControlNet checkpoint: {ckpt_path}")
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    if missing or unexpected:
        raise RuntimeError(
            "ControlNet checkpoint does not exactly match the current "
            "ControlNet architecture."
        )


def load_hint(path, image_size, device):
    image = Image.open(path).convert("RGB")

    if image.size != (image_size, image_size):
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.BILINEAR
        else:
            resample = Image.BILINEAR
        image = image.resize((image_size, image_size), resample=resample)

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
    # Evenly spaced DDIM subsequence, including t=0 and t=T-1.
    steps = np.linspace(
        0,
        num_train_steps - 1,
        num_ddim_steps,
        dtype=np.int64,
    )
    steps = np.unique(steps)

    if len(steps) != num_ddim_steps:
        raise RuntimeError(
            f"Could only construct {len(steps)} unique DDIM steps "
            f"from requested {num_ddim_steps}."
        )

    return torch.from_numpy(steps).long().to(device)


@torch.no_grad()
def ddim_sample(
    model,
    cond,
    uncond,
    shape,
    steps,
    guidance_scale,
    eta,
    generator,
    device,
):
    x = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    timesteps = make_timesteps(
        model.num_timesteps,
        steps,
        device,
    )

    alphas = model.alphas_cumprod.to(device=device, dtype=torch.float32)

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
            prev_t_value = int(timesteps[index - 1].item())
            alpha_prev = alphas[prev_t_value]
        else:
            alpha_prev = torch.tensor(
                1.0,
                device=device,
                dtype=torch.float32,
            )

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        pred_x0 = (
            x - sqrt_one_minus_alpha_t * eps
        ) / sqrt_alpha_t

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

        if index > 0 and eta > 0.0:
            noise = torch.randn(
                x.shape,
                generator=generator,
                device=device,
                dtype=x.dtype,
            )
        else:
            noise = torch.zeros_like(x)

        x = (
            torch.sqrt(alpha_prev) * pred_x0
            + direction
            + sigma * noise
        )

        done = len(timesteps) - index
        if done == 1 or done % 10 == 0 or index == 0:
            print(
                f"DDIM {done:3d}/{len(timesteps)} "
                f"(t={t_value:4d})"
            )

    return x


def main():
    parser = argparse.ArgumentParser(
        description="Generate one image with the HW2-3 ControlNet."
    )

    parser.add_argument(
        "--config",
        default="configs/controlnet/fill50k.yaml",
    )
    parser.add_argument(
        "--sd-ckpt",
        default="models/ldm/stable-diffusion-v1/model.ckpt",
    )
    parser.add_argument(
        "--control-ckpt",
        required=True,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Control/source image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference script.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    # Load pretrained SD first. ControlLatentDiffusion.load_state_dict()
    # then initializes the copied ControlNet encoder from SD.
    load_sd_checkpoint(model, args.sd_ckpt)

    # Now overwrite the ControlNet branch with the trained weights.
    load_control_checkpoint(model, args.control_ckpt)

    model = model.to(device)
    model.eval()
    model.control_model.eval()

    num_controls = len(model.model.diffusion_model.input_blocks) + 1
    model.control_scales = [args.control_scale] * num_controls

    hint = load_hint(
        args.source,
        args.image_size,
        device,
    )

    with torch.no_grad():
        cond_context = model.get_learned_conditioning([args.prompt])
        uncond_context = model.get_learned_conditioning([""])

    # ControlNet convention: conditional and unconditional CFG branches
    # receive the same spatial control image; only text differs.
    cond = {
        "c_crossattn": [cond_context],
        "c_control": [hint],
    }
    uncond = {
        "c_crossattn": [uncond_context],
        "c_control": [hint],
    }

    latent_size = args.image_size // 8
    shape = (1, 4, latent_size, latent_size)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    print("prompt:", args.prompt)
    print("source:", args.source)
    print("latent shape:", shape)
    print("steps:", args.steps)
    print("guidance scale:", args.guidance_scale)
    print("control scale:", args.control_scale)
    print("seed:", args.seed)

    latent = ddim_sample(
        model=model,
        cond=cond,
        uncond=uncond,
        shape=shape,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        generator=generator,
        device=device,
    )

    save_latent_image(
        model,
        latent,
        args.output,
    )

    print("saved:", args.output)


if __name__ == "__main__":
    main()