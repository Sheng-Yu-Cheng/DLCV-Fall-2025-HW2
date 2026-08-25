#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from fill50k_dataset import Fill50KDataset


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_sd_checkpoint(model, ckpt_path):
    checkpoint = torch_load_compat(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # ControlLatentDiffusion intentionally has new control_model.* parameters.
    # Its load_state_dict() implementation synchronizes the copied encoder
    # from the just-loaded pretrained SD U-Net.
    non_control_missing = [
        k for k in missing if not k.startswith("control_model.")
    ]

    print(f"checkpoint loaded: {ckpt_path}")
    print(f"missing keys: {len(missing)} "
          f"(non-ControlNet: {len(non_control_missing)})")
    print(f"unexpected keys: {len(unexpected)}")

    if non_control_missing:
        print("first non-ControlNet missing keys:")
        for key in non_control_missing[:20]:
            print("  ", key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlnet/fill50k.yaml",
    )
    parser.add_argument(
        "--sd-ckpt",
        default="models/ldm/stable-diffusion-v1/model.ckpt",
    )
    parser.add_argument(
        "--data-root",
        default="../hw2_data/fill50k/training",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    load_sd_checkpoint(model, args.sd_ckpt)
    model = model.to(device)
    model.eval()
    model.control_model.eval()

    dataset = Fill50KDataset(
        args.data_root,
        image_size=512,
        max_samples=1,
    )
    item = dataset[0]

    target = item["target"].unsqueeze(0).to(device)
    hint = item["hint"].unsqueeze(0).to(device)
    prompts = [item["prompt"]]

    print("sample:")
    print("  source:", item["source_name"])
    print("  target:", item["target_name"])
    print("  prompt:", prompts[0])
    print("target:", tuple(target.shape), target.dtype,
          f"range=[{target.min().item():.3f}, {target.max().item():.3f}]")
    print("hint:  ", tuple(hint.shape), hint.dtype,
          f"range=[{hint.min().item():.3f}, {hint.max().item():.3f}]")

    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            posterior = model.encode_first_stage(target)
            z = model.get_first_stage_encoding(posterior).detach()
            context = model.get_learned_conditioning(prompts)

            t = torch.full(
                (1,),
                500,
                device=device,
                dtype=torch.long,
            )

            noise = torch.randn_like(z)
            x_noisy = model.q_sample(
                x_start=z,
                t=t,
                noise=noise,
            )

            controls = model.control_model(
                x=x_noisy,
                hint=hint,
                timesteps=t,
                context=context,
            )

    print("z:      ", tuple(z.shape))
    print("context:", tuple(context.shape))
    print("controls:", len(controls))

    if len(controls) != 13:
        raise RuntimeError(
            f"Expected 13 ControlNet residuals, got {len(controls)}"
        )

    for i, tensor in enumerate(controls):
        print(
            f"  control[{i:02d}] "
            f"shape={tuple(tensor.shape)} "
            f"max_abs={tensor.abs().max().item():.8f}"
        )

    max_control = max(
        tensor.abs().max().item() for tensor in controls
    )

    # Zero-convolution initialization should make every residual exactly zero.
    if max_control > 1e-6:
        raise RuntimeError(
            f"Zero-conv sanity check failed: max_abs={max_control}"
        )

    cond = {
        "c_crossattn": [context],
        "c_control": [hint],
    }

    # Check that zero-initialized ControlNet leaves SD output unchanged.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            eps_base = model.model.diffusion_model(
                x_noisy,
                t,
                context=context,
            )

            eps_control = model.apply_model(
                x_noisy,
                t,
                cond,
            )

    print("eps_base:   ", tuple(eps_base.shape))
    print("eps_control:", tuple(eps_control.shape))

    diff = (eps_base - eps_control).abs()
    print("zero-control equivalence:")
    print("  max_abs_diff :", diff.max().item())
    print("  mean_abs_diff:", diff.mean().item())

    if eps_control.shape != z.shape:
        raise RuntimeError(
            f"epsilon shape mismatch: eps={eps_control.shape}, z={z.shape}"
        )

    # With zero residuals, these should be numerically identical.
    if diff.max().item() > 1e-4:
        raise RuntimeError(
            "Zero-initialized ControlNet changed the pretrained SD output."
        )

    print()
    print("CONTROLNET SANITY TEST PASSED")


if __name__ == "__main__":
    main()