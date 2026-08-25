#!/usr/bin/env python3
import argparse
import random
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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

    non_control_missing = [
        k for k in missing if not k.startswith("control_model.")
    ]

    print(f"Loaded pretrained SD checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)} "
          f"(non-ControlNet: {len(non_control_missing)})")
    print(f"  unexpected keys: {len(unexpected)}")

    if non_control_missing:
        print("WARNING: non-ControlNet keys are missing:")
        for key in non_control_missing[:20]:
            print("  ", key)


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
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")


def save_control_checkpoint(model, output_path, epoch, step, args):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save only the trainable branch, not the frozen SD/VAE/CLIP weights.
    payload = {
        "control_model": model.control_model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "image_size": int(args.image_size),
        "text_dropout": float(args.text_dropout),
    }
    torch.save(payload, output_path)


def prompt_dropout(prompts, probability):
    if probability <= 0.0:
        return list(prompts)

    result = []
    for prompt in prompts:
        result.append("" if random.random() < probability else prompt)
    return result


def freeze_base_model(model):
    # Base denoiser is frozen in ControlLatentDiffusion.__init__, but do this
    # explicitly here as a safety check.
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.first_stage_model.parameters():
        param.requires_grad = False
    for param in model.cond_stage_model.parameters():
        param.requires_grad = False

    model.model.eval()
    model.first_stage_model.eval()
    model.cond_stage_model.eval()
    model.control_model.train()


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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
    parser.add_argument(
        "--output-dir",
        default="runs/controlnet_fill50k",
    )

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--text-dropout", type=float, default=0.5)

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N samples. Useful for overfit tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save a ControlNet checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also save a numbered checkpoint at every epoch boundary. Disabled by default to avoid huge checkpoint spam.",
    )
    parser.add_argument(
        "--resume-control",
        default=None,
        help="Optional ControlNet-only checkpoint to resume from.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA AMP.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.text_dropout <= 1.0:
        raise ValueError("--text-dropout must be in [0, 1].")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Full ControlNet training is intended "
            "to run on a CUDA GPU."
        )

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    # Important ordering:
    # 1. instantiate ControlLatentDiffusion
    # 2. load the pretrained SD checkpoint
    # 3. ControlLatentDiffusion.load_state_dict() syncs the copied
    #    encoder/middle weights into ControlNet
    load_sd_checkpoint(model, args.sd_ckpt)

    if args.resume_control is not None:
        load_control_checkpoint(model, args.resume_control)

    model = model.to(device)
    freeze_base_model(model)

    trainable = count_trainable(model.control_model)
    total_base = sum(p.numel() for p in model.model.parameters())

    print(f"ControlNet trainable params: {trainable / 1e6:.2f} M")
    print(f"Frozen SD denoiser params:   {total_base / 1e6:.2f} M")

    # Safety check: frozen SD must really be frozen.
    if any(p.requires_grad for p in model.model.parameters()):
        raise RuntimeError("Frozen Stable Diffusion U-Net has trainable params.")

    dataset = Fill50KDataset(
        args.data_root,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print("training samples:", len(dataset))
    print("steps per epoch:", len(loader))

    optimizer = torch.optim.AdamW(
        model.control_model.parameters(),
        lr=args.lr,
    )

    amp_enabled = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    stop_training = False

    for epoch in range(args.epochs):
        if stop_training:
            break

        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            step_start = time.time()

            target = batch["target"].to(
                device,
                non_blocking=True,
            )
            hint = batch["hint"].to(
                device,
                non_blocking=True,
            )

            prompts = prompt_dropout(
                batch["prompt"],
                args.text_dropout,
            )

            optimizer.zero_grad(set_to_none=True)

            # VAE and CLIP are frozen. No gradients should be stored for them.
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    posterior = model.encode_first_stage(target)
                    z = model.get_first_stage_encoding(posterior).detach()
                    context = model.get_learned_conditioning(prompts)

            t = torch.randint(
                low=0,
                high=model.num_timesteps,
                size=(z.shape[0],),
                device=device,
                dtype=torch.long,
            )

            cond = {
                "c_crossattn": [context],
                "c_control": [hint],
            }

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, loss_dict = model.p_losses(
                    z,
                    cond,
                    t,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {global_step}: {loss.item()}"
                )

            scaler.scale(loss).backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.control_model.parameters(),
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.time() - step_start
                loss_simple = loss_dict["train/loss_simple"].detach().item()
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"simple={loss_simple:.6f} "
                    f"time={elapsed:.2f}s"
                )

            if (
                args.save_every > 0
                and global_step % args.save_every == 0
            ):
                path = checkpoint_dir / f"step_{global_step:07d}.pth"
                save_control_checkpoint(
                    model,
                    path,
                    epoch=epoch,
                    step=global_step,
                    args=args,
                )
                print("saved:", path)

                latest = checkpoint_dir / "latest.pth"
                save_control_checkpoint(
                    model,
                    latest,
                    epoch=epoch,
                    step=global_step,
                    args=args,
                )

            if (
                args.max_steps is not None
                and global_step >= args.max_steps
            ):
                stop_training = True
                break

        # Do not save a giant checkpoint every epoch by default.
        # This matters especially for one-sample overfit runs where one epoch
        # is only one optimizer step.
        if args.save_every_epoch:
            epoch_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"
            save_control_checkpoint(
                model,
                epoch_path,
                epoch=epoch,
                step=global_step,
                args=args,
            )
            print("saved:", epoch_path)

        print(
            f"epoch {epoch + 1} finished in "
            f"{time.time() - epoch_start:.1f}s"
        )

    # Always save one final checkpoint.
    final_path = checkpoint_dir / "latest.pth"
    save_control_checkpoint(
        model,
        final_path,
        epoch=max(0, epoch),
        step=global_step,
        args=args,
    )

    print("training finished")
    print("final step:", global_step)
    print("checkpoint:", final_path)


if __name__ == "__main__":
    main()