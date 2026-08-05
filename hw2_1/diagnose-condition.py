"""Controlled condition diagnostic for the trained two-condition CFG model.

For each seed, this script:
1. Samples one Gaussian x_T.
2. Repeats the same x_T ten times.
3. Generates digit labels 0..9 with dataset fixed to MNIST-M.
4. Repeats the same experiment with dataset fixed to SVHN.
5. Measures how much the model's predicted noise changes at t=T-1 when only
   digit or dataset labels change.

The script reuses the exact checkpoint-loading and DDIM logic from
inference_fast.py, so training and diagnosis reconstruct the same model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import save_image

from inference import (
    DATASETS,
    autocast_context,
    load_diffusion,
    make_ddim_schedule,
    predict_noise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose digit/dataset conditioning with fixed x_T."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnose_condition"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-size", type=int, default=28)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be in [1, 1000]")
    if args.guidance_scale < 0:
        parser.error("--guidance-scale must be non-negative")
    if args.save_size < 0:
        parser.error("--save-size must be non-negative")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{path} is not empty. Add --overwrite or choose another directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def resize_for_saving(images: Tensor, size: int) -> Tensor:
    if size == 0 or images.shape[-2:] == (size, size):
        return images
    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def rms(x: Tensor) -> float:
    return torch.sqrt(torch.mean(x.float() ** 2)).item()


@torch.inference_mode()
def probe_initial_timestep(
    diffusion,
    base_noise: Tensor,
    use_bf16: bool,
) -> Dict[str, object]:
    """Directly test label sensitivity with identical x_T and identical t."""
    device = diffusion.device
    timestep = diffusion.n_timesteps - 1

    x_same = base_noise.repeat(10, 1, 1, 1)
    timesteps = torch.full(
        (10,),
        timestep,
        device=device,
        dtype=torch.long,
    )
    digits = torch.arange(10, device=device, dtype=torch.long)

    null_digits = torch.full(
        (10,),
        diffusion.null_digit_idx,
        device=device,
        dtype=torch.long,
    )
    null_datasets = torch.full(
        (10,),
        diffusion.null_dataset_idx,
        device=device,
        dtype=torch.long,
    )

    with autocast_context(device, use_bf16):
        eps_uncond = diffusion.model(
            x_same,
            timesteps,
            null_digits,
            null_datasets,
        ).float()

    predictions: Dict[str, Tensor] = {}
    digit_metrics: Dict[str, object] = {}

    for dataset_name, dataset_id in DATASETS:
        datasets = torch.full(
            (10,),
            dataset_id,
            device=device,
            dtype=torch.long,
        )
        with autocast_context(device, use_bf16):
            eps = diffusion.model(
                x_same,
                timesteps,
                digits,
                datasets,
            ).float()

        predictions[dataset_name] = eps

        pairwise = [
            rms(eps[i] - eps[j])
            for i in range(10)
            for j in range(i + 1, 10)
        ]
        vs_digit0 = [
            {
                "digit": digit,
                "rms_vs_digit0": rms(eps[digit] - eps[0]),
            }
            for digit in range(10)
        ]
        vs_uncond = [
            rms(eps[digit] - eps_uncond[digit])
            for digit in range(10)
        ]

        digit_metrics[dataset_name] = {
            "mean_pairwise_rms": sum(pairwise) / len(pairwise),
            "max_pairwise_rms": max(pairwise),
            "mean_cond_vs_uncond_rms": sum(vs_uncond) / len(vs_uncond),
            "relative_to_digit0": vs_digit0,
        }

    dataset_per_digit = [
        {
            "digit": digit,
            "rms_mnistm_vs_svhn": rms(
                predictions["mnistm"][digit] - predictions["svhn"][digit]
            ),
        }
        for digit in range(10)
    ]

    return {
        "timestep": timestep,
        "digit_condition": digit_metrics,
        "dataset_condition": {
            "mean_rms_mnistm_vs_svhn": sum(
                item["rms_mnistm_vs_svhn"]
                for item in dataset_per_digit
            ) / 10.0,
            "per_digit": dataset_per_digit,
        },
    }


@torch.inference_mode()
def ddim_sample_from_noise(
    diffusion,
    initial_noise: Tensor,
    digit_labels: Tensor,
    dataset_labels: Tensor,
    steps: int,
    guidance_scale: float,
    use_bf16: bool,
) -> Tensor:
    """Deterministic DDIM sampling from a caller-supplied x_T."""
    batch_size = initial_noise.shape[0]
    device = diffusion.device
    x_t = initial_noise.to(device=device, dtype=torch.float32).clone()

    schedule = make_ddim_schedule(
        diffusion.n_timesteps,
        steps,
        device,
    ).flip(0)

    for position, timestep_tensor in enumerate(schedule):
        timestep = int(timestep_tensor.item())
        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        with autocast_context(device, use_bf16):
            eps = predict_noise(
                diffusion=diffusion,
                x_t=x_t,
                timesteps=timesteps,
                digit_labels=digit_labels,
                dataset_labels=dataset_labels,
                guidance_scale=guidance_scale,
            )

        eps = eps.float()
        x_t = x_t.float()
        alpha_bar_t = diffusion.alpha_bars[timestep].float()

        if position + 1 < len(schedule):
            previous_timestep = int(schedule[position + 1].item())
            alpha_bar_prev = diffusion.alpha_bars[
                previous_timestep
            ].float()
        else:
            alpha_bar_prev = torch.tensor(
                1.0,
                device=device,
                dtype=torch.float32,
            )

        predicted_x0 = (
            x_t - torch.sqrt(1.0 - alpha_bar_t) * eps
        ) / torch.sqrt(alpha_bar_t).clamp(min=1e-12)

        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        x_t = (
            torch.sqrt(alpha_bar_prev) * predicted_x0
            + torch.sqrt(1.0 - alpha_bar_prev) * eps
        )

    return x_t.clamp(-1.0, 1.0)


def save_images(
    diffusion,
    images: Tensor,
    output_dir: Path,
    dataset_name: str,
    seed: int,
    save_size: int,
) -> Tensor:
    images = diffusion.denormalize(images.float())
    images = resize_for_saving(images, save_size)
    images = images.clamp(0.0, 1.0).cpu()

    image_dir = output_dir / f"seed_{seed}" / dataset_name
    image_dir.mkdir(parents=True, exist_ok=True)

    for digit, image in enumerate(images):
        save_image(image, image_dir / f"{digit}.png")

    save_image(
        images,
        output_dir / f"seed_{seed}_{dataset_name}.png",
        nrow=10,
    )
    return images


def metrics_to_rows(seed: int, metrics: Dict[str, object]) -> List[dict]:
    rows: List[dict] = []

    digit_metrics = metrics["digit_condition"]
    for dataset_name, values in digit_metrics.items():
        rows.append(
            {
                "seed": seed,
                "type": "digit",
                "dataset": dataset_name,
                "digit": "",
                "metric": "mean_pairwise_rms",
                "value": values["mean_pairwise_rms"],
            }
        )
        rows.append(
            {
                "seed": seed,
                "type": "digit",
                "dataset": dataset_name,
                "digit": "",
                "metric": "mean_cond_vs_uncond_rms",
                "value": values["mean_cond_vs_uncond_rms"],
            }
        )
        for item in values["relative_to_digit0"]:
            rows.append(
                {
                    "seed": seed,
                    "type": "digit",
                    "dataset": dataset_name,
                    "digit": item["digit"],
                    "metric": "rms_vs_digit0",
                    "value": item["rms_vs_digit0"],
                }
            )

    for item in metrics["dataset_condition"]["per_digit"]:
        rows.append(
            {
                "seed": seed,
                "type": "dataset",
                "dataset": "mnistm_vs_svhn",
                "digit": item["digit"],
                "metric": "rms_mnistm_vs_svhn",
                "value": item["rms_mnistm_vs_svhn"],
            }
        )
    return rows


def print_summary(seed: int, metrics: Dict[str, object]) -> None:
    print(f"\nSeed {seed}")
    for dataset_name in ("mnistm", "svhn"):
        values = metrics["digit_condition"][dataset_name]
        print(
            f"  {dataset_name:6s} digit mean pairwise RMS: "
            f"{values['mean_pairwise_rms']:.8f}"
        )
        print(
            f"  {dataset_name:6s} conditional-vs-null RMS: "
            f"{values['mean_cond_vs_uncond_rms']:.8f}"
        )
    print(
        "  MNIST-M-vs-SVHN mean RMS: "
        f"{metrics['dataset_condition']['mean_rms_mnistm_vs_svhn']:.8f}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prepare_output_dir(args.output_dir, args.overwrite)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    diffusion, checkpoint = load_diffusion(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint global_step: {checkpoint.get('global_step')}")
    print(f"Device: {device}")
    print(f"DDIM steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Seeds: {args.seeds}")

    digits = torch.arange(10, device=device, dtype=torch.long)
    all_metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seeds": {},
    }
    csv_rows: List[dict] = []

    for seed in args.seeds:
        diffusion.set_seed(seed)

        # One x_T is shared by all 20 generated images for this seed.
        base_noise = diffusion.sample_random_noise(1)
        repeated_noise = base_noise.repeat(10, 1, 1, 1)

        metrics = probe_initial_timestep(
            diffusion,
            base_noise,
            args.bf16,
        )

        saved = {}
        for dataset_name, dataset_id in DATASETS:
            datasets = torch.full(
                (10,),
                dataset_id,
                device=device,
                dtype=torch.long,
            )
            generated = ddim_sample_from_noise(
                diffusion=diffusion,
                initial_noise=repeated_noise,
                digit_labels=digits,
                dataset_labels=datasets,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                use_bf16=args.bf16,
            )

            if not torch.isfinite(generated).all():
                raise FloatingPointError(
                    f"Non-finite output: seed={seed}, dataset={dataset_name}"
                )

            saved[dataset_name] = save_images(
                diffusion,
                generated,
                args.output_dir,
                dataset_name,
                seed,
                args.save_size,
            )

        # Row 1 = MNIST-M 0..9; row 2 = SVHN 0..9.
        save_image(
            torch.cat([saved["mnistm"], saved["svhn"]], dim=0),
            args.output_dir / f"seed_{seed}_combined.png",
            nrow=10,
        )

        all_metrics["seeds"][str(seed)] = metrics
        csv_rows.extend(metrics_to_rows(seed, metrics))
        print_summary(seed, metrics)

    json_path = args.output_dir / "condition_metrics.json"
    json_path.write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    csv_path = args.output_dir / "condition_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "type",
                "dataset",
                "digit",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    (args.output_dir / "README.txt").write_text(
        "Each seed_<N>_combined.png contains two rows.\n"
        "Row 1: MNIST-M labels 0..9.\n"
        "Row 2: SVHN labels 0..9.\n\n"
        "All 20 outputs for a seed start from exactly the same x_T.\n"
        "condition_metrics.json/csv measure direct predicted-noise changes\n"
        "at t=T-1 while x_T and timestep are held fixed.\n"
        "There is no universal numeric threshold; compare the grids and\n"
        "metrics across multiple seeds.\n",
        encoding="utf-8",
    )

    print(f"\nSaved diagnostics to: {args.output_dir}")
    print(f"Combined grids: seed_<N>_combined.png")
    print(f"Metrics: {json_path}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()