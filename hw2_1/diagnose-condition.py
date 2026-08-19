"""Fixed-noise diagnosis for the joint-class conditional model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from inference import (
    DATASETS,
    autocast_context,
    ddim_sample,
    load_diffusion,
)


def rms(value):
    return torch.sqrt(torch.mean(value.float() ** 2)).item()


def resize(images, size):
    if size == 0 or images.shape[-2:] == (size, size):
        return images
    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


@torch.inference_mode()
def probe_initial_timestep(
    diffusion,
    base_noise,
    use_bf16,
):
    device = diffusion.device
    timestep = diffusion.n_timesteps - 1
    x_same = base_noise.repeat(10, 1, 1, 1)
    timesteps = torch.full(
        (10,),
        timestep,
        device=device,
        dtype=torch.long,
    )
    digits = torch.arange(10, device=device)

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
        unconditional = diffusion.model(
            x_same,
            timesteps,
            null_digits,
            null_datasets,
        ).float()

    predictions = {}
    digit_metrics = {}

    for dataset_name, dataset_id in DATASETS:
        datasets = torch.full(
            (10,),
            dataset_id,
            device=device,
            dtype=torch.long,
        )
        with autocast_context(device, use_bf16):
            epsilon = diffusion.model(
                x_same,
                timesteps,
                digits,
                datasets,
            ).float()

        predictions[dataset_name] = epsilon
        pairwise = [
            rms(epsilon[i] - epsilon[j])
            for i in range(10)
            for j in range(i + 1, 10)
        ]
        digit_metrics[dataset_name] = {
            "mean_pairwise_rms": sum(pairwise) / len(pairwise),
            "max_pairwise_rms": max(pairwise),
            "mean_cond_vs_uncond_rms": sum(
                rms(epsilon[i] - unconditional[i])
                for i in range(10)
            ) / 10,
        }

    per_digit = [
        {
            "digit": digit,
            "rms_mnistm_vs_svhn": rms(
                predictions["mnistm"][digit]
                - predictions["svhn"][digit]
            ),
        }
        for digit in range(10)
    ]

    return {
        "timestep": timestep,
        "digit_condition": digit_metrics,
        "dataset_condition": {
            "mean_rms_mnistm_vs_svhn": sum(
                row["rms_mnistm_vs_svhn"]
                for row in per_digit
            ) / 10,
            "per_digit": per_digit,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-size", type=int, default=28)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            "Output directory is not empty; add --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if args.device == "auto"
        else torch.device(args.device)
    )
    diffusion, checkpoint = load_diffusion(
        args.checkpoint,
        device,
    )

    digits = torch.arange(
        10,
        device=device,
        dtype=torch.long,
    )
    all_metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seeds": {},
    }
    csv_rows = []

    for seed in args.seeds:
        diffusion.set_seed(seed)
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
            generated = ddim_sample(
                diffusion,
                digits,
                datasets,
                args.steps,
                args.guidance_scale,
                args.bf16,
                initial_noise=repeated_noise,
            )

            images = diffusion.denormalize(generated.float())
            images = resize(images, args.save_size)
            images = images.clamp(0, 1).cpu()
            saved[dataset_name] = images

            save_image(
                images,
                args.output_dir
                / f"seed_{seed}_{dataset_name}.png",
                nrow=10,
            )

        save_image(
            torch.cat(
                [saved["mnistm"], saved["svhn"]],
                dim=0,
            ),
            args.output_dir / f"seed_{seed}_combined.png",
            nrow=10,
        )

        all_metrics["seeds"][str(seed)] = metrics
        for dataset_name in ("mnistm", "svhn"):
            values = metrics["digit_condition"][dataset_name]
            csv_rows.append(
                {
                    "seed": seed,
                    "type": "digit",
                    "dataset": dataset_name,
                    "metric": "mean_pairwise_rms",
                    "value": values["mean_pairwise_rms"],
                }
            )

        csv_rows.append(
            {
                "seed": seed,
                "type": "dataset",
                "dataset": "mnistm_vs_svhn",
                "metric": "mean_rms",
                "value": metrics[
                    "dataset_condition"
                ]["mean_rms_mnistm_vs_svhn"],
            }
        )

        print(f"\nSeed {seed}")
        for dataset_name in ("mnistm", "svhn"):
            values = metrics["digit_condition"][dataset_name]
            print(
                f"  {dataset_name:6s} digit mean pairwise RMS: "
                f"{values['mean_pairwise_rms']:.8f}"
            )
        print(
            "  MNIST-M-vs-SVHN mean RMS: "
            f"{metrics['dataset_condition']['mean_rms_mnistm_vs_svhn']:.8f}"
        )

    (args.output_dir / "condition_metrics.json").write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    with (
        args.output_dir / "condition_metrics.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "type",
                "dataset",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()