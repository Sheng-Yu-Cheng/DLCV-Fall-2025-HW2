"""Fast DDIM inference for the joint-class CFG digit model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import save_image

from UNet import UNet


MNISTM_ID = 0
SVHN_ID = 1
NULL_DIGIT_ID = 10
NULL_DATASET_ID = 2
DATASETS: Tuple[Tuple[str, int], ...] = (
    ("mnistm", MNISTM_ID),
    ("svhn", SVHN_ID),
)


@dataclass(frozen=True)
class OutputRecord:
    dataset_name: str
    dataset_id: int
    digit: int
    sample_index: int

    @property
    def filename(self):
        return f"{self.digit}_{self.sample_index:03d}.png"


def load_cfg_class(project_dir: Path) -> Type[nn.Module]:
    candidates = [
        project_dir / "classifier_free_diffusion.py",
        project_dir / "classifier-free-diffusion.py",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("CFG module not found")

    spec = importlib.util.spec_from_file_location("cfg_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CFGDiffusionImageGenerator


def clean_state_dict(state_dict: Dict[str, Tensor]):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    return cleaned


def load_diffusion(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")

    config = checkpoint.get("config", {})
    if config.get("model_type") != "joint_class_resblock":
        raise ValueError(
            "Checkpoint is not from the joint-class ResBlock model. "
            "Old spatial-map checkpoints are incompatible."
        )

    model = UNet(
        in_channel=3,
        num_digit_classes=int(
            config.get("num_digit_classes", 10)
        ),
        num_dataset_classes=int(
            config.get("num_dataset_classes", 2)
        ),
        null_digit_idx=int(
            config.get("null_digit_idx", NULL_DIGIT_ID)
        ),
        null_dataset_idx=int(
            config.get("null_dataset_idx", NULL_DATASET_ID)
        ),
    )

    CFG = load_cfg_class(Path(__file__).resolve().parent)
    diffusion = CFG(
        conditional_model=model,
        checkpoint_path=None,
        image_size=int(config.get("image_size", 32)),
        channels=3,
        n_timesteps=int(config.get("n_timesteps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        device=str(device),
        p_uncond=float(config.get("p_uncond", 0.1)),
        n_digit_classes=10,
        n_dataset_classes=2,
        null_digit_idx=NULL_DIGIT_ID,
        null_dataset_idx=NULL_DATASET_ID,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    diffusion.model.load_state_dict(
        clean_state_dict(state_dict),
        strict=True,
    )
    diffusion.eval()
    return diffusion, checkpoint


def autocast_context(device: torch.device, use_bf16: bool):
    if (
        use_bf16
        and device.type == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    return nullcontext()


def make_ddim_schedule(
    n_timesteps: int,
    sampling_steps: int,
    device: torch.device,
):
    return torch.unique_consecutive(
        torch.linspace(
            0,
            n_timesteps - 1,
            sampling_steps,
            device=device,
        ).round().long()
    )


@torch.inference_mode()
def predict_noise(
    diffusion,
    x_t,
    timesteps,
    digit_labels,
    dataset_labels,
    guidance_scale,
):
    batch_size = x_t.shape[0]

    if guidance_scale == 1.0:
        return diffusion.model(
            x_t,
            timesteps,
            digit_labels,
            dataset_labels,
        )

    null_digits = torch.full(
        (batch_size,),
        diffusion.null_digit_idx,
        device=x_t.device,
        dtype=torch.long,
    )
    null_datasets = torch.full(
        (batch_size,),
        diffusion.null_dataset_idx,
        device=x_t.device,
        dtype=torch.long,
    )

    if guidance_scale == 0.0:
        return diffusion.model(
            x_t,
            timesteps,
            null_digits,
            null_datasets,
        )

    predictions = diffusion.model(
        torch.cat([x_t, x_t], dim=0),
        torch.cat([timesteps, timesteps], dim=0),
        torch.cat([digit_labels, null_digits], dim=0),
        torch.cat([dataset_labels, null_datasets], dim=0),
    )
    eps_cond, eps_uncond = predictions.chunk(2, dim=0)
    return eps_uncond + guidance_scale * (
        eps_cond - eps_uncond
    )


@torch.inference_mode()
def ddim_sample(
    diffusion,
    digit_labels,
    dataset_labels,
    sampling_steps,
    guidance_scale,
    use_bf16,
    initial_noise=None,
):
    batch_size = digit_labels.shape[0]
    device = diffusion.device

    x_t = (
        diffusion.sample_random_noise(batch_size)
        if initial_noise is None
        else initial_noise.to(device).clone()
    )

    schedule = make_ddim_schedule(
        diffusion.n_timesteps,
        sampling_steps,
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
                diffusion,
                x_t,
                timesteps,
                digit_labels,
                dataset_labels,
                guidance_scale,
            )

        x_t = x_t.float()
        eps = eps.float()
        alpha_bar_t = diffusion.alpha_bars[timestep].float()

        if position + 1 < len(schedule):
            previous = int(schedule[position + 1].item())
            alpha_bar_prev = diffusion.alpha_bars[previous].float()
        else:
            alpha_bar_prev = torch.tensor(
                1.0,
                device=device,
            )

        predicted_x0 = (
            x_t - torch.sqrt(1 - alpha_bar_t) * eps
        ) / torch.sqrt(alpha_bar_t).clamp(min=1e-12)
        predicted_x0 = predicted_x0.clamp(-1, 1)

        x_t = (
            torch.sqrt(alpha_bar_prev) * predicted_x0
            + torch.sqrt(1 - alpha_bar_prev) * eps
        )

    return x_t.clamp(-1, 1)


def make_records(samples_per_digit: int):
    return [
        OutputRecord(name, dataset_id, digit, sample_index)
        for name, dataset_id in DATASETS
        for digit in range(10)
        for sample_index in range(1, samples_per_digit + 1)
    ]


def resize_for_saving(images, size):
    if size == 0 or images.shape[-2:] == (size, size):
        return images
    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
    )
    parser.add_argument("--samples-per-digit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-size", type=int, default=28)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
    diffusion.set_seed(args.seed)

    records = make_records(args.samples_per_digit)
    for name, _ in DATASETS:
        (args.output_folder / name).mkdir(
            parents=True,
            exist_ok=True,
        )

    existing = [
        args.output_folder
        / record.dataset_name
        / record.filename
        for record in records
        if (
            args.output_folder
            / record.dataset_name
            / record.filename
        ).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; add --overwrite"
        )

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda value, **kwargs: value

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Images: {len(records)}")
    print(f"DDIM steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    for start in tqdm(
        range(0, len(records), args.batch_size),
        desc="Generating",
    ):
        batch = records[start : start + args.batch_size]
        digits = torch.tensor(
            [record.digit for record in batch],
            device=device,
            dtype=torch.long,
        )
        datasets = torch.tensor(
            [record.dataset_id for record in batch],
            device=device,
            dtype=torch.long,
        )

        images = ddim_sample(
            diffusion,
            digits,
            datasets,
            args.steps,
            args.guidance_scale,
            args.bf16,
        )
        if not torch.isfinite(images).all():
            raise FloatingPointError("Generated NaN/Inf")

        images = diffusion.denormalize(images.float())
        images = resize_for_saving(images, args.save_size)
        images = images.clamp(0, 1).cpu()

        for image, record in zip(images, batch):
            save_image(
                image,
                args.output_folder
                / record.dataset_name
                / record.filename,
            )

    with (args.output_folder / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["relative_path", "dataset", "digit"])
        for record in records:
            writer.writerow(
                [
                    str(
                        Path(record.dataset_name)
                        / record.filename
                    ),
                    record.dataset_name,
                    record.digit,
                ]
            )

    print(f"Saved {len(records)} images to {args.output_folder}")


if __name__ == "__main__":
    main()