"""Fast DDIM inference for the two-condition CFG digit model.

Compared with the original inference.py:
1. Uses DDIM with a configurable number of steps instead of 1000-step DDPM.
2. Computes conditional and unconditional predictions in one concatenated
   model call when guidance_scale is not 0 or 1.
3. Mixes all (dataset, digit) conditions in normal batches instead of launching
   a separate diffusion trajectory for every digit.
4. Clips predicted x0 to [-1, 1] during sampling.

Output format:
output_folder/
├── mnistm/0_001.png ... 9_050.png
└── svhn/0_001.png  ... 9_050.png
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import save_image

from train import (
    ConditionalUNet,
    load_cfg_class,
    MNISTM_ID,
    SVHN_ID,
    NULL_DIGIT_ID,
    NULL_DATASET_ID,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass(frozen=True)
class OutputRecord:
    dataset_name: str
    dataset_id: int
    digit: int
    sample_index: int

    @property
    def filename(self) -> str:
        return f"{self.digit}_{self.sample_index:03d}.png"


DATASETS: Tuple[Tuple[str, int], ...] = (
    ("mnistm", MNISTM_ID),
    ("svhn", SVHN_ID),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast DDIM inference for the trained two-condition CFG model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-folder", type=Path, default=Path("output_folder")
    )
    parser.add_argument("--samples-per-digit", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Number of final images generated in one diffusion batch. "
            "For CFG scale > 1, the denoiser internally sees twice this batch."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of DDIM denoising steps. Try 50 first; 100 may improve quality.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help=(
            "0=unconditional, 1=ordinary conditional, >1=CFG. "
            "Use 1.0 to inspect an early checkpoint; high CFG amplifies errors."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Examples: "auto", "cuda", "cuda:0", "cpu".',
    )
    parser.add_argument(
        "--save-size",
        type=int,
        default=28,
        help="Saved PNG size. Use 0 to preserve the model size.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 autocast on a supported CUDA GPU.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="Apply torch.compile to the conditional denoiser.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.samples_per_digit <= 0:
        parser.error("--samples-per-digit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be in [1, 1000]")
    if args.guidance_scale < 0:
        parser.error("--guidance-scale must be non-negative")
    if args.save_size < 0:
        parser.error("--save-size must be non-negative")

    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def clean_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def load_diffusion(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    image_size = int(config.get("image_size", 32))
    n_timesteps = int(config.get("n_timesteps", 1000))
    beta_start = float(config.get("beta_start", 1e-4))
    beta_end = float(config.get("beta_end", 2e-2))
    condition_dim = int(config.get("condition_dim", 16))
    p_uncond = float(config.get("p_uncond", 0.1))

    conditional_unet = ConditionalUNet(
        image_channels=3,
        condition_dim=condition_dim,
        num_digit_tokens=11,
        num_dataset_tokens=3,
    )

    project_dir = Path(__file__).resolve().parent
    CFGDiffusionImageGenerator = load_cfg_class(project_dir)

    diffusion = CFGDiffusionImageGenerator(
        conditional_model=conditional_unet,
        checkpoint_path=None,
        image_size=image_size,
        channels=3,
        n_timesteps=n_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=str(device),
        p_uncond=p_uncond,
        n_digit_classes=10,
        n_dataset_classes=2,
        null_digit_idx=NULL_DIGIT_ID,
        null_dataset_idx=NULL_DATASET_ID,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    diffusion.model.load_state_dict(
        clean_state_dict(state_dict),
        strict=True,
    )
    diffusion.eval()
    diffusion.set_seed(0)

    return diffusion, checkpoint


def autocast_context(device: torch.device, use_bf16: bool):
    if (
        use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        )
    return nullcontext()


def make_records(samples_per_digit: int) -> List[OutputRecord]:
    records: List[OutputRecord] = []
    for dataset_name, dataset_id in DATASETS:
        for digit in range(10):
            for sample_index in range(1, samples_per_digit + 1):
                records.append(
                    OutputRecord(
                        dataset_name=dataset_name,
                        dataset_id=dataset_id,
                        digit=digit,
                        sample_index=sample_index,
                    )
                )
    return records


def prepare_output(
    output_folder: Path,
    records: Sequence[OutputRecord],
    overwrite: bool,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for dataset_name, _ in DATASETS:
        (output_folder / dataset_name).mkdir(parents=True, exist_ok=True)

    existing = [
        output_folder / record.dataset_name / record.filename
        for record in records
        if (output_folder / record.dataset_name / record.filename).exists()
    ]

    if existing and not overwrite:
        examples = "\n".join(str(path) for path in existing[:5])
        raise FileExistsError(
            "Target files already exist. Add --overwrite to replace them.\n"
            f"Examples:\n{examples}"
        )


def make_ddim_schedule(
    n_timesteps: int,
    sampling_steps: int,
    device: torch.device,
) -> Tensor:
    # Ascending unique indices, then sampling traverses them in reverse.
    indices = torch.linspace(
        0,
        n_timesteps - 1,
        sampling_steps,
        device=device,
    ).round().long()
    return torch.unique_consecutive(indices)


@torch.inference_mode()
def predict_noise(
    diffusion,
    x_t: Tensor,
    timesteps: Tensor,
    digit_labels: Tensor,
    dataset_labels: Tensor,
    guidance_scale: float,
) -> Tensor:
    batch_size = x_t.shape[0]

    if guidance_scale == 1.0:
        # Ordinary conditional prediction: one B-sized model call.
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

    # One 2B-sized call instead of two separate B-sized calls.
    model_input = torch.cat([x_t, x_t], dim=0)
    model_timesteps = torch.cat([timesteps, timesteps], dim=0)
    model_digits = torch.cat([digit_labels, null_digits], dim=0)
    model_datasets = torch.cat([dataset_labels, null_datasets], dim=0)

    predicted = diffusion.model(
        model_input,
        model_timesteps,
        model_digits,
        model_datasets,
    )
    eps_cond, eps_uncond = predicted.chunk(2, dim=0)

    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.inference_mode()
def ddim_sample(
    diffusion,
    digit_labels: Tensor,
    dataset_labels: Tensor,
    sampling_steps: int,
    guidance_scale: float,
    use_bf16: bool,
) -> Tensor:
    batch_size = digit_labels.shape[0]
    device = diffusion.device

    x_t = diffusion.sample_random_noise(batch_size)
    schedule = make_ddim_schedule(
        n_timesteps=diffusion.n_timesteps,
        sampling_steps=sampling_steps,
        device=device,
    )

    descending = schedule.flip(0)

    for position, timestep_value in enumerate(descending):
        timestep = int(timestep_value.item())
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

        if position + 1 < len(descending):
            previous_timestep = int(descending[position + 1].item())
            alpha_bar_prev = diffusion.alpha_bars[previous_timestep].float()
        else:
            # The clean-image endpoint before timestep zero.
            alpha_bar_prev = torch.tensor(
                1.0,
                device=device,
                dtype=torch.float32,
            )

        predicted_x0 = (
            x_t - torch.sqrt(1.0 - alpha_bar_t) * eps
        ) / torch.sqrt(alpha_bar_t).clamp(min=1e-12)

        # Dynamic errors from an undertrained model otherwise accumulate badly.
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

        # Deterministic DDIM update (eta = 0).
        x_t = (
            torch.sqrt(alpha_bar_prev) * predicted_x0
            + torch.sqrt(1.0 - alpha_bar_prev) * eps
        )

    return x_t.clamp(-1.0, 1.0)


def resize_for_saving(images: Tensor, save_size: int) -> Tensor:
    if save_size == 0 or images.shape[-2:] == (save_size, save_size):
        return images

    return F.interpolate(
        images,
        size=(save_size, save_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def save_batch(
    diffusion,
    images: Tensor,
    records: Sequence[OutputRecord],
    output_folder: Path,
    save_size: int,
) -> None:
    images = diffusion.denormalize(images.float())
    images = resize_for_saving(images, save_size)
    images = images.clamp(0.0, 1.0).cpu()

    for image, record in zip(images, records):
        output_path = output_folder / record.dataset_name / record.filename
        save_image(image, output_path)


def write_manifest(
    output_folder: Path,
    records: Sequence[OutputRecord],
) -> None:
    with (output_folder / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["relative_path", "dataset", "digit"])
        for record in records:
            relative_path = Path(record.dataset_name) / record.filename
            writer.writerow(
                [str(relative_path), record.dataset_name, record.digit]
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    diffusion, checkpoint = load_diffusion(args.checkpoint, device)

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch version does not provide torch.compile")
        diffusion.model = torch.compile(
            diffusion.model,
            mode="reduce-overhead",
        )

    records = make_records(args.samples_per_digit)
    prepare_output(args.output_folder, records, args.overwrite)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint global_step: {checkpoint.get('global_step')}")
    print(f"Device: {device}")
    print(f"Images: {len(records):,}")
    print(f"DDIM steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Batch size: {args.batch_size}")
    print(f"BF16: {args.bf16}")

    diffusion.set_seed(args.seed)

    progress = tqdm(
        range(0, len(records), args.batch_size),
        desc="Generating",
    )

    for start in progress:
        batch_records = records[start : start + args.batch_size]
        digit_labels = torch.tensor(
            [record.digit for record in batch_records],
            device=device,
            dtype=torch.long,
        )
        dataset_labels = torch.tensor(
            [record.dataset_id for record in batch_records],
            device=device,
            dtype=torch.long,
        )

        images = ddim_sample(
            diffusion=diffusion,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            sampling_steps=args.steps,
            guidance_scale=args.guidance_scale,
            use_bf16=args.bf16,
        )

        if not torch.isfinite(images).all():
            raise FloatingPointError(
                f"Non-finite generated images in output batch starting at {start}"
            )

        save_batch(
            diffusion=diffusion,
            images=images,
            records=batch_records,
            output_folder=args.output_folder,
            save_size=args.save_size,
        )

    write_manifest(args.output_folder, records)

    actual_count = sum(1 for _ in args.output_folder.glob("*/*.png"))
    expected_count = len(records)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} PNG files, found {actual_count}"
        )

    print(f"Done. Saved {actual_count:,} images to {args.output_folder}")


if __name__ == "__main__":
    main()