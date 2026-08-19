"""Train the joint-class CFG diffusion model from scratch.

The model maps:
    joint_class = dataset_label * 10 + digit_label

Joint classes:
    0..9   = MNIST-M digits 0..9
    10..19 = SVHN digits 0..9
    20     = null CFG condition

Unlike the previous spatial-map wrapper, UNet.py adds the joint-class
embedding to the timestep embedding and injects it into every ResBlock.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms
from torchvision.utils import save_image

from UNet import UNet


MNISTM_ID = 0
SVHN_ID = 1
NULL_DIGIT_ID = 10
NULL_DATASET_ID = 2


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    digit_label: int
    dataset_label: int


class DigitDomainDataset(Dataset):
    def __init__(self, data_root: Path, image_size: int = 32):
        self.data_root = Path(data_root)
        self.records: List[ImageRecord] = []

        self.records.extend(
            self._read_split(
                self.data_root / "mnistm",
                MNISTM_ID,
            )
        )
        self.records.extend(
            self._read_split(
                self.data_root / "svhn",
                SVHN_ID,
            )
        )

        if not self.records:
            raise RuntimeError(f"No records found under {data_root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ),
            ]
        )

    @staticmethod
    def _read_split(
        dataset_dir: Path,
        dataset_label: int,
    ) -> List[ImageRecord]:
        csv_path = dataset_dir / "train.csv"
        image_dir = dataset_dir / "data"

        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        records: List[ImageRecord] = []
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            required = {"image_name", "label"}
            if reader.fieldnames is None or not required.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"{csv_path} must contain {sorted(required)}"
                )

            for row_number, row in enumerate(reader, start=2):
                image_name = str(row["image_name"]).strip()
                digit = int(row["label"])
                if digit == 10:
                    digit = 0
                if not 0 <= digit <= 9:
                    raise ValueError(
                        f"Invalid digit at {csv_path}:{row_number}: {digit}"
                    )

                image_path = image_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                records.append(
                    ImageRecord(
                        image_path=image_path,
                        digit_label=digit,
                        dataset_label=dataset_label,
                    )
                )
        return records

    def condition_counts(self) -> Counter:
        return Counter(
            (r.dataset_label, r.digit_label)
            for r in self.records
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return (
            image,
            torch.tensor(record.digit_label, dtype=torch.long),
            torch.tensor(record.dataset_label, dtype=torch.long),
        )


@dataclass
class TrainingConfig:
    data_root: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    image_size: int
    n_timesteps: int
    beta_start: float
    beta_end: float
    p_uncond: float
    gradient_clip: float
    save_every: int
    sample_every: int
    guidance_scale: float
    samples_per_condition: int
    balanced_sampling: bool
    amp: bool
    device: str
    resume: Optional[str]


def load_cfg_class(project_dir: Path) -> Type[nn.Module]:
    candidates = [
        project_dir / "classifier_free_diffusion.py",
        project_dir / "classifier-free-diffusion.py",
    ]
    module_path = next(
        (path for path in candidates if path.is_file()),
        None,
    )
    if module_path is None:
        raise FileNotFoundError(
            "Expected classifier_free_diffusion.py or "
            "classifier-free-diffusion.py"
        )

    spec = importlib.util.spec_from_file_location(
        "cfg_diffusion_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(module_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CFGDiffusionImageGenerator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_balanced_sampler(
    dataset: DigitDomainDataset,
    seed: int,
):
    counts = dataset.condition_counts()
    weights = [
        1.0 / counts[
            (record.dataset_label, record.digit_label)
        ]
        for record in dataset.records
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def save_label_sanity_grid(
    dataset: DigitDomainDataset,
    output_path: Path,
):
    """Save one training image for every dataset/digit pair.

    Row 1: MNIST-M 0..9
    Row 2: SVHN 0..9
    """
    found = {}
    for index, record in enumerate(dataset.records):
        key = (record.dataset_label, record.digit_label)
        if key not in found:
            image, _, _ = dataset[index]
            found[key] = image
        if len(found) == 20:
            break

    missing = [
        (dataset_id, digit)
        for dataset_id in (MNISTM_ID, SVHN_ID)
        for digit in range(10)
        if (dataset_id, digit) not in found
    ]
    if missing:
        raise RuntimeError(f"Missing label pairs: {missing}")

    images = torch.stack(
        [
            found[(dataset_id, digit)]
            for dataset_id in (MNISTM_ID, SVHN_ID)
            for digit in range(10)
        ]
    )
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, output_path, nrow=10)


def save_checkpoint(
    path: Path,
    diffusion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: TrainingConfig,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": diffusion.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": {
                **asdict(config),
                "model_type": "joint_class_resblock",
                "num_digit_classes": 10,
                "num_dataset_classes": 2,
                "null_digit_idx": NULL_DIGIT_ID,
                "null_dataset_idx": NULL_DATASET_ID,
            },
        },
        path,
    )


def resume_checkpoint(
    path: Path,
    diffusion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location=diffusion.device)
    config = checkpoint.get("config", {})
    if config.get("model_type") != "joint_class_resblock":
        raise ValueError(
            "This is not a joint-class checkpoint. Old spatial-map "
            "checkpoints cannot be resumed with this model."
        )

    diffusion.model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
    )


def save_condition_grid(
    diffusion,
    output_path: Path,
    samples_per_condition: int,
    guidance_scale: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = diffusion.sample_all_20_conditions(
        samples_per_condition=samples_per_condition,
        guidance_scale=guidance_scale,
    )
    save_image(
        diffusion.denormalize(samples),
        output_path,
        nrow=10 * samples_per_condition,
    )


def print_dataset_summary(dataset: DigitDomainDataset):
    counts = dataset.condition_counts()
    print(f"Total images: {len(dataset):,}")
    for dataset_id, name in (
        (MNISTM_ID, "MNIST-M"),
        (SVHN_ID, "SVHN"),
    ):
        row = [
            counts[(dataset_id, digit)]
            for digit in range(10)
        ]
        print(f"{name:7s}: {row}  total={sum(row):,}")


def train(config: TrainingConfig):
    project_dir = Path(__file__).resolve().parent
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)

    device = (
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if config.device == "auto"
        else torch.device(config.device)
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = DigitDomainDataset(
        Path(config.data_root),
        image_size=config.image_size,
    )
    print_dataset_summary(dataset)
    save_label_sanity_grid(
        dataset,
        output_dir / "dataset_label_sanity.png",
    )
    print(
        "Saved label sanity grid: "
        f"{output_dir / 'dataset_label_sanity.png'}"
    )

    sampler = (
        build_balanced_sampler(dataset, config.seed)
        if config.balanced_sampling
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )

    model = UNet(
        in_channel=3,
        num_digit_classes=10,
        num_dataset_classes=2,
        null_digit_idx=NULL_DIGIT_ID,
        null_dataset_idx=NULL_DATASET_ID,
    )

    CFGDiffusionImageGenerator = load_cfg_class(project_dir)
    diffusion = CFGDiffusionImageGenerator(
        conditional_model=model,
        image_size=config.image_size,
        channels=3,
        n_timesteps=config.n_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=str(device),
        p_uncond=config.p_uncond,
        n_digit_classes=10,
        n_dataset_classes=2,
        null_digit_idx=NULL_DIGIT_ID,
        null_dataset_idx=NULL_DATASET_ID,
    )

    diffusion.set_seed(config.seed)
    optimizer = torch.optim.AdamW(
        diffusion.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Only BF16 AMP is enabled. FP16 was unstable with this attention code.
    use_bf16 = (
        config.amp
        and device.type == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if config.amp and not use_bf16:
        print("BF16 AMP unavailable; training in FP32.")

    start_epoch = 0
    global_step = 0
    if config.resume is not None:
        start_epoch, global_step = resume_checkpoint(
            Path(config.resume),
            diffusion,
            optimizer,
        )
        print(
            f"Resumed at epoch {start_epoch + 1}, "
            f"global_step={global_step}"
        )

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "model_type": "joint_class_resblock",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_path = output_dir / "train_log.csv"
    if start_epoch == 0 or not log_path.exists():
        log_path.write_text(
            "epoch,global_step,mean_loss,learning_rate,seconds\n",
            encoding="utf-8",
        )

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda value, **kwargs: value

    for epoch in range(start_epoch, config.epochs):
        diffusion.train()
        start_time = time.time()
        running_loss = 0.0
        seen_batches = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
        )

        for images, digits, datasets in progress:
            images = images.to(device, non_blocking=True)
            digits = digits.to(device, non_blocking=True)
            datasets = datasets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
                if use_bf16
                else nullcontext()
            )
            with context:
                loss = diffusion.training_loss(
                    x_0=images,
                    digit_labels=digits,
                    dataset_labels=datasets,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, "
                    f"step {global_step}: {loss.item()}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                diffusion.model.parameters(),
                config.gradient_clip,
            )
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError(
                    f"Non-finite gradient norm: {grad_norm}"
                )

            optimizer.step()
            global_step += 1

            running_loss += float(loss.item())
            seen_batches += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mean=f"{running_loss / seen_batches:.4f}",
                    step=global_step,
                )

        mean_loss = running_loss / max(seen_batches, 1)
        seconds = time.time() - start_time
        learning_rate = optimizer.param_groups[0]["lr"]

        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                f"{epoch},{global_step},{mean_loss:.8f},"
                f"{learning_rate:.10g},{seconds:.3f}\n"
            )

        print(
            f"Epoch {epoch + 1}: mean_loss={mean_loss:.6f}, "
            f"time={seconds:.1f}s"
        )

        epoch_number = epoch + 1
        if (
            epoch_number % config.save_every == 0
            or epoch_number == config.epochs
        ):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch_number:04d}.pth",
                diffusion,
                optimizer,
                epoch,
                global_step,
                config,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pth",
                diffusion,
                optimizer,
                epoch,
                global_step,
                config,
            )

        if config.sample_every > 0 and (
            epoch_number % config.sample_every == 0
            or epoch_number == config.epochs
        ):
            diffusion.eval()
            save_condition_grid(
                diffusion,
                sample_dir / f"epoch_{epoch_number:04d}.png",
                config.samples_per_condition,
                config.guidance_scale,
            )

    print(f"Training complete: {output_dir}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="hw2_data/digits")
    parser.add_argument(
        "--output-dir",
        default="runs/cfg_joint_class",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--p-uncond", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--balanced-sampling",
        dest="balanced_sampling",
        action="store_true",
    )
    parser.add_argument(
        "--no-balanced-sampling",
        dest="balanced_sampling",
        action="store_false",
    )
    parser.set_defaults(balanced_sampling=True)

    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
    )
    parser.set_defaults(amp=False)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0 <= args.p_uncond < 1:
        parser.error("--p-uncond must satisfy 0 <= p < 1")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.sample_every < 0:
        parser.error("--sample-every must be non-negative")

    return TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        n_timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        p_uncond=args.p_uncond,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        sample_every=args.sample_every,
        guidance_scale=args.guidance_scale,
        samples_per_condition=args.samples_per_condition,
        balanced_sampling=args.balanced_sampling,
        amp=args.amp,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    train(parse_args()) 