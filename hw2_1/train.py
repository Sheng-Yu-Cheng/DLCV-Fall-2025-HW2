"""Train a two-condition classifier-free-guidance DDPM.

Conditions
----------
1. digit:   0..9, with null token 10
2. dataset: 0=MNIST-M, 1=SVHN, with null token 2

Expected project layout
-----------------------
project/
├── UNet.py
├── diffusion.py
├── classifier-free-diffusion.py   # or classifier_free_diffusion.py
├── train.py
└── hw2_data/
    └── digits/
        ├── mnistm/
        │   ├── data/
        │   └── train.csv          # image_name,label
        └── svhn/
            ├── data/
            └── train.csv          # image_name,label

Example
-------
python train_multicond.py \
    --data-root hw2_data/digits \
    --output-dir runs/cfg_digits_multicond \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --no-amp \
    --init-model-weights runs/cfg_digits_fp32/checkpoints/latest.pth
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image

from UNet import UNet

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


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
    """Combined MNIST-M and SVHN dataset with two condition labels."""

    def __init__(
        self,
        data_root: Path,
        image_size: int = 32,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.records: List[ImageRecord] = []

        self.records.extend(
            self._read_split(
                dataset_dir=data_root / "mnistm",
                dataset_label=MNISTM_ID,
            )
        )
        self.records.extend(
            self._read_split(
                dataset_dir=data_root / "svhn",
                dataset_label=SVHN_ID,
            )
        )

        if not self.records:
            raise RuntimeError(f"No training records found under {data_root}")

        # Do not use horizontal flips: a flipped digit may change its meaning.
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    @staticmethod
    def _read_split(dataset_dir: Path, dataset_label: int) -> List[ImageRecord]:
        csv_path = dataset_dir / "train.csv"
        image_dir = dataset_dir / "data"

        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        records: List[ImageRecord] = []

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {"image_name", "label"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"{csv_path} must contain columns {sorted(required_columns)}, "
                    f"got {reader.fieldnames}"
                )

            for row_index, row in enumerate(reader, start=2):
                image_name = str(row["image_name"]).strip()
                if not image_name:
                    raise ValueError(f"Empty image_name at {csv_path}:{row_index}")

                try:
                    digit_label = int(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid label at {csv_path}:{row_index}: {row['label']!r}"
                    ) from error

                # Some original SVHN annotations use 10 to represent digit zero.
                if digit_label == 10:
                    digit_label = 0

                if not 0 <= digit_label <= 9:
                    raise ValueError(
                        f"Digit label must be in 0..9 at {csv_path}:{row_index}, "
                        f"got {digit_label}"
                    )

                image_path = image_dir / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image referenced by CSV does not exist: {image_path}"
                    )

                records.append(
                    ImageRecord(
                        image_path=image_path,
                        digit_label=digit_label,
                        dataset_label=dataset_label,
                    )
                )

        return records

    def condition_counts(self) -> Counter:
        return Counter(
            (record.dataset_label, record.digit_label)
            for record in self.records
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]

        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        return (
            image_tensor,
            torch.tensor(record.digit_label, dtype=torch.long),
            torch.tensor(record.dataset_label, dtype=torch.long),
        )


class ConditionalUNet(nn.Module):
    """Add digit and dataset conditioning to the supplied UNet.

    The existing UNet already handles timestep conditioning. To avoid rewriting
    its long forward method, this wrapper converts both discrete conditions into
    learned spatial feature maps and concatenates them with x_t before passing
    the result to the original UNet.

    Interface required by CFGDiffusionImageGenerator:
        forward(x_t, timesteps, digit_labels, dataset_labels) -> epsilon_hat
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_dim: int = 16,
        num_digit_tokens: int = 11,
        num_dataset_tokens: int = 3,
    ) -> None:
        super().__init__()

        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")

        self.image_channels = image_channels
        self.condition_dim = condition_dim
        self.num_digit_tokens = num_digit_tokens
        self.num_dataset_tokens = num_dataset_tokens

        self.digit_embedding = nn.Embedding(num_digit_tokens, condition_dim)
        self.dataset_embedding = nn.Embedding(num_dataset_tokens, condition_dim)

        combined_dim = 2 * condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.SiLU(),
            nn.Linear(combined_dim, combined_dim),
        )

        self.backbone = UNet(in_channel=image_channels + combined_dim)

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        digit_labels: Tensor,
        dataset_labels: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = x_t.shape

        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [batch_size]")
        if digit_labels.shape != (batch_size,):
            raise ValueError("digit_labels must have shape [batch_size]")
        if dataset_labels.shape != (batch_size,):
            raise ValueError("dataset_labels must have shape [batch_size]")

        digit_condition = self.digit_embedding(digit_labels.long())
        dataset_condition = self.dataset_embedding(dataset_labels.long())
        condition = torch.cat([digit_condition, dataset_condition], dim=1)
        condition = self.condition_mlp(condition)

        condition_map = condition[:, :, None, None].expand(
            -1, -1, height, width
        )
        conditioned_input = torch.cat([x_t, condition_map], dim=1)

        return self.backbone(conditioned_input, timesteps)


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
    p_drop_digit_only: float
    p_drop_dataset_only: float
    condition_dim: int
    gradient_clip: float
    gradient_accumulation_steps: int
    save_every: int
    sample_every: int
    guidance_scale: float
    samples_per_condition: int
    balanced_sampling: bool
    amp: bool
    device: str
    resume: Optional[str]
    init_model_weights: Optional[str]


def load_cfg_class(project_dir: Path) -> Type[nn.Module]:
    """Load CFGDiffusionImageGenerator from either supported filename.

    The screenshot uses `classifier-free-diffusion.py`, whose hyphens prevent a
    normal Python import. Dynamic loading lets the script work without renaming
    that file. Renaming it to classifier_free_diffusion.py also works.
    """

    candidates = [
        project_dir / "classifier_free_diffusion_multicond.py",
        project_dir / "classifier_free_diffusion.py",
        project_dir / "classifier-free-diffusion.py",
    ]

    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        searched = ", ".join(str(path.name) for path in candidates)
        raise FileNotFoundError(
            f"Could not find the CFG module. Expected one of: {searched}"
        )

    spec = importlib.util.spec_from_file_location(
        "cfg_diffusion_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CFGDiffusionImageGenerator"):
        raise AttributeError(
            f"{module_path} does not define CFGDiffusionImageGenerator"
        )

    return module.CFGDiffusionImageGenerator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_balanced_sampler(
    dataset: DigitDomainDataset,
    seed: int,
) -> WeightedRandomSampler:
    """Balance all 20 (dataset, digit) condition pairs by oversampling."""

    counts = dataset.condition_counts()
    weights = [
        1.0 / counts[(record.dataset_label, record.digit_label)]
        for record in dataset.records
    ]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def save_checkpoint(
    path: Path,
    diffusion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # The inherited load_checkpoint() recognizes this key.
        "model_state_dict": diffusion.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": asdict(config),
    }
    torch.save(checkpoint, path)


def resume_checkpoint(
    path: Path,
    diffusion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location=diffusion.device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} has no model_state_dict")

    diffusion.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, global_step


def initialize_model_weights(
    path: Path,
    diffusion: nn.Module,
) -> Tuple[int, int]:
    """Load model weights only and intentionally reset optimizer/scaler.

    This is used to continue from the old jointly-dropped CFG checkpoint while
    starting a new training phase with partial-condition dropout. The model
    architecture is unchanged, so strict loading is expected to succeed.

    Returns the source checkpoint's epoch and global_step for logging.
    """
    checkpoint = torch.load(path, map_location=diffusion.device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must be a dictionary")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned_state_dict[new_key] = value

    diffusion.model.load_state_dict(cleaned_state_dict, strict=True)
    source_epoch = int(checkpoint.get("epoch", -1))
    source_global_step = int(checkpoint.get("global_step", 0))
    return source_epoch, source_global_step


def save_condition_grid(
    diffusion: nn.Module,
    output_path: Path,
    samples_per_condition: int,
    guidance_scale: float,
) -> None:
    """Generate all 20 (dataset, digit) combinations and save one grid.

    Rows correspond to datasets:
      first rows: MNIST-M digits 0..9
      next rows:  SVHN digits 0..9

    With samples_per_condition > 1, each condition is repeated consecutively.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, digit_labels, dataset_labels = diffusion.sample_all_20_conditions(
        samples_per_condition=samples_per_condition,
        guidance_scale=guidance_scale,
        return_labels=True,
    )
    samples_01 = diffusion.denormalize(samples)

    # 10 digits per dataset; repetitions remain adjacent.
    nrow = 10 * samples_per_condition
    save_image(samples_01, output_path, nrow=nrow)

    metadata_path = output_path.with_suffix(".json")
    metadata = [
        {
            "index": index,
            "digit": int(digit),
            "dataset": "mnistm" if int(dataset_id) == MNISTM_ID else "svhn",
        }
        for index, (digit, dataset_id) in enumerate(
            zip(digit_labels.detach().cpu(), dataset_labels.detach().cpu())
        )
    ]
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def print_dataset_summary(dataset: DigitDomainDataset) -> None:
    counts = dataset.condition_counts()

    print(f"Total images: {len(dataset):,}")
    for dataset_id, dataset_name in [(MNISTM_ID, "MNIST-M"), (SVHN_ID, "SVHN")]:
        row = [counts.get((dataset_id, digit), 0) for digit in range(10)]
        print(f"{dataset_name:7s}: {row}  total={sum(row):,}")


def train(config: TrainingConfig) -> None:
    project_dir = Path(__file__).resolve().parent
    data_root = Path(config.data_root)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = DigitDomainDataset(
        data_root=data_root,
        image_size=config.image_size,
    )
    print_dataset_summary(dataset)

    sampler = None
    shuffle = True
    if config.balanced_sampling:
        sampler = build_balanced_sampler(dataset, config.seed)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )

    conditional_unet = ConditionalUNet(
        image_channels=3,
        condition_dim=config.condition_dim,
        num_digit_tokens=11,
        num_dataset_tokens=3,
    )

    CFGDiffusionImageGenerator = load_cfg_class(project_dir)
    diffusion = CFGDiffusionImageGenerator(
        conditional_model=conditional_unet,
        image_size=config.image_size,
        channels=3,
        n_timesteps=config.n_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=str(device),
        p_uncond=config.p_uncond,
        p_drop_digit_only=config.p_drop_digit_only,
        p_drop_dataset_only=config.p_drop_dataset_only,
        n_digit_classes=10,
        n_dataset_classes=2,
        null_digit_idx=NULL_DIGIT_ID,
        null_dataset_idx=NULL_DATASET_ID,
    )

    # Your current CFG class refers to this DDPM coefficient during sampling,
    # while diffusion.py does not register it. Add it here so preview sampling
    # and later inference work without modifying either existing file.
    if not hasattr(diffusion, "sqrt_recip_alphas"):
        diffusion.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt(1.0 / diffusion.alphas),
        )

    diffusion.set_seed(config.seed)

    optimizer = torch.optim.AdamW(
        diffusion.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    use_amp = config.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # compatibility with older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0

    if config.init_model_weights is not None:
        init_path = Path(config.init_model_weights)
        source_epoch, source_global_step = initialize_model_weights(
            init_path,
            diffusion,
        )
        global_step = source_global_step
        print(
            f"Initialized model weights from {init_path}: "
            f"source_epoch={source_epoch + 1}, "
            f"source_global_step={source_global_step}. "
            "Optimizer and GradScaler were reset."
        )

    elif config.resume is not None:
        resume_path = Path(config.resume)
        start_epoch, global_step = resume_checkpoint(
            resume_path,
            diffusion,
            optimizer,
            scaler,
        )
        print(
            f"Resumed from {resume_path}: "
            f"start_epoch={start_epoch}, global_step={global_step}"
        )

    probabilities = diffusion.condition_state_probabilities()
    print(
        "Condition-state probabilities: "
        f"full={probabilities['full']:.3f}, "
        f"dataset_only={probabilities['dataset_only']:.3f}, "
        f"digit_only={probabilities['digit_only']:.3f}, "
        f"unconditional={probabilities['unconditional']:.3f}"
    )

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    log_path = output_dir / "train_log.csv"
    if start_epoch == 0 or not log_path.exists():
        log_path.write_text(
            "epoch,global_step,mean_loss,learning_rate,seconds\n",
            encoding="utf-8",
        )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, config.epochs):
        diffusion.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen_batches = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            leave=True,
        )

        for batch_index, (images, digit_labels, dataset_labels) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            digit_labels = digit_labels.to(device, non_blocking=True)
            dataset_labels = dataset_labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                loss = diffusion.training_loss(
                    x_0=images,
                    digit_labels=digit_labels,
                    dataset_labels=dataset_labels,
                )
                scaled_loss = loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            should_step = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(loader)
            )

            if should_step:
                scaler.unscale_(optimizer)
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        diffusion.model.parameters(),
                        max_norm=config.gradient_clip,
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().item())
            seen_batches += 1

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mean=f"{running_loss / seen_batches:.4f}",
                    step=global_step,
                )

        mean_loss = running_loss / max(seen_batches, 1)
        elapsed = time.time() - epoch_start
        learning_rate = optimizer.param_groups[0]["lr"]

        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                f"{epoch},{global_step},{mean_loss:.8f},"
                f"{learning_rate:.10g},{elapsed:.3f}\n"
            )

        print(
            f"Epoch {epoch + 1}: mean_loss={mean_loss:.6f}, "
            f"time={elapsed:.1f}s"
        )

        epoch_number = epoch + 1

        if epoch_number % config.save_every == 0 or epoch_number == config.epochs:
            epoch_checkpoint = checkpoint_dir / f"epoch_{epoch_number:04d}.pth"
            latest_checkpoint = checkpoint_dir / "latest.pth"

            save_checkpoint(
                epoch_checkpoint,
                diffusion,
                optimizer,
                scaler,
                epoch,
                global_step,
                config,
            )
            save_checkpoint(
                latest_checkpoint,
                diffusion,
                optimizer,
                scaler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved checkpoint: {epoch_checkpoint}")

        if config.sample_every > 0 and (
            epoch_number % config.sample_every == 0
            or epoch_number == config.epochs
        ):
            diffusion.eval()
            sample_path = sample_dir / f"epoch_{epoch_number:04d}.png"
            save_condition_grid(
                diffusion=diffusion,
                output_path=sample_path,
                samples_per_condition=config.samples_per_condition,
                guidance_scale=config.guidance_scale,
            )
            print(f"Saved sample grid: {sample_path}")

    print(f"Training complete. Outputs are in: {output_dir}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a digit+dataset conditioned CFG DDPM on MNIST-M and SVHN."
    )

    parser.add_argument("--data-root", type=str, default="hw2_data/digits")
    parser.add_argument("--output-dir", type=str, default="runs/cfg_digits")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument(
        "--p-uncond",
        type=float,
        default=0.1,
        help="Probability of dropping both digit and dataset conditions.",
    )
    parser.add_argument(
        "--p-drop-digit-only",
        type=float,
        default=0.1,
        help="Probability of replacing digit with null while keeping dataset.",
    )
    parser.add_argument(
        "--p-drop-dataset-only",
        type=float,
        default=0.1,
        help="Probability of replacing dataset with null while keeping digit.",
    )
    parser.add_argument("--condition-dim", type=int, default=16)

    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Set to 0 to disable slow 1000-step preview sampling during training.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--samples-per-condition", type=int, default=1)

    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance all 20 (dataset, digit) condition combinations.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when CUDA is available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-model-weights",
        type=str,
        default=None,
        help=(
            "Load only model_state_dict from an existing checkpoint, reset "
            "optimizer/scaler, and train --epochs new epochs. Use this for "
            "the new multi-condition CFG phase."
        ),
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    for name, value in (
        ("--p-uncond", args.p_uncond),
        ("--p-drop-digit-only", args.p_drop_digit_only),
        ("--p-drop-dataset-only", args.p_drop_dataset_only),
    ):
        if not 0.0 <= value < 1.0:
            parser.error(f"{name} must satisfy 0 <= probability < 1")
    if (
        args.p_uncond
        + args.p_drop_digit_only
        + args.p_drop_dataset_only
        >= 1.0
    ):
        parser.error(
            "--p-uncond + --p-drop-digit-only + "
            "--p-drop-dataset-only must be less than 1"
        )
    if args.resume is not None and args.init_model_weights is not None:
        parser.error(
            "Use either --resume or --init-model-weights, not both"
        )
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.sample_every < 0:
        parser.error("--sample-every must be non-negative")
    if args.samples_per_condition <= 0:
        parser.error("--samples-per-condition must be positive")

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
        p_drop_digit_only=args.p_drop_digit_only,
        p_drop_dataset_only=args.p_drop_dataset_only,
        condition_dim=args.condition_dim,
        gradient_clip=args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
        sample_every=args.sample_every,
        guidance_scale=args.guidance_scale,
        samples_per_condition=args.samples_per_condition,
        balanced_sampling=args.balanced_sampling,
        amp=args.amp,
        device=args.device,
        resume=args.resume,
        init_model_weights=args.init_model_weights,
    )


if __name__ == "__main__":
    train(parse_args())