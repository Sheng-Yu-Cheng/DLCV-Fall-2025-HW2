import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _pil_resampling(name):
    # Pillow >= 9 uses Image.Resampling; older Pillow exposes constants directly.
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


class Fill50KDataset(Dataset):
    """
    Fill50k JSONL dataset used by DLCV HW2-3.

    Expected root:
        training/
          source/
          target/
          prompt.json

    Each line in prompt.json:
        {"source": "source/0.png",
         "target": "target/0.png",
         "prompt": "..."}

    Returned tensors:
        target: float32 CHW in [-1, 1]
        hint:   float32 CHW in [0, 1]
        prompt: str
    """

    def __init__(self, root, image_size=512, max_samples=None):
        self.root = Path(root)
        self.image_size = int(image_size)

        prompt_path = self.root / "prompt.json"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"prompt.json not found: {prompt_path}")

        records = []
        with prompt_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for key in ("source", "target", "prompt"):
                    if key not in item:
                        raise KeyError(
                            f"{prompt_path}:{line_no} is missing key '{key}'"
                        )
                records.append(item)

        if max_samples is not None:
            records = records[: int(max_samples)]

        if len(records) == 0:
            raise RuntimeError(f"No samples found in {prompt_path}")

        self.records = records

    def __len__(self):
        return len(self.records)

    def _load_rgb(self, rel_path, is_control):
        path = self.root / rel_path
        if not path.is_file():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")

        if image.size != (self.image_size, self.image_size):
            # Target is a natural RGB image; use bicubic.
            # Control image is geometric; bilinear preserves edges without
            # introducing the blockiness of nearest-neighbor resize.
            resample = _pil_resampling("BILINEAR" if is_control else "BICUBIC")
            image = image.resize(
                (self.image_size, self.image_size),
                resample=resample,
            )

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        if is_control:
            return tensor

        return tensor * 2.0 - 1.0

    def __getitem__(self, index):
        item = self.records[index]

        hint = self._load_rgb(item["source"], is_control=True)
        target = self._load_rgb(item["target"], is_control=False)

        return {
            "hint": hint,
            "target": target,
            "prompt": item["prompt"],
            "source_name": item["source"],
            "target_name": item["target"],
        }