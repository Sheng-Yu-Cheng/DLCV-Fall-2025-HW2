import argparse
import os
import random
from typing import List

from utils import beta_scheduler

import torch
from torchvision.utils import save_image

from UNet import UNet


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def load_model(model_path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(model_path, map_location=device)

    # Robust loading for common checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip common prefixes if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_state_dict[nk] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def make_ddim_timesteps(num_ddim_timesteps: int = 50, num_ddpm_timesteps: int = 1000) -> List[int]:
    """
    Uniform time-step scheduler required by the homework.
    Returns descending timesteps: 981, 961, ..., 21, 1
    """
    if num_ddpm_timesteps % num_ddim_timesteps != 0:
        c = num_ddpm_timesteps // num_ddim_timesteps
    else:
        c = num_ddpm_timesteps // num_ddim_timesteps

    # 0, 20, 40, ..., 980 -> +1 => 1, 21, ..., 981 -> reverse for sampling
    timesteps = list(range(0, num_ddpm_timesteps, c))[:num_ddim_timesteps]
    timesteps = [t + 1 for t in timesteps]
    timesteps = timesteps[::-1]
    return timesteps


def load_noises(noise_dir: str, device: torch.device) -> torch.Tensor:
    xs = []
    for i in range(10):
        path = os.path.join(noise_dir, f"{i:02d}.pt")
        x = torch.load(path, map_location=device)

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a tensor in {path}, got {type(x)}")

        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected tensor shape in {path}: {tuple(x.shape)}")

        xs.append(x)

    return torch.cat(xs, dim=0).to(device=device, dtype=torch.float32)


@torch.no_grad()
def ddim_sample(
    model: UNet,
    x: torch.Tensor,
    alpha_bars: torch.Tensor,
    timesteps: List[int],
    eta: float = 0.0,
) -> torch.Tensor:
    """
    DDIM sampling.
    x: initial noise x_T, shape [B, C, H, W]
    alpha_bars: cumulative product of alphas, shape [1000]
    timesteps: descending list, e.g. [981, 961, ..., 21, 1]
    """
    device = x.device
    batch_size = x.shape[0]

    for idx, t in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps = model(x, t_tensor)

        alpha_t = alpha_bars[t].to(dtype=x.dtype)

        if idx + 1 < len(timesteps):
            prev_t = timesteps[idx + 1]
            alpha_prev = alpha_bars[prev_t].to(dtype=x.dtype)
        else:
            # Final step: go from t=1 to the previous selected step t=0
            alpha_prev = alpha_bars[0].to(dtype=x.dtype)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        # Predicted clean image x0
        pred_x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        # DDIM variance term
        if eta > 0.0:
            sigma_t = eta * torch.sqrt(
                ((1.0 - alpha_prev) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_prev)
            )
            noise = torch.randn_like(x)
        else:
            sigma_t = torch.zeros((), device=device, dtype=x.dtype)
            noise = torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0)) * eps
        x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise

    return x


def save_outputs(images: torch.Tensor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0

    for i in range(images.shape[0]):
        out_path = os.path.join(output_dir, f"{i:02d}.png")
        save_image(images[i], out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="HW2-2 DDIM inference")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to noise directory containing 00.pt ~ 09.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save 00.png ~ 09.png")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained UNet.pt")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta. Use 0.0 for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    betas = beta_scheduler().to(device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    timesteps = make_ddim_timesteps(num_ddim_timesteps=50, num_ddpm_timesteps=1000)

    x = load_noises(args.noise_dir, device)
    images = ddim_sample(model, x, alpha_bars, timesteps, eta=args.eta)
    save_outputs(images, args.output_dir)

    print(f"Saved {images.shape[0]} images to {args.output_dir}")


if __name__ == "__main__":
    main()