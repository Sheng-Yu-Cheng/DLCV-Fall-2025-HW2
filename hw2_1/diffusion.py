"""DDPM wrapper for the provided UNet.py.

The UNet is trained to predict epsilon from (x_t, t).
Images should be float tensors normalized to [-1, 1].
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.utils import save_image

from UNet import UNet


class DiffusionImageGenerator(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM).

    This class owns:
      1. the noise-prediction UNet epsilon_theta(x_t, t),
      2. the forward diffusion schedule q(x_t | x_0),
      3. the training objective,
      4. the reverse sampling procedure.

    The provided UNet has five downsampling stages, so image_size=32 is the
    safe choice. A 28x28 image can be resized to 32x32 for training, and the
    generated image can later be resized back to 28x28 by the classifier.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        image_size: int = 32,
        channels: int = 3,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError("Require 0 < beta_start < beta_end < 1")

        self.image_size = int(image_size)
        self.channels = int(channels)
        self.n_timesteps = int(n_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # epsilon_theta(x_t, t): predicts the Gaussian noise added to x_0.
        self.model = UNet(in_channel=self.channels)

        # Linear beta schedule from the original DDPM paper.
        betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.n_timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0
        )

        # Buffers move with .to(device), enter state_dict, and are not trained.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer(
            "sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars)
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas)
        )

        # q(x_{t-1} | x_t, x_0) posterior.
        posterior_variance = (
            betas
            * (1.0 - alpha_bars_prev)
            / (1.0 - alpha_bars)
        )
        posterior_mean_coef1 = (
            betas
            * torch.sqrt(alpha_bars_prev)
            / (1.0 - alpha_bars)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_bars_prev)
            * torch.sqrt(alphas)
            / (1.0 - alpha_bars)
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        self.to(resolved_device)

        # A dedicated RNG makes sampling repeatable after set_seed().
        self.generator = torch.Generator(device=self.device)
        self.set_seed(0)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    @property
    def device(self) -> torch.device:
        """Current device of the UNet parameters."""
        return next(self.model.parameters()).device

    def set_seed(self, seed: int) -> None:
        """Seed Python, NumPy, PyTorch, and this sampler's RNG."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.generator.manual_seed(seed)

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, x_shape: torch.Size) -> Tensor:
        """Gather one schedule value per batch item and reshape for broadcast.

        values:     [T]
        timesteps:  [B]
        return:     [B, 1, 1, 1] for image tensors
        """
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch_size]")
        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must have dtype torch.long")

        gathered = values.gather(0, timesteps)
        return gathered.reshape(
            timesteps.shape[0], *([1] * (len(x_shape) - 1))
        )

    def _randn_like(self, x: Tensor) -> Tensor:
        return torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self.generator,
        )

    def _validate_images(self, images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if images.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {images.shape[1]}"
            )
        if images.device != self.device:
            raise ValueError(
                f"Images are on {images.device}, but model is on {self.device}"
            )
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors")

    def forward(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        """Predict epsilon from a noisy image x_t and timestep t."""
        self._validate_images(x_t)
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape [batch_size]")
        if timesteps.device != x_t.device:
            raise ValueError("timesteps and x_t must be on the same device")
        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()
        return self.model(x_t, timesteps)

    def sample_random_noise(self, batch_size: int = 1) -> Tensor:
        """Sample x_T ~ N(0, I), shape [B, C, H, W]."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.randn(
            (
                batch_size,
                self.channels,
                self.image_size,
                self.image_size,
            ),
            device=self.device,
            dtype=torch.float32,
            generator=self.generator,
        )

    def q_sample(
        self,
        x_0: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample x_t directly from q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0
            + sqrt(1 - alpha_bar_t) * epsilon
        """
        self._validate_images(x_0)
        if timesteps.shape != (x_0.shape[0],):
            raise ValueError("timesteps must have shape [batch_size]")
        timesteps = timesteps.to(device=x_0.device, dtype=torch.long)

        if noise is None:
            noise = self._randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError("noise and x_0 must have the same shape")

        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars, timesteps, x_0.shape
        )
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, x_0.shape
        )

        return (
            sqrt_alpha_bar_t * x_0
            + sqrt_one_minus_alpha_bar_t * noise
        )

    def training_loss(
        self,
        x_0: Tensor,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute DDPM's simplified epsilon-prediction MSE objective."""
        self._validate_images(x_0)
        batch_size = x_0.shape[0]

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.n_timesteps,
                (batch_size,),
                device=x_0.device,
                dtype=torch.long,
            )
        else:
            timesteps = timesteps.to(device=x_0.device, dtype=torch.long)

        if noise is None:
            noise = self._randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError("noise and x_0 must have the same shape")

        x_t = self.q_sample(x_0, timesteps, noise)
        predicted_noise = self(x_t, timesteps)
        return F.mse_loss(predicted_noise, noise, reduction=reduction)

    def predict_x0_from_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Tensor,
    ) -> Tensor:
        """Recover the model's estimate of x_0 from epsilon prediction."""
        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars, timesteps, x_t.shape
        )
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, x_t.shape
        )
        return (
            x_t - sqrt_one_minus_alpha_bar_t * predicted_noise
        ) / sqrt_alpha_bar_t.clamp(min=1e-12)

    def p_mean_variance(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute p_theta(x_{t-1} | x_t).

        Returns:
            model_mean, model_variance, predicted_x0, predicted_noise
        """
        predicted_noise = self(x_t, timesteps)
        predicted_x0 = self.predict_x0_from_noise(
            x_t, timesteps, predicted_noise
        )

        if clip_denoised:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

        coef1_t = self._extract(
            self.posterior_mean_coef1, timesteps, x_t.shape
        )
        coef2_t = self._extract(
            self.posterior_mean_coef2, timesteps, x_t.shape
        )
        model_mean = coef1_t * predicted_x0 + coef2_t * x_t

        model_variance = self._extract(
            self.posterior_variance, timesteps, x_t.shape
        )

        return model_mean, model_variance, predicted_x0, predicted_noise

    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        clip_denoised: bool = True,
    ) -> Tensor:
        """Perform one reverse step x_t -> x_{t-1}."""
        model_mean, model_variance, _, _ = self.p_mean_variance(
            x_t, timesteps, clip_denoised=clip_denoised
        )

        noise = self._randn_like(x_t)
        nonzero_mask = (timesteps != 0).to(x_t.dtype).reshape(
            x_t.shape[0], *([1] * (x_t.ndim - 1))
        )

        return (
            model_mean
            + nonzero_mask
            * torch.sqrt(model_variance.clamp(min=1e-20))
            * noise
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        clip_denoised: bool = True,
        return_all_steps: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Sequence[Tensor]]]:
        """Generate images by running x_T -> ... -> x_0.

        Returned final images remain in model space, approximately [-1, 1].
        Use denormalize() before saving or displaying them.
        """
        was_training = self.training
        self.eval()

        x = self.sample_random_noise(batch_size)
        trajectory = [x.detach().cpu()] if return_all_steps else None

        for timestep in reversed(range(self.n_timesteps)):
            t = torch.full(
                (batch_size,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            x = self.p_sample(x, t, clip_denoised=clip_denoised)
            if trajectory is not None:
                trajectory.append(x.detach().cpu())

        if was_training:
            self.train()

        if trajectory is not None:
            return x, trajectory
        return x

    @staticmethod
    def normalize(images: Tensor) -> Tensor:
        """Convert [0, 1] float images to [-1, 1]."""
        return images * 2.0 - 1.0

    @staticmethod
    def denormalize(images: Tensor) -> Tensor:
        """Convert model-space images from [-1, 1] to [0, 1]."""
        return ((images.clamp(-1.0, 1.0) + 1.0) / 2.0)

    @torch.no_grad()
    def save_samples(
        self,
        output_dir: Union[str, Path],
        batch_size: int,
        prefix: str = "sample",
        start_index: int = 0,
    ) -> Tensor:
        """Generate and save individual PNG files.

        This UNet is unconditional. The prefix is only a filename prefix; it
        does not tell the model which digit to generate.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = self.sample(batch_size=batch_size)
        assert isinstance(samples, Tensor)
        samples_01 = self.denormalize(samples)

        for index, image in enumerate(samples_01):
            filename = f"{prefix}_{start_index + index:05d}.png"
            save_image(image, output_path / filename)

        return samples

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights, configuration, and optional optimizer state."""
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "image_size": self.image_size,
                "channels": self.channels,
                "n_timesteps": self.n_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
            },
            "epoch": epoch,
            "global_step": global_step,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            checkpoint["extra"] = extra

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load several common checkpoint formats.

        Supported formats:
          - {"model_state_dict": ...}
          - {"state_dict": ...}
          - raw UNet state_dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a state_dict or checkpoint dict")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove common wrappers such as DataParallel's `module.` and a
        # whole-diffusion checkpoint's `model.` prefix.
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            cleaned_state_dict[new_key] = value

        self.model.load_state_dict(cleaned_state_dict, strict=strict)
        return checkpoint


def example_training_step() -> None:
    """A small usage example; this is not a full dataset training script."""
    diffusion = DiffusionImageGenerator(
        image_size=32,
        channels=3,
        n_timesteps=1000,
    )
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

    # Replace this with a real DataLoader batch normalized to [-1, 1].
    images = torch.rand(2, 3, 32, 32, device=diffusion.device)
    images = diffusion.normalize(images)

    diffusion.train()
    optimizer.zero_grad(set_to_none=True)
    loss = diffusion.training_loss(images)
    loss.backward()
    optimizer.step()

    print(f"training loss: {loss.item():.6f}")


if __name__ == "__main__":
    # Runs one sanity-check training step. It does not train a useful model.
    example_training_step()
