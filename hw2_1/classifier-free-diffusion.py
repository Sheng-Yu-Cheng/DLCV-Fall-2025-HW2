from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from diffusion import DiffusionImageGenerator


LabelLike = Union[int, Sequence[int], torch.Tensor]


class CFGDiffusionImageGenerator(DiffusionImageGenerator):
    """Two-condition classifier-free-guidance diffusion model.

    Conditions
    ----------
    digit:
        0..9, null token = 10
    dataset:
        0 = MNIST-M, 1 = SVHN, null token = 2

    Training uses four condition states:
        1. (digit, dataset)
        2. (null_digit, dataset)
        3. (digit, null_dataset)
        4. (null_digit, null_dataset)

    Sampling supports hierarchical guidance:

        eps = eps_uncond
            + dataset_scale * (eps_dataset - eps_uncond)
            + digit_scale * (eps_full - eps_dataset)

    where:
        eps_uncond = model(x_t, t, null_digit, null_dataset)
        eps_dataset = model(x_t, t, null_digit, dataset)
        eps_full = model(x_t, t, digit, dataset)

    Setting digit_scale == dataset_scale == s is exactly the ordinary CFG
    expression eps_uncond + s * (eps_full - eps_uncond).
    """

    DATASET_MNISTM = 0
    DATASET_SVHN = 1

    def __init__(
        self,
        conditional_model=None,
        checkpoint_path: Optional[str] = None,
        image_size: int = 32,
        channels: int = 3,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[str] = None,
        p_uncond: float = 0.1,
        p_drop_digit_only: float = 0.1,
        p_drop_dataset_only: float = 0.1,
        n_digit_classes: int = 10,
        n_dataset_classes: int = 2,
        null_digit_idx: int = 10,
        null_dataset_idx: int = 2,
    ) -> None:
        super().__init__(
            checkpoint_path=None,
            image_size=image_size,
            channels=channels,
            n_timesteps=n_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )

        for name, value in {
            "p_uncond": p_uncond,
            "p_drop_digit_only": p_drop_digit_only,
            "p_drop_dataset_only": p_drop_dataset_only,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= {name} < 1")

        probability_sum = (
            p_uncond + p_drop_digit_only + p_drop_dataset_only
        )
        if probability_sum >= 1.0:
            raise ValueError(
                "p_uncond + p_drop_digit_only + "
                "p_drop_dataset_only must be less than 1"
            )

        self.n_timesteps = int(n_timesteps)
        self.p_uncond = float(p_uncond)
        self.p_drop_digit_only = float(p_drop_digit_only)
        self.p_drop_dataset_only = float(p_drop_dataset_only)
        self.p_keep_full = 1.0 - probability_sum

        self.n_digit_classes = int(n_digit_classes)
        self.n_dataset_classes = int(n_dataset_classes)
        self.null_digit_idx = int(null_digit_idx)
        self.null_dataset_idx = int(null_dataset_idx)

        if conditional_model is not None:
            self.model = conditional_model.to(self.device)

        # diffusion.py does not register this coefficient, while the DDPM
        # epsilon-parameterized reverse mean below uses it.
        if not hasattr(self, "sqrt_recip_alphas"):
            self.register_buffer(
                "sqrt_recip_alphas",
                torch.sqrt(1.0 / self.alphas),
            )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def condition_state_probabilities(self) -> dict[str, float]:
        return {
            "full": self.p_keep_full,
            "dataset_only": self.p_drop_digit_only,
            "digit_only": self.p_drop_dataset_only,
            "unconditional": self.p_uncond,
        }

    def _make_null_digit_labels(
        self,
        batch_size: int,
        device=None,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            self.null_digit_idx,
            dtype=torch.long,
            device=device or self.device,
        )

    def _make_null_dataset_labels(
        self,
        batch_size: int,
        device=None,
    ) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            self.null_dataset_idx,
            dtype=torch.long,
            device=device or self.device,
        )

    def _to_label_tensor(
        self,
        labels: Optional[LabelLike],
        batch_size: int,
        null_value: int,
        name: str,
        device=None,
    ) -> torch.Tensor:
        device = device or self.device

        if labels is None:
            return torch.full(
                (batch_size,),
                null_value,
                dtype=torch.long,
                device=device,
            )

        if isinstance(labels, int):
            return torch.full(
                (batch_size,),
                labels,
                dtype=torch.long,
                device=device,
            )

        if isinstance(labels, (list, tuple)):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported type for {name}: {type(labels)}")

        labels = labels.reshape(-1)
        if labels.numel() == 1 and batch_size > 1:
            labels = labels.repeat(batch_size)
        if labels.numel() != batch_size:
            raise ValueError(
                f"{name} must have length 1 or batch_size={batch_size}, "
                f"but got {labels.numel()}"
            )
        return labels

    def _drop_conditions(
        self,
        digit_labels: torch.Tensor,
        dataset_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one of four condition states for every training sample.

        The categorical intervals are:
            [0, p_drop_digit_only):             dataset-only
            next p_drop_dataset_only:            digit-only
            next p_uncond:                       unconditional
            remaining probability:               full condition
        """
        batch_size = digit_labels.shape[0]
        device = digit_labels.device
        random_values = torch.rand(batch_size, device=device)

        digit_cutoff = self.p_drop_digit_only
        dataset_cutoff = digit_cutoff + self.p_drop_dataset_only
        both_cutoff = dataset_cutoff + self.p_uncond

        drop_digit_only = random_values < digit_cutoff
        drop_dataset_only = (
            (random_values >= digit_cutoff)
            & (random_values < dataset_cutoff)
        )
        drop_both = (
            (random_values >= dataset_cutoff)
            & (random_values < both_cutoff)
        )

        conditioned_digits = digit_labels.clone()
        conditioned_datasets = dataset_labels.clone()

        conditioned_digits[drop_digit_only | drop_both] = self.null_digit_idx
        conditioned_datasets[
            drop_dataset_only | drop_both
        ] = self.null_dataset_idx

        return conditioned_digits, conditioned_datasets

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: torch.Tensor,
        dataset_labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            x_t,
            timesteps,
            digit_labels,
            dataset_labels,
        )

    def training_loss(
        self,
        x_0: torch.Tensor,
        digit_labels: LabelLike,
        dataset_labels: LabelLike,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x_0.shape[0]
        device = x_0.device

        digit_labels = self._to_label_tensor(
            digit_labels,
            batch_size,
            self.null_digit_idx,
            "digit_labels",
            device,
        )
        dataset_labels = self._to_label_tensor(
            dataset_labels,
            batch_size,
            self.null_dataset_idx,
            "dataset_labels",
            device,
        )

        conditioned_digits, conditioned_datasets = self._drop_conditions(
            digit_labels,
            dataset_labels,
        )

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.n_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, timesteps, noise)
        predicted_noise = self(
            x_t,
            timesteps,
            conditioned_digits,
            conditioned_datasets,
        )
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def predict_noise_hierarchical_cfg(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: LabelLike,
        dataset_labels: LabelLike,
        digit_guidance_scale: float = 1.0,
        dataset_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict epsilon with separately controllable dataset/digit CFG."""
        batch_size = x_t.shape[0]
        device = x_t.device

        digits = self._to_label_tensor(
            digit_labels,
            batch_size,
            self.null_digit_idx,
            "digit_labels",
            device,
        )
        datasets = self._to_label_tensor(
            dataset_labels,
            batch_size,
            self.null_dataset_idx,
            "dataset_labels",
            device,
        )
        null_digits = self._make_null_digit_labels(batch_size, device)
        null_datasets = self._make_null_dataset_labels(batch_size, device)

        eps_full = self(x_t, timesteps, digits, datasets)
        eps_dataset = self(x_t, timesteps, null_digits, datasets)
        eps_uncond = self(x_t, timesteps, null_digits, null_datasets)

        return (
            eps_uncond
            + dataset_guidance_scale * (eps_dataset - eps_uncond)
            + digit_guidance_scale * (eps_full - eps_dataset)
        )

    @torch.no_grad()
    def predict_noise_cfg(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: LabelLike,
        dataset_labels: LabelLike,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """Backward-compatible ordinary CFG shorthand.

        Equal hierarchical scales collapse algebraically to ordinary CFG.
        """
        return self.predict_noise_hierarchical_cfg(
            x_t=x_t,
            timesteps=timesteps,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            digit_guidance_scale=guidance_scale,
            dataset_guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: Optional[LabelLike] = None,
        dataset_labels: Optional[LabelLike] = None,
        guidance_scale: float = 3.0,
        digit_guidance_scale: Optional[float] = None,
        dataset_guidance_scale: Optional[float] = None,
    ):
        if digit_guidance_scale is None:
            digit_guidance_scale = guidance_scale
        if dataset_guidance_scale is None:
            dataset_guidance_scale = guidance_scale

        if digit_labels is None:
            digit_labels = self._make_null_digit_labels(
                x_t.shape[0],
                x_t.device,
            )
        if dataset_labels is None:
            dataset_labels = self._make_null_dataset_labels(
                x_t.shape[0],
                x_t.device,
            )

        predicted_noise = self.predict_noise_hierarchical_cfg(
            x_t=x_t,
            timesteps=timesteps,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            digit_guidance_scale=digit_guidance_scale,
            dataset_guidance_scale=dataset_guidance_scale,
        )

        beta_t = self._extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            x_t.shape,
        )
        sqrt_recip_alpha_t = self._extract(
            self.sqrt_recip_alphas,
            timesteps,
            x_t.shape,
        )

        model_mean = sqrt_recip_alpha_t * (
            x_t
            - beta_t
            * predicted_noise
            / sqrt_one_minus_alpha_bar_t
        )
        model_variance = self._extract(
            self.posterior_variance,
            timesteps,
            x_t.shape,
        )
        return model_mean, model_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: Optional[LabelLike] = None,
        dataset_labels: Optional[LabelLike] = None,
        guidance_scale: float = 3.0,
        digit_guidance_scale: Optional[float] = None,
        dataset_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        model_mean, model_variance = self.p_mean_variance(
            x_t=x_t,
            timesteps=timesteps,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            guidance_scale=guidance_scale,
            digit_guidance_scale=digit_guidance_scale,
            dataset_guidance_scale=dataset_guidance_scale,
        )

        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=self.generator,
        )
        nonzero_mask = (timesteps != 0).to(x_t.dtype).reshape(
            x_t.shape[0],
            *([1] * (x_t.ndim - 1)),
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
        batch_size: Optional[int] = None,
        digit_labels: Optional[LabelLike] = None,
        dataset_labels: Optional[LabelLike] = None,
        guidance_scale: float = 3.0,
        digit_guidance_scale: Optional[float] = None,
        dataset_guidance_scale: Optional[float] = None,
        return_labels: bool = False,
    ):
        if batch_size is None:
            for value in (digit_labels, dataset_labels):
                if isinstance(value, torch.Tensor):
                    batch_size = value.numel()
                    break
                if isinstance(value, (list, tuple)):
                    batch_size = len(value)
                    break
            else:
                batch_size = 1

        digits = self._to_label_tensor(
            digit_labels,
            batch_size,
            self.null_digit_idx,
            "digit_labels",
            self.device,
        )
        datasets = self._to_label_tensor(
            dataset_labels,
            batch_size,
            self.null_dataset_idx,
            "dataset_labels",
            self.device,
        )

        was_training = self.training
        self.eval()
        x = self.sample_random_noise(batch_size)

        for timestep in reversed(range(self.n_timesteps)):
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            x = self.p_sample(
                x_t=x,
                timesteps=timesteps,
                digit_labels=digits,
                dataset_labels=datasets,
                guidance_scale=guidance_scale,
                digit_guidance_scale=digit_guidance_scale,
                dataset_guidance_scale=dataset_guidance_scale,
            )

        if was_training:
            self.train()

        if return_labels:
            return x, digits, datasets
        return x

    @torch.no_grad()
    def sample_all_digits_for_dataset(
        self,
        dataset_label: int,
        samples_per_digit: int = 1,
        guidance_scale: float = 3.0,
        digit_guidance_scale: Optional[float] = None,
        dataset_guidance_scale: Optional[float] = None,
        return_labels: bool = False,
    ):
        digits = torch.arange(
            self.n_digit_classes,
            device=self.device,
            dtype=torch.long,
        ).repeat_interleave(samples_per_digit)
        datasets = torch.full(
            (digits.shape[0],),
            dataset_label,
            device=self.device,
            dtype=torch.long,
        )
        return self.sample(
            batch_size=digits.shape[0],
            digit_labels=digits,
            dataset_labels=datasets,
            guidance_scale=guidance_scale,
            digit_guidance_scale=digit_guidance_scale,
            dataset_guidance_scale=dataset_guidance_scale,
            return_labels=return_labels,
        )

    @torch.no_grad()
    def sample_all_20_conditions(
        self,
        samples_per_condition: int = 1,
        guidance_scale: float = 3.0,
        digit_guidance_scale: Optional[float] = None,
        dataset_guidance_scale: Optional[float] = None,
        return_labels: bool = False,
    ):
        digit_list = []
        dataset_list = []
        for dataset_label in (self.DATASET_MNISTM, self.DATASET_SVHN):
            for digit in range(self.n_digit_classes):
                digit_list.extend([digit] * samples_per_condition)
                dataset_list.extend([dataset_label] * samples_per_condition)

        digits = torch.tensor(
            digit_list,
            dtype=torch.long,
            device=self.device,
        )
        datasets = torch.tensor(
            dataset_list,
            dtype=torch.long,
            device=self.device,
        )
        return self.sample(
            batch_size=digits.shape[0],
            digit_labels=digits,
            dataset_labels=datasets,
            guidance_scale=guidance_scale,
            digit_guidance_scale=digit_guidance_scale,
            dataset_guidance_scale=dataset_guidance_scale,
            return_labels=return_labels,
        )


if __name__ == "__main__":
    pass