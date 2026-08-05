from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from diffusion import DiffusionImageGenerator


LabelLike = Union[int, Sequence[int], torch.Tensor]


class CFGDiffusionImageGenerator(DiffusionImageGenerator):
    """
    Classifier-Free Guidance diffusion wrapper.

    Assumptions
    -----------
    1. Base class `DiffusionImageGenerator` is the fuller version I gave before
       (has self.n_timesteps, self.device, self.generator, q_sample(), _extract(),
       betas/alphas/alpha_bars/posterior_variance, etc.).

    2. `self.model` must be a conditional denoiser with interface:
           model(x_t, timesteps, digit_labels, dataset_labels) -> predicted_noise

    3. Conditions:
           digit_labels   : 0~9, null digit = 10
           dataset_labels : 0=MNIST-M, 1=SVHN, null dataset = 2
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
        n_digit_classes: int = 10,
        n_dataset_classes: int = 2,
        null_digit_idx: int = 10,
        null_dataset_idx: int = 2,
    ):
        # 先用 base class 建好 diffusion schedule
        # 注意：checkpoint_path 先不要交給 super，避免它先拿舊 UNet 去 load 壞掉
        super().__init__(
            checkpoint_path=None,
            image_size=image_size,
            channels=channels,
            n_timesteps=n_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )

        self.n_timesteps = n_timesteps
        self.p_uncond = p_uncond

        self.n_digit_classes = n_digit_classes
        self.n_dataset_classes = n_dataset_classes

        self.null_digit_idx = null_digit_idx
        self.null_dataset_idx = null_dataset_idx

        # 之後你會把這裡換成 conditional UNet
        if conditional_model is not None:
            self.model = conditional_model.to(self.device)

        # 如果有 checkpoint，就在 model 換好之後再載入
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    # =========================================================
    # Helper functions
    # =========================================================

    def _make_null_digit_labels(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or self.device
        return torch.full(
            (batch_size,),
            self.null_digit_idx,
            dtype=torch.long,
            device=device,
        )

    def _make_null_dataset_labels(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or self.device
        return torch.full(
            (batch_size,),
            self.null_dataset_idx,
            dtype=torch.long,
            device=device,
        )

    def _to_label_tensor(
        self,
        labels: Optional[LabelLike],
        batch_size: int,
        null_value: int,
        name: str,
        device=None,
    ) -> torch.Tensor:
        """
        Convert int / list / tensor -> shape [B] long tensor.

        Rules:
        - None      -> all null labels
        - int       -> repeat to batch_size
        - len == 1  -> repeat to batch_size
        - len == B  -> use as-is
        """
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

        labels = labels.view(-1)

        if labels.numel() == 1 and batch_size > 1:
            labels = labels.repeat(batch_size)

        if labels.numel() != batch_size:
            raise ValueError(
                f"{name} must have length 1 or batch_size={batch_size}, "
                f"but got {labels.numel()}."
            )

        return labels

    def _drop_conditions(
        self,
        digit_labels: torch.Tensor,
        dataset_labels: torch.Tensor,
    ):
        """
        CFG training:
        with probability p_uncond, drop BOTH conditions for a sample.
        """
        batch_size = digit_labels.shape[0]
        device = digit_labels.device

        drop_mask = torch.rand(batch_size, device=device) < self.p_uncond

        cond_digit_labels = digit_labels.clone()
        cond_dataset_labels = dataset_labels.clone()

        cond_digit_labels[drop_mask] = self.null_digit_idx
        cond_dataset_labels[drop_mask] = self.null_dataset_idx

        return cond_digit_labels, cond_dataset_labels

    # =========================================================
    # Model forward
    # =========================================================

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: torch.Tensor,
        dataset_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional denoiser forward.

        Required backbone signature:
            model(x_t, timesteps, digit_labels, dataset_labels)
        """
        return self.model(x_t, timesteps, digit_labels, dataset_labels)

    # =========================================================
    # Training
    # =========================================================

    def training_loss(
        self,
        x_0: torch.Tensor,
        digit_labels: LabelLike,
        dataset_labels: LabelLike,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDPM / CFG training loss.

        Inputs
        ------
        x_0            : [B, 3, H, W], normalized to [-1, 1]
        digit_labels   : [B] or int/list
        dataset_labels : [B] or int/list

        Training trick
        --------------
        With probability p_uncond, replace BOTH conditions with null tokens.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        digit_labels = self._to_label_tensor(
            digit_labels,
            batch_size=batch_size,
            null_value=self.null_digit_idx,
            name="digit_labels",
            device=device,
        )

        dataset_labels = self._to_label_tensor(
            dataset_labels,
            batch_size=batch_size,
            null_value=self.null_dataset_idx,
            name="dataset_labels",
            device=device,
        )

        # classifier-free dropout
        cond_digit_labels, cond_dataset_labels = self._drop_conditions(
            digit_labels,
            dataset_labels,
        )

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.n_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(
            x_0=x_0,
            timesteps=timesteps,
            noise=noise,
        )

        predicted_noise = self(
            x_t,
            timesteps,
            cond_digit_labels,
            cond_dataset_labels,
        )

        return F.mse_loss(predicted_noise, noise)

    # =========================================================
    # CFG noise prediction
    # =========================================================

    @torch.no_grad()
    def predict_noise_cfg(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: LabelLike,
        dataset_labels: LabelLike,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Compute classifier-free guided noise.

        We use the common implementation form:
            eps_guided = eps_uncond + s * (eps_cond - eps_uncond)

        Meaning:
            s = 0   -> unconditional
            s = 1   -> ordinary conditional model
            s > 1   -> classifier-free guidance

        Paper notation:
            eps_tilde = (1 + w) eps_cond - w eps_uncond
        Correspondence:
            guidance_scale = 1 + w
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        digit_labels = self._to_label_tensor(
            digit_labels,
            batch_size=batch_size,
            null_value=self.null_digit_idx,
            name="digit_labels",
            device=device,
        )

        dataset_labels = self._to_label_tensor(
            dataset_labels,
            batch_size=batch_size,
            null_value=self.null_dataset_idx,
            name="dataset_labels",
            device=device,
        )

        null_digit_labels = self._make_null_digit_labels(batch_size, device=device)
        null_dataset_labels = self._make_null_dataset_labels(batch_size, device=device)

        eps_cond = self(
            x_t,
            timesteps,
            digit_labels,
            dataset_labels,
        )

        eps_uncond = self(
            x_t,
            timesteps,
            null_digit_labels,
            null_dataset_labels,
        )

        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        return eps_guided

    # =========================================================
    # Reverse diffusion
    # =========================================================

    @torch.no_grad()
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        digit_labels: Optional[LabelLike] = None,
        dataset_labels: Optional[LabelLike] = None,
        guidance_scale: float = 3.0,
    ):
        """
        Compute p_theta(x_{t-1} | x_t) using CFG predicted noise.
        """
        if digit_labels is None or dataset_labels is None:
            # unconditional fallback
            batch_size = x_t.shape[0]
            digit_labels = self._make_null_digit_labels(batch_size, device=x_t.device)
            dataset_labels = self._make_null_dataset_labels(batch_size, device=x_t.device)

        predicted_noise = self.predict_noise_cfg(
            x_t=x_t,
            timesteps=timesteps,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            guidance_scale=guidance_scale,
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
            x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t
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
    ) -> torch.Tensor:
        """
        One reverse step:
            x_t -> x_{t-1}
        """
        model_mean, model_variance = self.p_mean_variance(
            x_t=x_t,
            timesteps=timesteps,
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            guidance_scale=guidance_scale,
        )

        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=self.generator,
        )

        nonzero_mask = (timesteps != 0).float().reshape(
            x_t.shape[0],
            *([1] * (x_t.ndim - 1)),
        )

        x_prev = (
            model_mean
            + nonzero_mask * torch.sqrt(model_variance.clamp(min=1e-20)) * noise
        )

        return x_prev

    # =========================================================
    # Sampling
    # =========================================================

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        digit_labels: Optional[LabelLike] = None,
        dataset_labels: Optional[LabelLike] = None,
        guidance_scale: float = 3.0,
        return_labels: bool = False,
    ):
        """
        Sample images conditioned on (digit, dataset).

        Parameters
        ----------
        batch_size:
            If None, infer from labels.

        digit_labels:
            int / list / tensor
            Example:
                3
                [0,1,2,3]
                torch.tensor([7,7,7,7])

        dataset_labels:
            int / list / tensor
            dataset ids:
                0 = MNIST-M
                1 = SVHN

        guidance_scale:
            common CFG convention
                0.0 -> unconditional
                1.0 -> normal conditional
                >1  -> CFG

        Returns
        -------
        images: [B, 3, H, W]
            normalized images in approximately [-1, 1]
        """
        # infer batch size
        if batch_size is None:
            if isinstance(digit_labels, torch.Tensor):
                batch_size = digit_labels.numel()
            elif isinstance(digit_labels, (list, tuple)):
                batch_size = len(digit_labels)
            elif isinstance(dataset_labels, torch.Tensor):
                batch_size = dataset_labels.numel()
            elif isinstance(dataset_labels, (list, tuple)):
                batch_size = len(dataset_labels)
            else:
                batch_size = 1

        digit_labels = self._to_label_tensor(
            digit_labels,
            batch_size=batch_size,
            null_value=self.null_digit_idx,
            name="digit_labels",
            device=self.device,
        )

        dataset_labels = self._to_label_tensor(
            dataset_labels,
            batch_size=batch_size,
            null_value=self.null_dataset_idx,
            name="dataset_labels",
            device=self.device,
        )

        was_training = self.training
        self.eval()

        x = self.sample_random_noise(batch_size=batch_size)

        for timestep in reversed(range(self.n_timesteps)):
            t = torch.full(
                (batch_size,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )

            x = self.p_sample(
                x_t=x,
                timesteps=t,
                digit_labels=digit_labels,
                dataset_labels=dataset_labels,
                guidance_scale=guidance_scale,
            )

        if was_training:
            self.train()

        if return_labels:
            return x, digit_labels, dataset_labels
        return x

    @torch.no_grad()
    def sample_all_digits_for_dataset(
        self,
        dataset_label: int,
        samples_per_digit: int = 1,
        guidance_scale: float = 3.0,
        return_labels: bool = False,
    ):
        """
        Convenience method:
        generate digits 0~9 for ONE dataset condition.

        Example:
            sample_all_digits_for_dataset(dataset_label=0)  # MNIST-M
            sample_all_digits_for_dataset(dataset_label=1)  # SVHN
        """
        digit_labels = torch.arange(
            0,
            self.n_digit_classes,
            device=self.device,
            dtype=torch.long,
        ).repeat_interleave(samples_per_digit)

        dataset_labels = torch.full(
            (digit_labels.shape[0],),
            dataset_label,
            device=self.device,
            dtype=torch.long,
        )

        return self.sample(
            batch_size=digit_labels.shape[0],
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            guidance_scale=guidance_scale,
            return_labels=return_labels,
        )

    @torch.no_grad()
    def sample_all_20_conditions(
        self,
        samples_per_condition: int = 1,
        guidance_scale: float = 3.0,
        return_labels: bool = False,
    ):
        """
        Convenience method:
        generate all 20 combinations:
            digits 0~9 x datasets {MNIST-M, SVHN}

        Output order:
            (digit=0, dataset=MNIST-M)
            ...
            (digit=9, dataset=MNIST-M)
            (digit=0, dataset=SVHN)
            ...
            (digit=9, dataset=SVHN)
        """
        digit_list = []
        dataset_list = []

        for dataset_label in [self.DATASET_MNISTM, self.DATASET_SVHN]:
            for digit in range(self.n_digit_classes):
                digit_list.extend([digit] * samples_per_condition)
                dataset_list.extend([dataset_label] * samples_per_condition)

        digit_labels = torch.tensor(
            digit_list,
            dtype=torch.long,
            device=self.device,
        )

        dataset_labels = torch.tensor(
            dataset_list,
            dtype=torch.long,
            device=self.device,
        )

        return self.sample(
            batch_size=digit_labels.shape[0],
            digit_labels=digit_labels,
            dataset_labels=dataset_labels,
            guidance_scale=guidance_scale,
            return_labels=return_labels,
        )

if __name__ == "__main__":
    pass