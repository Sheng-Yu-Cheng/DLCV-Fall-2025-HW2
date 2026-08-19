# Joint-class conditional U-Net for HW2 Problem 1.

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


swish = F.silu


@torch.no_grad()
def variance_scaling_init_(
    tensor,
    scale=1,
    mode="fan_avg",
    distribution="uniform",
):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in
    elif mode == "fan_out":
        scale /= fan_out
    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        return tensor.normal_(0, math.sqrt(scale))

    bound = math.sqrt(3 * scale)
    return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    variance_scaling_init_(conv.weight, scale, mode=mode)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    layer = nn.Linear(in_channel, out_channel)
    variance_scaling_init_(layer.weight, scale, mode=mode)
    nn.init.zeros_(layer.bias)
    return layer


class Swish(nn.Module):
    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        )


class Downsample(nn.Sequential):
    def __init__(self, channel):
        super().__init__(
            conv2d(channel, channel, 3, stride=2, padding=1)
        )


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        use_affine_time=False,
        dropout=0,
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        # This projection now receives timestep + joint-class embedding.
        self.time = nn.Sequential(
            Swish(),
            linear(time_dim, time_out_dim, scale=time_scale),
        )

        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(
            out_channel,
            out_channel,
            3,
            padding=1,
            scale=1e-10,
        )

        self.skip = (
            conv2d(in_channel, out_channel, 1)
            if in_channel != out_channel
            else None
        )

    def forward(self, input, condition):
        batch = input.shape[0]
        out = self.conv1(self.activation1(self.norm1(input)))

        projected = self.time(condition).view(batch, -1, 1, 1)
        if self.use_affine_time:
            gamma, beta = projected.chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta
        else:
            out = self.norm2(out + projected)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)
        return out + input


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        if in_channel % n_head != 0:
            raise ValueError("in_channel must be divisible by n_head")

        self.n_head = n_head
        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(
            batch,
            n_head,
            head_dim * 3,
            height,
            width,
        )
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum(
            "bnchw,bncyx->bnhwyx",
            query,
            key,
        ).contiguous() / math.sqrt(head_dim)

        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(
            batch,
            n_head,
            height,
            width,
            height,
            width,
        )

        out = torch.einsum(
            "bnhwyx,bncyx->bnchw",
            attn,
            value,
        ).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.reshape(-1).float(), self.inv_freq)
        pos_emb = torch.cat(
            [sinusoid_in.sin(), sinusoid_in.cos()],
            dim=-1,
        )
        return pos_emb.view(*shape, self.dim)


class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel,
            out_channel,
            time_dim,
            use_affine_time,
            dropout,
        )
        self.attention = (
            SelfAttention(out_channel, n_head=attention_head)
            if use_attention
            else None
        )

    def forward(self, input, condition):
        out = self.resblocks(input, condition)
        if self.attention is not None:
            out = self.attention(out)
        return out


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(
            batch,
            -1,
            height * unfold,
            width * unfold,
        )
    )


class UNet(nn.Module):
    """U-Net with a 20-way joint condition injected into every ResBlock.

    Joint class mapping:
        0..9   = MNIST-M digits 0..9
        10..19 = SVHN digits 0..9
        20     = null condition for classifier-free guidance

    The public forward interface remains:
        model(x_t, timesteps, digit_labels, dataset_labels)
    """

    def __init__(
        self,
        in_channel=3,
        channel=128,
        attn_heads=1,
        use_affine_time=False,
        dropout=0,
        num_digit_classes=10,
        num_dataset_classes=2,
        null_digit_idx=10,
        null_dataset_idx=2,
    ):
        super().__init__()

        self.num_digit_classes = int(num_digit_classes)
        self.num_dataset_classes = int(num_dataset_classes)
        self.null_digit_idx = int(null_digit_idx)
        self.null_dataset_idx = int(null_dataset_idx)
        self.num_joint_classes = (
            self.num_digit_classes * self.num_dataset_classes
        )
        self.null_joint_idx = self.num_joint_classes

        time_dim = channel * 4

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        self.class_embedding = nn.Embedding(
            self.num_joint_classes + 1,
            time_dim,
        )
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        self.class_mlp = nn.Sequential(
            Swish(),
            linear(time_dim, time_dim),
        )

        self.down1 = conv2d(in_channel, channel, 3, padding=1)
        self.down2 = ResBlockWithAttention(128, 128, time_dim, dropout)
        self.down3 = ResBlockWithAttention(128, 128, time_dim, dropout)
        self.down4 = Downsample(128)
        self.down5 = ResBlockWithAttention(128, 128, time_dim, dropout)
        self.down6 = ResBlockWithAttention(128, 128, time_dim, dropout)
        self.down7 = Downsample(128)
        self.down8 = ResBlockWithAttention(128, 256, time_dim, dropout)
        self.down9 = ResBlockWithAttention(256, 256, time_dim, dropout)
        self.down10 = Downsample(256)
        self.down11 = ResBlockWithAttention(256, 256, time_dim, dropout)
        self.down12 = ResBlockWithAttention(256, 256, time_dim, dropout)
        self.down13 = Downsample(256)
        self.down14 = ResBlockWithAttention(
            256,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down15 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down16 = Downsample(512)
        self.down17 = ResBlockWithAttention(512, 512, time_dim, dropout)
        self.down18 = ResBlockWithAttention(512, 512, time_dim, dropout)

        self.mid1 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.mid2 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_affine_time=use_affine_time,
        )

        self.up1 = ResBlockWithAttention(1024, 512, time_dim, dropout)
        self.up2 = ResBlockWithAttention(1024, 512, time_dim, dropout)
        self.up3 = ResBlockWithAttention(1024, 512, time_dim, dropout)
        self.up4 = Upsample(512)
        self.up5 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up6 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up7 = ResBlockWithAttention(
            768,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up8 = Upsample(512)
        self.up9 = ResBlockWithAttention(768, 256, time_dim, dropout)
        self.up10 = ResBlockWithAttention(512, 256, time_dim, dropout)
        self.up11 = ResBlockWithAttention(512, 256, time_dim, dropout)
        self.up12 = Upsample(256)
        self.up13 = ResBlockWithAttention(512, 256, time_dim, dropout)
        self.up14 = ResBlockWithAttention(512, 256, time_dim, dropout)
        self.up15 = ResBlockWithAttention(384, 256, time_dim, dropout)
        self.up16 = Upsample(256)
        self.up17 = ResBlockWithAttention(384, 128, time_dim, dropout)
        self.up18 = ResBlockWithAttention(256, 128, time_dim, dropout)
        self.up19 = ResBlockWithAttention(256, 128, time_dim, dropout)
        self.up20 = Upsample(128)
        self.up21 = ResBlockWithAttention(256, 128, time_dim, dropout)
        self.up22 = ResBlockWithAttention(256, 128, time_dim, dropout)
        self.up23 = ResBlockWithAttention(256, 128, time_dim, dropout)

        self.out = nn.Sequential(
            nn.GroupNorm(32, 128),
            Swish(),
            conv2d(128, 3, 3, padding=1, scale=1e-10),
        )

    def _make_joint_labels(
        self,
        digit_labels: Optional[torch.Tensor],
        dataset_labels: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if digit_labels is None and dataset_labels is None:
            return torch.full(
                (batch_size,),
                self.null_joint_idx,
                dtype=torch.long,
                device=device,
            )
        if digit_labels is None or dataset_labels is None:
            raise ValueError(
                "digit_labels and dataset_labels must both be provided or both be None"
            )

        digit_labels = digit_labels.to(device=device, dtype=torch.long)
        dataset_labels = dataset_labels.to(device=device, dtype=torch.long)

        if digit_labels.shape != (batch_size,):
            raise ValueError("digit_labels must have shape [batch_size]")
        if dataset_labels.shape != (batch_size,):
            raise ValueError("dataset_labels must have shape [batch_size]")

        digit_is_null = digit_labels == self.null_digit_idx
        dataset_is_null = dataset_labels == self.null_dataset_idx

        if not torch.equal(digit_is_null, dataset_is_null):
            raise ValueError(
                "Partial null conditions are unsupported: digit and dataset "
                "must either both be present or both be null"
            )

        valid = ~digit_is_null
        if valid.any():
            if not torch.all(
                (digit_labels[valid] >= 0)
                & (digit_labels[valid] < self.num_digit_classes)
            ):
                raise ValueError("digit labels must be in 0..9")
            if not torch.all(
                (dataset_labels[valid] >= 0)
                & (dataset_labels[valid] < self.num_dataset_classes)
            ):
                raise ValueError("dataset labels must be 0 or 1")

        joint_labels = (
            dataset_labels.clamp(min=0) * self.num_digit_classes
            + digit_labels.clamp(min=0)
        )
        joint_labels = joint_labels.clone()
        joint_labels[digit_is_null] = self.null_joint_idx
        return joint_labels

    def forward(
        self,
        x,
        time,
        digit_labels=None,
        dataset_labels=None,
    ):
        batch_size = x.shape[0]
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch_size]")

        time_embed = self.time(time)
        joint_labels = self._make_joint_labels(
            digit_labels,
            dataset_labels,
            batch_size,
            x.device,
        )
        class_embed = self.class_mlp(
            self.class_embedding(joint_labels)
        )
        condition_embed = time_embed + class_embed

        feats = []

        x = self.down1(x)
        feats.append(x)
        x = self.down2(x, condition_embed)
        feats.append(x)
        x = self.down3(x, condition_embed)
        feats.append(x)
        x = self.down4(x)
        feats.append(x)
        x = self.down5(x, condition_embed)
        feats.append(x)
        x = self.down6(x, condition_embed)
        feats.append(x)
        x = self.down7(x)
        feats.append(x)
        x = self.down8(x, condition_embed)
        feats.append(x)
        x = self.down9(x, condition_embed)
        feats.append(x)
        x = self.down10(x)
        feats.append(x)
        x = self.down11(x, condition_embed)
        feats.append(x)
        x = self.down12(x, condition_embed)
        feats.append(x)
        x = self.down13(x)
        feats.append(x)
        x = self.down14(x, condition_embed)
        feats.append(x)
        x = self.down15(x, condition_embed)
        feats.append(x)
        x = self.down16(x)
        feats.append(x)
        x = self.down17(x, condition_embed)
        feats.append(x)
        x = self.down18(x, condition_embed)
        feats.append(x)

        x = self.mid1(x, condition_embed)
        x = self.mid2(x, condition_embed)

        x = self.up1(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up2(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up3(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up4(x)
        x = self.up5(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up6(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up7(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up8(x)
        x = self.up9(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up10(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up11(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up12(x)
        x = self.up13(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up14(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up15(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up16(x)
        x = self.up17(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up18(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up19(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up20(x)
        x = self.up21(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up22(torch.cat((x, feats.pop()), 1), condition_embed)
        x = self.up23(torch.cat((x, feats.pop()), 1), condition_embed)

        out = self.out(x)
        return spatial_unfold(out, 1)


if __name__ == "__main__":
    model = UNet()
    image = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    digits = torch.tensor([3, 7], dtype=torch.long)
    datasets = torch.tensor([0, 1], dtype=torch.long)
    output = model(image, timesteps, digits, datasets)
    print(output.shape)