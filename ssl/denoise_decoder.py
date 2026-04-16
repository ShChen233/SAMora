import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, Sequence[torch.Tensor]]


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize feature tensors into NCHW format.

    Supported inputs:
    - [B, C, H, W]
    - [B, H, W, C]
    - [B, N, C]  -> reshaped to square grid when possible
    """
    if x.ndim == 4:
        # BCHW
        if x.shape[1] <= 4096 and x.shape[2] > 4 and x.shape[3] > 4:
            return x
        # BHWC
        return x.permute(0, 3, 1, 2).contiguous()

    if x.ndim == 3:
        b, n, c = x.shape
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(
                f"Cannot reshape token sequence of length {n} into a square feature map."
            )
        return x.transpose(1, 2).reshape(b, c, side, side).contiguous()

    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def _pick_feature(x: TensorLike) -> torch.Tensor:
    """
    Select the most useful feature tensor from a nested encoder output.

    Typical supported inputs:
    - tensor
    - list/tuple of tensors
    """
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Received an empty feature sequence.")
        # Prefer the last tensor-like item, which is typically the highest-level feature.
        for item in reversed(x):
            if isinstance(item, torch.Tensor):
                return item
        raise ValueError("No tensor found in feature sequence.")

    raise TypeError(f"Unsupported feature container: {type(x)}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride=stride, padding=padding, bias=False),
            norm_layer(out_chans),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_chans, out_chans, 3, norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ConvNormAct(out_chans, out_chans, 3, norm_layer=norm_layer, act_layer=act_layer)

        if in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
                norm_layer(out_chans),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_chans,
            out_chans,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.refine = ResidualConvBlock(
            out_chans,
            out_chans,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.refine(x)
        return x


class DenoiseUNetDecoder(nn.Module):
    """
    Lightweight U-Net-like decoder for SAMora pixel-level denoising.

    This decoder accepts a single encoder feature tensor and reconstructs an
    image-sized denoised output. It is intentionally tolerant to different
    encoder outputs used during refactoring:
    - image embeddings in BCHW
    - image embeddings in BHWC
    - flattened patch tokens in BNC
    - sequences/lists where the last tensor is the highest-level feature
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int = 3,
        hidden_chans: Sequence[int] = (256, 128, 64, 32, 16),
        final_activation: str = "sigmoid",
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()

        hidden_chans = tuple(hidden_chans)
        if len(hidden_chans) < 2:
            raise ValueError("hidden_chans must contain at least two channel sizes.")

        self.stem = ResidualConvBlock(
            in_chans,
            hidden_chans[0],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        up_blocks = []
        in_dims = hidden_chans[:-1]
        out_dims = hidden_chans[1:]
        for cin, cout in zip(in_dims, out_dims):
            up_blocks.append(
                UpBlock(
                    cin,
                    cout,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.head = nn.Sequential(
            ConvNormAct(
                hidden_chans[-1],
                hidden_chans[-1],
                kernel_size=3,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
            nn.Conv2d(hidden_chans[-1], out_chans, kernel_size=1, bias=True),
        )

        final_activation = final_activation.lower()
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation in {"identity", "none"}:
            self.final_activation = nn.Identity()
        else:
            raise ValueError(
                "final_activation must be one of {'sigmoid', 'tanh', 'identity', 'none'}."
            )

    def forward(
        self,
        encoder_features: TensorLike,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = _pick_feature(encoder_features)
        x = _to_nchw(x)

        x = self.stem(x)
        for block in self.up_blocks:
            x = block(x)

        if target_size is not None and tuple(x.shape[-2:]) != tuple(target_size):
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        x = self.head(x)
        x = self.final_activation(x)
        return x


class SAMoraPixelDenoiser(nn.Module):
    """
    Convenience wrapper used in stage-1 pixel-level pretraining.

    Expected usage:
        pred = denoiser(encoder_features, noisy_image)
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int = 3,
        hidden_chans: Sequence[int] = (256, 128, 64, 32, 16),
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.decoder = DenoiseUNetDecoder(
            in_chans=in_chans,
            out_chans=out_chans,
            hidden_chans=hidden_chans,
            final_activation=final_activation,
        )

    def forward(
        self,
        encoder_features: TensorLike,
        noisy_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_size = None
        if isinstance(noisy_image, torch.Tensor):
            target_size = tuple(noisy_image.shape[-2:])
        return self.decoder(encoder_features, target_size=target_size)


def build_denoise_decoder(
    in_chans: int,
    out_chans: int = 3,
    hidden_chans: Sequence[int] = (256, 128, 64, 32, 16),
    final_activation: str = "sigmoid",
) -> DenoiseUNetDecoder:
    return DenoiseUNetDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        hidden_chans=hidden_chans,
        final_activation=final_activation,
    )
