from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolType = Literal["mean", "max", "cls", "none"]


class MLPProjector(nn.Module):
    """
    Lightweight projector used by SAMora stage1 image/patch branches.

    Accepts features in one of the common SAM-style layouts:
    - [B, H, W, C]
    - [B, C, H, W]
    - [B, N, C]
    - [B, C]

    It optionally applies a pre-projection pooling step, then uses a small
    LayerNorm + MLP head to map features into a compact representation space.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_bn: bool = False,
        normalize_output: bool = False,
        pool_type: PoolType = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.normalize_output = normalize_output
        self.pool_type = pool_type

        layers = []
        current_dim = in_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        self.proj = nn.Sequential(*layers)

    def _flatten_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # [B, H, W, C] or [B, C, H, W]
            if x.shape[-1] == self.in_dim:
                return x.reshape(x.shape[0], -1, x.shape[-1])
            if x.shape[1] == self.in_dim:
                return x.flatten(2).transpose(1, 2)
            raise ValueError(
                f"Cannot infer feature layout from shape {tuple(x.shape)} for in_dim={self.in_dim}"
            )
        if x.ndim == 3:
            # [B, N, C]
            if x.shape[-1] != self.in_dim:
                raise ValueError(
                    f"Expected last dim={self.in_dim} for token features, got shape {tuple(x.shape)}"
                )
            return x
        if x.ndim == 2:
            # [B, C]
            if x.shape[-1] != self.in_dim:
                raise ValueError(
                    f"Expected last dim={self.in_dim} for vector features, got shape {tuple(x.shape)}"
                )
            return x.unsqueeze(1)
        raise ValueError(f"Unsupported feature ndim={x.ndim}, shape={tuple(x.shape)}")

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "none":
            if tokens.shape[1] != 1:
                raise ValueError(
                    "pool_type='none' requires a single token/vector per sample. "
                    f"Got tokens with shape {tuple(tokens.shape)}"
                )
            return tokens[:, 0, :]
        if self.pool_type == "mean":
            return tokens.mean(dim=1)
        if self.pool_type == "max":
            return tokens.max(dim=1).values
        if self.pool_type == "cls":
            return tokens[:, 0, :]
        raise ValueError(f"Unsupported pool_type: {self.pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._flatten_spatial(x)
        pooled = self._pool(tokens)
        out = self.proj(pooled)
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


class TokenProjector(nn.Module):
    """
    Token-wise projector for patch-level reconstruction/distillation.

    Input layouts:
    - [B, H, W, C]
    - [B, C, H, W]
    - [B, N, C]

    Output:
    - [B, N, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.normalize_output = normalize_output
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[-1] == self.in_dim:
                return x.reshape(x.shape[0], -1, x.shape[-1])
            if x.shape[1] == self.in_dim:
                return x.flatten(2).transpose(1, 2)
            raise ValueError(
                f"Cannot infer feature layout from shape {tuple(x.shape)} for in_dim={self.in_dim}"
            )
        if x.ndim == 3:
            if x.shape[-1] != self.in_dim:
                raise ValueError(
                    f"Expected last dim={self.in_dim} for token features, got shape {tuple(x.shape)}"
                )
            return x
        raise ValueError(f"Unsupported feature ndim={x.ndim}, shape={tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._to_tokens(x)
        out = self.net(tokens)
        if self.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


class ProjectorWithPredictor(nn.Module):
    """
    Convenience wrapper for SimCLR/BYOL-style stage1 branches.

    projector: maps encoder features -> representation space
    predictor: optional extra head for online branch prediction
    """

    def __init__(
        self,
        in_dim: int,
        projector_hidden_dim: int = 512,
        projector_out_dim: int = 256,
        predictor_hidden_dim: int = 512,
        predictor_out_dim: Optional[int] = None,
        normalize_projector_output: bool = True,
        normalize_predictor_output: bool = False,
        pool_type: PoolType = "mean",
    ) -> None:
        super().__init__()
        predictor_out_dim = predictor_out_dim or projector_out_dim

        self.projector = MLPProjector(
            in_dim=in_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
            num_layers=2,
            normalize_output=normalize_projector_output,
            pool_type=pool_type,
        )
        self.predictor = nn.Sequential(
            nn.Linear(projector_out_dim, predictor_hidden_dim),
            nn.LayerNorm(predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, predictor_out_dim),
        )
        self.normalize_predictor_output = normalize_predictor_output

    def forward(self, x: torch.Tensor, return_projected: bool = False):
        projected = self.projector(x)
        predicted = self.predictor(projected)
        if self.normalize_predictor_output:
            predicted = F.normalize(predicted, dim=-1)
        if return_projected:
            return predicted, projected
        return predicted


class SAMoraProjectorBundle(nn.Module):
    """
    Unified bundle for stage1 branches.

    image_projector: global pooled projector for image-level SSL.
    patch_projector: token-wise projector for patch-level SSL.
    lowres_projector: optional auxiliary projector for low-resolution / early features.
    """

    def __init__(
        self,
        image_in_dim: int,
        patch_in_dim: Optional[int] = None,
        lowres_in_dim: Optional[int] = None,
        hidden_dim: int = 512,
        out_dim: int = 256,
        pool_type: PoolType = "mean",
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.image_projector = MLPProjector(
            in_dim=image_in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            normalize_output=normalize_output,
            pool_type=pool_type,
        )
        self.patch_projector = None
        self.lowres_projector = None

        if patch_in_dim is not None:
            self.patch_projector = TokenProjector(
                in_dim=patch_in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                normalize_output=normalize_output,
            )
        if lowres_in_dim is not None:
            self.lowres_projector = MLPProjector(
                in_dim=lowres_in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                normalize_output=normalize_output,
                pool_type=pool_type,
            )

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_projector(x)

    def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_projector is None:
            raise RuntimeError("patch_projector is not initialized")
        return self.patch_projector(x)

    def forward_lowres(self, x: torch.Tensor) -> torch.Tensor:
        if self.lowres_projector is None:
            raise RuntimeError("lowres_projector is not initialized")
        return self.lowres_projector(x)


def infer_feature_dim(x: torch.Tensor) -> int:
    if x.ndim == 4:
        return x.shape[-1] if x.shape[-1] <= x.shape[1] or x.shape[-1] < 4096 else x.shape[1]
    if x.ndim in (2, 3):
        return x.shape[-1]
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def build_projector(
    task: str,
    in_dim: int,
    hidden_dim: int = 512,
    out_dim: int = 256,
    pool_type: PoolType = "mean",
    normalize_output: bool = True,
) -> nn.Module:
    task = task.lower()
    if task == "image":
        return MLPProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            normalize_output=normalize_output,
            pool_type=pool_type,
        )
    if task == "patch":
        return TokenProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            normalize_output=normalize_output,
        )
    if task == "image_with_predictor":
        return ProjectorWithPredictor(
            in_dim=in_dim,
            projector_hidden_dim=hidden_dim,
            projector_out_dim=out_dim,
            normalize_projector_output=normalize_output,
            pool_type=pool_type,
        )
    raise ValueError(f"Unsupported projector task: {task}")
