from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention fusion block for H-SAMora.

    Inputs are expected in SAM image-encoder token layout:
        - [B, H, W, C] or [B, N, C]

    Internally everything is flattened to [B, N, C], fused with
    MultiheadAttention, then reshaped back to the query feature layout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.out_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if x.ndim == 4:
            b, h, w, c = x.shape
            return x.reshape(b, h * w, c), (b, h, w, c)
        if x.ndim == 3:
            b, n, c = x.shape
            return x, (b, n, c)
        raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

    @staticmethod
    def _restore(x: torch.Tensor, shape_meta: Tuple[int, ...]) -> torch.Tensor:
        if len(shape_meta) == 4:
            b, h, w, c = shape_meta
            return x.reshape(b, h, w, c)
        if len(shape_meta) == 3:
            return x
        raise ValueError(f"Unsupported shape metadata: {shape_meta}")

    def forward(
        self,
        query_feat: torch.Tensor,
        key_feat: torch.Tensor,
        value_feat: torch.Tensor,
        need_weights: bool = False,
    ):
        q, q_shape = self._flatten(query_feat)
        k, _ = self._flatten(key_feat)
        v, _ = self._flatten(value_feat)

        q_in = self.query_norm(q)
        k_in = self.key_norm(k)
        v_in = self.value_norm(v)

        out, weights = self.attn(q_in, k_in, v_in, need_weights=need_weights)
        out = self.out_drop(self.out_proj(out))
        if self.use_residual:
            out = out + q
        out = self._restore(out, q_shape)

        if need_weights:
            return out, weights
        return out


class HLAttn(nn.Module):
    """
    Hierarchical LoRA Attention used by SAMora / H-SAMora.

    Default fusion order follows the paper intuition:
        1) fuse lower-level features first (patch <- pixel)
        2) inject higher-level image semantics afterwards (image <- fused_low)

    By default the returned tensor is the fused expert signal E_omega(x), which
    can later be added to the frozen base block output F_theta(x).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        fusion_order: Tuple[str, str, str] = ("pixel", "patch", "image"),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        return_only_fused_signal: bool = True,
    ) -> None:
        super().__init__()
        valid_orders = {
            ("pixel", "patch", "image"),
            ("patch", "pixel", "image"),
            ("pixel", "image", "patch"),
            ("patch", "image", "pixel"),
            ("image", "patch", "pixel"),
            ("image", "pixel", "patch"),
        }
        if fusion_order not in valid_orders:
            raise ValueError(f"Unsupported fusion_order: {fusion_order}")

        self.dim = dim
        self.fusion_order = fusion_order
        self.return_only_fused_signal = return_only_fused_signal

        self.low_level_fuser = CrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_residual=True,
        )
        self.high_level_fuser = CrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_residual=True,
        )

        self.low_gate = nn.Parameter(torch.tensor(1.0))
        self.high_gate = nn.Parameter(torch.tensor(1.0))
        self.final_norm = nn.LayerNorm(dim)

    def _select_inputs(
        self,
        image_feat: torch.Tensor,
        patch_feat: torch.Tensor,
        pixel_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "image": image_feat,
            "patch": patch_feat,
            "pixel": pixel_feat,
        }

    def forward(
        self,
        image_feat: torch.Tensor,
        patch_feat: torch.Tensor,
        pixel_feat: torch.Tensor,
        return_debug: bool = False,
    ):
        feats = self._select_inputs(image_feat, patch_feat, pixel_feat)
        level_a, level_b, level_c = self.fusion_order

        # First fusion: middle query attends to lower granularity signal.
        low_query = feats[level_b]
        low_key_value = feats[level_a]
        low_fused = self.low_level_fuser(
            query_feat=low_query,
            key_feat=low_key_value,
            value_feat=low_key_value,
            need_weights=False,
        )
        low_fused = low_query + self.low_gate * (low_fused - low_query)

        # Second fusion: highest-level semantics attends to the fused low-level signal.
        high_query = feats[level_c]
        high_fused = self.high_level_fuser(
            query_feat=high_query,
            key_feat=low_fused,
            value_feat=low_fused,
            need_weights=False,
        )
        high_fused = high_query + self.high_gate * (high_fused - high_query)
        fused_signal = self.final_norm(high_fused)

        if not self.return_only_fused_signal:
            fused_signal = fused_signal + image_feat

        if return_debug:
            return {
                "low_fused": low_fused,
                "high_fused": high_fused,
                "fused_signal": fused_signal,
                "fusion_order": self.fusion_order,
            }
        return fused_signal


class DualHLAttn(nn.Module):
    """
    Convenience wrapper to fuse Q and V branches independently.

    This mirrors the current transitional implementation in samora_lora_hsam.py,
    where q-perturbation and v-perturbation each use one HL-Attn module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        fusion_order: Tuple[str, str, str] = ("pixel", "patch", "image"),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.q_fuser = HLAttn(
            dim=dim,
            num_heads=num_heads,
            fusion_order=fusion_order,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            return_only_fused_signal=True,
        )
        self.v_fuser = HLAttn(
            dim=dim,
            num_heads=num_heads,
            fusion_order=fusion_order,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            return_only_fused_signal=True,
        )

    def forward(
        self,
        image_q: torch.Tensor,
        patch_q: torch.Tensor,
        pixel_q: torch.Tensor,
        image_v: torch.Tensor,
        patch_v: torch.Tensor,
        pixel_v: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "q": self.q_fuser(image_q, patch_q, pixel_q),
            "v": self.v_fuser(image_v, patch_v, pixel_v),
        }
