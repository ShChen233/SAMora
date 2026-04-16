# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d, MLPBlock


class ImageEncoderViT(nn.Module):
    """
    SAM / H-SAM image encoder with light refactors that make the backbone easier
    to extend for SAMora-style hierarchical fusion while keeping the original
    H-SAM interface intact.

    Backward compatibility:
    - By default, forward(x) still returns (image_embeddings, low_image_embeddings)
      exactly like the original H-SAM implementation.

    New capabilities:
    - Optional intermediate feature collection.
    - Optional block-level override hooks for future HL-Attn integration.
    - Cleaner separation between patch embedding, block traversal, and neck.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_chans = out_chans

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    # ------------------------------------------------------------------
    # public helpers for later SAMora integration
    # ------------------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return x

    def neck_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(x.permute(0, 3, 1, 2))

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
        return_all_blocks: bool = False,
        use_samora_path: bool = True,
    ):
        """
        Args:
            x: input image tensor in BxCxHxW.
            return_intermediate: when True, also return an auxiliary dictionary
                with block-wise features for future SAMora debugging/analysis.
            return_all_blocks: when True, include hidden states from every block
                in the returned auxiliary dictionary.
            use_samora_path: when True, allow block-level override hooks to run.
                This keeps the encoder ready for future HL-Attn integration.

        Returns:
            Default:
                image_embeddings, low_image_embeddings
            If return_intermediate=True:
                image_embeddings, low_image_embeddings, aux_dict
        """
        hidden = self.patchify(x)

        hidden_states: List[torch.Tensor] = [] if return_all_blocks else []
        low_level_index = max(len(self.blocks) - 2, 0)
        low_level_feat: Optional[torch.Tensor] = None

        for block_idx, blk in enumerate(self.blocks):
            if use_samora_path and hasattr(blk, "samora_forward") and callable(blk.samora_forward):
                hidden = blk.samora_forward(hidden)
            else:
                hidden = blk(hidden)

            if block_idx == low_level_index:
                low_level_feat = hidden
            if return_all_blocks:
                hidden_states.append(hidden)

        if low_level_feat is None:
            low_level_feat = hidden

        image_embeddings = self.neck_forward(hidden)

        if not return_intermediate:
            return image_embeddings, low_level_feat

        aux: Dict[str, Any] = {
            "tokens_before_neck": hidden,
            "low_level_tokens": low_level_feat,
            "image_embeddings": image_embeddings,
        }
        if return_all_blocks:
            aux["hidden_states"] = hidden_states
        return image_embeddings, low_level_feat, aux


class Block(nn.Module):
    """
    Transformer block with support for:
    - the original H-SAM forward path,
    - optional future SAMora block-level override hooks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

        # External override hook. Another module may assign a callable here later:
        #   block.samora_forward = my_custom_forward
        self.samora_forward = None

    def _partition_if_needed(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            return x, pad_hw, (h, w)
        return x, None, None

    def _unpartition_if_needed(
        self,
        x: torch.Tensor,
        pad_hw: Optional[Tuple[int, int]],
        hw: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if self.window_size > 0:
            assert pad_hw is not None
            assert hw is not None
            x = window_unpartition(x, self.window_size, pad_hw, hw)
        return x

    def forward_attn_only(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x_attn, pad_hw, hw = self._partition_if_needed(x_norm)
        x_attn = self.attn(x_attn)
        x_attn = self._unpartition_if_needed(x_attn, pad_hw, hw)
        return x_attn

    def forward_attn_from_qkv(self, x_norm: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
        x_attn, pad_hw, hw = self._partition_if_needed(x_norm)
        expected_hw = (x_attn.shape[1], x_attn.shape[2])
        x_attn = self.attn.forward_from_qkv(qkv, spatial_shape=expected_hw)
        x_attn = self._unpartition_if_needed(x_attn, pad_hw, hw)
        return x_attn

    def forward_mlp_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x))

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = shortcut + self.forward_attn_only(x)
        x = x + self.forward_mlp_only(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.samora_forward is not None:
            return self.samora_forward(x)
        return self.forward_base(x)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if rel_pos_zero_init:
                nn.init.zeros_(self.rel_pos_h)
                nn.init.zeros_(self.rel_pos_w)

    def _attention_from_qkv(
        self,
        qkv: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        b, h, w = qkv.shape[0], spatial_shape[0], spatial_shape[1]
        qkv = qkv.reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (h, w), (h, w))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        x = self.proj(x)
        return x

    def forward_from_qkv(
        self,
        qkv: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if spatial_shape is None:
            spatial_shape = (qkv.shape[1], qkv.shape[2])
        return self._attention_from_qkv(qkv, spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        qkv = self.qkv(x)
        return self._attention_from_qkv(qkv, (h, w))


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and remove padding.
    """
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
    query and key sizes.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed relative positional embeddings from MViTv2.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = (
        attn.view(b, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(b, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x
