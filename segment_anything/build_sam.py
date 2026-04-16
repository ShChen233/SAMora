from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

import torch

from .modeling import (
    ImageEncoderViT,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

# ---------------------------------------------------------------------
# optional decoder imports for compatibility with different forks
# ---------------------------------------------------------------------
try:
    from .modeling.mask_decoder_224 import MaskDecoder_224 as _MaskDecoder224
except Exception:
    _MaskDecoder224 = None

try:
    from .modeling.mask_decoder import MaskDecoder as _MaskDecoder
except Exception:
    _MaskDecoder = None


def _get_mask_decoder_cls():
    if _MaskDecoder224 is not None:
        return _MaskDecoder224
    if _MaskDecoder is not None:
        return _MaskDecoder
    raise ImportError("No compatible mask decoder implementation found.")


# ---------------------------------------------------------------------
# checkpoint loading helpers
# ---------------------------------------------------------------------
def _safe_load_state_dict(model: torch.nn.Module, checkpoint: Optional[str]) -> None:
    if checkpoint is None or checkpoint == "":
        return

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    # keep this permissive because many forks resize pos_embed / rel_pos later
    if len(missing) > 0:
        print(f"[build_sam] missing keys while loading checkpoint: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[build_sam] unexpected keys while loading checkpoint: {len(unexpected)}")


# ---------------------------------------------------------------------
# main builders
# ---------------------------------------------------------------------
def _build_single_decoder_sam(
    *,
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes,
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
    prompt_embed_dim: int = 256,
) -> Sam:
    """
    SAMora / SAMed-style single-decoder SAM.
    """
    mask_decoder_cls = _get_mask_decoder_cls()

    image_embedding_size = image_size // 16

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = mask_decoder_cls(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        num_classes=num_classes,
    )

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder2=None,
        mask_decoder2=None,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    _safe_load_state_dict(sam, checkpoint)
    return sam


def _build_dual_decoder_sam(
    *,
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes,
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
    prompt_embed_dim: int = 256,
) -> Sam:
    """
    H-SAMora / H-SAM-style dual-decoder SAM.
    """
    mask_decoder_cls = _get_mask_decoder_cls()

    image_embedding_size = image_size // 16

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    prompt_encoder2 = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = mask_decoder_cls(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        num_classes=num_classes,
    )

    mask_decoder2 = mask_decoder_cls(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        num_classes=num_classes,
    )

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder2=prompt_encoder2,
        mask_decoder2=mask_decoder2,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    _safe_load_state_dict(sam, checkpoint)
    return sam


# ---------------------------------------------------------------------
# public builders: vanilla aliases
# ---------------------------------------------------------------------
def build_sam_vit_b(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_sam_vit_l(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_sam_vit_h(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


# ---------------------------------------------------------------------
# SAMora: single-decoder builders
# ---------------------------------------------------------------------
def build_samora_vit_b(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_single_decoder_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_samora_vit_l(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_single_decoder_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_samora_vit_h(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_single_decoder_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


# ---------------------------------------------------------------------
# H-SAMora: explicit dual-decoder builders
# ---------------------------------------------------------------------
def build_hsamora_vit_b(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_hsamora_vit_l(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


def build_hsamora_vit_h(
    image_size: int = 224,
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    pixel_mean=(123.675, 116.28, 103.53),
    pixel_std=(58.395, 57.12, 57.375),
):
    return _build_dual_decoder_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )


sam_model_registry: Dict[str, Any] = {
    # backward-compatible defaults
    "vit_b": build_sam_vit_b,
    "vit_l": build_sam_vit_l,
    "vit_h": build_sam_vit_h,

    # explicit names
    "samora_vit_b": build_samora_vit_b,
    "samora_vit_l": build_samora_vit_l,
    "samora_vit_h": build_samora_vit_h,
    "hsamora_vit_b": build_hsamora_vit_b,
    "hsamora_vit_l": build_hsamora_vit_l,
    "hsamora_vit_h": build_hsamora_vit_h,
}