from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MAETeacher(nn.Module):
    """
    Lightweight MAE-style teacher for SAMora stage-1 patch-level pretraining.

    The teacher:
    - patchifies an input image
    - optionally masks a subset of patches
    - encodes visible tokens with a ViT-style encoder
    - predicts token-level patch representations or pixel reconstructions

    This implementation is intentionally lightweight and self-contained so it can
    be used without external timm dependencies.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        proj_dim: int = 256,
        norm_projection: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.norm_projection = norm_projection

        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * in_chans,
            bias=True,
        )

        self.token_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def patchify_pixels(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        if imgs.shape[-1] != self.img_size or imgs.shape[-2] != self.img_size:
            imgs = F.interpolate(
                imgs,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        b, c, h, w = imgs.shape
        gh = h // p
        gw = w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(b, gh * gw, p * p * c)
        return x

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, C]
        Returns:
            x_masked: visible tokens only
            mask: [B, N], 0 for keep, 1 for remove
            ids_restore: indices to restore original order
        """
        b, n, c = x.shape
        len_keep = max(1, int(n * (1 - mask_ratio)))

        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, c),
        )

        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_vis = torch.cat([cls_tokens, x_masked], dim=1)

        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)
        return x_vis, mask, ids_restore, x

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder_embed(latent)

        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        pred = self.decoder_pred(x[:, 1:, :])
        return pred

    def forward_tokens(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> dict:
        latent, mask, ids_restore, full_tokens = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        visible_tokens = latent[:, 1:, :]
        proj_tokens = self.token_projector(visible_tokens)
        if self.norm_projection:
            proj_tokens = F.normalize(proj_tokens, dim=-1)
        return {
            "visible_tokens": visible_tokens,
            "projected_tokens": proj_tokens,
            "mask": mask,
            "ids_restore": ids_restore,
            "full_tokens": full_tokens,
            "cls_token": latent[:, 0, :],
        }

    def forward_reconstruction(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> dict:
        latent, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify_pixels(imgs)
        return {
            "pred": pred,
            "target": target,
            "mask": mask,
            "latent": latent,
        }

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        return_reconstruction: bool = True,
        return_tokens: bool = True,
    ):
        token_dict = self.forward_tokens(imgs, mask_ratio=mask_ratio)
        if return_reconstruction:
            recon_dict = self.forward_reconstruction(imgs, mask_ratio=mask_ratio)
            token_dict.update(recon_dict)
        if return_tokens:
            return token_dict
        return {
            "pred": token_dict.get("pred"),
            "target": token_dict.get("target"),
            "mask": token_dict.get("mask"),
        }


class MAETeacherStudentAdapter(nn.Module):
    """
    Utility adapter for stage-1 patch-level distillation.

    The student is assumed to produce patch tokens. This adapter aligns teacher
    and student tokens with a student-side projector.
    """

    def __init__(
        self,
        teacher: MAETeacher,
        student_token_dim: int,
        proj_dim: int = 256,
        hidden_dim: int = 1024,
        normalize_student: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.normalize_student = normalize_student
        self.student_projector = nn.Sequential(
            nn.LayerNorm(student_token_dim),
            nn.Linear(student_token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward_teacher(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> dict:
        with torch.no_grad():
            return self.teacher(imgs, mask_ratio=mask_ratio, return_reconstruction=True, return_tokens=True)

    def forward_student(self, student_tokens: torch.Tensor) -> dict:
        proj = self.student_projector(student_tokens)
        if self.normalize_student:
            proj = F.normalize(proj, dim=-1)
        return {
            "student_tokens": student_tokens,
            "student_projection": proj,
        }


def build_mae_teacher(
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 1024,
    depth: int = 8,
    num_heads: int = 16,
    decoder_embed_dim: int = 512,
    decoder_depth: int = 2,
    decoder_num_heads: int = 8,
    proj_dim: int = 256,
) -> MAETeacher:
    return MAETeacher(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        proj_dim=proj_dim,
    )
