from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice loss for segmentation / reconstruction-style masks.

    Supports targets in either of these formats:
    - integer labels with shape [B, H, W]
    - one-hot / soft labels with shape [B, C, H, W]
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        smooth: float = 1e-5,
        include_background: bool = True,
        apply_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        self.apply_softmax = apply_softmax

    @staticmethod
    def _to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
        if target.dim() == 4:
            return target.float()
        if target.dim() != 3:
            raise ValueError(
                f"Expected target shape [B,H,W] or [B,C,H,W], got {tuple(target.shape)}"
            )
        one_hot = F.one_hot(target.long(), num_classes=num_classes)
        return one_hot.permute(0, 3, 1, 2).float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits shape [B,C,H,W], got {tuple(logits.shape)}")

        probs = F.softmax(logits, dim=1) if self.apply_softmax else logits
        num_classes = probs.shape[1] if self.num_classes is None else self.num_classes
        target_oh = self._to_one_hot(target, num_classes=num_classes).to(probs.device)

        if target_oh.shape != probs.shape:
            raise ValueError(
                f"Shape mismatch between probs {tuple(probs.shape)} and target {tuple(target_oh.shape)}"
            )

        start_c = 0 if self.include_background else 1
        probs = probs[:, start_c:]
        target_oh = target_oh[:, start_c:]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        denom = torch.sum(probs * probs, dims) + torch.sum(target_oh * target_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class SoftDiceLoss(nn.Module):
    """Binary/soft Dice loss for generic dense predictions."""

    def __init__(self, smooth: float = 1e-5, from_logits: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred = pred.float().reshape(pred.shape[0], -1)
        target = target.float().reshape(target.shape[0], -1)
        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float(), target.float())


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.float(), target.float())


def cosine_distill_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    detach_teacher: bool = True,
) -> torch.Tensor:
    if detach_teacher:
        teacher = teacher.detach()
    student = F.normalize(student.flatten(1), dim=1)
    teacher = F.normalize(teacher.flatten(1), dim=1)
    return 1.0 - (student * teacher).sum(dim=1).mean()


class DistillLoss(nn.Module):
    """
    Generic feature distillation loss.

    mode:
        - 'mse'     : mean squared error
        - 'l1'      : mean absolute error
        - 'cosine'  : 1 - cosine similarity
        - 'smoothl1': Smooth L1
    """

    def __init__(self, mode: str = "mse", detach_teacher: bool = True) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.detach_teacher = detach_teacher

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.detach_teacher:
            teacher = teacher.detach()

        if self.mode == "mse":
            return F.mse_loss(student.float(), teacher.float())
        if self.mode == "l1":
            return F.l1_loss(student.float(), teacher.float())
        if self.mode == "smoothl1":
            return F.smooth_l1_loss(student.float(), teacher.float())
        if self.mode == "cosine":
            return cosine_distill_loss(student, teacher, detach_teacher=False)
        raise ValueError(f"Unsupported distillation mode: {self.mode}")


class ReconstructionLoss(nn.Module):
    """
    Generic reconstruction loss for stage1 patch/pixel tasks.

    mode:
        - 'mse'
        - 'l1'
        - 'smoothl1'
        - 'bce'
    """

    def __init__(self, mode: str = "mse") -> None:
        super().__init__()
        self.mode = mode.lower()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        if self.mode == "mse":
            return F.mse_loss(pred, target)
        if self.mode == "l1":
            return F.l1_loss(pred, target)
        if self.mode == "smoothl1":
            return F.smooth_l1_loss(pred, target)
        if self.mode == "bce":
            return F.binary_cross_entropy_with_logits(pred, target)
        raise ValueError(f"Unsupported reconstruction mode: {self.mode}")


class DiceMSELoss(nn.Module):
    """
    Weighted Dice + MSE loss.

    This is useful for SAMora stage1 pixel-level supervision because the paper's
    supplementary material reports a Dice + MSE mixture with weights 0.9 / 0.1.fileciteturn0file1
    """

    def __init__(
        self,
        dice_weight: float = 0.9,
        mse_weight: float = 0.1,
        from_logits: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.mse_weight = float(mse_weight)
        self.dice = SoftDiceLoss(from_logits=from_logits)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_val = self.dice(pred, target)
        mse_val = F.mse_loss(pred.float(), target.float())
        return self.dice_weight * dice_val + self.mse_weight * mse_val


class HierarchicalStageLoss(nn.Module):
    """
    Optional weighted sum for multi-stage / hierarchical supervision.

    This matches the paper's idea of combining per-stage losses with a weighted
    coefficient that can decay over training.fileciteturn0file1
    """

    def __init__(self, base_loss: nn.Module, aux_weight: float = 0.4) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.aux_weight = aux_weight

    def set_aux_weight(self, aux_weight: float) -> None:
        self.aux_weight = float(aux_weight)

    def forward(
        self,
        main_pred: torch.Tensor,
        main_target: torch.Tensor,
        aux_pred: Optional[torch.Tensor] = None,
        aux_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.base_loss(main_pred, main_target)
        if aux_pred is not None and aux_target is not None and self.aux_weight > 0.0:
            loss = loss + self.aux_weight * self.base_loss(aux_pred, aux_target)
        return loss


class SAMoraStage1ImageLoss(nn.Module):
    """
    Image-level loss wrapper.

    Intended usage:
    - student_view1/student_view2 are projected student features
    - teacher_view1/teacher_view2 are teacher features (SimCLRv2 side)
    """

    def __init__(self, distill_mode: str = "mse", symmetrize: bool = True) -> None:
        super().__init__()
        self.distill = DistillLoss(mode=distill_mode, detach_teacher=True)
        self.symmetrize = symmetrize

    def forward(
        self,
        student_view1: torch.Tensor,
        student_view2: torch.Tensor,
        teacher_view1: torch.Tensor,
        teacher_view2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_12 = self.distill(student_view1, teacher_view1)
        loss_21 = self.distill(student_view2, teacher_view2)
        if self.symmetrize:
            total = 0.5 * (loss_12 + loss_21)
        else:
            total = loss_12
        return {
            "loss": total,
            "loss_view1": loss_12.detach(),
            "loss_view2": loss_21.detach(),
        }


class SAMoraStage1PatchLoss(nn.Module):
    """
    Patch-level loss wrapper for MAE-style reconstruction / distillation.
    """

    def __init__(
        self,
        recon_mode: str = "mse",
        distill_mode: Optional[str] = None,
        recon_weight: float = 1.0,
        distill_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.reconstruction = ReconstructionLoss(mode=recon_mode)
        self.distill = DistillLoss(mode=distill_mode or "mse")
        self.recon_weight = recon_weight
        self.distill_weight = distill_weight

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        student_tokens: Optional[torch.Tensor] = None,
        teacher_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon = self.reconstruction(reconstructed, target)
        total = self.recon_weight * recon
        out = {
            "loss": total,
            "loss_recon": recon.detach(),
        }
        if (
            self.distill_weight > 0.0
            and student_tokens is not None
            and teacher_tokens is not None
        ):
            distill = self.distill(student_tokens, teacher_tokens)
            total = total + self.distill_weight * distill
            out["loss"] = total
            out["loss_distill"] = distill.detach()
        return out


class SAMoraStage1PixelLoss(nn.Module):
    """
    Pixel-level loss wrapper.

    By default it uses Dice + MSE, which aligns with the supplementary settings.fileciteturn0file1
    """

    def __init__(
        self,
        dice_weight: float = 0.9,
        mse_weight: float = 0.1,
        use_hierarchical_aux: bool = False,
        aux_weight: float = 0.4,
    ) -> None:
        super().__init__()
        base_loss = DiceMSELoss(
            dice_weight=dice_weight,
            mse_weight=mse_weight,
            from_logits=False,
        )
        self.use_hierarchical_aux = use_hierarchical_aux
        self.loss_fn = (
            HierarchicalStageLoss(base_loss, aux_weight=aux_weight)
            if use_hierarchical_aux
            else base_loss
        )

    def set_aux_weight(self, aux_weight: float) -> None:
        if isinstance(self.loss_fn, HierarchicalStageLoss):
            self.loss_fn.set_aux_weight(aux_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_pred: Optional[torch.Tensor] = None,
        aux_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(self.loss_fn, HierarchicalStageLoss):
            loss = self.loss_fn(pred, target, aux_pred=aux_pred, aux_target=aux_target)
        else:
            loss = self.loss_fn(pred, target)
        return {"loss": loss}


def build_stage1_loss(task: str, **kwargs) -> nn.Module:
    task = task.lower()
    if task == "image":
        return SAMoraStage1ImageLoss(**kwargs)
    if task == "patch":
        return SAMoraStage1PatchLoss(**kwargs)
    if task == "pixel":
        return SAMoraStage1PixelLoss(**kwargs)
    raise ValueError(f"Unsupported stage1 task: {task}")
