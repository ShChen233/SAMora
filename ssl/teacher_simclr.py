from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 256,
        num_layers: int = 2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        dim_in = in_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleConvBackbone(nn.Module):
    """
    Lightweight fallback backbone used when torchvision/timm backbones
    are unavailable or intentionally avoided.
    """

    def __init__(self, in_chans: int = 3, base_chans: int = 64, out_dim: int = 2048) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_chans, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_chans),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_chans, base_chans * 2, stride=2)
        self.layer2 = self._make_layer(base_chans * 2, base_chans * 4, stride=2)
        self.layer3 = self._make_layer(base_chans * 4, base_chans * 8, stride=2)
        self.proj = nn.Sequential(
            nn.Conv2d(base_chans * 8, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    @staticmethod
    def _make_layer(in_chans: int, out_chans: int, stride: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return feat


class SimCLRTeacher(nn.Module):
    """
    Teacher network for SAMora stage-1 image-level pretraining.

    Default behavior:
    - tries to build a torchvision ResNet-style backbone when available
    - falls back to a lightweight internal conv backbone
    - returns both pooled feature and projection vector
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        in_chans: int = 3,
        feat_dim: int = 2048,
        proj_dim: int = 256,
        proj_hidden_dim: int = 2048,
        projector_layers: int = 2,
        normalize_projected: bool = True,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.normalize_projected = normalize_projected

        backbone, inferred_dim = self._build_backbone(
            backbone_name=self.backbone_name,
            in_chans=in_chans,
            feat_dim=feat_dim,
            pretrained=pretrained,
        )
        self.backbone = backbone
        self.feat_dim = inferred_dim

        self.projector = MLPHead(
            in_dim=self.feat_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim,
            num_layers=projector_layers,
            use_bn=True,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _build_backbone(
        self,
        backbone_name: str,
        in_chans: int,
        feat_dim: int,
        pretrained: bool,
    ) -> Tuple[nn.Module, int]:
        # torchvision path
        try:
            import torchvision.models as tv_models

            if backbone_name in {"resnet50", "resnet101", "resnet18", "resnet34"}:
                ctor = getattr(tv_models, backbone_name)
                try:
                    model = ctor(weights="DEFAULT" if pretrained else None)
                except Exception:
                    model = ctor(pretrained=pretrained)

                if in_chans != 3:
                    old_conv = model.conv1
                    model.conv1 = nn.Conv2d(
                        in_chans,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=False,
                    )

                out_dim = model.fc.in_features
                model.fc = nn.Identity()

                class _TorchvisionBackbone(nn.Module):
                    def __init__(self, net: nn.Module, dim: int) -> None:
                        super().__init__()
                        self.net = net
                        self.out_dim = dim

                    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
                        x = self.net.conv1(x)
                        x = self.net.bn1(x)
                        x = self.net.relu(x)
                        x = self.net.maxpool(x)
                        x = self.net.layer1(x)
                        x = self.net.layer2(x)
                        x = self.net.layer3(x)
                        x = self.net.layer4(x)
                        return x

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        feat = self.forward_features(x)
                        return F.adaptive_avg_pool2d(feat, 1).flatten(1)

                return _TorchvisionBackbone(model, out_dim), out_dim
        except Exception:
            pass

        # fallback path
        backbone = SimpleConvBackbone(in_chans=in_chans, out_dim=feat_dim)
        return backbone, backbone.out_dim

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        proj = self.projector(feat)
        if self.normalize_projected:
            proj = F.normalize(proj, dim=-1)
        return proj

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
        return_projection: bool = True,
    ):
        feat = self.forward_features(x)
        proj = self.projector(feat)
        if self.normalize_projected:
            proj = F.normalize(proj, dim=-1)

        if return_features and return_projection:
            return {
                "features": feat,
                "projection": proj,
            }
        if return_features:
            return feat
        if return_projection:
            return proj
        return {
            "features": feat,
            "projection": proj,
        }


class SimCLRTeacherStudentAdapter(nn.Module):
    """
    Utility wrapper for stage-1 image-level distillation.

    It does not define the student itself; instead it aligns teacher outputs
    with student outputs that are already projected by a student projector.
    """

    def __init__(
        self,
        teacher: SimCLRTeacher,
        student_feature_dim: int,
        student_proj_dim: int = 256,
        student_proj_hidden_dim: int = 1024,
        normalize_student: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.normalize_student = normalize_student
        self.student_projector = MLPHead(
            in_dim=student_feature_dim,
            hidden_dim=student_proj_hidden_dim,
            out_dim=student_proj_dim,
            num_layers=2,
            use_bn=True,
        )

    def forward_teacher(self, x: torch.Tensor):
        with torch.no_grad():
            return self.teacher(x, return_features=True, return_projection=True)

    def forward_student(self, student_features: torch.Tensor):
        proj = self.student_projector(student_features)
        if self.normalize_student:
            proj = F.normalize(proj, dim=-1)
        return {
            "features": student_features,
            "projection": proj,
        }


def build_simclr_teacher(
    backbone_name: str = "resnet50",
    in_chans: int = 3,
    feat_dim: int = 2048,
    proj_dim: int = 256,
    proj_hidden_dim: int = 2048,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> SimCLRTeacher:
    return SimCLRTeacher(
        backbone_name=backbone_name,
        in_chans=in_chans,
        feat_dim=feat_dim,
        proj_dim=proj_dim,
        proj_hidden_dim=proj_hidden_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
