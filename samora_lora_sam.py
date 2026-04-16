import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from segment_anything.modeling import HLAttn, Sam


class _LoRAExpert(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.linear_a_q = nn.Linear(dim, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, dim, bias=False)
        self.linear_a_v = nn.Linear(dim, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "q": self.linear_b_q(self.linear_a_q(x)),
            "v": self.linear_b_v(self.linear_a_v(x)),
        }


class _SAMoraBlockAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        num_heads: int,
        use_hl_attn: bool = True,
        fusion_order=("pixel", "patch", "image"),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_hl_attn = use_hl_attn
        self.stage = "stage2"
        self.active_expert = "image"

        self.experts = nn.ModuleDict(
            {
                "image": _LoRAExpert(dim, rank),
                "patch": _LoRAExpert(dim, rank),
                "pixel": _LoRAExpert(dim, rank),
            }
        )

        self.fuser = (
            HLAttn(
                dim=dim,
                num_heads=num_heads,
                fusion_order=fusion_order,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            if use_hl_attn
            else None
        )

        self.image_scale = nn.Parameter(torch.tensor(1.0))
        self.patch_scale = nn.Parameter(torch.tensor(1.0))
        self.pixel_scale = nn.Parameter(torch.tensor(1.0))

    def set_stage(self, stage: str, active_expert: Optional[str] = None) -> None:
        self.stage = stage
        if active_expert is not None:
            if active_expert not in self.experts:
                raise ValueError(f"Unknown expert: {active_expert}")
            self.active_expert = active_expert

    def _manual_attention_from_qkv(self, attn_module: nn.Module, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.ndim != 4:
            raise ValueError(f"Expected BHWC qkv tensor, got shape={tuple(qkv.shape)}")

        b, h, w, three_c = qkv.shape
        c = three_c // 3
        head_dim = c // self.num_heads
        scale = getattr(attn_module, "scale", head_dim ** -0.5)

        qkv = qkv.reshape(b, h * w, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, h, w, c)
        out = attn_module.proj(out)
        if hasattr(attn_module, "proj_drop") and attn_module.proj_drop is not None:
            out = attn_module.proj_drop(out)
        return out

    def _attn_from_qkv(self, block: nn.Module, norm_x: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
        if hasattr(block, "forward_attn_from_qkv"):
            attempts = [
                lambda: block.forward_attn_from_qkv(norm_x, qkv),
                lambda: block.forward_attn_from_qkv(qkv, norm_x),
                lambda: block.forward_attn_from_qkv(qkv),
            ]
            for fn in attempts:
                try:
                    out = fn()
                    if out is not None:
                        return out
                except TypeError:
                    pass

        if hasattr(block.attn, "forward_from_qkv"):
            attempts = [
                lambda: block.attn.forward_from_qkv(qkv, norm_x.shape[1], norm_x.shape[2]),
                lambda: block.attn.forward_from_qkv(qkv, (norm_x.shape[1], norm_x.shape[2])),
                lambda: block.attn.forward_from_qkv(qkv),
            ]
            for fn in attempts:
                try:
                    out = fn()
                    if out is not None:
                        return out
                except TypeError:
                    pass

        return self._manual_attention_from_qkv(block.attn, qkv)

    def _base_qkv(self, block: nn.Module, norm_x: torch.Tensor) -> torch.Tensor:
        qkv_module = getattr(block.attn, "qkv", None)
        if qkv_module is None or not callable(qkv_module):
            raise AttributeError("Block attention must expose a callable qkv module.")
        return qkv_module(norm_x)

    def _scaled_delta(self, expert_name: str, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.experts[expert_name](x)
        scale = {
            "image": self.image_scale,
            "patch": self.patch_scale,
            "pixel": self.pixel_scale,
        }[expert_name]
        return {k: scale * v for k, v in out.items()}

    def _inject_qv_delta(self, base_qkv: torch.Tensor, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        qkv = base_qkv.clone()
        qkv[..., : self.dim] = qkv[..., : self.dim] + delta["q"]
        qkv[..., -self.dim:] = qkv[..., -self.dim:] + delta["v"]
        return qkv

    def _run_attention(self, block: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        norm_x = block.norm1(x)
        base_qkv = self._base_qkv(block, norm_x)
        base_attn = self._attn_from_qkv(block, norm_x, base_qkv)

        residuals = {}
        expert_attn = {}
        for expert_name in ("image", "patch", "pixel"):
            delta = self._scaled_delta(expert_name, norm_x)
            expert_qkv = self._inject_qv_delta(base_qkv, delta)
            expert_out = self._attn_from_qkv(block, norm_x, expert_qkv)
            expert_attn[expert_name] = expert_out
            residuals[expert_name] = expert_out - base_attn

        return {
            "base_attn": base_attn,
            "residual_image": residuals["image"],
            "residual_patch": residuals["patch"],
            "residual_pixel": residuals["pixel"],
            "expert_image": expert_attn["image"],
            "expert_patch": expert_attn["patch"],
            "expert_pixel": expert_attn["pixel"],
        }

    def _fuse_residuals(
        self,
        residual_image: torch.Tensor,
        residual_patch: torch.Tensor,
        residual_pixel: torch.Tensor,
    ) -> torch.Tensor:
        if self.fuser is None or not self.use_hl_attn:
            return residual_image + residual_patch + residual_pixel
        return self.fuser(
            image_feat=residual_image,
            patch_feat=residual_patch,
            pixel_feat=residual_pixel,
        )

    def forward(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        attn_pack = self._run_attention(block, x)
        base_attn = attn_pack["base_attn"]

        if self.stage.startswith("stage1"):
            selected = attn_pack[f"expert_{self.active_expert}"]
            x = x + selected
        else:
            fused_residual = self._fuse_residuals(
                residual_image=attn_pack["residual_image"],
                residual_patch=attn_pack["residual_patch"],
                residual_pixel=attn_pack["residual_pixel"],
            )
            x = x + base_attn + fused_residual

        x = x + block.mlp(block.norm2(x))
        return x


class LoRA_Sam(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
        r: int,
        lora_layer: Optional[List[int]] = None,
        use_hl_attn: bool = True,
        stage: str = "stage2",
        train_prompt_encoder_in_stage2: bool = False,
        fusion_order=("pixel", "patch", "image"),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.sam = sam_model
        self.rank = r
        self.use_hl_attn = use_hl_attn
        self.stage = stage
        self.train_prompt_encoder_in_stage2 = train_prompt_encoder_in_stage2

        if lora_layer is None:
            self.lora_layer = list(range(len(self.sam.image_encoder.blocks)))
        else:
            self.lora_layer = lora_layer

        self.adapters: List[_SAMoraBlockAdapter] = []

        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

        for layer_idx, blk in enumerate(self.sam.image_encoder.blocks):
            if layer_idx not in self.lora_layer:
                continue

            qkv_module = getattr(blk.attn, "qkv", None)
            if qkv_module is None or not hasattr(qkv_module, "in_features"):
                raise AttributeError("Block attention qkv must expose in_features for SAMora adapter injection.")

            adapter = _SAMoraBlockAdapter(
                dim=qkv_module.in_features,
                rank=r,
                num_heads=blk.attn.num_heads,
                use_hl_attn=use_hl_attn,
                fusion_order=fusion_order,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            blk.samora_adapter = adapter
            self.adapters.append(adapter)

        if stage == "stage2":
            self.freeze_for_stage2()
        elif stage.startswith("stage1"):
            expert = stage.split("_")[-1]
            self.freeze_for_stage1(expert)
        else:
            raise ValueError("stage must be one of {'stage2', 'stage1_image', 'stage1_patch', 'stage1_pixel'}")

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    def _iter_stage2_modules(self) -> Iterable[nn.Module]:
        yield self.sam.mask_decoder
        if self.train_prompt_encoder_in_stage2:
            yield self.sam.prompt_encoder

    def freeze_for_stage1(self, expert: str) -> None:
        if expert not in {"image", "patch", "pixel"}:
            raise ValueError("expert must be one of {'image', 'patch', 'pixel'}")

        self.stage = f"stage1_{expert}"
        self._set_requires_grad(self.sam, False)

        for adapter in self.adapters:
            adapter.set_stage(self.stage, active_expert=expert)
            for expert_name, module in adapter.experts.items():
                self._set_requires_grad(module, expert_name == expert)
            self._set_requires_grad(adapter.fuser, False)
            adapter.image_scale.requires_grad = expert == "image"
            adapter.patch_scale.requires_grad = expert == "patch"
            adapter.pixel_scale.requires_grad = expert == "pixel"

    def freeze_for_stage2(self) -> None:
        self.stage = "stage2"
        self._set_requires_grad(self.sam, False)

        for adapter in self.adapters:
            adapter.set_stage("stage2")
            for module in adapter.experts.values():
                self._set_requires_grad(module, False)
            self._set_requires_grad(adapter.fuser, self.use_hl_attn)
            adapter.image_scale.requires_grad = False
            adapter.patch_scale.requires_grad = False
            adapter.pixel_scale.requires_grad = False

        for module in self._iter_stage2_modules():
            self._set_requires_grad(module, True)

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _collect_state_by_keywords(self, keywords: List[str]) -> Dict[str, torch.Tensor]:
        state = self.sam.state_dict()
        return {k: v.detach().cpu() for k, v in state.items() if any(word in k for word in keywords)}

    def _expert_keywords(self, expert: Optional[str] = None) -> List[str]:
        if expert is None:
            return [".samora_adapter.experts."]
        return [f".samora_adapter.experts.{expert}."]

    def _fusion_keywords(self) -> List[str]:
        return [".samora_adapter.fuser."]

    def _decoder_keywords(self) -> List[str]:
        keys = ["mask_decoder"]
        if self.train_prompt_encoder_in_stage2:
            keys += ["prompt_encoder"]
        return keys

    def save_stage1_parameters(self, filename: str, expert: str) -> None:
        if not filename.endswith((".pt", ".pth")):
            raise ValueError("filename must end with .pt or .pth")
        if expert not in {"image", "patch", "pixel"}:
            raise ValueError("expert must be one of {'image', 'patch', 'pixel'}")

        state = {
            "state_dict_type": "samora_stage1",
            "expert": expert,
            "rank": self.rank,
            "lora_layer": self.lora_layer,
            "model": self._collect_state_by_keywords(self._expert_keywords(expert)),
        }
        torch.save(state, filename)

    def save_lora_parameters(self, filename: str) -> None:
        if not filename.endswith((".pt", ".pth")):
            raise ValueError("filename must end with .pt or .pth")

        model_state = {}
        model_state.update(self._collect_state_by_keywords(self._expert_keywords(None)))
        model_state.update(self._collect_state_by_keywords(self._fusion_keywords()))
        model_state.update(self._collect_state_by_keywords(self._decoder_keywords()))

        for key, value in self.sam.state_dict().items():
            if key.endswith("image_scale") or key.endswith("patch_scale") or key.endswith("pixel_scale"):
                model_state[key] = value.detach().cpu()

        state = {
            "state_dict_type": "samora_stage2",
            "rank": self.rank,
            "lora_layer": self.lora_layer,
            "use_hl_attn": self.use_hl_attn,
            "train_prompt_encoder_in_stage2": self.train_prompt_encoder_in_stage2,
            "model": model_state,
        }
        torch.save(state, filename)

    def load_expert_parameters(
        self,
        image_ckpt: Optional[str] = None,
        patch_ckpt: Optional[str] = None,
        pixel_ckpt: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        for expert_name, ckpt_path in {"image": image_ckpt, "patch": patch_ckpt, "pixel": pixel_ckpt}.items():
            if ckpt_path is None:
                continue

            payload = torch.load(ckpt_path, map_location="cpu")
            state_dict = payload.get("model", payload)
            missing, unexpected = self.sam.load_state_dict(state_dict, strict=False)

            if strict:
                bad_missing = [k for k in missing if f"experts.{expert_name}." in k]
                bad_unexpected = [k for k in unexpected if f"experts.{expert_name}." in k]
                if bad_missing or bad_unexpected:
                    raise RuntimeError(
                        f"Failed to load {expert_name} expert. missing={bad_missing}, unexpected={bad_unexpected}"
                    )

    def load_stage2_parameters(self, filename: str, strict: bool = True) -> None:
        payload = torch.load(filename, map_location="cpu")
        state_dict = payload.get("model", payload)
        missing, unexpected = self.sam.load_state_dict(state_dict, strict=False)

        if strict:
            filtered_missing = [
                k for k in missing
                if (
                    ".samora_adapter.experts." in k
                    or ".samora_adapter.fuser." in k
                    or "mask_decoder" in k
                    or (self.train_prompt_encoder_in_stage2 and "prompt_encoder" in k)
                )
            ]
            filtered_unexpected = [
                k for k in unexpected
                if (
                    ".samora_adapter.experts." in k
                    or ".samora_adapter.fuser." in k
                    or "mask_decoder" in k
                    or "prompt_encoder" in k
                )
            ]
            if filtered_missing or filtered_unexpected:
                raise RuntimeError(
                    f"Failed to load stage2 checkpoint. missing={filtered_missing}, unexpected={filtered_unexpected}"
                )

    def load_lora_parameters(self, filename: str) -> None:
        self.load_stage2_parameters(filename, strict=False)

    def forward(
        self,
        batched_input,
        multimask_output,
        image_size,
        gt=None,
        mode: str = "train",
        stage: Optional[str] = None,
        ssl_task: Optional[str] = None,
    ):
        if stage is not None and stage != self.stage:
            if stage == "stage2":
                self.freeze_for_stage2()
            elif stage.startswith("stage1"):
                expert = ssl_task if ssl_task is not None else stage.split("_")[-1]
                self.freeze_for_stage1(expert)
            else:
                raise ValueError(f"Unsupported stage: {stage}")

        return self.sam(
            batched_input,
            multimask_output,
            image_size,
            gt=gt,
            mode=mode,
            stage=stage if stage is not None else self.stage,
            ssl_task=ssl_task,
        )


MultiExpertLoRASam = LoRA_Sam
