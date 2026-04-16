import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.hl_attn import HLAttn


class _LoRAExpert(nn.Module):
    """LoRA expert that perturbs Q and V projections for one transformer block."""

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
    """
    Block-level SAMora adapter.

    Stage 1:
        use exactly one expert branch and return that expert-conditioned
        attention output for the current block.

    Stage 2:
        freeze all experts, compute expert residuals relative to the frozen base
        attention, then fuse them with HL-Attn at block level.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        num_heads: int,
        use_hl_attn: bool = True,
        fusion_order: Tuple[str, str, str] = ("pixel", "patch", "image"),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
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

        self.image_scale = nn.Parameter(torch.tensor(1.0))
        self.patch_scale = nn.Parameter(torch.tensor(1.0))
        self.pixel_scale = nn.Parameter(torch.tensor(1.0))

        self.hl_attn = (
            HLAttn(
                dim=dim,
                num_heads=num_heads,
                fusion_order=fusion_order,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                return_only_fused_signal=True,
            )
            if use_hl_attn
            else None
        )

    def set_stage(self, stage: str, active_expert: Optional[str] = None) -> None:
        self.stage = stage
        if active_expert is not None:
            if active_expert not in self.experts:
                raise ValueError(f"Unknown expert: {active_expert}")
            self.active_expert = active_expert

    def _expert_scale(self, expert_name: str) -> torch.Tensor:
        if expert_name == "image":
            return self.image_scale
        if expert_name == "patch":
            return self.patch_scale
        if expert_name == "pixel":
            return self.pixel_scale
        raise ValueError(f"Unknown expert: {expert_name}")

    def _apply_expert_to_qkv(
        self,
        base_qkv: torch.Tensor,
        x_tokens: torch.Tensor,
        expert_name: str,
    ) -> torch.Tensor:
        expert = self.experts[expert_name]
        scale = self._expert_scale(expert_name)
        delta = expert(x_tokens)

        qkv = base_qkv.clone()
        qkv[..., : self.dim] = qkv[..., : self.dim] + scale * delta["q"]
        qkv[..., -self.dim :] = qkv[..., -self.dim :] + scale * delta["v"]
        return qkv

    def _expert_attention_output(
        self,
        block,
        x_partitioned: torch.Tensor,
        base_qkv: torch.Tensor,
        expert_name: str,
    ) -> torch.Tensor:
        qkv = self._apply_expert_to_qkv(base_qkv, x_partitioned, expert_name)
        spatial_shape = (x_partitioned.shape[1], x_partitioned.shape[2])
        return block.attn.forward_from_qkv(qkv, spatial_shape=spatial_shape)

    def forward(self, x: torch.Tensor, block) -> torch.Tensor:
        shortcut = x

        x_norm = block.norm1(x)
        x_part, pad_hw, hw = block._partition_if_needed(x_norm)
        base_qkv = block.attn.qkv(x_part)
        spatial_shape = (x_part.shape[1], x_part.shape[2])
        base_attn = block.attn.forward_from_qkv(base_qkv, spatial_shape=spatial_shape)

        if self.stage.startswith("stage1"):
            selected_attn = self._expert_attention_output(
                block=block,
                x_partitioned=x_part,
                base_qkv=base_qkv,
                expert_name=self.active_expert,
            )
            attn_out = selected_attn
        else:
            image_attn = self._expert_attention_output(block, x_part, base_qkv, "image")
            patch_attn = self._expert_attention_output(block, x_part, base_qkv, "patch")
            pixel_attn = self._expert_attention_output(block, x_part, base_qkv, "pixel")

            image_delta = image_attn - base_attn
            patch_delta = patch_attn - base_attn
            pixel_delta = pixel_attn - base_attn

            if self.use_hl_attn and self.hl_attn is not None:
                fused_signal = self.hl_attn(
                    image_feat=image_delta,
                    patch_feat=patch_delta,
                    pixel_feat=pixel_delta,
                )
            else:
                fused_signal = image_delta + patch_delta + pixel_delta

            attn_out = base_attn + fused_signal

        attn_out = block._unpartition_if_needed(attn_out, pad_hw, hw)
        x = shortcut + attn_out
        x = x + block.forward_mlp_only(x)
        return x


class LoRA_Sam(nn.Module):
    """
    Multi-expert SAMora wrapper for H-SAM.

    The main shift from the earlier qkv-only wrapper is that fusion now happens
    at transformer-block level through block.samora_forward hooks, which is
    closer to the paper's O(x) = F_theta(x) + E_omega(x) formulation.
    """

    def __init__(
        self,
        sam_model: Sam,
        r: int,
        lora_layer: Optional[List[int]] = None,
        use_hl_attn: bool = True,
        stage: str = "stage2",
        train_prompt_encoder_in_stage2: bool = False,
        fusion_order: Tuple[str, str, str] = ("pixel", "patch", "image"),
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
        self.fusion_order = fusion_order

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

            adapter = _SAMoraBlockAdapter(
                dim=blk.attn.qkv.in_features,
                rank=r,
                num_heads=blk.attn.num_heads,
                use_hl_attn=use_hl_attn,
                fusion_order=fusion_order,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            blk.samora_adapter = adapter
            blk.samora_forward = (lambda x, adapter=adapter, blk=blk: adapter(x, blk))
            self.adapters.append(adapter)

        if stage == "stage2":
            self.freeze_for_stage2()
        elif stage.startswith("stage1"):
            self.freeze_for_stage1(stage.split("_")[-1])
        else:
            raise ValueError(
                "stage must be one of {'stage2', 'stage1_image', 'stage1_patch', 'stage1_pixel'}"
            )

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    def _iter_decoder_modules(self) -> Iterable[nn.Module]:
        yield self.sam.mask_decoder
        yield self.sam.mask_decoder2
        if self.train_prompt_encoder_in_stage2:
            yield self.sam.prompt_encoder
            if self.sam.prompt_encoder2 is not self.sam.prompt_encoder:
                yield self.sam.prompt_encoder2

    def freeze_for_stage1(self, expert: str) -> None:
        if expert not in {"image", "patch", "pixel"}:
            raise ValueError("expert must be one of {'image', 'patch', 'pixel'}")

        self.stage = f"stage1_{expert}"
        self._set_requires_grad(self.sam, False)

        for adapter in self.adapters:
            adapter.set_stage(self.stage, active_expert=expert)
            for expert_name, module in adapter.experts.items():
                self._set_requires_grad(module, expert_name == expert)
            self._set_requires_grad(adapter.hl_attn, False)
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
            self._set_requires_grad(adapter.hl_attn, self.use_hl_attn)
            adapter.image_scale.requires_grad = False
            adapter.patch_scale.requires_grad = False
            adapter.pixel_scale.requires_grad = False

        for module in self._iter_decoder_modules():
            self._set_requires_grad(module, True)

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _collect_state_by_keywords(self, keywords: List[str]) -> Dict[str, torch.Tensor]:
        state = self.sam.state_dict()
        return {
            k: v.detach().cpu()
            for k, v in state.items()
            if any(word in k for word in keywords)
        }

    def _expert_keywords(self, expert: Optional[str] = None) -> List[str]:
        if expert is None:
            return [".samora_adapter.experts."]
        return [
            f".samora_adapter.experts.{expert}.",
            f".samora_adapter.{expert}_scale",
        ]

    def _fusion_keywords(self) -> List[str]:
        return [".samora_adapter.hl_attn."]

    def _decoder_keywords(self) -> List[str]:
        keys = ["mask_decoder", "mask_decoder2"]
        if self.train_prompt_encoder_in_stage2:
            keys += ["prompt_encoder", "prompt_encoder2"]
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
            "fusion_order": self.fusion_order,
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

        state = {
            "state_dict_type": "samora_stage2",
            "rank": self.rank,
            "lora_layer": self.lora_layer,
            "use_hl_attn": self.use_hl_attn,
            "fusion_order": self.fusion_order,
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
        for expert_name, ckpt_path in {
            "image": image_ckpt,
            "patch": patch_ckpt,
            "pixel": pixel_ckpt,
        }.items():
            if ckpt_path is None:
                continue
            payload = torch.load(ckpt_path, map_location="cpu")
            state_dict = payload.get("model", payload)
            missing, unexpected = self.sam.load_state_dict(state_dict, strict=False)
            if strict:
                bad_missing = [k for k in missing if f"samora_adapter.experts.{expert_name}." in k]
                bad_unexpected = [k for k in unexpected if f"samora_adapter.experts.{expert_name}." in k]
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
                    or ".samora_adapter.hl_attn." in k
                    or "mask_decoder" in k
                    or "mask_decoder2" in k
                    or (self.train_prompt_encoder_in_stage2 and ("prompt_encoder" in k or "prompt_encoder2" in k))
                )
            ]
            filtered_unexpected = [
                k for k in unexpected
                if (
                    ".samora_adapter.experts." in k
                    or ".samora_adapter.hl_attn." in k
                    or "mask_decoder" in k
                    or "mask_decoder2" in k
                    or "prompt_encoder" in k
                    or "prompt_encoder2" in k
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
            stage=self.stage if stage is None else stage,
            ssl_task=ssl_task,
        )


MultiExpertLoRASam = LoRA_Sam
