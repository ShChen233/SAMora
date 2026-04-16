from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


TensorOrBatch = Union[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]


class Sam(nn.Module):
    """
    Unified SAM wrapper that supports both:

    - SAMora / SAMed-style single-decoder training
    - H-SAMora / H-SAM-style dual-decoder training

    Behavior is inferred from whether prompt_encoder2 and mask_decoder2 are
    provided, but can also be overridden at call time with `variant`.
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        *,
        prompt_encoder2: Optional[nn.Module] = None,
        mask_decoder2: Optional[nn.Module] = None,
        pixel_mean=(123.675, 116.28, 103.53),
        pixel_std=(58.395, 57.12, 57.375),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.has_dual_prompt = prompt_encoder2 is not None
        self.has_dual_decoder = mask_decoder2 is not None
        self.dual_branch = self.has_dual_prompt and self.has_dual_decoder

        self.prompt_encoder2 = prompt_encoder if prompt_encoder2 is None else prompt_encoder2
        self.mask_decoder2 = mask_decoder if mask_decoder2 is None else mask_decoder2

        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    # ------------------------------------------------------------------
    # image utils
    # ------------------------------------------------------------------
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got shape={tuple(x.shape)}")
        x = (x - self.pixel_mean) / self.pixel_std

        target_size = int(getattr(self.image_encoder, "img_size", x.shape[-1]))
        h, w = x.shape[-2:]
        pad_h = max(target_size - h, 0)
        pad_w = max(target_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        target_size = int(getattr(self.image_encoder, "img_size", masks.shape[-1]))
        masks = F.interpolate(
            masks,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    # ------------------------------------------------------------------
    # input / encoder helpers
    # ------------------------------------------------------------------
    def _prepare_image_batch(self, batched_input: TensorOrBatch) -> torch.Tensor:
        if isinstance(batched_input, torch.Tensor):
            if batched_input.ndim == 3:
                batched_input = batched_input.unsqueeze(0)
            if batched_input.ndim != 4:
                raise ValueError(f"Expected BCHW tensor, got shape={tuple(batched_input.shape)}")
            return batched_input

        if isinstance(batched_input, dict):
            if "image" not in batched_input:
                raise KeyError("batched_input dict must contain key 'image'")
            image = batched_input["image"]
            if not isinstance(image, torch.Tensor):
                raise TypeError("batched_input['image'] must be a torch.Tensor")
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return image

        if isinstance(batched_input, list):
            images = []
            for item in batched_input:
                if not isinstance(item, dict) or "image" not in item:
                    raise ValueError("List input requires items like {'image': tensor}")
                image = item["image"]
                if image.ndim == 3:
                    images.append(image)
                elif image.ndim == 4 and image.shape[0] == 1:
                    images.append(image[0])
                else:
                    raise ValueError(f"Unsupported per-item image shape: {tuple(image.shape)}")
            return torch.stack(images, dim=0)

        raise TypeError(f"Unsupported batched_input type: {type(batched_input)}")

    def _extract_encoder_outputs(
        self,
        input_images: torch.Tensor,
        *,
        return_intermediate: bool = True,
        return_all_blocks: bool = False,
        use_samora_path: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        kwargs = {
            "return_intermediate": return_intermediate,
            "return_all_blocks": return_all_blocks,
            "use_samora_path": use_samora_path,
        }

        try:
            out = self.image_encoder(input_images, **kwargs)
        except TypeError:
            try:
                out = self.image_encoder(
                    input_images,
                    return_intermediate=return_intermediate,
                    return_all_blocks=return_all_blocks,
                )
            except TypeError:
                out = self.image_encoder(input_images)

        if isinstance(out, tuple):
            if len(out) == 3:
                image_embeddings, low_image_embeddings, aux = out
            elif len(out) == 2:
                image_embeddings, low_image_embeddings = out
                aux = {}
            else:
                raise RuntimeError(f"Unexpected image encoder output tuple length: {len(out)}")
        else:
            image_embeddings = out
            low_image_embeddings = out
            aux = {}

        if not isinstance(aux, dict):
            aux = {"encoder_aux": aux}

        aux.setdefault("image_embeddings", image_embeddings)
        aux.setdefault("low_image_embeddings", low_image_embeddings)
        return image_embeddings, low_image_embeddings, aux

    def _resolve_variant(self, variant: Optional[str] = None) -> str:
        if variant is not None:
            variant = str(variant).lower()
            if variant not in {"samora", "hsamora"}:
                raise ValueError(f"variant must be 'samora' or 'hsamora', got {variant}")
            return variant
        return "hsamora" if self.dual_branch else "samora"

    def _get_prompt_pe(self, prompt_encoder: nn.Module) -> torch.Tensor:
        if not hasattr(prompt_encoder, "get_dense_pe"):
            raise AttributeError("Prompt encoder must implement get_dense_pe().")
        return prompt_encoder.get_dense_pe()

    def _call_prompt_encoder(
        self,
        prompt_encoder: nn.Module,
        points=None,
        boxes=None,
        masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return prompt_encoder(points=points, boxes=boxes, masks=masks)

    def _call_decoder(
        self,
        decoder: nn.Module,
        *,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        gt: Optional[torch.Tensor] = None,
        low_image_embeddings: Optional[torch.Tensor] = None,
    ):
        attempts = [
            {
                "image_embeddings": image_embeddings,
                "image_pe": image_pe,
                "sparse_prompt_embeddings": sparse_prompt_embeddings,
                "dense_prompt_embeddings": dense_prompt_embeddings,
                "multimask_output": multimask_output,
                "gt": gt,
            },
            {
                "image_embeddings": image_embeddings,
                "low_image_embeddings": low_image_embeddings,
                "image_pe": image_pe,
                "sparse_prompt_embeddings": sparse_prompt_embeddings,
                "dense_prompt_embeddings": dense_prompt_embeddings,
                "multimask_output": multimask_output,
                "gt": gt,
            },
            {
                "image_embeddings": image_embeddings,
                "ps": low_image_embeddings,
                "image_pe": image_pe,
                "sparse_prompt_embeddings": sparse_prompt_embeddings,
                "dense_prompt_embeddings": dense_prompt_embeddings,
                "multimask_output": multimask_output,
                "gt": gt,
            },
        ]

        last_error = None
        for kwargs in attempts:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return decoder(**kwargs)
            except TypeError as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Decoder invocation failed unexpectedly.")

    @staticmethod
    def _unpack_decoder_output(out):
        if isinstance(out, tuple):
            if len(out) == 2:
                return out[0], out[1]
            if len(out) >= 1:
                return out[0], None
        return out, None

    # ------------------------------------------------------------------
    # stage1 ssl
    # ------------------------------------------------------------------
    def _pack_stage1_feature_dict(
        self,
        x: torch.Tensor,
        image_embeddings: torch.Tensor,
        low_image_embeddings: torch.Tensor,
        aux: Dict[str, Any],
        task_name: str,
    ) -> Dict[str, Any]:
        packed = {
            "task": task_name,
            "input": x,
            "image_embeddings": image_embeddings,
            "low_image_embeddings": low_image_embeddings,
            "encoder_aux": aux,
            "input_size": tuple(x.shape[-2:]),
        }
        if isinstance(aux, dict):
            packed.update({k: v for k, v in aux.items() if k not in packed})
        return packed

    def _forward_single_stage1_tensor(self, x: torch.Tensor, task_name: str) -> Dict[str, Any]:
        x = x.to(self.device)
        image_embeddings, low_image_embeddings, aux = self._extract_encoder_outputs(
            x,
            return_intermediate=True,
            return_all_blocks=False,
            use_samora_path=True,
        )
        return self._pack_stage1_feature_dict(
            x=x,
            image_embeddings=image_embeddings,
            low_image_embeddings=low_image_embeddings,
            aux=aux,
            task_name=task_name,
        )

    def forward_stage1_ssl(
        self,
        batched_input: TensorOrBatch,
        ssl_task: str,
        image_size: int,
    ) -> Dict[str, Any]:
        del image_size
        ssl_task = str(ssl_task).lower()

        if ssl_task not in {"image", "patch", "pixel"}:
            raise ValueError(f"ssl_task must be one of {{'image','patch','pixel'}}, got {ssl_task}")

        if ssl_task == "image":
            if isinstance(batched_input, dict):
                result = {"task": "image"}
                if "view1" in batched_input:
                    result["view1"] = self._forward_single_stage1_tensor(batched_input["view1"], "image")
                if "view2" in batched_input:
                    result["view2"] = self._forward_single_stage1_tensor(batched_input["view2"], "image")
                if "image" in batched_input and "view1" not in batched_input and "view2" not in batched_input:
                    result["image"] = self._forward_single_stage1_tensor(batched_input["image"], "image")
                if len(result) == 1:
                    raise KeyError("For ssl_task='image', batched_input must contain 'view1'/'view2' or 'image'.")
                return result

            if isinstance(batched_input, torch.Tensor):
                return {
                    "task": "image",
                    "image": self._forward_single_stage1_tensor(batched_input, "image"),
                }

            raise TypeError(f"Unsupported batched_input type for image ssl: {type(batched_input)}")

        if ssl_task == "patch":
            if isinstance(batched_input, dict):
                if "image" not in batched_input:
                    raise KeyError("For ssl_task='patch', batched_input must contain key 'image'.")
                return {
                    "task": "patch",
                    "image": self._forward_single_stage1_tensor(batched_input["image"], "patch"),
                    "mask": batched_input.get("mask", None),
                }

            if isinstance(batched_input, torch.Tensor):
                return {
                    "task": "patch",
                    "image": self._forward_single_stage1_tensor(batched_input, "patch"),
                    "mask": None,
                }

            raise TypeError(f"Unsupported batched_input type for patch ssl: {type(batched_input)}")

        if isinstance(batched_input, dict):
            noisy = batched_input.get("noisy_image", batched_input.get("image", None))
            if noisy is None:
                raise KeyError("For ssl_task='pixel', batched_input must contain 'noisy_image' or 'image'.")
            target = batched_input.get("image", None)
            packed_noisy = self._forward_single_stage1_tensor(noisy, "pixel")
            target_size = tuple(target.shape[-2:]) if isinstance(target, torch.Tensor) else packed_noisy["input_size"]
            return {
                "task": "pixel",
                "noisy": packed_noisy,
                "target": target.to(self.device) if isinstance(target, torch.Tensor) else None,
                "target_size": target_size,
            }

        if isinstance(batched_input, torch.Tensor):
            packed_noisy = self._forward_single_stage1_tensor(batched_input, "pixel")
            return {
                "task": "pixel",
                "noisy": packed_noisy,
                "target": None,
                "target_size": packed_noisy["input_size"],
            }

        raise TypeError(f"Unsupported batched_input type for pixel ssl: {type(batched_input)}")

    # ------------------------------------------------------------------
    # main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batched_input: TensorOrBatch,
        multimask_output: bool,
        image_size: int,
        gt: Optional[torch.Tensor] = None,
        mode: str = "train",
        stage: str = "stage2",
        ssl_task: Optional[str] = None,
        variant: Optional[str] = None,
    ):
        if stage.startswith("stage1"):
            return self.forward_stage1_ssl(
                batched_input=batched_input,
                ssl_task=ssl_task if ssl_task is not None else stage.split("_")[-1],
                image_size=image_size,
            )

        if mode == "test":
            return self.forward_test(
                batched_input=batched_input,
                multimask_output=multimask_output,
                image_size=image_size,
                gt=gt,
                variant=variant,
            )

        return self.forward_train(
            batched_input=batched_input,
            multimask_output=multimask_output,
            image_size=image_size,
            gt=gt,
            variant=variant,
        )

    def forward_train(
        self,
        batched_input: TensorOrBatch,
        multimask_output: bool,
        image_size: int,
        input_points=None,
        gt: Optional[torch.Tensor] = None,
        mode: str = "train",
        stage: str = "stage2",
        ssl_task: Optional[str] = None,
        variant: Optional[str] = None,
    ):
        del image_size, mode, stage, ssl_task
        resolved_variant = self._resolve_variant(variant)

        input_images = self._prepare_image_batch(batched_input).to(self.device)
        image_embeddings, low_image_embeddings, aux = self._extract_encoder_outputs(
            input_images,
            return_intermediate=True,
            return_all_blocks=False,
            use_samora_path=True,
        )

        sparse_embeddings1, dense_embeddings1 = self._call_prompt_encoder(
            self.prompt_encoder,
            points=input_points,
            boxes=None,
            masks=None,
        )
        image_pe1 = self._get_prompt_pe(self.prompt_encoder)

        if resolved_variant == "samora":
            out = self._call_decoder(
                self.mask_decoder,
                image_embeddings=image_embeddings,
                low_image_embeddings=low_image_embeddings,
                image_pe=image_pe1,
                sparse_prompt_embeddings=sparse_embeddings1,
                dense_prompt_embeddings=dense_embeddings1,
                multimask_output=multimask_output,
                gt=gt,
            )
            outputs, attn = self._unpack_decoder_output(out)
            return outputs, attn

        sparse_embeddings2, dense_embeddings2 = self._call_prompt_encoder(
            self.prompt_encoder2,
            points=input_points,
            boxes=None,
            masks=None,
        )
        image_pe2 = self._get_prompt_pe(self.prompt_encoder2)

        out1 = self._call_decoder(
            self.mask_decoder,
            image_embeddings=low_image_embeddings,
            low_image_embeddings=low_image_embeddings,
            image_pe=image_pe1,
            sparse_prompt_embeddings=sparse_embeddings1,
            dense_prompt_embeddings=dense_embeddings1,
            multimask_output=multimask_output,
            gt=gt,
        )
        outputs1, attn1 = self._unpack_decoder_output(out1)

        out2 = self._call_decoder(
            self.mask_decoder2,
            image_embeddings=image_embeddings,
            low_image_embeddings=low_image_embeddings,
            image_pe=image_pe2,
            sparse_prompt_embeddings=sparse_embeddings2,
            dense_prompt_embeddings=dense_embeddings2,
            multimask_output=multimask_output,
            gt=gt,
        )
        outputs2, attn2 = self._unpack_decoder_output(out2)

        return outputs1, outputs2, attn1, attn2

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: TensorOrBatch,
        multimask_output: bool,
        image_size: int,
        input_points=None,
        gt: Optional[torch.Tensor] = None,
        mode: str = "test",
        stage: str = "stage2",
        ssl_task: Optional[str] = None,
        variant: Optional[str] = None,
    ):
        del mode, stage, ssl_task
        return self.forward_train(
            batched_input=batched_input,
            multimask_output=multimask_output,
            image_size=image_size,
            input_points=input_points,
            gt=gt,
            variant=variant,
        )


    def freeze_encoder_and_prompt(self) -> None:
        modules = [self.image_encoder, self.prompt_encoder]
        if self.dual_branch:
            modules.append(self.prompt_encoder2)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_decoders(self, variant: Optional[str] = None) -> None:
        resolved_variant = self._resolve_variant(variant)
        modules = [self.mask_decoder] if resolved_variant == "samora" else [self.mask_decoder, self.mask_decoder2]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def freeze_for_stage2(self, variant: Optional[str] = None) -> None:
        self.freeze_encoder_and_prompt()
        self.unfreeze_decoders(variant=variant)
