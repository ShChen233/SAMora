import argparse
import importlib
import importlib.util
import random
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from segment_anything.build_sam import sam_model_registry


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_yaml_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config support. Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _merge_config_into_args(args, config: Dict[str, Any], parser):
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if getattr(action, "dest", None) not in (None, "help")
    }
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
    return args


def _load_trainer_module():
    """
    Prefer trainer_samora_ready.py when present, otherwise fall back to
    trainer_stage1_ready.py, then trainer.py.
    """
    repo_dir = Path(__file__).resolve().parent
    for name in ["trainer_samora_ready.py", "trainer_stage1_ready.py", "trainer.py"]:
        candidate = repo_dir / name
        if not candidate.exists():
            continue
        if name == "trainer.py":
            import trainer
            return trainer

        spec = importlib.util.spec_from_file_location(candidate.stem, str(candidate))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load trainer module from {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise FileNotFoundError("No trainer module found.")


def _build_snapshot_path(args) -> str:
    base = Path(args.output)
    base.mkdir(parents=True, exist_ok=True)

    tag_parts = [
        args.dataset,
        args.stage,
        args.variant,
        args.vit_name,
        f"img{args.img_size}",
        f"epo{args.max_epochs}",
        f"bs{args.batch_size}",
        f"lr{args.base_lr}",
        f"r{args.rank}",
        f"s{args.seed}",
    ]
    if args.stage.startswith("stage1"):
        tag_parts.append(f"proj{args.stage1_proj_dim}")
        if args.stage == "stage1_patch":
            tag_parts.append(f"mask{args.mask_ratio}")
    if args.stage == "stage2":
        tag_parts.append("hlattn" if args.use_hl_attn else "nofuse")
    if getattr(args, "experiment_name", ""):
        tag_parts.append(args.experiment_name)

    snapshot_path = base / "_".join(str(x) for x in tag_parts)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    return str(snapshot_path)


def _set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False


def _dataset_config(dataset_name: str) -> Dict[str, Any]:
    cfg = {
        "Synapse": {
            "root_path": "../data/Synapse/train_npz",
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 8,
            "z_spacing": 1,
        }
    }
    if dataset_name not in cfg:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return cfg[dataset_name]


def _resolve_stage_expert(stage: str) -> str:
    mapping = {
        "stage1_image": "image",
        "stage1_patch": "patch",
        "stage1_pixel": "pixel",
    }
    if stage not in mapping:
        raise ValueError(f"Stage {stage} does not correspond to a single expert.")
    return mapping[stage]


def _infer_variant(args) -> str:
    if getattr(args, "variant", None) not in (None, "", "auto"):
        variant = str(args.variant).lower()
        if variant not in {"samora", "hsamora"}:
            raise ValueError(f"variant must be one of {{'samora','hsamora','auto'}}, got {args.variant}")
        return variant

    module_name = str(getattr(args, "module", "")).lower()
    vit_name = str(getattr(args, "vit_name", "")).lower()

    if "hsam" in module_name or "hsam" in vit_name:
        return "hsamora"
    if "samora" in module_name or "samora" in vit_name:
        return "samora"

    # safe default for the original repo behavior
    return "hsamora"


def _build_model(args):
    if args.vit_name not in sam_model_registry:
        raise KeyError(f"Unknown vit_name={args.vit_name}. Available keys: {list(sam_model_registry.keys())}")

    sam = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )

    pkg = importlib.import_module(args.module)
    net = pkg.LoRA_Sam(
        sam,
        args.rank,
        use_hl_attn=args.use_hl_attn,
        stage=args.stage,
        train_prompt_encoder_in_stage2=args.train_prompt_encoder_in_stage2,
    )
    return net.cuda()


def _load_stage_specific_checkpoints(args, net) -> None:
    target = net.module if isinstance(net, torch.nn.DataParallel) else net

    image_ckpt = args.image_lora_ckpt
    patch_ckpt = args.patch_lora_ckpt
    pixel_ckpt = args.pixel_lora_ckpt

    if args.stage.startswith("stage1"):
        expert = _resolve_stage_expert(args.stage)
        if expert == "image" and args.resume_expert_ckpt and image_ckpt is None:
            image_ckpt = args.resume_expert_ckpt
        if expert == "patch" and args.resume_expert_ckpt and patch_ckpt is None:
            patch_ckpt = args.resume_expert_ckpt
        if expert == "pixel" and args.resume_expert_ckpt and pixel_ckpt is None:
            pixel_ckpt = args.resume_expert_ckpt

    if hasattr(target, "load_expert_parameters") and any(x is not None for x in [image_ckpt, patch_ckpt, pixel_ckpt]):
        target.load_expert_parameters(
            image_ckpt=image_ckpt,
            patch_ckpt=patch_ckpt,
            pixel_ckpt=pixel_ckpt,
            strict=args.strict_ckpt_loading,
        )

    if args.lora_ckpt:
        if hasattr(target, "load_lora_parameters"):
            target.load_lora_parameters(args.lora_ckpt)
        elif hasattr(target, "load_stage2_parameters"):
            target.load_stage2_parameters(args.lora_ckpt, strict=False)

    if args.stage == "stage2" and hasattr(target, "freeze_for_stage2"):
        try:
            target.freeze_for_stage2(variant=args.variant)
        except TypeError:
            target.freeze_for_stage2()
    elif args.stage.startswith("stage1") and hasattr(target, "freeze_for_stage1"):
        target.freeze_for_stage1(_resolve_stage_expert(args.stage))


def _dispatch_trainer(args, trainer_module) -> Tuple[Callable, bool]:
    if args.stage == "stage2":
        if args.variant == "samora":
            if not hasattr(trainer_module, "trainer_synapse_samora"):
                raise AttributeError("trainer module does not define trainer_synapse_samora")
            return trainer_module.trainer_synapse_samora, True

        if not hasattr(trainer_module, "trainer_synapse_hsamora"):
            raise AttributeError("trainer module does not define trainer_synapse_hsamora")
        return trainer_module.trainer_synapse_hsamora, True

    if args.stage == "stage1_image":
        return trainer_module.trainer_stage1_image, False
    if args.stage == "stage1_patch":
        return trainer_module.trainer_stage1_patch, False
    if args.stage == "stage1_pixel":
        return trainer_module.trainer_stage1_pixel, False

    raise ValueError(f"Unsupported stage: {args.stage}")


def get_parser():
    parser = argparse.ArgumentParser(description="SAMora / H-SAMora training entrypoint")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="Synapse")
    parser.add_argument("--output", type=str, default="./model_out")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--stage", type=str, default="stage2", choices=["stage2", "stage1_image", "stage1_patch", "stage1_pixel"])
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "samora", "hsamora"])

    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--list_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--low_res", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_gpu", type=int, default=1)

    parser.add_argument("--unlabeled_roots", type=str, default="")
    parser.add_argument("--stage1_max_samples", type=int, default=None)
    parser.add_argument("--stage1_num_workers", type=int, default=4)

    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--stop_epoch", type=int, default=30)
    parser.add_argument("--base_lr", type=float, default=2.5e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--warmup", type=_str2bool, default=True)
    parser.add_argument("--warmup_period", type=int, default=25)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--dice_param", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=_str2bool, default=True)

    parser.add_argument("--vit_name", type=str, default="vit_b")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--module", type=str, default="samora_lora_hsam")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--use_hl_attn", type=_str2bool, default=True)
    parser.add_argument("--train_prompt_encoder_in_stage2", type=_str2bool, default=False)
    parser.add_argument("--strict_ckpt_loading", type=_str2bool, default=False)

    parser.add_argument("--image_lora_ckpt", type=str, default=None)
    parser.add_argument("--patch_lora_ckpt", type=str, default=None)
    parser.add_argument("--pixel_lora_ckpt", type=str, default=None)
    parser.add_argument("--resume_expert_ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)

    parser.add_argument("--simclr_backbone", type=str, default="resnet50")
    parser.add_argument("--stage1_teacher_pretrained", type=_str2bool, default=False)
    parser.add_argument("--stage1_distill_mode", type=str, default="mse", choices=["mse", "cosine"])

    parser.add_argument("--stage1_proj_dim", type=int, default=256)
    parser.add_argument("--stage1_proj_hidden_dim", type=int, default=2048)

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--mae_embed_dim", type=int, default=1024)
    parser.add_argument("--mae_depth", type=int, default=8)
    parser.add_argument("--mae_num_heads", type=int, default=16)
    parser.add_argument("--mae_decoder_embed_dim", type=int, default=512)
    parser.add_argument("--mae_decoder_depth", type=int, default=2)
    parser.add_argument("--mae_decoder_num_heads", type=int, default=8)

    parser.add_argument("--pixel_decoder_channels", type=int, nargs="+", default=[256, 128, 64, 32, 16])
    parser.add_argument("--pixel_decoder_activation", type=str, default="sigmoid", choices=["sigmoid", "tanh", "identity", "none"])
    parser.add_argument("--pixel_dice_weight", type=float, default=0.9)
    parser.add_argument("--pixel_mse_weight", type=float, default=0.1)

    parser.add_argument("--AdamW", action="store_true")
    parser.add_argument("--multimask_output", type=int, default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.config is not None:
        config_dict = _load_yaml_config(args.config)
        args = _merge_config_into_args(args, config_dict, parser)

    args.variant = _infer_variant(args)

    dataset_cfg = _dataset_config(args.dataset)
    if not args.root_path:
        args.root_path = dataset_cfg["root_path"]
    if not args.list_dir:
        args.list_dir = dataset_cfg["list_dir"]
    if args.num_classes is None or args.num_classes <= 0:
        args.num_classes = dataset_cfg["num_classes"]

    snapshot_path = _build_snapshot_path(args)
    _set_random_seed(args.seed, args.deterministic)

    trainer_module = _load_trainer_module()
    trainer_fn, requires_stage2_args = _dispatch_trainer(args, trainer_module)

    net = _build_model(args)
    _load_stage_specific_checkpoints(args, net)

    if requires_stage2_args:
        trainer_fn(args, net, snapshot_path, args.multimask_output, args.low_res)
    else:
        trainer_fn(args, net, snapshot_path)


if __name__ == "__main__":
    main()
