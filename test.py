import argparse
import importlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything.build_sam import sam_model_registry
from utils import test_single_volume


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


def _dataset_config(dataset_name: str) -> Dict[str, Any]:
    cfg = {
        "Synapse": {
            "root_path": "../data/Synapse/test_vol_h5",
            "list_dir": "./lists/lists_Synapse",
            "split": "test_vol",
            "num_classes": 8,
            "z_spacing": 1,
        }
    }
    if dataset_name not in cfg:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return cfg[dataset_name]


def _resolve_variant(args) -> str:
    if args.variant != "auto":
        return args.variant

    module_name = str(args.module).lower()
    vit_name = str(args.vit_name).lower()

    if "samora_lora_sam" in module_name or vit_name.startswith("samora_"):
        return "samora"
    if "samora_lora_hsam" in module_name or vit_name.startswith("hsamora_"):
        return "hsamora"

    # fall back to dual-decoder default for backwards compatibility
    return "hsamora"


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


def _setup_logger(output_dir: str, args) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "test_log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(str(args))


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
        stage="stage2",
        train_prompt_encoder_in_stage2=args.train_prompt_encoder_in_stage2,
    )
    return net.cuda()


def _load_checkpoints(args, net, variant: str) -> None:
    target = net.module if isinstance(net, torch.nn.DataParallel) else net

    if hasattr(target, "load_expert_parameters"):
        if any(x is not None for x in [args.image_lora_ckpt, args.patch_lora_ckpt, args.pixel_lora_ckpt]):
            target.load_expert_parameters(
                image_ckpt=args.image_lora_ckpt,
                patch_ckpt=args.patch_lora_ckpt,
                pixel_ckpt=args.pixel_lora_ckpt,
                strict=args.strict_ckpt_loading,
            )

    if args.lora_ckpt:
        if hasattr(target, "load_lora_parameters"):
            target.load_lora_parameters(args.lora_ckpt)
        elif hasattr(target, "load_stage2_parameters"):
            target.load_stage2_parameters(args.lora_ckpt, strict=False)

    if hasattr(target, "freeze_for_stage2"):
        try:
            target.freeze_for_stage2(variant=variant)
        except TypeError:
            target.freeze_for_stage2()


def get_parser():
    parser = argparse.ArgumentParser(description="H-SAMora / SAMora testing entrypoint")

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="Synapse")
    parser.add_argument("--root_path", type=str, default="")
    parser.add_argument("--list_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./test_out")
    parser.add_argument("--is_savenii", type=_str2bool, default=False)
    parser.add_argument("--z_spacing", type=float, default=1.0)

    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--deterministic", type=_str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--module", type=str, default="samora_lora_hsam")
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "samora", "hsamora"])
    parser.add_argument("--vit_name", type=str, default="vit_b")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--use_hl_attn", type=_str2bool, default=True)
    parser.add_argument("--train_prompt_encoder_in_stage2", type=_str2bool, default=False)
    parser.add_argument("--strict_ckpt_loading", type=_str2bool, default=False)

    parser.add_argument("--image_lora_ckpt", type=str, default=None)
    parser.add_argument("--patch_lora_ckpt", type=str, default=None)
    parser.add_argument("--pixel_lora_ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)

    # only used by hsamora-style dual-decoder evaluation
    parser.add_argument(
        "--eval_decoder_stage",
        type=int,
        default=3,
        choices=[2, 3],
        help="For hsamora only: 2 uses outputs2, 3 averages outputs1 and outputs2 in utils.test_single_volume.",
    )

    return parser


def inference(args, model, test_save_path: Optional[str] = None):
    from datasets.dataset_synapse import Synapse_dataset

    db_test = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("%d test iterations per epoch", len(testloader))

    model.eval()
    metric_list = 0.0

    iterator = tqdm(testloader, ncols=100)
    for i_batch, sampled_batch in enumerate(iterator):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch["case_name"][0] if isinstance(sampled_batch["case_name"], list) else sampled_batch["case_name"]

        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes + 1,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
            stage=args.eval_decoder_stage if args.variant == "hsamora" else 2,
        )
        metric_list += np.array(metric_i)
        logging.info("idx %d case %s mean_dice %f mean_hd95 %f", i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])

    metric_list = metric_list / len(db_test)
    for class_i in range(1, args.num_classes + 1):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f", class_i, metric_list[class_i - 1][0], metric_list[class_i - 1][1])

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f", performance, mean_hd95)
    return performance, mean_hd95


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.config is not None:
        config_dict = _load_yaml_config(args.config)
        args = _merge_config_into_args(args, config_dict, parser)

    dataset_cfg = _dataset_config(args.dataset)
    if not args.root_path:
        args.root_path = dataset_cfg["root_path"]
    if not args.list_dir:
        args.list_dir = dataset_cfg["list_dir"]
    if not args.split:
        args.split = dataset_cfg["split"]
    if args.num_classes is None or args.num_classes <= 0:
        args.num_classes = dataset_cfg["num_classes"]
    if args.z_spacing == parser.get_default("z_spacing"):
        args.z_spacing = dataset_cfg["z_spacing"]

    args.variant = _resolve_variant(args)

    _set_random_seed(args.seed, args.deterministic)
    _setup_logger(args.output_dir, args)

    net = _build_model(args)
    _load_checkpoints(args, net, args.variant)

    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path=test_save_path)


if __name__ == "__main__":
    main()
