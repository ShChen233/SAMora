import importlib.util
import logging
import os
import random
import sys
from types import ModuleType
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------

def calc_loss(
    outputs,
    label_batch: torch.Tensor,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    dice_weight: float = 0.8,
):
    """Compute CE + Dice on outputs['low_res_logits']."""
    logits = outputs["low_res_logits"]
    loss_ce = ce_loss(logits, label_batch.long())
    loss_dice = dice_loss(logits, label_batch, softmax=True)
    loss = (1.0 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def _setup_logger(snapshot_path: str, args) -> None:
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def _worker_init_fn(seed: int):
    def _fn(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    return _fn


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _iter_trainable_named_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def _build_optimizer(args, params, base_lr: float):
    params = list(params)
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found. Check your stage-specific freezing logic.")

    use_adamw = getattr(args, "optimizer", "adamw").lower() == "adamw" or getattr(args, "AdamW", False)
    if use_adamw:
        return optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)
    return optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=1e-4)


def _adjust_learning_rate(args, optimizer, iter_num: int, max_iterations: int, base_lr: float) -> float:
    warmup = getattr(args, "warmup", False)
    warmup_period = int(getattr(args, "warmup_period", 0))

    if warmup and iter_num < warmup_period:
        lr_ = base_lr * float(iter_num + 1) / float(max(1, warmup_period))
    else:
        shift_iter = iter_num - warmup_period if warmup else iter_num
        shift_iter = max(shift_iter, 0)
        denom = max(1, max_iterations - (warmup_period if warmup else 0))
        lr_ = base_lr * (1.0 - float(shift_iter) / float(max(1, denom))) ** 0.9

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_
    return lr_


def _save_stage2_checkpoint(model: nn.Module, snapshot_path: str, epoch_num: int) -> str:
    save_mode_path = os.path.join(snapshot_path, f"epoch_{epoch_num}.pth")
    target = _unwrap_model(model)
    if not hasattr(target, "save_lora_parameters"):
        raise AttributeError("Model does not implement save_lora_parameters().")
    target.save_lora_parameters(save_mode_path)
    logging.info("save model to %s", save_mode_path)
    return save_mode_path


def _save_stage1_expert_checkpoint(model: nn.Module, snapshot_path: str, epoch_num: int, expert: str) -> str:
    save_mode_path = os.path.join(snapshot_path, f"{expert}_expert_epoch_{epoch_num}.pth")
    target = _unwrap_model(model)
    if not hasattr(target, "save_stage1_parameters"):
        raise AttributeError("Model does not implement save_stage1_parameters().")
    target.save_stage1_parameters(save_mode_path, expert=expert)
    logging.info("save %s expert to %s", expert, save_mode_path)
    return save_mode_path


def _resolve_unlabeled_roots(args) -> List[str]:
    raw = getattr(args, "unlabeled_roots", None)
    if raw is None or raw == "":
        root_path = getattr(args, "root_path", None)
        if root_path is None:
            raise ValueError("No unlabeled roots provided. Set args.unlabeled_roots or args.root_path.")
        return [root_path]

    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        return parts if len(parts) > 0 else [getattr(args, "root_path", ".")]

    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]

    return [str(raw)]


def _load_local_module(module_key: str, relative_path: str) -> ModuleType:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Required local module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage1_modules():
    dataset_module = _load_local_module("samora_dataset_unlabeled", "datasets/dataset_unlabeled.py")
    losses_module = _load_local_module("samora_ssl_losses", "ssl/losses.py")
    projector_module = _load_local_module("samora_ssl_projector", "ssl/projector.py")
    denoise_module = _load_local_module("samora_ssl_denoise_decoder", "ssl/denoise_decoder.py")
    simclr_module = _load_local_module("samora_ssl_teacher_simclr", "ssl/teacher_simclr.py")
    mae_module = _load_local_module("samora_ssl_teacher_mae", "ssl/teacher_mae.py")
    return dataset_module, losses_module, projector_module, denoise_module, simclr_module, mae_module


def _as_bnc(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        # BHWC
        if tokens.shape[-1] <= 4096:
            b, h, w, c = tokens.shape
            return tokens.reshape(b, h * w, c)
        # BCHW
        b, c, h, w = tokens.shape
        return tokens.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if tokens.ndim == 3:
        return tokens
    raise ValueError(f"Unsupported token shape: {tuple(tokens.shape)}")


def _forward_image_encoder(target_model: nn.Module, x: torch.Tensor):
    out = target_model.sam.image_encoder(
        x,
        return_intermediate=True,
        return_all_blocks=False,
        use_samora_path=True,
    )
    if isinstance(out, tuple) and len(out) == 3:
        image_embeddings, low_level_tokens, aux = out
    elif isinstance(out, tuple) and len(out) == 2:
        image_embeddings, low_level_tokens = out
        aux = {
            "tokens_before_neck": low_level_tokens,
            "low_level_tokens": low_level_tokens,
            "image_embeddings": image_embeddings,
        }
    else:
        raise RuntimeError("Unexpected image encoder output format.")
    return image_embeddings, low_level_tokens, aux


def _infer_stage1_feature_dims(target_model: nn.Module, img_size: int, device: torch.device):
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        image_embeddings, low_level_tokens, aux = _forward_image_encoder(target_model, dummy)

    image_embed_dim = image_embeddings.shape[1] if image_embeddings.ndim == 4 else image_embeddings.shape[-1]
    token_tensor = aux.get("tokens_before_neck", low_level_tokens)
    token_tensor = _as_bnc(token_tensor)
    token_dim = token_tensor.shape[-1]
    return image_embed_dim, token_dim


def _log_stage1_model_status(model: nn.Module) -> None:
    total_params, trainable_params = _count_parameters(model)
    logging.info("model_total_params: %d", total_params)
    logging.info("model_trainable_params: %d", trainable_params)
    logging.info("trainable parameter names: %s", [name for name, _ in _iter_trainable_named_parameters(model)])


def _save_aux_module_checkpoint(module: nn.Module, snapshot_path: str, epoch_num: int, prefix: str) -> str:
    ckpt_path = os.path.join(snapshot_path, f"{prefix}_epoch_{epoch_num}.pth")
    torch.save(module.state_dict(), ckpt_path)
    logging.info("save %s auxiliary module to %s", prefix, ckpt_path)
    return ckpt_path


def _build_stage1_dataloader(args, ssl_task: str):
    dataset_module, *_ = _load_stage1_modules()
    UnlabeledMedicalSliceDataset = dataset_module.UnlabeledMedicalSliceDataset

    root_dirs = _resolve_unlabeled_roots(args)
    max_samples = getattr(args, "stage1_max_samples", None)
    num_workers = int(getattr(args, "stage1_num_workers", 4))

    dataset = UnlabeledMedicalSliceDataset(
        root_dirs=root_dirs,
        image_size=int(args.img_size),
        ssl_task=ssl_task,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn(int(args.seed)),
    )
    return dataset, loader


# -----------------------------------------------------------------------------
# Stage 2: H-SAMora fine-tuning on Synapse
# -----------------------------------------------------------------------------

def trainer_synapse_hsamora(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    _setup_logger(snapshot_path, args)

    base_lr = float(args.base_lr)
    num_classes = int(args.num_classes)
    batch_size = int(args.batch_size) * int(args.n_gpu)

    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]
        ),
    )
    logging.info("The length of train set is: %d", len(db_train))

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=_worker_init_fn(args.seed),
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    target_model = _unwrap_model(model)
    if hasattr(target_model, "freeze_for_stage2"):
        target_model.freeze_for_stage2()

    model.train()

    total_params, trainable_params = _count_parameters(model)
    logging.info("model_total_params: %d", total_params)
    logging.info("model_trainable_params: %d", trainable_params)
    logging.info("trainable parameter names: %s", [name for name, _ in _iter_trainable_named_parameters(model)])

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    start_lr = base_lr / max(1, int(args.warmup_period)) if getattr(args, "warmup", False) else base_lr
    optimizer = _build_optimizer(args, filter(lambda p: p.requires_grad, model.parameters()), start_lr)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    max_epoch = int(args.max_epochs)
    stop_epoch = int(args.stop_epoch)
    max_iterations = max_epoch * len(trainloader)
    logging.info("%d iterations per epoch. %d max iterations", len(trainloader), max_iterations)

    save_interval = getattr(args, "save_interval", max(1, max_epoch // 3))
    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda(non_blocking=True)
            label_batch = sampled_batch["label"].cuda(non_blocking=True)
            low_res_label_batch = sampled_batch["low_res_label"].cuda(non_blocking=True)

            outputs1, outputs2, _, _ = model(
                image_batch,
                multimask_output,
                args.img_size,
                gt=low_res_label_batch,
                mode="train",
                stage="stage2",
                variant="hsamora",
            )

            loss1, loss_ce1, loss_dice1 = calc_loss(
                outputs1,
                low_res_label_batch,
                ce_loss,
                dice_loss,
                dice_weight=args.dice_param,
            )
            loss2, loss_ce2, loss_dice2 = calc_loss(
                outputs2,
                label_batch,
                ce_loss,
                dice_loss,
                dice_weight=args.dice_param,
            )

            weight = 0.6 ** (0.990 ** epoch_num)
            loss = (1.0 - weight) * loss1 + weight * loss2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr_ = _adjust_learning_rate(args, optimizer, iter_num, max_iterations, base_lr)
            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss1", loss1.item(), iter_num)
            writer.add_scalar("info/loss2", loss2.item(), iter_num)
            writer.add_scalar("info/loss_ce1", loss_ce1.item(), iter_num)
            writer.add_scalar("info/loss_dice1", loss_dice1.item(), iter_num)
            writer.add_scalar("info/loss_ce2", loss_ce2.item(), iter_num)
            writer.add_scalar("info/loss_dice2", loss_dice2.item(), iter_num)

            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

            logging.info(
                "epoch %d iter %d : loss=%f, loss1=%f, loss2=%f, loss_ce1=%f, loss_dice1=%f, loss_ce2=%f, loss_dice2=%f",
                epoch_num,
                i_batch,
                loss.item(),
                loss1.item(),
                loss2.item(),
                loss_ce1.item(),
                loss_dice1.item(),
                loss_ce2.item(),
                loss_dice2.item(),
            )

        num_batches = max(1, len(trainloader))
        avg_loss = epoch_loss / num_batches
        avg_loss1 = epoch_loss1 / num_batches
        avg_loss2 = epoch_loss2 / num_batches
        writer.add_scalar("epoch/avg_total_loss", avg_loss, epoch_num)
        writer.add_scalar("epoch/avg_loss1", avg_loss1, epoch_num)
        writer.add_scalar("epoch/avg_loss2", avg_loss2, epoch_num)
        logging.info(
            "epoch %d summary : avg_total_loss=%f, avg_loss1=%f, avg_loss2=%f",
            epoch_num,
            avg_loss,
            avg_loss1,
            avg_loss2,
        )

        if (epoch_num + 1) % save_interval == 0:
            _save_stage2_checkpoint(model, snapshot_path, epoch_num)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            _save_stage2_checkpoint(model, snapshot_path, epoch_num)
            iterator.close()
            break

    writer.close()
    return "H-SAMora Training Finished!"


# backward compatibility alias
trainer_synapse = trainer_synapse_hsamora


# -----------------------------------------------------------------------------
# Stage 2: SAMora fine-tuning on Synapse
# -----------------------------------------------------------------------------

def trainer_synapse_samora(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    _setup_logger(snapshot_path, args)

    base_lr = float(args.base_lr)
    num_classes = int(args.num_classes)
    batch_size = int(args.batch_size) * int(args.n_gpu)

    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]
        ),
    )
    logging.info("The length of train set is: %d", len(db_train))

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=_worker_init_fn(args.seed),
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    target_model = _unwrap_model(model)
    if hasattr(target_model, "freeze_for_stage2"):
        try:
            target_model.freeze_for_stage2(variant="samora")
        except TypeError:
            target_model.freeze_for_stage2()

    model.train()

    total_params, trainable_params = _count_parameters(model)
    logging.info("model_total_params: %d", total_params)
    logging.info("model_trainable_params: %d", trainable_params)
    logging.info("trainable parameter names: %s", [name for name, _ in _iter_trainable_named_parameters(model)])

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    start_lr = base_lr / max(1, int(args.warmup_period)) if getattr(args, "warmup", False) else base_lr
    optimizer = _build_optimizer(args, filter(lambda p: p.requires_grad, model.parameters()), start_lr)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    max_epoch = int(args.max_epochs)
    stop_epoch = int(args.stop_epoch)
    max_iterations = max_epoch * len(trainloader)
    logging.info("%d iterations per epoch. %d max iterations", len(trainloader), max_iterations)

    save_interval = getattr(args, "save_interval", max(1, max_epoch // 3))
    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda(non_blocking=True)
            label_batch = sampled_batch["label"].cuda(non_blocking=True)

            outputs, _ = model(
                image_batch,
                multimask_output,
                args.img_size,
                gt=label_batch,
                mode="train",
                stage="stage2",
                variant="samora",
            )

            loss, loss_ce, loss_dice = calc_loss(
                outputs,
                label_batch,
                ce_loss,
                dice_loss,
                dice_weight=args.dice_param,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr_ = _adjust_learning_rate(args, optimizer, iter_num, max_iterations, base_lr)
            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("info/loss_dice", loss_dice.item(), iter_num)

            epoch_loss += loss.item()

            logging.info(
                "epoch %d iter %d : loss=%f, loss_ce=%f, loss_dice=%f",
                epoch_num,
                i_batch,
                loss.item(),
                loss_ce.item(),
                loss_dice.item(),
            )

        num_batches = max(1, len(trainloader))
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("epoch/avg_total_loss", avg_loss, epoch_num)
        logging.info(
            "epoch %d summary : avg_total_loss=%f",
            epoch_num,
            avg_loss,
        )

        if (epoch_num + 1) % save_interval == 0:
            _save_stage2_checkpoint(model, snapshot_path, epoch_num)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            _save_stage2_checkpoint(model, snapshot_path, epoch_num)
            iterator.close()
            break

    writer.close()
    return "SAMora Training Finished!"


# -----------------------------------------------------------------------------
# Stage 1 image-level
# -----------------------------------------------------------------------------

def trainer_stage1_image(args, model, snapshot_path):
    _setup_logger(snapshot_path, args)
    _, losses_module, projector_module, _, simclr_module, _ = _load_stage1_modules()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_model = _unwrap_model(model)
    target_model.freeze_for_stage1("image")
    model.train()

    _, trainloader = _build_stage1_dataloader(args, ssl_task="image")
    image_embed_dim, _ = _infer_stage1_feature_dims(target_model, int(args.img_size), device)

    teacher = simclr_module.build_simclr_teacher(
        backbone_name=getattr(args, "simclr_backbone", "resnet50"),
        in_chans=3,
        feat_dim=int(getattr(args, "simclr_feat_dim", 2048)),
        proj_dim=int(getattr(args, "stage1_proj_dim", 256)),
        proj_hidden_dim=int(getattr(args, "stage1_proj_hidden_dim", 2048)),
        pretrained=bool(getattr(args, "stage1_teacher_pretrained", False)),
        freeze_backbone=True,
    ).to(device)
    teacher.eval()

    student_projector = projector_module.build_projector(
        task="image",
        in_dim=image_embed_dim,
        hidden_dim=int(getattr(args, "stage1_proj_hidden_dim", 512)),
        out_dim=int(getattr(args, "stage1_proj_dim", 256)),
        pool_type="mean",
        normalize_output=True,
    ).to(device)

    criterion = losses_module.build_stage1_loss(
        "image",
        distill_mode=getattr(args, "stage1_distill_mode", "mse"),
        symmetrize=True,
    ).to(device)

    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(student_projector.parameters())
    optimizer = _build_optimizer(args, params, float(args.base_lr))

    _log_stage1_model_status(model)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    max_epoch = int(args.max_epochs)
    stop_epoch = int(args.stop_epoch)
    max_iterations = max_epoch * len(trainloader)
    save_interval = getattr(args, "save_interval", max(1, max_epoch // 3))
    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0

        for i_batch, batch in enumerate(trainloader):
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)

            student_feat1, _, _ = _forward_image_encoder(target_model, view1)
            student_feat2, _, _ = _forward_image_encoder(target_model, view2)
            student_proj1 = student_projector(student_feat1)
            student_proj2 = student_projector(student_feat2)

            with torch.no_grad():
                teacher_proj1 = teacher(view1, return_features=False, return_projection=True)
                teacher_proj2 = teacher(view2, return_features=False, return_projection=True)

            loss_dict = criterion(student_proj1, student_proj2, teacher_proj1, teacher_proj2)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr_ = _adjust_learning_rate(args, optimizer, iter_num, max_iterations, float(args.base_lr))
            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/stage1_image_loss", loss.item(), iter_num)
            epoch_loss += loss.item()

            logging.info(
                "stage1_image epoch %d iter %d : loss=%f",
                epoch_num,
                i_batch,
                loss.item(),
            )

        avg_loss = epoch_loss / max(1, len(trainloader))
        writer.add_scalar("epoch/stage1_image_avg_loss", avg_loss, epoch_num)
        logging.info("stage1_image epoch %d summary : avg_loss=%f", epoch_num, avg_loss)

        if (epoch_num + 1) % save_interval == 0:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="image")
            _save_aux_module_checkpoint(student_projector, snapshot_path, epoch_num, prefix="image_projector")

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="image")
            _save_aux_module_checkpoint(student_projector, snapshot_path, epoch_num, prefix="image_projector")
            iterator.close()
            break

    writer.close()
    return "Stage1 Image Training Finished!"


# -----------------------------------------------------------------------------
# Stage 1 patch-level
# -----------------------------------------------------------------------------

def trainer_stage1_patch(args, model, snapshot_path):
    _setup_logger(snapshot_path, args)
    _, losses_module, projector_module, _, _, mae_module = _load_stage1_modules()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_model = _unwrap_model(model)
    target_model.freeze_for_stage1("patch")
    model.train()

    _, trainloader = _build_stage1_dataloader(args, ssl_task="patch")
    _, token_dim = _infer_stage1_feature_dims(target_model, int(args.img_size), device)

    teacher = mae_module.build_mae_teacher(
        img_size=int(args.img_size),
        patch_size=int(getattr(args, "patch_size", 16)),
        in_chans=3,
        embed_dim=int(getattr(args, "mae_embed_dim", 1024)),
        depth=int(getattr(args, "mae_depth", 8)),
        num_heads=int(getattr(args, "mae_num_heads", 16)),
        decoder_embed_dim=int(getattr(args, "mae_decoder_embed_dim", 512)),
        decoder_depth=int(getattr(args, "mae_decoder_depth", 2)),
        decoder_num_heads=int(getattr(args, "mae_decoder_num_heads", 8)),
        proj_dim=int(getattr(args, "stage1_proj_dim", 256)),
    ).to(device)
    teacher.eval()

    student_projector = projector_module.build_projector(
        task="patch",
        in_dim=token_dim,
        hidden_dim=int(getattr(args, "stage1_proj_hidden_dim", 512)),
        out_dim=int(getattr(args, "stage1_proj_dim", 256)),
        normalize_output=True,
    ).to(device)

    criterion = losses_module.build_stage1_loss(
        "patch",
        recon_mode="mse",
        recon_weight=0.0,
        distill_weight=1.0,
    ).to(device)

    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(student_projector.parameters())
    optimizer = _build_optimizer(args, params, float(args.base_lr))

    _log_stage1_model_status(model)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    max_epoch = int(args.max_epochs)
    stop_epoch = int(args.stop_epoch)
    max_iterations = max_epoch * len(trainloader)
    save_interval = getattr(args, "save_interval", max(1, max_epoch // 3))
    mask_ratio = float(getattr(args, "mask_ratio", 0.75))
    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0

        for i_batch, batch in enumerate(trainloader):
            image = batch["image"].to(device, non_blocking=True)

            _, _, aux = _forward_image_encoder(target_model, image)
            student_tokens_full = _as_bnc(aux["tokens_before_neck"])

            with torch.no_grad():
                teacher_out = teacher.forward_tokens(image, mask_ratio=mask_ratio)
                teacher_proj = teacher_out["projected_tokens"]
                mask = teacher_out["mask"]

            len_keep = teacher_proj.shape[1]
            ids_keep = torch.argsort(mask, dim=1)[:, :len_keep]
            ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, student_tokens_full.shape[-1])
            student_visible_tokens = torch.gather(student_tokens_full, 1, ids_keep_expanded)
            student_proj = student_projector(student_visible_tokens)

            loss_dict = criterion(
                reconstructed=student_proj,
                target=teacher_proj,
                student_tokens=student_proj,
                teacher_tokens=teacher_proj,
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr_ = _adjust_learning_rate(args, optimizer, iter_num, max_iterations, float(args.base_lr))
            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/stage1_patch_loss", loss.item(), iter_num)
            epoch_loss += loss.item()

            logging.info(
                "stage1_patch epoch %d iter %d : loss=%f",
                epoch_num,
                i_batch,
                loss.item(),
            )

        avg_loss = epoch_loss / max(1, len(trainloader))
        writer.add_scalar("epoch/stage1_patch_avg_loss", avg_loss, epoch_num)
        logging.info("stage1_patch epoch %d summary : avg_loss=%f", epoch_num, avg_loss)

        if (epoch_num + 1) % save_interval == 0:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="patch")
            _save_aux_module_checkpoint(student_projector, snapshot_path, epoch_num, prefix="patch_projector")

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="patch")
            _save_aux_module_checkpoint(student_projector, snapshot_path, epoch_num, prefix="patch_projector")
            iterator.close()
            break

    writer.close()
    return "Stage1 Patch Training Finished!"


# -----------------------------------------------------------------------------
# Stage 1 pixel-level
# -----------------------------------------------------------------------------

def trainer_stage1_pixel(args, model, snapshot_path):
    _setup_logger(snapshot_path, args)
    _, losses_module, _, denoise_module, _, _ = _load_stage1_modules()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    target_model = _unwrap_model(model)
    target_model.freeze_for_stage1("pixel")
    model.train()

    _, trainloader = _build_stage1_dataloader(args, ssl_task="pixel")
    image_embed_dim, _ = _infer_stage1_feature_dims(target_model, int(args.img_size), device)

    denoiser = denoise_module.build_denoise_decoder(
        in_chans=image_embed_dim,
        out_chans=3,
        hidden_chans=tuple(getattr(args, "pixel_decoder_channels", (256, 128, 64, 32, 16))),
        final_activation=getattr(args, "pixel_decoder_activation", "sigmoid"),
    ).to(device)

    criterion = losses_module.build_stage1_loss(
        "pixel",
        dice_weight=float(getattr(args, "pixel_dice_weight", 0.9)),
        mse_weight=float(getattr(args, "pixel_mse_weight", 0.1)),
        use_hierarchical_aux=False,
    ).to(device)

    params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(denoiser.parameters())
    optimizer = _build_optimizer(args, params, float(args.base_lr))

    _log_stage1_model_status(model)
    logging.info("pixel-level denoiser trainable params: %d", sum(p.numel() for p in denoiser.parameters()))

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    iter_num = 0
    max_epoch = int(args.max_epochs)
    stop_epoch = int(args.stop_epoch)
    max_iterations = max_epoch * len(trainloader)
    save_interval = getattr(args, "save_interval", max(1, max_epoch // 3))
    iterator = tqdm(range(max_epoch), ncols=100)

    for epoch_num in iterator:
        epoch_loss = 0.0

        for i_batch, batch in enumerate(trainloader):
            clean_image = batch["image"].to(device, non_blocking=True)
            noisy_image = batch["noisy_image"].to(device, non_blocking=True)

            image_embeddings, _, _ = _forward_image_encoder(target_model, noisy_image)
            pred = denoiser(image_embeddings, target_size=tuple(clean_image.shape[-2:]))

            loss_dict = criterion(pred, clean_image)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr_ = _adjust_learning_rate(args, optimizer, iter_num, max_iterations, float(args.base_lr))
            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/stage1_pixel_loss", loss.item(), iter_num)
            epoch_loss += loss.item()

            logging.info(
                "stage1_pixel epoch %d iter %d : loss=%f",
                epoch_num,
                i_batch,
                loss.item(),
            )

        avg_loss = epoch_loss / max(1, len(trainloader))
        writer.add_scalar("epoch/stage1_pixel_avg_loss", avg_loss, epoch_num)
        logging.info("stage1_pixel epoch %d summary : avg_loss=%f", epoch_num, avg_loss)

        if (epoch_num + 1) % save_interval == 0:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="pixel")
            _save_aux_module_checkpoint(denoiser, snapshot_path, epoch_num, prefix="pixel_denoiser")

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            _save_stage1_expert_checkpoint(model, snapshot_path, epoch_num, expert="pixel")
            _save_aux_module_checkpoint(denoiser, snapshot_path, epoch_num, prefix="pixel_denoiser")
            iterator.close()
            break

    writer.close()
    return "Stage1 Pixel Training Finished!"
