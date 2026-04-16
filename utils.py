import csv
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from medpy import metric
from scipy.ndimage import zoom


class Focal_loss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, num_classes: int = 3, size_average: bool = True):
        super().__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        bsz, height, width = labels.shape
        assert bsz * height * width == preds.shape[0]
        assert preds.shape[-1] == self.num_classes

        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    @staticmethod
    def _dice_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight=None, softmax: bool = False) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    if pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    return 0.0, 0.0


def _extract_masks_from_dict(output: Any) -> Union[torch.Tensor, None]:
    if isinstance(output, dict) and 'masks' in output:
        return output['masks']
    return None


def _select_output_masks(model_output: Any, decoder_stage: Union[int, str] = 3) -> torch.Tensor:
    """
    Robustly resolve masks from both old H-SAM and the refactored H-SAMora paths.

    Supported patterns:
    - dict with key 'masks'
    - tuple/list whose first two items are dicts with 'masks'
    - tuple/list whose first item is a dict with 'masks'

    decoder_stage:
    - 2: use outputs2 only
    - 3: average outputs1 and outputs2
    """
    if isinstance(decoder_stage, str):
        try:
            decoder_stage = int(decoder_stage)
        except ValueError:
            decoder_stage = 3

    if isinstance(model_output, dict):
        masks = _extract_masks_from_dict(model_output)
        if masks is None:
            raise ValueError('Model returned a dict without a "masks" key.')
        return masks

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError('Model returned an empty tuple/list.')

        first_masks = _extract_masks_from_dict(model_output[0])
        second_masks = _extract_masks_from_dict(model_output[1]) if len(model_output) > 1 else None

        if first_masks is not None and second_masks is not None:
            if decoder_stage == 2:
                return second_masks
            return (first_masks + second_masks) / 2.0

        if first_masks is not None:
            return first_masks

    raise TypeError(
        'Unable to resolve segmentation masks from model output. '
        f'Got type={type(model_output)}.'
    )


def _run_slice_inference(
    net: torch.nn.Module,
    inputs: torch.Tensor,
    multimask_output: bool,
    image_size: int,
    mode: str,
    decoder_stage: Union[int, str],
) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, image_size, gt=None, mode=mode)
        output_masks = _select_output_masks(outputs, decoder_stage=decoder_stage)
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        return out.detach().cpu().numpy()


def test_single_volume(
    image: torch.Tensor,
    label: torch.Tensor,
    net: torch.nn.Module,
    classes: int,
    multimask_output: bool,
    patch_size: Sequence[int] = (256, 256),
    input_size: Sequence[int] = (224, 224),
    test_save_path: str = None,
    case: str = None,
    z_spacing: float = 1,
    stage: Union[int, str] = 3,
    mode: str = 'test',
):
    """
    stage here means decoder selection during inference, not the training stage.

    - 2: use outputs2 only
    - 3: use average(outputs1, outputs2)

    This keeps compatibility with the legacy H-SAM testing code while also
    matching the refactored test.py entrypoint that passes eval_decoder_stage.
    """
    lab = label.squeeze(0)
    image_np = image.squeeze(0).detach().cpu().numpy()
    label_np = label.squeeze(0).detach().cpu().numpy()

    if len(image_np.shape) == 3:
        prediction = np.zeros_like(label_np)
        for ind in range(image_np.shape[0]):
            slice_np = image_np[ind, :, :]
            x, y = slice_np.shape[0], slice_np.shape[1]

            if x != input_size[0] or y != input_size[1]:
                slice_np = zoom(slice_np, (input_size[0] / x, input_size[1] / y), order=3)
            new_x, new_y = slice_np.shape[0], slice_np.shape[1]
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slice_np = zoom(slice_np, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)

            inputs = torch.from_numpy(slice_np).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            out = _run_slice_inference(
                net=net,
                inputs=inputs,
                multimask_output=multimask_output,
                image_size=patch_size[0],
                mode=mode,
                decoder_stage=stage,
            )
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred
    else:
        x, y = image_np.shape[-2:]
        working = image_np
        if x != patch_size[0] or y != patch_size[1]:
            working = zoom(working, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(working).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        prediction = _run_slice_inference(
            net=net,
            inputs=inputs,
            multimask_output=multimask_output,
            image_size=patch_size[0],
            mode=mode,
            decoder_stage=stage,
        )
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    metric_list = []
    metric_list_dice = []
    metric_list_hd = []
    for i in range(1, classes + 1):
        dice_i, hd_i = calculate_metric_percase(prediction == i, label_np == i)
        metric_list_dice.append(dice_i)
        metric_list_hd.append(hd_i)
        metric_list.append((dice_i, hd_i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image_np.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_np.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f'{case}_pred.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f'{case}_img.nii.gz'))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f'{case}_gt.nii.gz'))
        with open(os.path.join(test_save_path, 'dice.csv'), 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(metric_list_dice)
    return metric_list


def mask_latent_code_spatial_wise(
    latent_code: torch.Tensor,
    loss: torch.Tensor,
    percentile: float = 1 / 3.0,
    random: bool = False,
    loss_type: str = 'corr',
    if_detach: bool = True,
    if_soft: bool = False,
):
    del loss_type  # kept for backward compatibility
    code = latent_code
    num_images = code.size(0)
    spatial_size = code.size(2) * code.size(3)
    height, width = code.size(2), code.size(3)

    gradient = torch.autograd.grad(loss, [code])[0]
    spatial_mean = torch.mean(gradient, dim=1, keepdim=True)
    spatial_mean = spatial_mean.squeeze().view(num_images, spatial_size)

    if random:
        percentile = np.random.rand() * percentile

    vector_thresh_percent = int(spatial_size * percentile)
    vector_thresh_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
    vector_thresh_value = vector_thresh_value.view(num_images, 1).expand(num_images, spatial_size)

    if if_soft:
        vector = torch.where(
            spatial_mean > vector_thresh_value,
            0.5 * torch.rand_like(spatial_mean),
            torch.ones_like(spatial_mean),
        )
    else:
        vector = torch.where(
            spatial_mean > vector_thresh_value,
            torch.zeros_like(spatial_mean),
            torch.ones_like(spatial_mean),
        )

    mask_all = vector.view(num_images, 1, height, width)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()  # type: ignore[name-defined]
    except Exception:
        pass
    return masked_latent_code, mask_all


def set_grad(module: nn.Module, requires_grad: bool = False) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad
