"""
Loss functions and evaluation metrics for binary segmentation.

Replaces duplicated implementations in:
  - models/train_roi_efficientunet_BCSS.py
  - models/backbone_efficientunet_inference_BCSS.py
"""

from __future__ import annotations

import numpy as np
import torch


# ── Differentiable losses ─────────────────────────────────────────────────────

def dice_coeff(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth_num: float = 1.0,
    smooth_den: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice coefficient.

    Args:
        y_true:     ground-truth mask, any shape.
        y_pred:     predicted mask (probabilities or binary), same shape.
        smooth_num: additive smoothing in numerator (prevents div/0).
        smooth_den: additive smoothing in denominator.

    Returns:
        Scalar tensor in [0, 1].
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth_num) / (
        torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth_den
    )


def dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth_num: float = 1.0,
    smooth_den: float = 1.0,
) -> torch.Tensor:
    """1 - dice_coeff. Use as training loss."""
    return 1.0 - dice_coeff(y_true, y_pred, smooth_num, smooth_den)


# ── NumPy metrics (evaluation, not differentiable) ───────────────────────────

def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_iou(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    """
    Intersection-over-Union for a single class in a binary prediction.

    Returns float('nan') when the class is absent in both pred and target
    (standard mIoU convention — nan values are excluded from the mean).
    """
    pred_cls   = (pred == cls)
    target_cls = (target == cls)
    intersection = np.logical_and(pred_cls, target_cls).sum()
    union        = np.logical_or(pred_cls, target_cls).sum()
    return float("nan") if union == 0 else float(intersection / union)


def compute_miou_binary(
    pred:   np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> float:
    """
    Mean IoU for binary segmentation (foreground class only).

    Both pred and target are thresholded at 0.5 before evaluation.
    Accepts NumPy arrays or PyTorch tensors.
    """
    pred   = (_to_numpy(pred)   > 0.5).astype(int)
    target = (_to_numpy(target) > 0.5).astype(int)
    return compute_iou(pred, target, cls=1)


# ── Torch metrics ─────────────────────────────────────────────────────────────

def pixel_accuracy(
    preds:  torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Pixel-level accuracy for a batch of binary predictions.

    Args:
        preds:  [B, 1, H, W] or [B, H, W] — model output (probabilities or logits).
        labels: [B, 1, H, W] or [B, H, W] — ground-truth binary mask.

    Returns:
        (per_sample_accuracy [B], mean_accuracy scalar)
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    preds  = (preds > 0.5).float()
    labels = labels.float()

    correct       = (preds == labels).float().sum(dim=[1, 2])
    total_pixels  = labels.size(1) * labels.size(2)
    per_sample    = correct / total_pixels
    return per_sample, per_sample.mean().item()
