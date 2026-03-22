"""
ROI crop helpers for ProGIS interactive inference.

Two crops are used at different stages:

  roi_crop_for_prototype()
      Called once at the start of inference per batch.
      Centres the 256×256 window on the fg guiding signal.
      Generates a fresh skeleton signal from the largest CC of the GT mask.
      Returns everything needed by ProGISModel.forward_prototype().

  roi_crop_for_correction()
      Called inside the iterative correction loop.
      Centres the 256×256 window on the error-region centroid produced by
      process_masks() / process_masks_gpu().
      Returns image/prev_mask/signal crops ready for ProGISModel.segment().

Replaces ROI_crop_signal_line() and the inline crop code inside the
correction while-loop in backbone_efficientunet_inference_BCSS.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .signals import _generate_guiding_signal_tensor, _largest_connected_component


# ── Shared crop-window helper ─────────────────────────────────────────────────

def _crop_window(
    cy: int, cx: int,
    H: int, W: int,
    crop: int,
) -> tuple[int, int, int, int]:
    """
    Compute (sy, sx, ey, ex) for a crop×crop window centred on (cy, cx).
    Clamps to image boundaries while keeping exact crop size.
    """
    sy = min(max(cy - crop // 2, 0), H - crop)
    sx = min(max(cx - crop // 2, 0), W - crop)
    return sy, sx, sy + crop, sx + crop


# ── Initial ROI crop (for prototype initialisation) ───────────────────────────

@dataclass
class PrototypeCropBatch:
    """
    Outputs of roi_crop_for_prototype(). All tensors are on the same device
    as the input images.

    Attributes:
        roi_images:  [B, 3, crop, crop]   — cropped RGB images.
        roi_signals: [B, 2, crop, crop]   — skeleton signals from largest CC
                                            of the GT mask within the ROI.
        roi_masks:   [B, 1, crop, crop]   — GT masks within the ROI.
        mask_box:    [B, 1, H, W]         — binary flag: 1 inside the ROI window.
    """
    roi_images:  torch.Tensor
    roi_signals: torch.Tensor
    roi_masks:   torch.Tensor
    mask_box:    torch.Tensor


def roi_crop_for_prototype(
    images:    torch.Tensor,   # [B, 3, H, W]
    signals:   torch.Tensor,   # [B, 2, H, W]  pre-computed guiding signal
    masks:     torch.Tensor,   # [B, 1, H, W]  GT masks
    crop_size: int = 256,
) -> PrototypeCropBatch:
    """
    Crop 256×256 ROI around the fg guiding signal for prototype initialisation.

    Per sample:
      1. Find centroid of fg signal (signals[:, 0, :, :]).
         If signal is empty → random position.
      2. Crop image and mask to crop_size × crop_size.
      3. Extract the largest connected component of the GT mask in the ROI.
      4. Generate a skeleton guiding signal from that CC.

    Args:
        images:    [B, 3, H, W] float32.
        signals:   [B, 2, H, W] float32 — pre-computed skeleton signals.
        masks:     [B, 1, H, W] float32 — GT binary masks.
        crop_size: ROI window size (default 256).

    Returns:
        PrototypeCropBatch with roi_images, roi_signals, roi_masks, mask_box.
    """
    B, _, H, W = images.shape
    device = images.device
    crop = crop_size

    roi_images:  list[torch.Tensor] = []
    roi_signals: list[torch.Tensor] = []
    roi_masks:   list[torch.Tensor] = []
    mask_box = torch.zeros(B, 1, H, W, device=device)

    for b in range(B):
        fg_ys, fg_xs = torch.where(signals[b, 0] == 1)

        if fg_ys.numel() > 0:
            cy = int(fg_ys.float().mean().round().item())
            cx = int(fg_xs.float().mean().round().item())
        else:
            cy = int(torch.randint(0, max(H - crop, 1), (1,)).item()) + crop // 2
            cx = int(torch.randint(0, max(W - crop, 1), (1,)).item()) + crop // 2

        sy, sx, ey, ex = _crop_window(cy, cx, H, W, crop)
        mask_box[b, :, sy:ey, sx:ex] = 1.0

        roi_img  = images[b, :, sy:ey, sx:ex]              # [3, crop, crop]
        roi_mask = masks[b, :, sy:ey, sx:ex]               # [1, crop, crop]

        # Largest CC of the GT mask in the ROI → fresh skeleton signal
        largest_cc, _ = _largest_connected_component(roi_mask[0])  # [H, W]
        fg_skel = _generate_guiding_signal_tensor(largest_cc)       # [H, W]
        roi_sig = torch.stack([fg_skel, torch.zeros_like(fg_skel)]) # [2, crop, crop]

        roi_images.append(roi_img)
        roi_signals.append(roi_sig)
        roi_masks.append(roi_mask)

    return PrototypeCropBatch(
        roi_images  = torch.stack(roi_images),
        roi_signals = torch.stack(roi_signals),
        roi_masks   = torch.stack(roi_masks),
        mask_box    = mask_box,
    )


# ── Correction crop (for iterative loop) ─────────────────────────────────────

@dataclass
class CorrectionCropBatch:
    """
    Outputs of roi_crop_for_correction(). All tensors are on the same device
    as the input images.

    Attributes:
        roi_images:     [B, 3, crop, crop] — cropped RGB images.
        roi_prev_masks: [B, 1, crop, crop] — current prediction crop.
        roi_signals:    [B, 2, crop, crop] — union signal crop.
    """
    roi_images:     torch.Tensor
    roi_prev_masks: torch.Tensor
    roi_signals:    torch.Tensor


def roi_crop_for_correction(
    images:       torch.Tensor,              # [B, 3, H, W]
    pred_masks:   torch.Tensor,              # [B, 1, H, W]  current prediction
    union_signal: torch.Tensor,              # [B, 2, H, W]  accumulated signal
    centers:      list[tuple[int,int] | None],  # from process_masks
    crop_size:    int = 256,
) -> CorrectionCropBatch:
    """
    Crop 256×256 ROI around the error-region centroid for the correction step.

    Per sample:
      - Centre = error centroid from process_masks() (cy, cx).
      - Fallback to image centre when no error region exists (perfect prediction).

    Args:
        images:       [B, 3, H, W] full image.
        pred_masks:   [B, 1, H, W] current binary prediction.
        union_signal: [B, 2, H, W] accumulated fg+bg guiding signal.
        centers:      list of (cy, cx) int tuples or None (B elements).
        crop_size:    ROI window size (default 256).

    Returns:
        CorrectionCropBatch with roi_images, roi_prev_masks, roi_signals.
    """
    B, _, H, W = images.shape
    crop = crop_size

    roi_images:     list[torch.Tensor] = []
    roi_prev_masks: list[torch.Tensor] = []
    roi_signals:    list[torch.Tensor] = []

    for b in range(B):
        center = centers[b] if centers[b] is not None else (H // 2, W // 2)
        cy, cx = int(center[0]), int(center[1])
        sy, sx, ey, ex = _crop_window(cy, cx, H, W, crop)

        roi_images.append(images[b, :, sy:ey, sx:ex])
        roi_prev_masks.append(pred_masks[b, :, sy:ey, sx:ex])
        roi_signals.append(union_signal[b, :, sy:ey, sx:ex])

    return CorrectionCropBatch(
        roi_images     = torch.stack(roi_images),
        roi_prev_masks = torch.stack(roi_prev_masks),
        roi_signals    = torch.stack(roi_signals),
    )


# ── Paste crop back into full-resolution mask ─────────────────────────────────

def paste_crop_into_mask(
    full_mask:  torch.Tensor,              # [B, 1, H, W]  modified in-place
    crop_pred:  torch.Tensor,              # [B, 1, crop, crop]
    centers:    list[tuple[int,int] | None],
    H: int, W: int,
    crop_size:  int = 256,
    threshold:  float = 0.5,
) -> torch.Tensor:
    """
    Write thresholded crop_pred back into the corresponding ROI of full_mask.

    Args:
        full_mask:  [B, 1, H, W] — output tensor, updated in-place.
        crop_pred:  [B, 1, crop, crop] — segment() output for the crop.
        centers:    list of (cy, cx) from roi_crop_for_correction.
        threshold:  binarisation threshold (default 0.5).

    Returns:
        full_mask (same tensor, updated in-place).
    """
    B = full_mask.shape[0]
    crop = crop_size
    bin_pred = (crop_pred > threshold).float()

    for b in range(B):
        center = centers[b] if centers[b] is not None else (H // 2, W // 2)
        cy, cx = int(center[0]), int(center[1])
        sy, sx, ey, ex = _crop_window(cy, cx, H, W, crop)
        full_mask[b, :, sy:ey, sx:ex] = bin_pred[b]

    return full_mask
