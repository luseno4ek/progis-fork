"""
Error-region guiding signal generation for the interactive correction loop.

Used in both training (CU-Training: two forward passes) and inference
(iterative correction: up to 20 rounds).

Replaces duplicate / old implementations in:
  - models/train_roi_efficientunet_BCSS.py  (processMasks, processMasks_gpu, _edt_skeleton_gpu)
  - models/backbone_efficientunet_inference_BCSS.py  (same three functions + old variants)

Public API
----------
process_masks(pred, gt)           → (signal [B,2,H,W], centers list)  — CPU
process_masks_gpu(pred, gt)       → (signal [B,2,H,W], centers list)  — GPU
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as skimage_label
from skimage.measure import label as label_1
from skimage.measure import regionprops
from skimage.morphology import skeletonize


# ── Stroke-length limiter ─────────────────────────────────────────────────────

def _sample_stroke_bfs(skel_np: np.ndarray, max_length: int) -> np.ndarray:
    """
    Return a connected sub-stroke of at most *max_length* pixels from skeleton.

    Algorithm: pick a random starting pixel on the skeleton, then grow
    outward via 8-connected BFS until *max_length* pixels are collected.

    Args:
        skel_np:    [H, W] binary ndarray (skeleton).
        max_length: maximum number of pixels to keep.

    Returns:
        [H, W] binary ndarray with at most max_length pixels set.
    """
    ys, xs = np.where(skel_np)
    if len(ys) == 0 or len(ys) <= max_length:
        return skel_np

    H, W = skel_np.shape
    start_i = np.random.randint(len(ys))
    start   = (int(ys[start_i]), int(xs[start_i]))

    visited: set[tuple[int, int]] = {start}
    queue   = deque([start])
    result  = np.zeros_like(skel_np)

    while queue and len(visited) < max_length:
        cy, cx = queue.popleft()
        result[cy, cx] = 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                nb = (ny, nx)
                if 0 <= ny < H and 0 <= nx < W and nb not in visited and skel_np[ny, nx]:
                    visited.add(nb)
                    queue.append(nb)

    return result


# ── Per-pixel skeleton signal (torch, used by CPU process_masks) ──────────────

def _generate_guiding_signal_tensor(
    binary_mask:      torch.Tensor,
    max_stroke_length: int | None = None,
) -> torch.Tensor:
    """
    Skeleton guiding signal from a binary [H, W] tensor.

    Mirrors generate_guiding_signal from data/signal_utils.py but accepts and
    returns torch tensors (device-aware).

    Args:
        binary_mask: [H, W] float or uint8 tensor.

    Returns:
        [H, W] float32 skeleton signal (values 0 or 1).
    """
    bm = binary_mask.to(torch.uint8)

    if bm.sum() <= 1:
        return torch.zeros_like(bm, dtype=torch.float32)

    dist_np = distance_transform_edt(bm.cpu().numpy())
    dist    = torch.tensor(dist_np, dtype=torch.float32, device=binary_mask.device)

    # Compute mean/std only over foreground pixels — background has dist=0
    # and would drag the mean down, making thresh near 0 → entire region passes.
    fg_dist = dist[bm.bool()]
    mean_d = float(fg_dist.mean().cpu())
    std_d  = float(fg_dist.std().cpu())

    thresh = float(np.random.uniform(mean_d - std_d, mean_d + std_d))
    if thresh < 0:
        thresh = float(np.random.uniform(mean_d / 2, mean_d + std_d / 2))
    thresh_t = torch.tensor(thresh, device=binary_mask.device)

    new_mask = dist > thresh_t
    if new_mask.sum() == 0:
        new_mask = dist > (thresh_t / 2)
    if new_mask.sum() == 0:
        new_mask = bm.bool()

    skel = skeletonize(new_mask.cpu().numpy())
    if max_stroke_length is not None:
        skel = _sample_stroke_bfs(skel, max_stroke_length)
    return torch.tensor(skel, dtype=torch.float32, device=binary_mask.device)


# ── CPU process_masks ─────────────────────────────────────────────────────────

def process_masks(
    pred_mask_all:     torch.Tensor,
    gt_mask_all:       torch.Tensor,
    max_stroke_length: int | None = None,
) -> tuple[torch.Tensor, list]:
    """
    Compute error-region guiding signals for a batch (CPU implementation).

    For each sample:
      - fg error = GT==1 & pred==0  (missed foreground)
      - bg error = GT==0 & pred==1  (false positive)
      - Selects whichever has the larger connected component.
      - Skeletonises it and places in the corresponding output channel.
      - Returns the centroid of the selected region.

    Args:
        pred_mask_all: [B, 1, H, W] float — model output (thresholded at 0.5 inside).
        gt_mask_all:   [B, 1, H, W] float — ground-truth binary mask.

    Returns:
        signal:  [B, 2, H, W] float32 — ch0 = fg guidance, ch1 = bg guidance.
        centers: list of length B — each entry is (cy, cx) int tuple or None
                 (None means no error region for that sample).
    """
    pred_mask_all = (pred_mask_all > 0.5).float()
    B, _, H, W = pred_mask_all.shape

    output  = torch.zeros(B, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    centers: list = []

    for i in range(B):
        pred = pred_mask_all[i, 0]   # [H, W]
        gt   = gt_mask_all[i, 0]     # [H, W]

        fg = ((gt == 1) & (pred == 0)).float()
        bg = ((gt == 0) & (pred == 1)).float()

        fg_largest, fg_center = _largest_connected_component(fg)
        bg_largest, bg_center = _largest_connected_component(bg)

        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            skel = _generate_guiding_signal_tensor(fg_largest, max_stroke_length) if fg_area > 0 else torch.zeros_like(pred)
            output[i, 0] = skel
            centers.append(fg_center)
        else:
            skel = _generate_guiding_signal_tensor(bg_largest, max_stroke_length) if bg_area > 0 else torch.zeros_like(pred)
            output[i, 1] = skel
            centers.append(bg_center)

    return output, centers


def _largest_connected_component(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """
    Keep only the largest 4-connected component of a binary [H, W] mask.

    Returns:
        (largest_mask [H, W] float32, centroid (cy, cx) int tuple or None)
    """
    if mask.sum() == 0:
        return torch.zeros_like(mask), None

    labeled = skimage_label(mask.cpu().numpy(), connectivity=1)
    regions = regionprops(labeled)

    if not regions:
        return torch.zeros_like(mask), None

    largest = max(regions, key=lambda r: r.area)
    largest_np = (labeled == largest.label).astype(np.float32)
    cy, cx = largest.centroid
    center = (round(cy), round(cx))

    largest_t = torch.from_numpy(largest_np).to(mask.device, dtype=torch.float32)
    return largest_t, center


# ── GPU process_masks ─────────────────────────────────────────────────────────

def _edt_skeleton_gpu(mask: torch.Tensor, max_steps: int = 80) -> torch.Tensor:
    """
    Approximate skeleton of binary mask regions via EDT ridge (fully on GPU).

    Algorithm:
      1. Approximate EDT via successive 3×3 min-pool erosion — no CPU sync in loop.
      2. Random threshold in [mean-std, mean+std] of EDT values.
      3. Local maxima of EDT within the thresholded core = ridge ≈ medial axis.

    Args:
        mask:      [B, 1, H, W] float32 binary tensor.
        max_steps: upper bound on erosion iterations (≈ max region radius in px).

    Returns:
        [B, 1, H, W] float32 binary skeleton.
    """
    B      = mask.shape[0]
    device = mask.device

    # ── 1. Approximate EDT via erosion ─────────────────────────────────────
    edt     = torch.zeros_like(mask)
    current = mask.clone()

    for step in range(1, max_steps + 1):
        next_c       = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        newly_removed = (current > 0) & (next_c == 0)
        edt          = edt + newly_removed.float() * step
        current      = next_c

    edt = edt + (current > 0).float() * (max_steps + 1)

    # ── 2. Random threshold (vectorised over batch) ─────────────────────
    mask_bool   = mask > 0
    edt_in_mask = edt * mask

    n_pixels = mask.flatten(1).sum(1).clamp(min=1)          # [B]
    edt_sum  = edt_in_mask.flatten(1).sum(1)
    edt_sq   = edt_in_mask.pow(2).flatten(1).sum(1)

    means  = edt_sum / n_pixels
    stds   = (edt_sq / n_pixels - means.pow(2)).clamp(min=0).sqrt()
    rand   = torch.rand(B, 1, 1, 1, device=device)
    thresh = (means.view(B, 1, 1, 1) - stds.view(B, 1, 1, 1)
              + rand * 2 * stds.view(B, 1, 1, 1)).clamp(min=0)

    core = (edt > thresh) & mask_bool

    # Fallback: empty core → use max-EDT pixels
    need_fallback = (core.float().flatten(1).sum(1) == 0) & (mask.flatten(1).sum(1) > 0)
    max_edt       = (edt * mask).flatten(1).max(1).values.view(B, 1, 1, 1)
    fallback_core = (edt >= max_edt - 1e-6) & mask_bool
    core = torch.where(need_fallback.view(B, 1, 1, 1), fallback_core, core)

    # ── 3. Ridge = local maxima of EDT within core ──────────────────────
    core_edt  = edt * core.float()
    local_max = F.max_pool2d(core_edt, kernel_size=3, stride=1, padding=1)
    ridge     = (core_edt >= local_max - 1e-6) & core

    empty_ridge = (ridge.float().flatten(1).sum(1) == 0) & (core.float().flatten(1).sum(1) > 0)
    ridge = torch.where(empty_ridge.view(B, 1, 1, 1), core, ridge)

    return ridge.float()


def process_masks_gpu(
    pred_mask_all:     torch.Tensor,
    gt_mask_all:       torch.Tensor,
    max_edt_steps:     int      = 80,
    max_stroke_length: int | None = None,
) -> tuple[torch.Tensor, list]:
    """
    GPU drop-in replacement for process_masks. No scipy/skimage calls.

    Mirrors CPU logic:
      - fg error (GT=1, pred=0) and bg error (GT=0, pred=1).
      - Selects the larger region per sample (total area, not largest CC).
      - EDT skeleton via _edt_skeleton_gpu.
      - Places skeleton in the correct output channel.
      - Returns vectorised centroids.

    Args:
        pred_mask_all:  [B, 1, H, W] float.
        gt_mask_all:    [B, 1, H, W] float.
        max_edt_steps:  erosion iterations budget.

    Returns:
        signal:  [B, 2, H, W] float32.
        centers: list of length B — (cy, cx) int tuple or None.
    """
    device = pred_mask_all.device
    B, _, H, W = pred_mask_all.shape

    pred_bin = (pred_mask_all > 0.5).float()
    fg = ((gt_mask_all == 1) & (pred_bin == 0)).float()
    bg = ((gt_mask_all == 0) & (pred_bin == 1)).float()

    fg_area = fg.flatten(1).sum(1)                        # [B]
    bg_area = bg.flatten(1).sum(1)
    use_fg  = (fg_area >= bg_area).float().view(B, 1, 1, 1)

    selected = fg * use_fg + bg * (1 - use_fg)           # [B, 1, H, W]
    skel     = _edt_skeleton_gpu(selected, max_edt_steps) # [B, 1, H, W]

    if max_stroke_length is not None:
        # Radius = half the target length (skeleton can curve, diameter ≈ length)
        radius = max_stroke_length / 2.0
        ys = torch.arange(H, device=device).float().view(1, 1, H, 1)
        xs = torch.arange(W, device=device).float().view(1, 1, 1, W)
        # Pick centroid of skeleton as crop centre (vectorised)
        skel_n   = skel.flatten(1).sum(1).clamp(min=1)            # [B]
        ctr_y    = (skel * ys).flatten(1).sum(1) / skel_n         # [B]
        ctr_x    = (skel * xs).flatten(1).sum(1) / skel_n         # [B]
        dist2    = (ys - ctr_y.view(B, 1, 1, 1)).pow(2) + \
                   (xs - ctr_x.view(B, 1, 1, 1)).pow(2)
        skel = skel * (dist2 <= radius ** 2).float()

    output = torch.zeros(B, 2, H, W, device=device)
    output[:, 0:1] = skel * use_fg
    output[:, 1:2] = skel * (1 - use_fg)

    # Vectorised centroids (3 tiny CPU syncs total)
    ys       = torch.arange(H, device=device).float().view(1, 1, H, 1)
    xs       = torch.arange(W, device=device).float().view(1, 1, 1, W)
    n        = selected.flatten(1).sum(1).clamp(min=1)
    cy       = (selected * ys).flatten(1).sum(1) / n
    cx       = (selected * xs).flatten(1).sum(1) / n
    has_rgn  = selected.flatten(1).sum(1) > 0

    cy_list  = cy.round().long().tolist()
    cx_list  = cx.round().long().tolist()
    has_list = has_rgn.tolist()
    centers  = [(cy_list[b], cx_list[b]) if has_list[b] else None for b in range(B)]

    return output, centers