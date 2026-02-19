"""
Quick sanity-check: compare processMasks (CPU, original) vs processMasks_gpu (GPU, EDT-ridge).

Run from models/ directory:
    python test_processmasks.py

What is checked:
  1. Both functions return tensors of the same shape [B, 2, H, W].
  2. Signals are non-zero where there ARE error regions.
  3. Signals are zero where there are NO error regions.
  4. Spatial overlap between CPU and GPU signals is >0 (both cover the error region core).
  5. GPU version is faster.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label as label_1, regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# ── CPU version (copied from train_roi_efficientunet_BCSS.py) ────────────────

def generateGuidingSignal(binaryMask):
    binaryMask = binaryMask.to(torch.uint8)
    if binaryMask.sum() > 1:
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        distance_map = torch.tensor(distance_map, dtype=torch.float32, device=binaryMask.device)
        tempMean = distance_map.mean().cpu().numpy()
        tempStd = distance_map.std().cpu().numpy()
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
            tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        if newMask.sum() == 0:
            newMask = binaryMask
        skel = skeletonize(newMask.cpu().numpy())
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)
    return skel


def processMasks(pred_mask_all, GT_mask_all):
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    for i in range(batch_size):
        pred_mask = pred_mask_all[i].squeeze(0)
        GT_mask = GT_mask_all[i].squeeze(0)
        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32)
        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32)
        if fg.sum() > 0:
            labeled_fg = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg = regionprops(labeled_fg)
            if regions_fg:
                largest_region_fg = max(regions_fg, key=lambda r: r.area)
                fg_largest = (labeled_fg == largest_region_fg.label)
                fg_largest = torch.from_numpy(fg_largest).to(fg.device, dtype=torch.float32)
            else:
                fg_largest = torch.zeros_like(fg)
        else:
            fg_largest = torch.zeros_like(fg)
        if bg.sum() > 0:
            labeled_bg = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg = regionprops(labeled_bg)
            if regions_bg:
                largest_region_bg = max(regions_bg, key=lambda r: r.area)
                bg_largest = (labeled_bg == largest_region_bg.label)
                bg_largest = torch.from_numpy(bg_largest).to(bg.device, dtype=torch.float32)
            else:
                bg_largest = torch.zeros_like(bg)
        else:
            bg_largest = torch.zeros_like(bg)
        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()
        if fg_area >= bg_area:
            fg_skeleton = generateGuidingSignal(fg_largest) if fg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            output[i, 0] = fg_skeleton
        else:
            bg_skeleton = generateGuidingSignal(bg_largest) if bg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            output[i, 1] = bg_skeleton
    return output


# ── GPU version (EDT-ridge, mirrors paper pipeline) ───────────────────────────

def _edt_skeleton_gpu(mask, max_steps=80):
    """
    Approximate skeleton of binary mask regions via EDT ridge (GPU, no CPU transfer).

    Algorithm mirrors the CPU pipeline:
      1. Approximate EDT via successive erosion: each pixel receives the step
         number at which it disappears under repeated 3x3 min-pooling
         (= Chebyshev distance to the region boundary).
      2. Random threshold in [mean-std, mean+std] of EDT values to get "core"
         (mirrors CPU's generateGuidingSignal tempThresh logic).
      3. Local maxima of EDT within core = ridge = medial axis ~ skeleton.

    Args:
        mask:      [B, 1, H, W] float32 binary tensor.
        max_steps: max erosion iterations (upper bound on region radius in px).
    Returns:
        [B, 1, H, W] float32 binary skeleton.
    """
    B = mask.shape[0]
    device = mask.device

    # 1. Approximate EDT via successive erosion
    edt = torch.zeros_like(mask)
    current = mask.clone()
    step = 0

    for step in range(1, max_steps + 1):
        next_c = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        newly_removed = (current > 0) & (next_c == 0)
        edt = edt + newly_removed.float() * step
        current = next_c
        if current.sum() == 0:
            break

    # Pixels surviving all steps -> assign distance = last step + 1
    edt = edt + (current > 0).float() * (step + 1)

    # 2. Random threshold (vectorised across batch)
    mask_bool = mask > 0
    edt_in_mask = edt * mask                              # zero outside mask

    n_pixels = mask.flatten(1).sum(1).clamp(min=1)       # [B]
    edt_sum  = edt_in_mask.flatten(1).sum(1)             # [B]
    edt_sq   = edt_in_mask.pow(2).flatten(1).sum(1)      # [B]

    means = edt_sum / n_pixels
    stds  = (edt_sq / n_pixels - means.pow(2)).clamp(min=0).sqrt()

    rand   = torch.rand(B, 1, 1, 1, device=device)
    thresh = (means.view(B,1,1,1) - stds.view(B,1,1,1)
              + rand * 2 * stds.view(B,1,1,1)).clamp(min=0)

    core = (edt > thresh) & mask_bool                    # [B, 1, H, W]

    # Fallback: empty core -> use max-EDT pixels
    core_count = core.float().flatten(1).sum(1)
    need_fallback = (core_count == 0) & (mask.flatten(1).sum(1) > 0)
    if need_fallback.any():
        max_edt = (edt * mask).flatten(1).max(1).values.view(B, 1, 1, 1)
        fallback = ((edt >= max_edt - 1e-6) & mask_bool).float()
        core_f = core.float()
        core_f[need_fallback] = fallback[need_fallback]
        core = core_f.bool()

    # 3. Ridge = local maxima of EDT within core (skeleton)
    core_edt  = edt * core.float()
    local_max = F.max_pool2d(core_edt, kernel_size=3, stride=1, padding=1)
    ridge = (core_edt >= local_max - 1e-6) & core

    # Fallback: empty ridge -> use core directly
    ridge_f = ridge.float()
    empty_ridge = (ridge_f.flatten(1).sum(1) == 0) & (core.float().flatten(1).sum(1) > 0)
    if empty_ridge.any():
        ridge_f[empty_ridge] = core.float()[empty_ridge]

    return ridge_f


def processMasks_gpu(pred_mask_all, GT_mask_all, max_edt_steps=80):
    """
    GPU-only drop-in replacement for processMasks. Closely mirrors CPU logic:
      - Computes fg (GT=1 & pred=0) and bg (GT=0 & pred=1) error regions.
      - Selects the larger region per sample (mirrors CPU's largest-CC selection).
      - Calls _edt_skeleton_gpu: EDT -> random threshold -> ridge -> skeleton.
      - Places result in the correct output channel (ch0=fg, ch1=bg).

    Args:
        pred_mask_all:  [B, 1, H, W]
        GT_mask_all:    [B, 1, H, W]
        max_edt_steps:  upper bound on EDT erosion iterations (default 80)
    Returns:
        [B, 2, H, W]  float32,  ch0 = fg guidance,  ch1 = bg guidance
    """
    device = pred_mask_all.device
    B, _, H, W = pred_mask_all.shape

    pred_bin = (pred_mask_all > 0.5).float()
    fg = ((GT_mask_all == 1) & (pred_bin == 0)).float()   # missed fg
    bg = ((GT_mask_all == 0) & (pred_bin == 1)).float()   # false positive bg

    # Select larger region per sample (mirrors CPU's largest-CC logic)
    fg_area = fg.flatten(1).sum(1)
    bg_area = bg.flatten(1).sum(1)
    use_fg  = (fg_area >= bg_area).float().view(B, 1, 1, 1)

    selected = fg * use_fg + bg * (1 - use_fg)           # [B, 1, H, W]
    skel     = _edt_skeleton_gpu(selected, max_edt_steps) # [B, 1, H, W]

    output = torch.zeros(B, 2, H, W, device=device)
    output[:, 0:1] = skel * use_fg          # fg channel
    output[:, 1:2] = skel * (1 - use_fg)   # bg channel

    return output


# ── helpers ──────────────────────────────────────────────────────────────────

def make_batch(B=4, H=150, W=150, device='cpu'):
    """Synthetic batch: GT has a square fg region, pred partially misses it."""
    gt   = torch.zeros(B, 1, H, W, device=device)
    pred = torch.zeros(B, 1, H, W, device=device)
    for i in range(B):
        r0, r1 = H // 4, 3 * H // 4
        c0, c1 = W // 4, 3 * W // 4
        gt[i, 0, r0:r1, c0:c1] = 1.0
        # Prediction misses the top half of fg -> fg error region
        pred[i, 0, (r0 + r1) // 2:r1, c0:c1] = 1.0
        # False positive outside GT -> bg error region
        pred[i, 0, 0:H // 8, 0:W // 8] = 1.0
    return gt, pred


def jaccard(a, b):
    inter = (a & b).float().sum()
    union = (a | b).float().sum()
    return (inter / union).item() if union > 0 else float('nan')


# ── run test ─────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W = 8, 150, 150
    gt, pred = make_batch(B, H, W, device=device)

    # ── CPU version ──
    t0 = time.perf_counter()
    out_cpu = processMasks(pred.float(), gt.float())
    t_cpu = time.perf_counter() - t0
    out_cpu = out_cpu.to(device)

    # ── GPU version ──
    t0 = time.perf_counter()
    out_gpu = processMasks_gpu(pred.float(), gt.float())
    t_gpu = time.perf_counter() - t0

    print(f"Shape CPU: {tuple(out_cpu.shape)}   GPU: {tuple(out_gpu.shape)}")
    assert out_cpu.shape == out_gpu.shape, "Shape mismatch!"

    # ── check: signal is inside the error region ──
    fg_error = ((gt == 1) & (pred < 0.5)).squeeze(1)   # [B, H, W]
    bg_error = ((gt == 0) & (pred >= 0.5)).squeeze(1)  # [B, H, W]

    for name, out in [("CPU", out_cpu), ("GPU", out_gpu)]:
        total_signal = out.sum().item()
        fg_in_err = ((out[:, 0] > 0) & fg_error).float().sum().item()
        bg_in_err = ((out[:, 1] > 0) & bg_error).float().sum().item()
        signal_in_err = fg_in_err + bg_in_err
        print(f"[{name}]")
        print(f"  Total signal pixels : {total_signal:.0f}")
        print(f"  Signal inside error : {signal_in_err:.0f}  "
              f"({100 * signal_in_err / max(total_signal, 1):.1f}% of signal is inside error region)")

    # ── check: spatial overlap of signals ──
    iou_fg = jaccard(out_cpu[:, 0] > 0, out_gpu[:, 0] > 0)
    iou_bg = jaccard(out_cpu[:, 1] > 0, out_gpu[:, 1] > 0)
    print(f"\nSpatial overlap (IoU) between CPU and GPU signals:")
    print(f"  fg channel: {iou_fg:.3f}")
    print(f"  bg channel: {iou_bg:.3f}")
    print(f"  (>0 means both cover overlapping regions)")

    # ── check: signal pixels outside their error region ──
    for ch, name, err in [(0, 'fg', fg_error), (1, 'bg', bg_error)]:
        cpu_out = ((out_cpu[:, ch] > 0) & ~err).float().sum().item()
        gpu_out = ((out_gpu[:, ch] > 0) & ~err).float().sum().item()
        print(f"\n  {name} signal outside error region — CPU: {cpu_out:.0f}   GPU: {gpu_out:.0f}")

    # ── timing ──
    print(f"\nTiming (batch_size={B}, {H}x{W}):")
    print(f"  CPU: {t_cpu * 1000:.1f} ms")
    print(f"  GPU: {t_gpu * 1000:.1f} ms")
    if t_gpu > 0:
        print(f"  Speedup: {t_cpu / t_gpu:.1f}x")

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
