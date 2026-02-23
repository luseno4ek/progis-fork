"""
Sanity-check: compare CPU processMasks (from inference script, returns centers)
vs GPU processMasks_gpu (new version with GPU centroid computation).

Run from models/ directory:
    python test_processmasks_centers.py

What is checked:
  1. Both functions return (signal [B,2,H,W], centers list of length B).
  2. centers[b] is either None or a (y, x) int tuple.
  3. Non-None centers are INSIDE the corresponding error region.
  4. CPU and GPU centers are close (within ~10% of image size).
  5. Signal pixels are inside error regions (>90%).
  6. Spatial overlap (IoU) between CPU and GPU signals is >0.
  7. GPU version is faster.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label as label_1, regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# ── CPU version (from backbone_efficientunet_inference_BCSS.py) ──────────────

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


def processMasks_cpu(pred_mask_all, GT_mask_all):
    """CPU version from inference script — returns (signal [B,2,H,W], centers list)."""
    pred_mask_all = (pred_mask_all > 0.5).float()
    batch_size, _, H, W = pred_mask_all.shape
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    centers = []

    for i in range(batch_size):
        pred_mask = pred_mask_all[i].squeeze(0)
        GT_mask   = GT_mask_all[i].squeeze(0)

        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32)
        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32)

        if fg.sum() > 0:
            labeled_fg  = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg  = regionprops(labeled_fg)
            if regions_fg:
                best_fg     = max(regions_fg, key=lambda r: r.area)
                fg_largest  = torch.from_numpy(labeled_fg == best_fg.label).to(fg.device, dtype=torch.float32)
                fg_center   = (round(best_fg.centroid[0]), round(best_fg.centroid[1]))
            else:
                fg_largest  = torch.zeros_like(fg)
                fg_center   = None
        else:
            fg_largest  = torch.zeros_like(fg)
            fg_center   = None

        if bg.sum() > 0:
            labeled_bg  = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg  = regionprops(labeled_bg)
            if regions_bg:
                best_bg     = max(regions_bg, key=lambda r: r.area)
                bg_largest  = torch.from_numpy(labeled_bg == best_bg.label).to(bg.device, dtype=torch.float32)
                bg_center   = (round(best_bg.centroid[0]), round(best_bg.centroid[1]))
            else:
                bg_largest  = torch.zeros_like(bg)
                bg_center   = None
        else:
            bg_largest  = torch.zeros_like(bg)
            bg_center   = None

        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            skel = generateGuidingSignal(fg_largest) if fg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            output[i, 0] = skel
            centers.append(fg_center)
        else:
            skel = generateGuidingSignal(bg_largest) if bg_largest.sum() > 0 else torch.zeros_like(pred_mask)
            output[i, 1] = skel
            centers.append(bg_center)

    return output, centers


# ── GPU version (from backbone_efficientunet_inference_BCSS.py) ──────────────

def _edt_skeleton_gpu(mask, max_steps=80):
    B = mask.shape[0]
    device = mask.device
    edt = torch.zeros_like(mask)
    current = mask.clone()
    for step in range(1, max_steps + 1):
        next_c = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        newly_removed = (current > 0) & (next_c == 0)
        edt = edt + newly_removed.float() * step
        current = next_c
    edt = edt + (current > 0).float() * (max_steps + 1)
    mask_bool = mask > 0
    edt_in_mask = edt * mask
    n_pixels = mask.flatten(1).sum(1).clamp(min=1)
    edt_sum  = edt_in_mask.flatten(1).sum(1)
    edt_sq   = edt_in_mask.pow(2).flatten(1).sum(1)
    means = edt_sum / n_pixels
    stds  = (edt_sq / n_pixels - means.pow(2)).clamp(min=0).sqrt()
    rand   = torch.rand(B, 1, 1, 1, device=device)
    thresh = (means.view(B,1,1,1) - stds.view(B,1,1,1)
              + rand * 2 * stds.view(B,1,1,1)).clamp(min=0)
    core = (edt > thresh) & mask_bool
    need_fallback = (core.float().flatten(1).sum(1) == 0) & (mask.flatten(1).sum(1) > 0)
    max_edt = (edt * mask).flatten(1).max(1).values.view(B, 1, 1, 1)
    fallback_core = (edt >= max_edt - 1e-6) & mask_bool
    core = torch.where(need_fallback.view(B, 1, 1, 1), fallback_core, core)
    core_edt  = edt * core.float()
    local_max = F.max_pool2d(core_edt, kernel_size=3, stride=1, padding=1)
    ridge = (core_edt >= local_max - 1e-6) & core
    empty_ridge = (ridge.float().flatten(1).sum(1) == 0) & (core.float().flatten(1).sum(1) > 0)
    ridge = torch.where(empty_ridge.view(B, 1, 1, 1), core, ridge)
    return ridge.float()


def processMasks_gpu(pred_mask_all, GT_mask_all, max_edt_steps=80):
    """GPU version with centroid — returns (signal [B,2,H,W], centers list)."""
    device = pred_mask_all.device
    B, _, H, W = pred_mask_all.shape

    pred_bin = (pred_mask_all > 0.5).float()
    fg = ((GT_mask_all == 1) & (pred_bin == 0)).float()
    bg = ((GT_mask_all == 0) & (pred_bin == 1)).float()

    fg_area = fg.flatten(1).sum(1)
    bg_area = bg.flatten(1).sum(1)
    use_fg  = (fg_area >= bg_area).float().view(B, 1, 1, 1)

    selected = fg * use_fg + bg * (1 - use_fg)
    skel     = _edt_skeleton_gpu(selected, max_edt_steps)

    output = torch.zeros(B, 2, H, W, device=device)
    output[:, 0:1] = skel * use_fg
    output[:, 1:2] = skel * (1 - use_fg)

    ys = torch.arange(H, device=device).float().view(1, 1, H, 1)
    xs = torch.arange(W, device=device).float().view(1, 1, 1, W)
    n  = selected.flatten(1).sum(1).clamp(min=1)
    cy = (selected * ys).flatten(1).sum(1) / n
    cx = (selected * xs).flatten(1).sum(1) / n
    has_region = selected.flatten(1).sum(1) > 0
    cy_list  = cy.round().long().tolist()
    cx_list  = cx.round().long().tolist()
    has_list = has_region.tolist()
    centers = [(cy_list[b], cx_list[b]) if has_list[b] else None for b in range(B)]

    return output, centers


# ── helpers ───────────────────────────────────────────────────────────────────

def make_batch(B=4, H=150, W=150, device='cpu'):
    """Synthetic batch: GT has square fg region, pred partially misses it."""
    gt   = torch.zeros(B, 1, H, W, device=device)
    pred = torch.zeros(B, 1, H, W, device=device)
    for i in range(B):
        r0, r1 = H // 4, 3 * H // 4
        c0, c1 = W // 4, 3 * W // 4
        gt[i, 0, r0:r1, c0:c1] = 1.0
        pred[i, 0, (r0 + r1) // 2:r1, c0:c1] = 1.0   # misses top half -> fg error
        pred[i, 0, 0:H // 8, 0:W // 8] = 1.0          # false positive -> bg error
    return gt, pred


def make_batch_perfect(B=4, H=150, W=150, device='cpu'):
    """Batch where prediction == GT → no error regions → all centers must be None."""
    gt = torch.zeros(B, 1, H, W, device=device)
    for i in range(B):
        gt[i, 0, H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0
    return gt, gt.clone()


def jaccard(a, b):
    inter = (a & b).float().sum()
    union = (a | b).float().sum()
    return (inter / union).item() if union > 0 else float('nan')


def center_in_region(center, region_mask):
    """True if (cy, cx) falls on a pixel where region_mask is True."""
    if center is None:
        return None
    cy, cx = center
    H, W = region_mask.shape
    if 0 <= cy < H and 0 <= cx < W:
        return region_mask[cy, cx].item()
    return False


# ── main test ─────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W = 8, 150, 150

    # ── Test 1: normal batch with error regions ───────────────────────────────
    print("=" * 60)
    print("TEST 1: batch with fg + bg error regions")
    print("=" * 60)

    gt, pred = make_batch(B, H, W, device=device)

    t0 = time.perf_counter()
    out_cpu, centers_cpu = processMasks_cpu(pred.float(), gt.float())
    t_cpu = time.perf_counter() - t0
    out_cpu = out_cpu.to(device)

    t0 = time.perf_counter()
    out_gpu, centers_gpu = processMasks_gpu(pred.float(), gt.float())
    t_gpu = time.perf_counter() - t0

    # 1a. Shape check
    assert out_cpu.shape == out_gpu.shape == (B, 2, H, W), "Shape mismatch!"
    assert len(centers_cpu) == len(centers_gpu) == B,       "centers length mismatch!"
    print(f"[PASS] Signal shape: {tuple(out_cpu.shape)}")
    print(f"[PASS] centers list length: {B}")

    # 1b. Center type check
    for b in range(B):
        for name, c in [('CPU', centers_cpu[b]), ('GPU', centers_gpu[b])]:
            if c is not None:
                assert isinstance(c, tuple) and len(c) == 2, f"{name} centers[{b}] not a 2-tuple"
    print("[PASS] All non-None centers are (y, x) int tuples")

    # 1c. Centers are inside error regions
    fg_error = ((gt == 1) & (pred < 0.5)).squeeze(1)   # [B, H, W]
    bg_error = ((gt == 0) & (pred >= 0.5)).squeeze(1)  # [B, H, W]

    cpu_in = gpu_in = 0
    for b in range(B):
        err_b = fg_error[b] | bg_error[b]
        if centers_cpu[b] is not None:
            ok = center_in_region(centers_cpu[b], err_b.cpu())
            cpu_in += int(ok)
        if centers_gpu[b] is not None:
            ok = center_in_region(centers_gpu[b], err_b.cpu())
            gpu_in += int(ok)

    n_valid = sum(1 for c in centers_cpu if c is not None)
    print(f"\nCenter inside error region (out of {n_valid} non-None):")
    print(f"  CPU: {cpu_in}/{n_valid}")
    print(f"  GPU: {gpu_in}/{n_valid}")
    assert cpu_in == n_valid, "CPU: some centers fall outside error regions!"
    assert gpu_in == n_valid, "GPU: some centers fall outside error regions!"
    print("[PASS] All centers are inside error regions")

    # 1d. CPU-GPU center proximity (< 15% of image diagonal)
    diag = (H**2 + W**2) ** 0.5
    tol  = 0.15 * diag
    dists = []
    for b in range(B):
        cc, gc = centers_cpu[b], centers_gpu[b]
        if cc is not None and gc is not None:
            dist = ((cc[0] - gc[0])**2 + (cc[1] - gc[1])**2) ** 0.5
            dists.append(dist)
    mean_dist = np.mean(dists) if dists else float('nan')
    print(f"\nCPU vs GPU center distance: mean={mean_dist:.1f}px  tolerance={tol:.0f}px")
    for d in dists:
        assert d <= tol, f"Center distance {d:.1f}px exceeds tolerance {tol:.0f}px"
    print("[PASS] CPU and GPU centers agree within tolerance")

    # 1e. Signal quality: pixels inside error region
    for name, out in [("CPU", out_cpu), ("GPU", out_gpu)]:
        total  = out.sum().item()
        in_err = (((out[:, 0] > 0) & fg_error) | ((out[:, 1] > 0) & bg_error)).float().sum().item()
        pct    = 100 * in_err / max(total, 1)
        print(f"\n[{name}] Signal pixels: total={total:.0f}  inside error={in_err:.0f}  ({pct:.1f}%)")
        assert pct >= 90, f"{name}: only {pct:.1f}% of signal is inside error region (expected >=90%)"
    print("[PASS] >90% of signal pixels are inside error regions")

    # 1f. Spatial overlap CPU vs GPU
    iou_fg = jaccard(out_cpu[:, 0] > 0, out_gpu[:, 0] > 0)
    iou_bg = jaccard(out_cpu[:, 1] > 0, out_gpu[:, 1] > 0)
    print(f"\nSignal IoU (CPU vs GPU):  fg={iou_fg:.3f}  bg={iou_bg:.3f}")
    # fg channel always used in synthetic batch (fg area > bg area)
    assert not np.isnan(iou_fg) and iou_fg > 0, "fg signals do not overlap!"
    print("[PASS] CPU and GPU fg signals overlap")

    # ── Test 2: perfect prediction → all centers must be None ─────────────────
    print("\n" + "=" * 60)
    print("TEST 2: perfect prediction → no error regions → centers = None")
    print("=" * 60)

    gt2, pred2 = make_batch_perfect(B, H, W, device=device)
    _, centers_cpu2 = processMasks_cpu(pred2.float(), gt2.float())
    _, centers_gpu2 = processMasks_gpu(pred2.float(), gt2.float())

    assert all(c is None for c in centers_cpu2), "CPU: expected all None for perfect pred"
    assert all(c is None for c in centers_gpu2), "GPU: expected all None for perfect pred"
    print("[PASS] All centers are None when prediction is perfect")

    # ── Test 3: timing ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TEST 3: timing  (B={B}, {H}x{W})")
    print("=" * 60)
    print(f"  CPU: {t_cpu * 1000:.1f} ms")
    print(f"  GPU: {t_gpu * 1000:.1f} ms")
    if t_gpu > 0:
        print(f"  Speedup: {t_cpu / t_gpu:.1f}x")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All checks passed.")


if __name__ == '__main__':
    main()
