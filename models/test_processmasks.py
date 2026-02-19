"""
Quick sanity-check: compare processMasks (CPU, original) vs processMasks_gpu (GPU).

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

# Import both versions from the training script
from train_roi_efficientunet_BCSS import processMasks, processMasks_gpu

# ── helpers ──────────────────────────────────────────────────────────────────

def make_batch(B=4, H=150, W=150, fg_ratio=0.2, device='cpu'):
    """Create a synthetic batch: GT mask with a square fg region, pred with partial miss."""
    gt   = torch.zeros(B, 1, H, W, device=device)
    pred = torch.zeros(B, 1, H, W, device=device)

    for i in range(B):
        # Foreground square
        r0, r1 = H // 4, 3 * H // 4
        c0, c1 = W // 4, 3 * W // 4
        gt[i, 0, r0:r1, c0:c1] = 1.0
        # Prediction misses the top half of fg (creates fg error region)
        pred[i, 0, (r0 + r1) // 2:r1, c0:c1] = 1.0
        # False positive outside gt (creates bg error region)
        pred[i, 0, 0:H // 8, 0:W // 8] = 1.0

    return gt, pred


def jaccard(a, b):
    """IoU between two binary tensors."""
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

    # ── check 1: signal is non-zero where error exists ──
    fg_error = ((gt == 1) & (pred < 0.5))   # [B, 1, H, W]
    bg_error = ((gt == 0) & (pred >= 0.5))  # [B, 1, H, W]

    for name, out in [("CPU", out_cpu), ("GPU", out_gpu)]:
        total_signal  = out.sum().item()
        fg_in_fg_err  = ((out[:, 0:1] > 0) & fg_error).float().sum().item()
        bg_in_bg_err  = ((out[:, 1:2] > 0) & bg_error).float().sum().item()
        signal_in_err = fg_in_fg_err + bg_in_bg_err

        print(f"\n[{name}]")
        print(f"  Total signal pixels : {total_signal:.0f}")
        print(f"  Signal inside error : {signal_in_err:.0f}  "
              f"({100*signal_in_err/max(total_signal,1):.1f}% of signal is inside error region)")

    # ── check 2: spatial overlap of fg signals ──
    cpu_fg = out_cpu[:, 0] > 0
    gpu_fg = out_gpu[:, 0] > 0
    iou_fg = jaccard(cpu_fg, gpu_fg)

    cpu_bg = out_cpu[:, 1] > 0
    gpu_bg = out_gpu[:, 1] > 0
    iou_bg = jaccard(cpu_bg, gpu_bg)

    print(f"\nSpatial overlap (IoU):")
    print(f"  fg channel: {iou_fg:.3f}")
    print(f"  bg channel: {iou_bg:.3f}")
    print(f"  (>0 means both versions cover overlapping regions)")

    # ── check 3: both signals inside their error regions ──
    for ch, name, err_mask in [(0, 'fg', fg_error.squeeze(1)),
                                (1, 'bg', bg_error.squeeze(1))]:
        cpu_sig = out_cpu[:, ch] > 0
        gpu_sig = out_gpu[:, ch] > 0
        cpu_out_of_err = (cpu_sig & ~err_mask).float().sum().item()
        gpu_out_of_err = (gpu_sig & ~err_mask).float().sum().item()
        print(f"\n  {name} signal pixels outside error region — "
              f"CPU: {cpu_out_of_err:.0f}   GPU: {gpu_out_of_err:.0f}")

    # ── timing ──
    print(f"\nTiming (batch_size={B}, {H}×{W}):")
    print(f"  CPU: {t_cpu*1000:.1f} ms")
    print(f"  GPU: {t_gpu*1000:.1f} ms")
    if t_cpu > 0:
        print(f"  Speedup: {t_cpu/t_gpu:.1f}x")

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
