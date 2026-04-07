"""
Visualisation of ProGIS iterative segmentation on BCSS val patches.

Uses the full ProGIS inference pipeline (prototype navigation on 512×512 +
iterative 256×256 ROI corrections), unlike the old models/visualize_inference.py
which passed the whole patch directly to segment_part.

Finds patches that appear in multiple tissue classes (same image file,
multiple GT masks), runs 20-iteration inference for each class, and produces:
  1. Grid PNG:  original | GT_combined | GT_per_class |
                pred@iter0 (prototype) | pred@iter1 | ... | pred@iter20
  2. (opt) GIF: animated prediction + accumulated scribbles over all iterations

Usage (from project root):
    # Via YAML config (recommended — reads data.patches_dir, data.splits_json,
    # data.fold, data.crop_size, eval.threshold, eval.device automatically):
    python -m progis_rework.inference.visualize \\
        --config    progis_rework/configs/server_bcss.yaml \\
        --roi_ckpt  runs/stage2/fold1/.../stage2_best.pth \\
        --n_samples 2 --animate

    # With SimCLR backbone:
    python -m progis_rework.inference.visualize \\
        --config    progis_rework/configs/server_bcss.yaml \\
        --roi_ckpt  runs/stage2/fold1/.../stage2_best.pth \\
        --backbone  simclr \\
        --proj_ckpt runs/simclr_proj/fold1/.../simclr_proj_best.pth

    # All flags explicit (no config):
    python -m progis_rework.inference.visualize \\
        --roi_ckpt    runs/stage2/fold1/.../stage2_best.pth \\
        --patches_dir /srv/.../BCSS/patches \\
        --splits_json /srv/.../BCSS/patches/fold_splits.json \\
        --fold 1 --device cuda:1 --n_samples 2 --animate
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from progis_rework.data.dataset import get_wsi_stem, load_fold_splits
from progis_rework.interactive.roi import (
    roi_crop_for_correction,
    roi_crop_for_prototype,
    paste_crop_into_mask,
    paste_crop_soft,
)
from progis_rework.interactive.signals import process_masks
from progis_rework.models.progis import ProGISModel


# ── Visual identity per class ─────────────────────────────────────────────────

CLASS_COLORS = {
    'tumor':                     np.array([0.86, 0.20, 0.20]),   # red
    'stroma':                    np.array([0.20, 0.40, 0.86]),   # blue
    'inflammatory_infiltration': np.array([0.20, 0.72, 0.20]),   # green
    'necrosis':                  np.array([0.90, 0.72, 0.00]),   # yellow
    'others':                    np.array([0.60, 0.20, 0.80]),   # purple
    # LumenStone S1
    'chalcopyrite': np.array([1.00, 0.65, 0.00]),
    'galena':       np.array([0.60, 0.80, 0.20]),
    'bornite':      np.array([0.00, 0.75, 1.00]),
    'pyrite':       np.array([0.18, 0.31, 0.31]),
    'sphalerite':   np.array([0.93, 0.51, 0.93]),
    'tenantite':    np.array([0.28, 0.24, 0.55]),
}
CLASS_SHORT = {
    'tumor':                     'tumor',
    'stroma':                    'stroma',
    'inflammatory_infiltration': 'inflam.',
    'necrosis':                  'necros.',
    'others':                    'others',
}
# Higher priority → drawn last (overwrites lower in combined overlay)
CLASS_PRIORITY = ['stroma', 'others', 'inflammatory_infiltration', 'necrosis', 'tumor',
                  'galena', 'bornite', 'pyrite', 'sphalerite', 'tenantite', 'chalcopyrite']


def _get_color(cls: str) -> np.ndarray:
    """Return color for a class, generating a deterministic fallback if unknown."""
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    rng = np.random.RandomState(abs(hash(cls)) % (2**31))
    return rng.uniform(0.3, 0.9, size=3)


# ── Data helpers ──────────────────────────────────────────────────────────────

def find_multiclass_patches(
    patches_dir: str,
    fold_splits: dict,
    fold: int,
    split: str,
    min_classes: int = 2,
    n: int = 3,
) -> list[tuple[str, list[str]]]:
    """
    Return up to n patch filenames that have GT masks in >= min_classes classes.
    Returns list of (fname, [cls1, cls2, ...]) sorted by descending class count.
    """
    valid_stems = set(fold_splits[f'fold_{fold}'][split])
    patches_dir = Path(patches_dir)
    fname_to_classes: dict = defaultdict(list)

    available = [d.name for d in patches_dir.iterdir()
                 if d.is_dir() and d.name != 'all' and (d / 'mask_npy').exists()]
    for cls in sorted(available):
        mask_dir = patches_dir / cls / 'mask_npy'
        if not mask_dir.exists():
            continue
        for f in mask_dir.glob('*.npy'):
            if get_wsi_stem(f.name) in valid_stems:
                fname_to_classes[f.name].append(cls)

    multi = [
        (fname, classes)
        for fname, classes in fname_to_classes.items()
        if len(classes) >= min_classes
    ]
    multi.sort(key=lambda x: len(x[1]), reverse=True)
    return multi[:n]


def load_patch_data(
    patches_dir: str,
    fname: str,
    cls: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (image [H,W,3], mask [H,W], signal [2,H,W]) for one patch."""
    d = Path(patches_dir)
    image  = np.load(d / 'all' / 'image_npy'         / fname).astype(np.float32)
    mask   = np.load(d / cls  / 'mask_npy'            / fname).astype(np.float32)
    signal = np.load(d / cls  / 'signal_all_line_npy' / fname).astype(np.float32)
    return image, mask, signal


def to_display(image_np: np.ndarray) -> np.ndarray:
    """Normalise image to [0,1] for display regardless of input range."""
    img = image_np.astype(np.float32)
    if img.max() > 1.5:   # likely [0, 255]
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:             ProGISModel,
    image_t:           torch.Tensor,   # [3, H, W]
    gt_mask_t:         torch.Tensor,   # [1, H, W]
    signal_t:          torch.Tensor,   # [2, H, W]
    device:            str,
    n_iters:           int       = 20,
    crop_size:         int       = 256,
    threshold:         float     = 0.5,
    max_stroke_length: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Full ProGIS inference pipeline for a single patch:
      1. Prototype initialisation: roi_crop_for_prototype → forward_prototype.
      2. Iterative correction loop (n_iters rounds):
           process_masks → roi_crop_for_correction → segment → paste back.

    Returns:
        all_masks   — list of n_iters+1 numpy arrays [H,W]:
                      index 0 = prototype init, index k = after k-th correction.
        all_signals — list of n_iters+1 numpy arrays [2,H,W]:
                      accumulated union signal at each step.
    """
    images  = image_t.unsqueeze(0).to(device)    # [1, 3, H, W]
    masks   = gt_mask_t.unsqueeze(0).to(device)  # [1, 1, H, W]
    signals = signal_t.unsqueeze(0).to(device)   # [1, 2, H, W]

    all_masks:   list[np.ndarray] = []
    all_signals: list[np.ndarray] = []

    # ── Step 0: prototype initialisation ──────────────────────────────────────
    proto_crop = roi_crop_for_prototype(images, signals, masks, crop_size)
    proto_out  = model.forward_prototype(
        roi_input  = proto_crop.roi_images,
        roi_signal = proto_crop.roi_signals,
        full_image = images,
        mask_box   = proto_crop.mask_box,
        threshold  = threshold,
    )
    current_mask = proto_out.prototype_mask.clone()    # [1, 1, H, W]
    all_masks.append(current_mask.squeeze().cpu().numpy())
    # Step 0: only fg channel — ch1 of signal_all_line_npy is a pre-computed bg
    # skeleton from preprocessing, no bg corrections applied yet.
    sig0 = signals.squeeze().cpu().numpy().copy()
    sig0[1] = 0.0
    all_signals.append(sig0)

    # ── Steps 1..n_iters: iterative correction ────────────────────────────────
    error_signal, centers = process_masks(current_mask, masks, max_stroke_length)
    union_signal = torch.bitwise_or(
        error_signal.to(torch.uint8),
        signals.to(torch.uint8),
    ).float()

    for _ in range(n_iters):
        crop_batch = roi_crop_for_correction(
            images, current_mask, union_signal, centers, crop_size,
        )
        crop_pred = model.segment(
            crop_batch.roi_images,
            crop_batch.roi_prev_masks,
            crop_batch.roi_signals,
        )
        paste_crop_into_mask(current_mask, crop_pred, centers,
                             images.shape[2], images.shape[3], crop_size)

        all_masks.append(current_mask.squeeze().cpu().numpy())
        all_signals.append(union_signal.squeeze().cpu().numpy())

        error_signal, centers = process_masks(current_mask, masks, max_stroke_length)
        union_signal = torch.bitwise_or(
            error_signal.to(torch.uint8),
            union_signal.to(torch.uint8),
        ).float()

    return all_masks, all_signals


@torch.no_grad()
def run_inference_multiclass(
    model:             ProGISModel,
    image_t:           torch.Tensor,         # [3, H, W]
    gt_masks_t:        list[torch.Tensor],   # n_cls × [1, H, W]
    signals_t:         list[torch.Tensor],   # n_cls × [2, H, W]
    available_classes: list[str],
    device:            str,
    n_iters:           int       = 20,
    crop_size:         int       = 256,
    threshold:         float     = 0.5,
    max_stroke_length: int | None = None,
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    """
    Multi-class inference: proto navigation without pixel overlaps.

    Proto step: all classes together via forward_prototype_multiclass()
      → argmax over per-class similarity → no pixel assigned to 2+ classes.
    Correction step: each class independently (overlap impossible by design).

    Returns:
        dict cls → (all_masks, all_signals), same format as run_inference().
    """
    images = image_t.unsqueeze(0).to(device)   # [1, 3, H, W]
    n_cls  = len(available_classes)

    masks_list   = [gt_masks_t[k].unsqueeze(0).to(device)  for k in range(n_cls)]
    signals_list = [signals_t[k].unsqueeze(0).to(device)   for k in range(n_cls)]

    # ── Proto step: all classes at once ───────────────────────────────────────
    proto_crops = [
        roi_crop_for_prototype(images, signals_list[k], masks_list[k], crop_size)
        for k in range(n_cls)
    ]
    proto_outputs = model.forward_prototype_multiclass(
        roi_inputs  = [pc.roi_images  for pc in proto_crops],
        roi_signals = [pc.roi_signals for pc in proto_crops],
        full_image  = images,
        mask_boxes  = [pc.mask_box    for pc in proto_crops],
        threshold   = threshold,
    )

    # ── Correction step: joint argmax conflict resolution ─────────────────────
    # Maintain one soft probability map per class [1, 1, H, W].
    # After each round all classes paste their raw predictions, then argmax
    # assigns each pixel to at most one class (or background if max < 0.5).
    H, W = images.shape[2], images.shape[3]

    current_probs = [proto_outputs[k].prototype_mask.float().clone() for k in range(n_cls)]
    current_masks = [proto_outputs[k].prototype_mask.clone()         for k in range(n_cls)]

    # Per-class history and accumulated union signal.
    # Step 0 = prototype init: only fg channel (ch0) is shown — ch1 in the raw
    # signal_all_line_npy file is a pre-computed bg skeleton from dataset
    # preprocessing, but no bg corrections have been applied yet at this stage.
    all_masks_h: list[list[np.ndarray]] = [[cm.squeeze().cpu().numpy()] for cm in current_masks]
    all_signals_h: list[list[np.ndarray]] = []
    for k in range(n_cls):
        sig_np = signals_list[k].squeeze().cpu().numpy().copy()  # [2, H, W]
        sig_np[1] = 0.0  # zero bg channel: no bg corrections at prototype init
        all_signals_h.append([sig_np])

    union_signals: list[torch.Tensor]          = []
    centers_list:  list[list[tuple | None]]    = []
    for k in range(n_cls):
        err, ctr = process_masks(current_masks[k], masks_list[k], max_stroke_length)
        union_signals.append(
            torch.bitwise_or(err.to(torch.uint8), signals_list[k].to(torch.uint8)).float()
        )
        centers_list.append(ctr)

    for _ in range(n_iters):
        # 1. Collect soft predictions for every class into updated prob maps
        new_probs = [p.clone() for p in current_probs]

        for k in range(n_cls):
            crop_batch = roi_crop_for_correction(
                images, current_masks[k], union_signals[k], centers_list[k], crop_size,
            )
            crop_pred = model.segment(
                crop_batch.roi_images,
                crop_batch.roi_prev_masks,
                crop_batch.roi_signals,
            )
            paste_crop_soft(new_probs[k], crop_pred, centers_list[k], H, W, crop_size)

        # 2. Argmax conflict resolution: each pixel → at most one class
        prob_stack = torch.cat(new_probs, dim=1)          # [1, n_cls, H, W]
        best_prob, best_cls_map = prob_stack.max(dim=1, keepdim=True)  # [1, 1, H, W]

        current_probs = new_probs
        for k in range(n_cls):
            current_masks[k] = ((best_cls_map == k) & (best_prob > 0.5)).float()

        # 3. Record masks + update union signals
        for k in range(n_cls):
            all_masks_h[k].append(current_masks[k].squeeze().cpu().numpy())
            all_signals_h[k].append(union_signals[k].squeeze().cpu().numpy())

            err, ctr = process_masks(current_masks[k], masks_list[k], max_stroke_length)
            union_signals[k] = torch.bitwise_or(
                err.to(torch.uint8), union_signals[k].to(torch.uint8),
            ).float()
            centers_list[k] = ctr

    return {
        cls: (all_masks_h[k], all_signals_h[k])
        for k, cls in enumerate(available_classes)
    }


# ── Overlay helpers ───────────────────────────────────────────────────────────

def apply_mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:
    """Blend binary mask over image with given color."""
    out = img.copy()
    where = mask > 0.5
    for c in range(3):
        out[:, :, c] = np.where(
            where,
            (1 - alpha) * img[:, :, c] + alpha * color[c],
            img[:, :, c],
        )
    return out


def combined_overlay(
    img: np.ndarray,
    masks_per_class: dict[str, np.ndarray],
    alpha: float = 0.55,
) -> np.ndarray:
    """Multi-class overlay in CLASS_PRIORITY order (tumour wins over stroma)."""
    out = img.copy()
    all_cls = list(CLASS_PRIORITY) + [c for c in masks_per_class if c not in CLASS_PRIORITY]
    for cls in all_cls:
        mask = masks_per_class.get(cls)
        if mask is None:
            continue
        color = _get_color(cls)
        where = mask > 0.5
        for c in range(3):
            out[:, :, c] = np.where(
                where,
                (1 - alpha) * out[:, :, c] + alpha * color[c],
                out[:, :, c],
            )
    return out


def scribble_overlay(
    img: np.ndarray,
    signals_per_class: dict[str, np.ndarray],
    radius: int = 3,
) -> np.ndarray:
    """
    Overlay fg scribbles (class colour) and bg scribbles (white).
    signal shape: [2, H, W], ch0=fg, ch1=bg.
    """
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    out = img.copy()
    all_cls = list(CLASS_PRIORITY) + [c for c in signals_per_class if c not in CLASS_PRIORITY]
    for cls in all_cls:
        sig = signals_per_class.get(cls)
        if sig is None:
            continue
        color = _get_color(cls)
        fg_px = binary_dilation(sig[0] > 0, structure=struct)
        bg_px = binary_dilation(sig[1] > 0, structure=struct)
        for c in range(3):
            out[:, :, c] = np.where(fg_px, color[c], out[:, :, c])
        out[bg_px] = [1.0, 1.0, 1.0]
    return out


# ── Grid figure ───────────────────────────────────────────────────────────────

def plot_grid(
    image_np:           np.ndarray,
    gt_per_class:       dict[str, np.ndarray],
    all_masks_per_class: dict[str, list[np.ndarray]],
    available_classes:  list[str],
    snap_indices:       list[int],
    out_path:           Path,
) -> None:
    """
    Layout:
      Row 0  : wide original image + legend
      Row 1  : GT (combined + per-class)
      Row 2+ : one row per snapshot index (0 = prototype init, k = after k corrections)
    """
    n_cls  = len(available_classes)
    n_cols = 1 + n_cls
    n_snaps = len(snap_indices)
    n_rows  = 1 + 1 + n_snaps

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))

    # Row 0: wide original
    ax_orig = fig.add_subplot(n_rows, 1, 1)
    ax_orig.imshow(image_np)
    ax_orig.set_title('Original patch (512×512)', fontsize=10, fontweight='bold')
    ax_orig.axis('off')
    handles = [
        mpatches.Patch(color=_get_color(cls), label=CLASS_SHORT.get(cls, cls))
        for cls in available_classes
    ]
    ax_orig.legend(handles=handles, loc='lower right', fontsize=8,
                   framealpha=0.8, ncol=len(available_classes))

    # Rows 1+: GT then snapshots
    iter_labels = (
        ['GT']
        + ['proto']
        + [f'iter {idx}' for idx in snap_indices if idx > 0]
    )
    masks_rows = (
        [gt_per_class]
        + [
            {cls: all_masks_per_class[cls][idx] for cls in available_classes
             if cls in all_masks_per_class}
            for idx in snap_indices
        ]
    )

    for row_i, (row_label, masks_dict) in enumerate(zip(iter_labels, masks_rows)):
        base_row = row_i + 1

        ax = fig.add_subplot(n_rows, n_cols, base_row * n_cols + 1)
        ax.imshow(combined_overlay(image_np, masks_dict))
        ax.set_title(f'{row_label} | combined', fontsize=7)
        ax.axis('off')

        for col_i, cls in enumerate(available_classes):
            ax = fig.add_subplot(n_rows, n_cols, base_row * n_cols + 2 + col_i)
            mask = masks_dict.get(cls)
            if mask is not None:
                ax.imshow(apply_mask_overlay(image_np, mask, _get_color(cls)))
            else:
                ax.imshow(image_np)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title(f'{row_label} | {CLASS_SHORT.get(cls, cls)}', fontsize=7)
            ax.axis('off')

    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Grid saved → {out_path}")


# ── Animation ─────────────────────────────────────────────────────────────────

def make_animation(
    image_np:             np.ndarray,
    all_masks_per_class:  dict[str, list[np.ndarray]],
    all_signals_per_class: dict[str, list[np.ndarray]],
    available_classes:    list[str],
    gt_per_class:         dict[str, np.ndarray],
    out_path:             Path,
    fps:                  int = 2,
    scribble_radius:      int = 3,
) -> None:
    """
    GIF with 3 panels: GT (static) | combined prediction | accumulated scribbles.
    Frame 0 = prototype initialisation, frames 1..n = iterative corrections.
    """
    n_frames = len(next(iter(all_masks_per_class.values())))

    fig, (ax_gt, ax_pred, ax_sig) = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05)

    gt_frame = combined_overlay(image_np, gt_per_class)

    def _pred_frame(i):
        return combined_overlay(image_np,
                                {cls: all_masks_per_class[cls][i]
                                 for cls in available_classes
                                 if cls in all_masks_per_class})

    def _sig_frame(i):
        return scribble_overlay(image_np,
                                {cls: all_signals_per_class[cls][i]
                                 for cls in available_classes
                                 if cls in all_signals_per_class},
                                radius=scribble_radius)

    ax_gt.imshow(gt_frame)
    im_pred = ax_pred.imshow(_pred_frame(0))
    im_sig  = ax_sig.imshow(_sig_frame(0))

    ax_gt.set_title('Ground truth',           fontsize=11, fontweight='bold')
    ax_pred.set_title('Prediction',            fontsize=11)
    ax_sig.set_title('Accumulated scribbles', fontsize=11)
    for ax in (ax_gt, ax_pred, ax_sig):
        ax.axis('off')

    ttl = fig.suptitle('Prototype init', fontsize=12)

    handles = [
        mpatches.Patch(color=_get_color(cls), label=CLASS_SHORT.get(cls, cls))
        for cls in available_classes
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(available_classes),
               fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.85)

    def update(frame):
        im_pred.set_data(_pred_frame(frame))
        im_sig.set_data(_sig_frame(frame))
        label = 'Prototype init' if frame == 0 else f'Iteration {frame} / {n_frames - 1}'
        ttl.set_text(label)
        return [im_pred, im_sig, ttl]

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Animation saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cfg_from_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    d = raw.get("data",  {})
    m = raw.get("model", {})
    e = raw.get("eval",  {})
    return {
        "patches_dir": d.get("patches_dir"),
        "splits_json": d.get("splits_json"),
        "fold":        d.get("fold",       1),
        "crop_size":   d.get("crop_size",  256),
        "backbone":    m.get("backbone",   "efficientunet"),
        "roi_ckpt":    m.get("roi_ckpt",   ""),
        "proj_ckpt":   m.get("proj_ckpt",  ""),
        "threshold":   e.get("threshold",  0.5),
        "device":      e.get("device",     None),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Visualise ProGIS iterative inference on BCSS patches.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',      default=None,
                   help='Path to a YAML config file. Individual flags override YAML values.')
    # Required unless provided via config
    p.add_argument('--roi_ckpt',    default=None,
                   help='Path to segment_part .pth checkpoint.')
    p.add_argument('--patches_dir', default=None,
                   help='Path to data/patches/ root.')
    p.add_argument('--splits_json', default=None,
                   help='Path to fold_splits.json.')
    # Optional overrides
    p.add_argument('--backbone',    default=None,
                   choices=['efficientunet', 'simclr', 'petroscope_resnet34'])
    p.add_argument('--proj_ckpt',   default=None,
                   help='SimCLR projection head checkpoint (simclr only).')
    p.add_argument('--fold',        type=int, default=None)
    p.add_argument('--device',      default=None,
                   help='Torch device, e.g. cpu, cuda, cuda:1.')
    p.add_argument('--crop_size',   type=int, default=None,
                   help='ROI crop size for segment_part.')
    p.add_argument('--threshold',   type=float, default=None,
                   help='Prototype similarity threshold.')
    # Visualisation-specific (no YAML equivalent)
    p.add_argument('--split',       default='val', choices=['train', 'val'])
    p.add_argument('--n_samples',   type=int, default=2,
                   help='Number of multi-class patches to visualise.')
    p.add_argument('--min_classes', type=int, default=2,
                   help='Minimum number of tissue classes per patch.')
    p.add_argument('--n_iters',     type=int, default=20,
                   help='Number of correction iterations.')
    p.add_argument('--snap_iters',  type=int, nargs='+',
                   default=[0, 1, 5, 10, 20],
                   help='Iteration indices to show in grid (0 = prototype init).')
    p.add_argument('--out_dir',     default='visualizations',
                   help='Output directory for PNG/GIF files.')
    p.add_argument('--animate',          action='store_true',
                   help='Also save a GIF animation.')
    p.add_argument('--scribble_radius', type=int, default=3,
                   help='Dilation radius for scribble display in pixels (default 3 → 7×7).')
    p.add_argument('--max_stroke', type=int, default=None,
                   help='Max stroke length in pixels per correction step '
                        '(None = unlimited). E.g. --max_stroke 50.')
    p.add_argument('--multiclass_proto', action='store_true',
                   help='Use joint multi-class prototype navigation (no pixel overlaps). '
                        'Default: per-class independent (paper-faithful).')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    defaults = _cfg_from_yaml(args.config) if args.config else {}

    def get(key, cast=None, fallback=None):
        cli_val = getattr(args, key, None)
        val = cli_val if cli_val is not None else defaults.get(key, fallback)
        return cast(val) if (cast and val is not None) else val

    patches_dir = get("patches_dir")
    splits_json = get("splits_json")
    roi_ckpt    = get("roi_ckpt", str, "")

    if not patches_dir or not splits_json:
        _build_parser().error(
            "--patches_dir and --splits_json are required "
            "(pass them directly or via --config)."
        )
    if not roi_ckpt:
        _build_parser().error(
            "--roi_ckpt is required (pass directly or set model.roi_ckpt in the YAML)."
        )

    device = get("device") or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    backbone    = get("backbone", str, "efficientunet")
    proj_ckpt   = get("proj_ckpt", str, "")
    crop_size   = get("crop_size", int, 256)
    threshold   = get("threshold", float, 0.5)
    fold        = get("fold", int, 1)

    backbone_kwargs = {}
    if proj_ckpt:
        backbone_kwargs['proj_ckpt'] = proj_ckpt

    model = ProGISModel.from_checkpoint(
        backbone_name   = backbone,
        roi_ckpt        = roi_ckpt,
        backbone_kwargs = backbone_kwargs,
    ).to(device)

    # ── Find multi-class patches ──────────────────────────────────────────────
    fold_splits = load_fold_splits(splits_json)
    patches = find_multiclass_patches(
        patches_dir, fold_splits, fold, args.split,
        min_classes=args.min_classes, n=args.n_samples,
    )
    if not patches:
        print(f"No patches found with >= {args.min_classes} classes. "
              f"Try --min_classes 1")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_indices = sorted(set(
        min(i, args.n_iters) for i in args.snap_iters
    ))

    for sample_i, (fname, available_classes) in enumerate(patches):
        print(f"\n[{sample_i+1}/{len(patches)}] {fname}")
        print(f"  Classes: {available_classes}")

        image_raw, _, _ = load_patch_data(patches_dir, fname, available_classes[0])
        image_np = to_display(image_raw)   # [H, W, 3] in [0, 1]  — display only
        # Model expects the same range as RoISegDataset (raw .npy, typically [0, 255])
        image_t_base = torch.tensor(image_raw.transpose(2, 0, 1), dtype=torch.float32)

        gt_per_class:           dict[str, np.ndarray]        = {}
        all_masks_per_class:    dict[str, list[np.ndarray]]  = {}
        all_signals_per_class:  dict[str, list[np.ndarray]]  = {}

        if args.multiclass_proto and len(available_classes) > 1:
            gt_masks_t = []
            signals_t  = []
            for cls in available_classes:
                _, mask_np, signal_np = load_patch_data(patches_dir, fname, cls)
                gt_per_class[cls] = mask_np
                gt_masks_t.append(torch.tensor(mask_np,   dtype=torch.float32).unsqueeze(0))
                signals_t.append( torch.tensor(signal_np, dtype=torch.float32))

            mc_results = run_inference_multiclass(
                model, image_t_base, gt_masks_t, signals_t, available_classes, device,
                n_iters=args.n_iters, crop_size=crop_size, threshold=threshold,
                max_stroke_length=args.max_stroke,
            )
            for cls, (all_masks, all_sigs) in mc_results.items():
                all_masks_per_class[cls]   = all_masks
                all_signals_per_class[cls] = all_sigs
        else:
            for cls in tqdm(available_classes, desc='  Classes', leave=False):
                _, mask_np, signal_np = load_patch_data(patches_dir, fname, cls)

                mask_t   = torch.tensor(mask_np,   dtype=torch.float32).unsqueeze(0)
                signal_t = torch.tensor(signal_np, dtype=torch.float32)

                all_masks, all_sigs = run_inference(
                    model, image_t_base, mask_t, signal_t, device,
                    n_iters=args.n_iters, crop_size=crop_size,
                    threshold=threshold, max_stroke_length=args.max_stroke,
                )

                gt_per_class[cls]          = mask_np
                all_masks_per_class[cls]   = all_masks
                all_signals_per_class[cls] = all_sigs

        stem = Path(fname).stem
        grid_path = out_dir / f'sample_{sample_i:02d}_{stem}_grid.png'
        plot_grid(
            image_np, gt_per_class, all_masks_per_class,
            available_classes, snap_indices, grid_path,
        )

        if args.animate:
            anim_path = out_dir / f'sample_{sample_i:02d}_{stem}_anim.gif'
            make_animation(
                image_np, all_masks_per_class, all_signals_per_class,
                available_classes, gt_per_class, anim_path,
                scribble_radius=args.scribble_radius,
            )

    print(f"\nDone. Results → {out_dir}/")


if __name__ == '__main__':
    main()
