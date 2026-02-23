"""
Visualization of ProGIS iterative segmentation on BCSS val patches.

Finds patches that appear in multiple tissue classes (same image, multiple GT masks),
runs 20-iteration inference for each class, and produces:
  1. Grid PNG:  original | GT_combined | GT_per_class | pred@iter1 | pred@iter5 | ...
  2. (opt) GIF: animated mask+scribble overlay for all 20 iterations

Usage (from project root):
    python models/visualize_inference.py \\
        --checkpoint data/patches/fold_1/ROI_ckpt/BCSS_effi-Unet_roi_best.pth \\
        --patches_dir data/patches \\
        --splits_json data/processed/fold_splits.json \\
        --fold 1 --n_samples 2 --animate
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from efficientunet import get_efficientunet_b0
from dataset import load_fold_splits, ALL_CLASSES, get_wsi_stem


# ── Visual identity per class ─────────────────────────────────────────────────

CLASS_COLORS = {
    'tumor':                     np.array([0.86, 0.20, 0.20]),   # red
    'stroma':                    np.array([0.20, 0.40, 0.86]),   # blue
    'inflammatory_infiltration': np.array([0.20, 0.72, 0.20]),   # green
    'necrosis':                  np.array([0.90, 0.72, 0.00]),   # yellow
    'others':                    np.array([0.60, 0.20, 0.80]),   # purple
}
CLASS_SHORT = {
    'tumor':                     'tumor',
    'stroma':                    'stroma',
    'inflammatory_infiltration': 'inflam.',
    'necrosis':                  'necros.',
    'others':                    'others',
}
# Higher index → drawn last (higher priority overwrites lower in combined overlay)
CLASS_PRIORITY = ['stroma', 'others', 'inflammatory_infiltration', 'necrosis', 'tumor']

# Iteration indices (0-based) to show in the grid (displayed as iter 1, 5, 10, 15, 20)
SNAPSHOT_ITERS = [0, 4, 9, 14, 19]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_segment_part(checkpoint_path: str, device: str):
    """Load only the ROI-Seg (segment_part) from a saved state_dict."""
    model = get_efficientunet_b0(
        out_channels=1, concat_input=True, pretrained=False, backbone=False
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    return model.eval().to(device)


# ── GPU processMasks (no CPU-GPU sync, returns centroid) ──────────────────────

def _edt_skeleton_gpu(mask: torch.Tensor, max_steps: int = 80) -> torch.Tensor:
    B = mask.shape[0]
    device = mask.device
    edt = torch.zeros_like(mask)
    current = mask.clone()
    for step in range(1, max_steps + 1):
        next_c = -F.max_pool2d(-current, kernel_size=3, stride=1, padding=1)
        newly_removed = (current > 0) & (next_c == 0)
        edt += newly_removed.float() * step
        current = next_c
    edt += (current > 0).float() * (max_steps + 1)
    mask_bool = mask > 0
    n_pixels = mask.flatten(1).sum(1).clamp(min=1)
    edt_in   = edt * mask
    means = edt_in.flatten(1).sum(1) / n_pixels
    stds  = ((edt_in.pow(2).flatten(1).sum(1) / n_pixels) - means.pow(2)).clamp(min=0).sqrt()
    rand   = torch.rand(B, 1, 1, 1, device=device)
    thresh = (means.view(B,1,1,1) - stds.view(B,1,1,1) + rand * 2 * stds.view(B,1,1,1)).clamp(min=0)
    core = (edt > thresh) & mask_bool
    need_fb = (core.float().flatten(1).sum(1) == 0) & (mask.flatten(1).sum(1) > 0)
    max_edt = (edt * mask).flatten(1).max(1).values.view(B, 1, 1, 1)
    core = torch.where(need_fb.view(B,1,1,1), (edt >= max_edt - 1e-6) & mask_bool, core)
    core_edt  = edt * core.float()
    local_max = F.max_pool2d(core_edt, kernel_size=3, stride=1, padding=1)
    ridge = (core_edt >= local_max - 1e-6) & core
    empty = (ridge.float().flatten(1).sum(1) == 0) & (core.float().flatten(1).sum(1) > 0)
    ridge = torch.where(empty.view(B,1,1,1), core, ridge)
    return ridge.float()


def processMasks_gpu(pred: torch.Tensor, gt: torch.Tensor):
    """Returns (signal [B,2,H,W], centers list). Fully on GPU."""
    device = pred.device
    B, _, H, W = pred.shape
    pred_bin = (pred > 0.5).float()
    fg = ((gt == 1) & (pred_bin == 0)).float()
    bg = ((gt == 0) & (pred_bin == 1)).float()
    use_fg = (fg.flatten(1).sum(1) >= bg.flatten(1).sum(1)).float().view(B, 1, 1, 1)
    selected = fg * use_fg + bg * (1 - use_fg)
    skel = _edt_skeleton_gpu(selected)
    out = torch.zeros(B, 2, H, W, device=device)
    out[:, 0:1] = skel * use_fg
    out[:, 1:2] = skel * (1 - use_fg)
    return out


# ── Data helpers ──────────────────────────────────────────────────────────────

def find_multiclass_patches(patches_dir: str, fold_splits: dict,
                            fold: int, split: str,
                            min_classes: int = 2, n: int = 3) -> list:
    """
    Return up to n patch filenames that have GT masks in ≥ min_classes tissue classes.
    Returns list of (fname, [class1, class2, ...]) sorted by descending class count.
    """
    valid_stems = set(fold_splits[f'fold_{fold}'][split])
    patches_dir = Path(patches_dir)
    fname_to_classes: dict = defaultdict(list)

    for cls in ALL_CLASSES:
        mask_dir = patches_dir / cls / 'mask_npy'
        if not mask_dir.exists():
            continue
        for f in mask_dir.glob('*.npy'):
            if get_wsi_stem(f.name) in valid_stems:
                fname_to_classes[f.name].append(cls)

    multi = [(fname, classes)
             for fname, classes in fname_to_classes.items()
             if len(classes) >= min_classes]
    multi.sort(key=lambda x: len(x[1]), reverse=True)
    return multi[:n]


def load_patch_data(patches_dir: str, fname: str, cls: str):
    """Load (image [H,W,3], mask [H,W], signal [2,H,W]) for one patch."""
    d = Path(patches_dir)
    image  = np.load(d / 'all' / 'image_npy'          / fname).astype(np.float32)
    mask   = np.load(d / cls  / 'mask_npy'             / fname).astype(np.float32)
    signal = np.load(d / cls  / 'signal_all_line_npy'  / fname).astype(np.float32)
    return image, mask, signal


def to_display(image_np: np.ndarray) -> np.ndarray:
    """Normalise image to [0,1] for display regardless of input range."""
    img = image_np.astype(np.float32)
    vmax = img.max()
    if vmax > 1.5:          # likely [0, 255]
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, image_t, gt_mask_t, init_signal_t, device, n_iters=20):
    """
    Run n_iters of iterative inference.

    Returns:
        snapshots  – list of 5 numpy arrays [H,W], pred masks at SNAPSHOT_ITERS
        all_masks  – list of 20 numpy arrays [H,W], pred masks at every iter
        all_signals – list of 20 numpy arrays [2,H,W], accumulated signals at every iter
    """
    img    = image_t.unsqueeze(0).to(device)        # [1,3,H,W]
    gt     = gt_mask_t.unsqueeze(0).to(device)      # [1,1,H,W]
    u_sig  = init_signal_t.unsqueeze(0).float().to(device)  # [1,2,H,W]
    prev   = torch.zeros_like(gt)                   # [1,1,H,W]

    all_masks   = []
    all_signals = []

    for it in range(n_iters):
        inp  = torch.cat([img, prev, u_sig], dim=1)   # [1,6,H,W]
        pred = model(inp)                              # [1,1,H,W]

        all_masks.append((pred > 0.5).squeeze().cpu().numpy().astype(np.float32))
        all_signals.append(u_sig.squeeze().cpu().numpy())   # [2,H,W]

        # Prepare next iteration
        sig   = processMasks_gpu(pred, gt)
        u_sig = torch.bitwise_or(sig.to(torch.uint8), u_sig.to(torch.uint8)).float()
        prev  = (pred > 0.5).float()

    snapshots = [all_masks[i] for i in SNAPSHOT_ITERS]
    return snapshots, all_masks, all_signals


# ── Overlay helpers ───────────────────────────────────────────────────────────

def apply_mask_overlay(img: np.ndarray, mask: np.ndarray,
                       color: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    """Blend binary mask over image with given color."""
    out = img.copy()
    where = mask > 0.5
    for c in range(3):
        out[:, :, c] = np.where(where,
                                (1 - alpha) * img[:, :, c] + alpha * color[c],
                                img[:, :, c])
    return out


def combined_overlay(img: np.ndarray, masks_per_class: dict,
                     alpha: float = 0.55) -> np.ndarray:
    """
    Build multi-class overlay. Draws classes in CLASS_PRIORITY order
    so tumour (highest priority) always wins over stroma.
    """
    out = img.copy()
    for cls in CLASS_PRIORITY:
        mask = masks_per_class.get(cls)
        if mask is None:
            continue
        color = CLASS_COLORS[cls]
        where = mask > 0.5
        for c in range(3):
            out[:, :, c] = np.where(where,
                                    (1 - alpha) * img[:, :, c] + alpha * color[c],
                                    out[:, :, c])
    return out


def _dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Thicken a binary mask with a disk-shaped structuring element."""
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask > 0, structure=struct)


def scribble_overlay(img: np.ndarray, signals_per_class: dict,
                     scribble_radius: int = 3) -> np.ndarray:
    """
    Overlay accumulated scribbles: fg scribbles in class colour, bg scribbles in white.
    signal shape: [2, H, W], ch0=fg, ch1=bg.
    Strokes are thickened by scribble_radius pixels for visibility.
    """
    out = img.copy()
    for cls in CLASS_PRIORITY:
        sig = signals_per_class.get(cls)
        if sig is None:
            continue
        color = CLASS_COLORS[cls]
        fg_px = _dilate(sig[0], scribble_radius)
        bg_px = _dilate(sig[1], scribble_radius)
        for c in range(3):
            out[:, :, c] = np.where(fg_px, color[c], out[:, :, c])
        out[bg_px] = [1.0, 1.0, 1.0]   # bg scribbles white
    return out


# ── Grid figure ───────────────────────────────────────────────────────────────

def plot_grid(image_np, gt_per_class, snapshots_per_class,
              available_classes, out_path: Path):
    """
    Layout (columns per row):
      col 0   : combined overlay
      col 1…K : per-class overlays

    Rows:
      row 0        : original image (full width) + legend
      row 1        : GT
      row 2…6      : iter 1, 5, 10, 15, 20
    """
    n_cls  = len(available_classes)
    n_cols = 1 + n_cls
    # rows: [original row] + [GT row] + [5 snapshot rows]
    n_rows = 1 + 1 + len(SNAPSHOT_ITERS)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))

    # ── Row 0: wide original image ────────────────────────────────────────────
    ax_orig = fig.add_subplot(n_rows, 1, 1)
    ax_orig.imshow(image_np)
    ax_orig.set_title('Original patch', fontsize=10, fontweight='bold')
    ax_orig.axis('off')

    # Legend anchored inside the original panel
    handles = [
        mpatches.Patch(color=CLASS_COLORS[cls], label=CLASS_SHORT.get(cls, cls))
        for cls in available_classes
    ]
    ax_orig.legend(handles=handles, loc='lower right', fontsize=8,
                   framealpha=0.8, ncol=len(available_classes))

    # ── Rows 1…: GT + snapshots ───────────────────────────────────────────────
    row_labels  = ['GT'] + [f'iter {SNAPSHOT_ITERS[i]+1}' for i in range(len(SNAPSHOT_ITERS))]
    masks_rows  = [gt_per_class] + [
        {cls: snapshots_per_class[cls][snap_i] for cls in available_classes
         if cls in snapshots_per_class}
        for snap_i in range(len(SNAPSHOT_ITERS))
    ]

    for row_i, (row_label, masks_dict) in enumerate(zip(row_labels, masks_rows)):
        base_row = row_i + 1   # subplot row index (1-based after the wide original)

        # Combined column
        ax = fig.add_subplot(n_rows, n_cols, base_row * n_cols + 1)
        ax.imshow(combined_overlay(image_np, masks_dict))
        ax.set_title(f'{row_label} | combined', fontsize=7)
        ax.axis('off')

        # Per-class columns
        for col_i, cls in enumerate(available_classes):
            ax = fig.add_subplot(n_rows, n_cols, base_row * n_cols + 2 + col_i)
            mask = masks_dict.get(cls)
            if mask is not None:
                ax.imshow(apply_mask_overlay(image_np, mask, CLASS_COLORS[cls]))
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

def make_animation(image_np, all_masks_per_class, all_signals_per_class,
                   available_classes, gt_per_class: dict,
                   out_path: Path, fps: int = 1):
    """
    20-frame GIF with 3 panels:
      left   – GT (static, for reference)
      middle – combined prediction at current iteration
      right  – accumulated scribbles at current iteration
    """
    n_iters = len(next(iter(all_masks_per_class.values())))

    fig, (ax_gt, ax_pred, ax_sig) = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05)

    # Pre-render the static GT frame once
    gt_frame = combined_overlay(image_np, gt_per_class)

    def _pred_frame(it):
        return combined_overlay(image_np,
                                {cls: all_masks_per_class[cls][it]
                                 for cls in available_classes
                                 if cls in all_masks_per_class})

    def _sig_frame(it):
        return scribble_overlay(image_np,
                                {cls: all_signals_per_class[cls][it]
                                 for cls in available_classes
                                 if cls in all_signals_per_class})

    ax_gt.imshow(gt_frame)           # static — no update needed
    im_pred = ax_pred.imshow(_pred_frame(0))
    im_sig  = ax_sig.imshow(_sig_frame(0))

    ax_gt.set_title('Ground truth', fontsize=11, fontweight='bold')
    ax_pred.set_title('Prediction',  fontsize=11)
    ax_sig.set_title('Accumulated scribbles', fontsize=11)
    for ax in (ax_gt, ax_pred, ax_sig):
        ax.axis('off')
    ttl = fig.suptitle('Iteration 1 / 20', fontsize=12)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[cls], label=CLASS_SHORT.get(cls, cls))
               for cls in available_classes]
    fig.legend(handles=handles, loc='lower center', ncol=len(available_classes),
               fontsize=9, bbox_to_anchor=(0.5, 0.0), framealpha=0.85)

    def update(frame):
        # GT panel stays constant — no set_data needed
        im_pred.set_data(_pred_frame(frame))
        im_sig.set_data(_sig_frame(frame))
        ttl.set_text(f'Iteration {frame + 1} / {n_iters}')
        return [im_pred, im_sig, ttl]

    anim = FuncAnimation(fig, update, frames=n_iters, interval=1000 // fps, blit=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Animation saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise ProGIS iterative inference')
    parser.add_argument('--checkpoint',  required=True,
                        help='Path to segment_part .pth checkpoint')
    parser.add_argument('--patches_dir', required=True,
                        help='Path to data/patches/')
    parser.add_argument('--splits_json', required=True,
                        help='Path to fold_splits.json')
    parser.add_argument('--fold',        type=int, default=1)
    parser.add_argument('--split',       default='val', choices=['train', 'val'])
    parser.add_argument('--n_samples',   type=int, default=2,
                        help='Number of multi-class patches to visualise')
    parser.add_argument('--min_classes', type=int, default=2,
                        help='Minimum number of tissue classes per patch')
    parser.add_argument('--n_iters',     type=int, default=20,
                        help='Number of correction iterations')
    parser.add_argument('--out_dir',     default='../visualizations',
                        help='Output directory for PNG/GIF files')
    parser.add_argument('--animate',     action='store_true',
                        help='Also save a GIF animation')
    parser.add_argument('--device',      default=None,
                        help='cuda / cpu (auto-detected if omitted)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_segment_part(args.checkpoint, device)

    # Find patches with multiple tissue classes
    fold_splits = load_fold_splits(args.splits_json)
    patches = find_multiclass_patches(
        args.patches_dir, fold_splits, args.fold, args.split,
        min_classes=args.min_classes, n=args.n_samples
    )

    if not patches:
        print(f"No patches found with ≥{args.min_classes} classes. "
              f"Try --min_classes 1")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_i, (fname, available_classes) in enumerate(patches):
        print(f"\n[{sample_i+1}/{len(patches)}] {fname}")
        print(f"  Available classes: {available_classes}")

        # Load shared image (same for all classes)
        image_raw, _, _ = load_patch_data(args.patches_dir, fname, available_classes[0])
        image_np = to_display(image_raw)  # [H,W,3] in [0,1]

        gt_per_class:       dict = {}
        snapshots_per_class: dict = {}
        all_masks_per_class: dict = {}
        all_signals_per_class: dict = {}

        for cls in tqdm(available_classes, desc='  Classes', leave=False):
            _, mask_np, signal_np = load_patch_data(args.patches_dir, fname, cls)

            image_t  = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32)
            mask_t   = torch.tensor(mask_np,   dtype=torch.float32).unsqueeze(0)
            signal_t = torch.tensor(signal_np, dtype=torch.float32)

            snaps, all_masks, all_sigs = run_inference(
                model, image_t, mask_t, signal_t, device, n_iters=args.n_iters
            )

            gt_per_class[cls]        = mask_np
            snapshots_per_class[cls] = snaps          # list of 5 numpy [H,W]
            all_masks_per_class[cls] = all_masks      # list of 20 numpy [H,W]
            all_signals_per_class[cls] = all_sigs     # list of 20 numpy [2,H,W]

        stem = Path(fname).stem
        grid_path = out_dir / f'sample_{sample_i:02d}_{stem}_grid.png'
        plot_grid(image_np, gt_per_class, snapshots_per_class,
                  available_classes, grid_path)

        if args.animate:
            anim_path = out_dir / f'sample_{sample_i:02d}_{stem}_anim.gif'
            make_animation(image_np, all_masks_per_class, all_signals_per_class,
                           available_classes, gt_per_class, anim_path)

    print(f"\nAll done. Results → {out_dir}/")


if __name__ == '__main__':
    main()
