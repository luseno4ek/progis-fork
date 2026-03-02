"""
Visualization of ProGIS iterative segmentation on BCSS val patches.

Finds patches that appear in multiple tissue classes (same image, multiple GT masks),
runs 20-iteration inference for each class, and produces:
  1. Grid PNG:  original | GT_combined | GT_per_class | pred@iter1 | pred@iter5 | ...
  2. (opt) GIF: animated mask+scribble overlay for all 20 iterations

Usage (from project root):
    python models/visualize_inference.py \
        --checkpoint /srv/data1/data_repository/BCSS/patches/fold_1/ROI_ckpt/BCSS_effi-Unet_roi_best_dice0.9772_epoch19.pth \
        --patches_dir /srv/data1/data_repository/BCSS/patches \
        --splits_json /srv/data1/data_repository/BCSS/patches/fold_splits.json \
        --fold 1 --n_samples 2 --animate
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

from skimage.measure import label as label_1
from skimage.measure import regionprops
import cv2
import numpy as np
from skimage.morphology import skeletonize
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
def _largest_cc_edt_cpu(binary_mask_np):
    """Find largest connected component and compute exact EDT using cv2 (fast C++).

    Replaces the old GPU-only EDT-ridge skeleton which had two bugs:
      1. Processed ALL error pixels instead of only the largest connected component.
      2. EDT via iterative erosion saturated at max_steps=80, causing flat plateaux
         that made the 'ridge' degenerate to the entire error region.

    Args:
        binary_mask_np: uint8 numpy array [H, W], values in {0, 1}
    Returns:
        largest_cc  : uint8 [H, W]  — mask of the largest CC
        edt         : float32 [H, W] — exact Euclidean distance transform
        centroid    : (cy, cx) int tuple, or None if no foreground
        area        : int, pixel count of the largest CC
    """
    H, W = binary_mask_np.shape
    empty = (np.zeros((H, W), dtype=np.uint8),
             np.zeros((H, W), dtype=np.float32), None, 0)

    if binary_mask_np.sum() == 0:
        return empty

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask_np, connectivity=4)

    if n_labels <= 1:          # only background label
        return empty

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_cc  = (labels == largest_idx).astype(np.uint8)
    area        = int(stats[largest_idx, cv2.CC_STAT_AREA])
    cy = int(round(centroids[largest_idx][1]))   # row
    cx = int(round(centroids[largest_idx][0]))   # col

    edt = cv2.distanceTransform(largest_cc, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return largest_cc, edt, (cy, cx), area


def processMasks_gpu(pred_mask_all, GT_mask_all, max_edt_steps=80):
    """
    Hybrid CPU/GPU replacement for processMasks + generateGuidingSignal.

    Algorithm (matches the original CPU pipeline):
      1. cv2.connectedComponentsWithStats — find the largest error CC per sample.
      2. cv2.distanceTransform — exact Euclidean EDT (no saturation).
      3. GPU: random threshold in [mean-std, mean+std] → core region.
      4. GPU: local-maxima of EDT within core → ridge ≈ skeletonize output.

    Fixes vs old GPU-only version:
      - Uses only the LARGEST connected component (not all error pixels).
      - Exact EDT without max_steps saturation for large regions.

    Args:
        pred_mask_all:  [B, 1, H, W]
        GT_mask_all:    [B, 1, H, W]
        max_edt_steps:  kept for API compatibility, no longer used
    Returns:
        [B, 2, H, W]  float32,  ch0 = fg guidance,  ch1 = bg guidance
    """
    device = pred_mask_all.device
    B, _, H, W = pred_mask_all.shape

    pred_np = (pred_mask_all > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
    gt_np   = GT_mask_all.squeeze(1).cpu().numpy().astype(np.uint8)

    output = torch.zeros(B, 2, H, W, device=device)

    for b in range(B):
        fg_mask = ((gt_np[b] == 1) & (pred_np[b] == 0)).astype(np.uint8)
        bg_mask = ((gt_np[b] == 0) & (pred_np[b] == 1)).astype(np.uint8)

        fg_cc, fg_edt, _, fg_area = _largest_cc_edt_cpu(fg_mask)
        bg_cc, bg_edt, _, bg_area = _largest_cc_edt_cpu(bg_mask)

        if fg_area >= bg_area:
            cc, edt_np, channel = fg_cc, fg_edt, 0
        else:
            cc, edt_np, channel = bg_cc, bg_edt, 1

        if cc.sum() == 0:
            continue

        # ── GPU: threshold + ridge (mirrors generateGuidingSignal) ────────────
        edt_t  = torch.tensor(edt_np, dtype=torch.float32, device=device)
        mask_t = torch.tensor(cc,     dtype=torch.float32, device=device)

        vals  = edt_t[mask_t > 0]
        mu    = vals.mean()
        sigma = vals.std()
        rand  = torch.rand(1, device=device).item()
        thresh = float((mu - sigma + rand * 2 * sigma).clamp(min=0))

        core = (edt_t > thresh) & (mask_t > 0)
        if core.sum() == 0:
            core = (edt_t > thresh / 2) & (mask_t > 0)
        if core.sum() == 0:
            core = mask_t > 0

        # Local-maxima ridge ≈ medial axis (approximates skeletonize)
        core_edt  = edt_t * core.float()
        local_max = F.max_pool2d(
            core_edt.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0).squeeze(0)
        ridge = (core_edt >= local_max - 0.5) & core
        if ridge.sum() == 0:
            ridge = core

        output[b, channel] = ridge.float()

    return output


def generateGuidingSignal(binaryMask):
    # binaryMask = binaryMask.squeeze(0)  # Remove the batch dimension if it's (1, H, W)
    binaryMask = binaryMask.to(torch.uint8)
    
    if binaryMask.sum() > 1:
        # Compute distance transform (move to CPU for NumPy operations)
        distance_map = distance_transform_edt(binaryMask.cpu().numpy())
        distance_map = torch.tensor(distance_map, dtype=torch.float32, device=binaryMask.device)
        
        # Calculate mean and std (ensure they are on CPU before NumPy operations)
        tempMean = distance_map.mean().cpu().numpy()
        tempStd = distance_map.std().cpu().numpy()
        
        # Random threshold based on mean and std
        tempThresh = np.random.uniform(tempMean - tempStd, tempMean + tempStd)
        tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        if tempThresh < 0:
            tempThresh = np.random.uniform(tempMean / 2, tempMean + tempStd / 2)
            tempThresh = torch.tensor(tempThresh, device=binaryMask.device)
        
        # Apply threshold to get new mask
        newMask = distance_map > tempThresh
        if newMask.sum() == 0:
            newMask = distance_map > (tempThresh / 2)
        
        if newMask.sum() == 0:
            newMask = binaryMask

        # Skeletonize (use skimage and convert back to tensor)
        skel = skeletonize(newMask.cpu().numpy())
        skel = torch.tensor(skel, dtype=torch.float32, device=binaryMask.device)
    else:
        skel = torch.zeros_like(binaryMask, dtype=torch.float32, device=binaryMask.device)

    return skel

def processMasks(pred_mask_all, GT_mask_all):
    """
    批量处理GT_mask和pred_mask，计算每个样本的前景和背景骨架信号。
    参数:
        pred_masks: 预测的mask，形状为 [batch_size, 1, H, W]。
        GT_masks: 真值mask，形状为 [batch_size, 1, H, W]。
    返回:
        输出张量，形状为 [batch_size, 2, H, W]。
        每个样本的第0通道为前景区域骨架信号，第1通道为背景区域骨架信号。
    """
    pred_mask_all = (pred_mask_all > 0.5).float()

    batch_size, _, H, W = pred_mask_all.shape

    # 初始化输出张量
    output = torch.zeros(batch_size, 2, H, W, device=pred_mask_all.device, dtype=torch.float32)
    # centers = []  # 存储每个样本的最大错误连通域中心坐标

    for i in range(batch_size):
        # 取出当前样本的预测和真值mask
        pred_mask = pred_mask_all[i].squeeze(0)  # [H, W]
        GT_mask = GT_mask_all[i].squeeze(0)      # [H, W]

        # 计算前景区域 (GT_mask为1且pred_mask为0)
        fg = (GT_mask == 1) & (pred_mask == 0)
        fg = fg.to(torch.float32)  # [H, W]

        # 计算背景区域 (GT_mask为0且pred_mask为1)
        bg = (GT_mask == 0) & (pred_mask == 1)
        bg = bg.to(torch.float32)  # [H, W]
        
        # 找出前景的最大连通域
        if fg.sum() > 0:
            labeled_fg = label_1(fg.cpu().numpy(), connectivity=1)
            regions_fg = regionprops(labeled_fg)
            if regions_fg:
                largest_region_fg = max(regions_fg, key=lambda r: r.area)
                fg_largest = (labeled_fg == largest_region_fg.label)
                fg_largest = torch.from_numpy(fg_largest).to(fg.device, dtype=torch.float32)
                # fg_center = largest_region_fg.centroid
                # fg_center = (round(fg_center[0]), round(fg_center[1]))  # 四舍五入
            else:
                fg_largest = torch.zeros_like(fg)
                fg_center = None
        else:
            fg_largest = torch.zeros_like(fg)
            fg_center = None

        # 找出背景的最大连通域
        if bg.sum() > 0:
            labeled_bg = label_1(bg.cpu().numpy(), connectivity=1)
            regions_bg = regionprops(labeled_bg)
            if regions_bg:
                largest_region_bg = max(regions_bg, key=lambda r: r.area)
                bg_largest = (labeled_bg == largest_region_bg.label)
                bg_largest = torch.from_numpy(bg_largest).to(bg.device, dtype=torch.float32)
                # bg_center = largest_region_bg.centroid  # 获取背景最大连通域的中心坐标 (y, x)
                # bg_center = (round(bg_center[0]), round(bg_center[1]))  # 四舍五入
            else:
                bg_largest = torch.zeros_like(bg)
                bg_center = None
        else:
            bg_largest = torch.zeros_like(bg)
            bg_center = None
        
        # 比较前景和背景的最大连通域面积
        fg_area = fg_largest.sum().item()
        bg_area = bg_largest.sum().item()

        if fg_area >= bg_area:
            largest_connected = fg_largest
            # 计算前景区域的骨架信号
            fg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
            output[i, 0] = fg_skeleton  # 前景骨架信号
            # centers.append(fg_center)  # 保存中心坐标
        else:
            largest_connected = bg_largest
            # 计算背景区域的骨架信号
            bg_skeleton = generateGuidingSignal(largest_connected) if largest_connected.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
            output[i, 1] = bg_skeleton  # 背景骨架信号
            # centers.append(bg_center)  # 保存中心坐标
            
        
        # # 计算前景区域的骨架信号
        # fg_skeleton = generateGuidingSignal(fg) if fg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果fg为全0，创建一个全零张量
        
        # # 计算背景区域的骨架信号
        # bg_skeleton = generateGuidingSignal(bg) if bg.sum() > 0 else torch.zeros_like(pred_mask, dtype=torch.float32, device=pred_mask.device)  # 如果bg为全0，创建一个全零张量

        # # 合并前景和背景骨架信号到输出
        # output[i, 0] = fg_skeleton  # 前景骨架信号
        # output[i, 1] = bg_skeleton  # 背景骨架信号
         
    return output


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
        sig   = processMasks(pred, gt)
        u_sig = torch.bitwise_or(sig.to(torch.uint8), u_sig.to(torch.uint8)).float()
        prev  = (pred > 0.5).float()

    # Pick 5 evenly-spaced snapshot indices regardless of n_iters
    n = len(all_masks)
    snap_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    snapshots = [all_masks[i] for i in snap_indices]
    return snapshots, snap_indices, all_masks, all_signals


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
              available_classes, snap_indices: list, out_path: Path):
    """
    Layout (columns per row):
      col 0   : combined overlay
      col 1…K : per-class overlays

    Rows:
      row 0      : original image (full width) + legend
      row 1      : GT
      row 2…N+1  : one row per snapshot (iter numbers from snap_indices)
    """
    n_cls  = len(available_classes)
    n_cols = 1 + n_cls
    n_snaps = len(snap_indices)
    # rows: [original row] + [GT row] + [snapshot rows]
    n_rows = 1 + 1 + n_snaps

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
    row_labels = ['GT'] + [f'iter {idx + 1}' for idx in snap_indices]
    masks_rows = [gt_per_class] + [
        {cls: snapshots_per_class[cls][snap_i] for cls in available_classes
         if cls in snapshots_per_class}
        for snap_i in range(n_snaps)
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
                   out_path: Path, fps: int = 2):
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

    # blit=False required for PillowWriter: blit=True skips frame 0 when saving to GIF
    anim = FuncAnimation(fig, update, frames=n_iters, interval=1000 // fps, blit=False)
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

            snaps, snap_indices, all_masks, all_sigs = run_inference(
                model, image_t, mask_t, signal_t, device, n_iters=args.n_iters
            )

            gt_per_class[cls]          = mask_np
            snapshots_per_class[cls]   = snaps       # list of N numpy [H,W]
            all_masks_per_class[cls]   = all_masks   # list of n_iters numpy [H,W]
            all_signals_per_class[cls] = all_sigs    # list of n_iters numpy [2,H,W]

        stem = Path(fname).stem
        grid_path = out_dir / f'CPU_sample_{sample_i:02d}_{stem}_grid.png'
        plot_grid(image_np, gt_per_class, snapshots_per_class,
                  available_classes, snap_indices, grid_path)

        if args.animate:
            anim_path = out_dir / f'CPU_sample_{sample_i:02d}_{stem}_anim.gif'
            make_animation(image_np, all_masks_per_class, all_signals_per_class,
                           available_classes, gt_per_class, anim_path)

    print(f"\nAll done. Results → {out_dir}/")


if __name__ == '__main__':
    main()
