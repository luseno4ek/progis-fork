"""
Конвертация BCSS данных из PNG в NPY формат для ProGIS.

Ключевые принципы:
  - Каждое изображение WSI сохраняется ОДИН раз (all/image_npy/)
  - Маски и сигналы сохраняются отдельно для каждого класса
  - Разбиение на фолды хранится в fold_splits.json (а не дублируется на диске)
  - Нет папок fold_1 ... fold_5 → нет 5x дублирования данных

Выходная структура:
  data/processed/
    fold_splits.json          ← {fold_1: {train: [...], val: [...]}, ...}
    all/
      image_npy/              ← RGB изображения [H, W, 3], float32
    tumor/
      mask_npy/               ← бинарная маска [H, W], float32
      signal_all_line_npy/    ← guiding signals [2, H, W], float32
    stroma/
      mask_npy/
      signal_all_line_npy/
    inflammatory_infiltration/
      ...
    necrosis/
      ...
    others/
      ...

Использование:
  python convert_bcss_to_npy.py \
      --images_dir data/raw/images \
      --masks_dir  data/raw/masks \
      --output_dir data/processed
"""

import json
import numpy as np
import cv2
from PIL import Image, ImageFile
from pathlib import Path
import argparse
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Allow PIL to load truncated/incomplete PNG files
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Downsampling ──────────────────────────────────────────────────────────────

def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample an RGB image by `factor` using LANCZOS (anti-aliased).
    Preferred over bilinear for integer downsampling ratios ≥ 2.
    """
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    pil = pil.resize((w // factor, h // factor), Image.LANCZOS)
    return np.array(pil)


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a label mask by `factor` using NEAREST (preserves label IDs).
    """
    h, w = mask.shape
    pil = Image.fromarray(mask.astype(np.uint8))
    pil = pil.resize((w // factor, h // factor), Image.NEAREST)
    return np.array(pil)


# ── Reinhard stain normalisation ─────────────────────────────────────────────
#
# Reference: Reinhard et al. "Color transfer between images", IEEE CGA 2001.
# Standard in computational pathology for H&E normalisation.
#
# Workflow:
#   1. Compute LAB statistics of a reference image once.
#   2. For each WSI: convert to LAB, rescale mean/std to match reference,
#      convert back to RGB.
#
# The reference image should be a representative H&E slide from the dataset.
# Pass --reference_image to convert_bcss.py; if omitted, normalisation is
# skipped (images saved as-is).

def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """RGB uint8 → CIE LAB float32.

    Uses OpenCV (COLOR_RGB2LAB) which encodes a/b with neutral=128,
    so all three channels are valid unsigned uint8 values and mean/std
    statistics are correct for Reinhard normalization.
    (PIL's LAB uses neutral=0 for a/b, causing uint8 wrap-around.)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE LAB float32 → RGB uint8 (clipped to [0, 255])."""
    lab_clipped = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_clipped, cv2.COLOR_LAB2RGB)


def compute_lab_stats(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel (mean, std) in LAB space for an RGB uint8 image."""
    lab = _rgb_to_lab(image).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0)


def reinhard_normalise(
    image:    np.ndarray,
    ref_mean: np.ndarray,
    ref_std:  np.ndarray,
) -> np.ndarray:
    """
    Apply Reinhard colour normalisation to an RGB uint8 image.

    Args:
        image:    [H, W, 3] uint8 source image.
        ref_mean: [3] float32 target LAB channel means.
        ref_std:  [3] float32 target LAB channel stds.

    Returns:
        [H, W, 3] uint8 normalised image.
    """
    lab = _rgb_to_lab(image)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std  = lab.reshape(-1, 3).std(axis=0).clip(min=1e-6)

    lab_norm = (lab - src_mean) / src_std * ref_std + ref_mean
    return _lab_to_rgb(lab_norm)


# ── BCSS pixel label → ProGIS category ──────────────────────────────────────
#
# BCSS raw labels:
#   0   outside_roi  → фон, игнорируется
#   1   tumor
#   2   stroma
#   3   lymphocytic_infiltrate  → inflammatory_infiltration
#   4   necrosis_or_debris      → necrosis
#   5-21 всё остальное          → others

CLASSES = {
    'tumor':                      [1],
    'stroma':                     [2],
    'inflammatory_infiltration':  [3],
    'necrosis':                   [4],
    'others':                     list(range(5, 22)),
}

N_FOLDS = 5


# ── signal generation ────────────────────────────────────────────────────────

def generate_guiding_signal(binary_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Distance-transform skeleton signal. Returns float32 [H, W]."""
    np.random.seed(seed)
    bm = (binary_mask > 0.5).astype(np.uint8)
    if bm.sum() == 0:
        return np.zeros_like(bm, dtype=np.float32)

    dist = distance_transform_edt(bm)
    nonzero = dist[dist > 0]
    if len(nonzero) == 0:
        return np.zeros_like(bm, dtype=np.float32)

    mean_d, std_d = nonzero.mean(), nonzero.std()
    thresh = max(0.0, float(np.random.uniform(mean_d - std_d, mean_d + std_d)))
    skel_mask = (dist > thresh).astype(np.uint8)
    if skel_mask.sum() == 0:
        skel_mask = bm

    return skeletonize(skel_mask).astype(np.float32)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_binary_mask(mc_mask: np.ndarray, label_ids: list) -> np.ndarray:
    out = np.zeros(mc_mask.shape, dtype=np.float32)
    for lid in label_ids:
        out[mc_mask == lid] = 1.0
    return out


def resize_to_multiple16(image: np.ndarray, mask: np.ndarray):
    h, w = image.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if h == new_h and w == new_w:
        return image, mask
    img_r  = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    mask_r = np.array(Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST))
    return img_r, mask_r


def make_5fold_splits(stems: list, n_folds: int = 5) -> dict:
    """
    Детерминированное разбиение на фолды.
    stems: список имён WSI, уже отсортированных.
    Возвращает: {'fold_1': {'train': [...], 'val': [...]}, ...}
    """
    n = len(stems)
    fold_size = n // n_folds
    splits = {}
    for k in range(n_folds):
        val_start = k * fold_size
        val_end   = val_start + fold_size if k < n_folds - 1 else n
        val   = stems[val_start:val_end]
        train = stems[:val_start] + stems[val_end:]
        splits[f'fold_{k+1}'] = {'train': train, 'val': val}
    return splits


# ── main pipeline ─────────────────────────────────────────────────────────────

def convert_bcss(images_dir: str, masks_dir: str, output_dir: str,
                 n_folds: int = 5, resize: bool = True,
                 downsample: int = 4,
                 reference_image: str | None = None):
    """
    Args:
        downsample:       Factor to reduce spatial resolution before patching.
                          4 converts 40x (MPP=0.25) → 10x (MPP=1.0), matching
                          the authors' BCSS_x10_reinhard_cut protocol.
                          Set to 1 to skip downsampling.
        reference_image:  Path to a representative RGB PNG used as the Reinhard
                          normalisation target. If None, stain normalisation is
                          skipped (not recommended for multi-scanner datasets).
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)
    output_dir = Path(output_dir)

    image_files = sorted(images_dir.glob('*.png'))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {images_dir}")

    # ── Reinhard reference stats (computed once) ──────────────────────────
    ref_mean = ref_std = None
    if reference_image is not None:
        ref_img = np.array(Image.open(reference_image).convert('RGB'))
        if downsample > 1:
            ref_img = downsample_image(ref_img, downsample)
        ref_mean, ref_std = compute_lab_stats(ref_img)
        print(f"Reinhard reference: {reference_image}")
        print(f"  LAB mean={ref_mean.round(2)}  std={ref_std.round(2)}")

    print(f"\n{'='*60}")
    print(f"BCSS → NPY  |  {n_folds}-fold CV  |  {len(CLASSES)} classes")
    print(f"{'='*60}")
    print(f"Images : {images_dir}  ({len(image_files)} WSI)")
    print(f"Masks  : {masks_dir}")
    print(f"Output : {output_dir}")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Downsample ×{downsample}  |  Reinhard: {'yes' if ref_mean is not None else 'no'}")
    print(f"\nStorage advantage: image saved ONCE per WSI, not {len(CLASSES)}×")
    print(f"{'='*60}\n")

    # Создаём выходные директории
    all_img_dir = output_dir / 'all' / 'image_npy'
    all_img_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in CLASSES:
        (output_dir / cls_name / 'mask_npy').mkdir(parents=True, exist_ok=True)
        (output_dir / cls_name / 'signal_all_line_npy').mkdir(parents=True, exist_ok=True)

    stems = []

    for wsi_idx, img_file in enumerate(image_files):
        stem = f'sample_{wsi_idx:04d}'
        filename = f'{stem}.npy'

        # Пропускаем уже обработанные WSI (resume при перезапуске)
        img_dst = all_img_dir / filename
        if img_dst.exists():
            print(f"  [skip] {stem} already processed")
            stems.append(stem)  # включаем в fold_splits
            continue

        # Загрузка
        try:
            image = np.array(Image.open(img_file).convert('RGB'))
        except Exception as e:
            print(f"  ⚠  failed to load image: {img_file.name} ({e}), skipping")
            continue
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            print(f"  ⚠  mask not found: {img_file.name}, skipping")
            continue
        try:
            mc_mask = np.array(Image.open(mask_file))
        except Exception as e:
            print(f"  ⚠  failed to load mask: {mask_file.name} ({e}), skipping")
            continue

        # Downsample: 40x → 10x (factor=4). Image: LANCZOS. Mask: NEAREST.
        if downsample > 1:
            image   = downsample_image(image,   downsample)
            mc_mask = downsample_mask(mc_mask,  downsample)

        # Reinhard stain normalisation (requires reference_image)
        if ref_mean is not None:
            image = reinhard_normalise(image, ref_mean, ref_std)

        if resize:
            image, mc_mask = resize_to_multiple16(image, mc_mask)

        stems.append(stem)

        # Сохраняем изображение ОДИН раз
        np.save(img_dst, image.astype(np.float32))

        # Сохраняем маски и сигналы для каждого класса
        class_results = {}
        for cls_name, label_ids in CLASSES.items():
            binary_mask = make_binary_mask(mc_mask, label_ids)
            if binary_mask.sum() == 0:
                class_results[cls_name] = '–'
                continue

            fg_signal = generate_guiding_signal(binary_mask, seed=wsi_idx * 10)
            bg_signal = generate_guiding_signal(
                (mc_mask == 0).astype(np.float32), seed=wsi_idx * 10 + 1
            )
            signal = np.stack([fg_signal, bg_signal], axis=0)  # [2, H, W]

            np.save(output_dir / cls_name / 'mask_npy' / filename, binary_mask)
            np.save(output_dir / cls_name / 'signal_all_line_npy' / filename, signal)
            class_results[cls_name] = '✓'

        status = '  '.join(f"{c}:{class_results[c]}" for c in CLASSES)
        print(f"  [{wsi_idx+1:3d}/{len(image_files)}] {img_file.name} → {stem}  |  {status}")

    # Генерируем fold_splits.json
    fold_splits = make_5fold_splits(stems, n_folds)
    splits_path = output_dir / 'fold_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(fold_splits, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"  WSI processed : {len(image_files)}")
    print(f"  fold_splits   : {splits_path}")
    print(f"{'='*60}")

    # Статистика фолдов
    print(f"\nFold split summary:")
    for fold_name, split in fold_splits.items():
        print(f"  {fold_name}: train={len(split['train'])}  val={len(split['val'])}")

    print(f"\nNext step:")
    print(f"  python create_patches.py --input_dir {output_dir} --output_dir data/patches")


def main():
    parser = argparse.ArgumentParser(
        description='Convert BCSS PNG to NPY. Images stored once, fold splits in JSON.'
    )
    parser.add_argument('--images_dir',       default='data/raw/images')
    parser.add_argument('--masks_dir',        default='data/raw/masks')
    parser.add_argument('--output_dir',       default='data/processed')
    parser.add_argument('--n_folds',          type=int,   default=5)
    parser.add_argument('--no_resize',        action='store_true')
    parser.add_argument('--downsample',       type=int,   default=4,
                        help='Spatial downsampling factor (default 4: 40x→10x). '
                             'Set to 1 to skip.')
    parser.add_argument('--reference_image',  default=None,
                        help='Path to a representative PNG used as the Reinhard '
                             'stain normalisation target. Omit to skip normalisation.')
    args = parser.parse_args()

    convert_bcss(
        args.images_dir, args.masks_dir, args.output_dir,
        n_folds=args.n_folds, resize=not args.no_resize,
        downsample=args.downsample,
        reference_image=args.reference_image,
    )


if __name__ == '__main__':
    main()
