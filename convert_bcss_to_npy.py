"""
Конвертация BCSS данных из PNG в NPY формат для ProGIS

Создаёт 5-fold CV разбиение:
  - 125 WSI, отсортированных по имени
  - fold_k: val = WSI[k*25:(k+1)*25], train = остальные 100 WSI
  - Для каждого WSI генерируются маски и сигналы для всех 5 классов

Использование:
  python convert_bcss_to_npy.py \
      --images_dir data/raw/images \
      --masks_dir  data/raw/masks \
      --output_dir data/processed

Структура выходных данных:
  data/processed/
    fold_1/ ... fold_5/
      train/ val/
        tumor/ stroma/ inflammatory_infiltration/ necrosis/ others/
          image_npy/  mask_npy/  signal_all_line_npy/
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


# ── BCSS pixel label → ProGIS category ──────────────────────────────────────
#
# BCSS raw labels (pixel values in mask PNG):
#   0  outside_roi            → ignored (not tissue)
#   1  tumor
#   2  stroma
#   3  lymphocytic_infiltrate → inflammatory_infiltration
#   4  necrosis_or_debris     → necrosis
#   5-21 everything else      → others
#
# ProGIS paper: "5 categories: tumor, stroma, inflammatory infiltration,
#                necrosis, and others"  (following [28])

CLASSES = {
    'tumor':                      [1],
    'stroma':                     [2],
    'inflammatory_infiltration':  [3],
    'necrosis':                   [4],
    'others':                     list(range(5, 22)),   # 5..21 inclusive
}

N_FOLDS  = 5
VAL_SIZE = 25   # WSI per fold used as validation


# ── signal generation ────────────────────────────────────────────────────────

def generate_guiding_signal(binary_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Distance-transform skeleton signal from a binary mask [H, W].
    Returns float32 array [H, W].
    """
    np.random.seed(seed)
    binary_mask = (binary_mask > 0.5).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)

    dist = distance_transform_edt(binary_mask)
    nonzero = dist[dist > 0]
    if len(nonzero) == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)

    mean_d, std_d = nonzero.mean(), nonzero.std()
    thresh = max(0.0, float(np.random.uniform(mean_d - std_d, mean_d + std_d)))

    skel_mask = (dist > thresh).astype(np.uint8)
    if skel_mask.sum() == 0:
        skel_mask = binary_mask

    skeleton = skeletonize(skel_mask)
    return skeleton.astype(np.float32)


# ── mask helpers ─────────────────────────────────────────────────────────────

def make_binary_mask(multiclass_mask: np.ndarray, label_ids: list) -> np.ndarray:
    """Return float32 binary mask: 1 where pixel ∈ label_ids, else 0."""
    out = np.zeros(multiclass_mask.shape, dtype=np.float32)
    for lid in label_ids:
        out[multiclass_mask == lid] = 1.0
    return out


# ── 5-fold split ─────────────────────────────────────────────────────────────

def make_5fold_splits(all_files: list, n_folds: int = 5) -> list:
    """
    Split sorted file list into n_folds groups.
    Returns list of dicts: [{'train': [...], 'val': [...]}, ...]
    """
    files = sorted(all_files, key=lambda p: p.name)
    n = len(files)
    fold_size = n // n_folds

    splits = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end   = val_start + fold_size if k < n_folds - 1 else n  # last fold takes remainder
        val_files   = files[val_start:val_end]
        train_files = files[:val_start] + files[val_end:]
        splits.append({'train': train_files, 'val': val_files})

    return splits


# ── single WSI processing ────────────────────────────────────────────────────

def process_wsi(img_file: Path, masks_dir: Path, out_base: Path,
                wsi_idx: int, resize: bool) -> dict:
    """
    Process one WSI for all 5 classes.
    Returns dict {class_name: True/False} indicating which classes were saved.
    """
    # Load image
    image = np.array(Image.open(img_file).convert('RGB'))

    # Load mask
    mask_file = masks_dir / img_file.name
    if not mask_file.exists():
        print(f"    ⚠  mask not found for {img_file.name}, skipping")
        return {}

    mc_mask = np.array(Image.open(mask_file))

    # Optional: resize to multiple of 16
    if resize:
        image, mc_mask = resize_to_multiple16(image, mc_mask)

    filename = f"sample_{wsi_idx:04d}.npy"
    saved = {}

    for cls_name, label_ids in CLASSES.items():
        binary_mask = make_binary_mask(mc_mask, label_ids)

        if binary_mask.sum() == 0:
            saved[cls_name] = False
            continue  # class absent in this WSI

        # Signals
        fg_signal = generate_guiding_signal(binary_mask,      seed=wsi_idx * 10)
        bg_mask   = 1.0 - (mc_mask > 0).astype(np.float32)   # all non-tissue as background
        bg_signal = generate_guiding_signal(bg_mask,          seed=wsi_idx * 10 + 1)
        signal    = np.stack([fg_signal, bg_signal], axis=0)  # [2, H, W]

        # Output dirs
        cls_dir = out_base / cls_name
        (cls_dir / 'image_npy').mkdir(parents=True, exist_ok=True)
        (cls_dir / 'mask_npy').mkdir(parents=True, exist_ok=True)
        (cls_dir / 'signal_all_line_npy').mkdir(parents=True, exist_ok=True)

        np.save(cls_dir / 'image_npy'           / filename, image)
        np.save(cls_dir / 'mask_npy'            / filename, binary_mask)
        np.save(cls_dir / 'signal_all_line_npy' / filename, signal)

        saved[cls_name] = True

    return saved


def resize_to_multiple16(image: np.ndarray, mask: np.ndarray):
    h, w = image.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if h == new_h and w == new_w:
        return image, mask
    img_pil  = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    return np.array(img_pil), np.array(mask_pil)


# ── main pipeline ─────────────────────────────────────────────────────────────

def convert_bcss(images_dir: str, masks_dir: str, output_dir: str,
                 n_folds: int = 5, resize: bool = True):
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)
    output_dir = Path(output_dir)

    image_files = sorted(images_dir.glob('*.png'))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {images_dir}")

    print(f"\n{'='*60}")
    print(f"BCSS → NPY  |  {n_folds}-fold CV")
    print(f"{'='*60}")
    print(f"Images : {images_dir}  ({len(image_files)} WSI)")
    print(f"Masks  : {masks_dir}")
    print(f"Output : {output_dir}")
    print(f"Classes: {', '.join(CLASSES.keys())}")
    print(f"{'='*60}\n")

    splits = make_5fold_splits(image_files, n_folds)

    for fold_idx, split in enumerate(splits):
        fold_num = fold_idx + 1
        print(f"\n── fold_{fold_num}  "
              f"(train={len(split['train'])} WSI, val={len(split['val'])} WSI) ──")

        for split_name in ('train', 'val'):
            wsi_list = split[split_name]
            out_base  = output_dir / f'fold_{fold_num}' / split_name

            counts = {cls: 0 for cls in CLASSES}
            for wsi_idx, img_file in enumerate(wsi_list):
                saved = process_wsi(img_file, masks_dir, out_base, wsi_idx, resize)
                for cls, ok in saved.items():
                    if ok:
                        counts[cls] += 1

            print(f"  {split_name:5s}: " +
                  "  ".join(f"{cls}={counts[cls]}" for cls in CLASSES))

    print(f"\n{'='*60}")
    print(f"✓ Done! Output: {output_dir}")
    print(f"{'='*60}")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    print(f"  ├── fold_1/ ... fold_{n_folds}/")
    print(f"  │     ├── train/")
    print(f"  │     │     ├── tumor/{{image_npy, mask_npy, signal_all_line_npy}}")
    print(f"  │     │     ├── stroma/")
    print(f"  │     │     ├── inflammatory_infiltration/")
    print(f"  │     │     ├── necrosis/")
    print(f"  │     │     └── others/")
    print(f"  │     └── val/  (same structure)")
    print(f"\nNext step:")
    print(f"  python create_patches.py --input_dir {output_dir} "
          f"--output_dir data/patches")


def main():
    parser = argparse.ArgumentParser(
        description='Convert BCSS PNG to NPY with 5-fold CV split'
    )
    parser.add_argument('--images_dir', default='data/raw/images',
                        help='Directory with WSI images (.png)')
    parser.add_argument('--masks_dir',  default='data/raw/masks',
                        help='Directory with annotation masks (.png)')
    parser.add_argument('--output_dir', default='data/processed',
                        help='Output directory for NPY files')
    parser.add_argument('--n_folds',    type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--no_resize',  action='store_true',
                        help='Skip resize to multiple of 16')
    args = parser.parse_args()

    convert_bcss(
        args.images_dir,
        args.masks_dir,
        args.output_dir,
        n_folds=args.n_folds,
        resize=not args.no_resize,
    )


if __name__ == '__main__':
    main()
