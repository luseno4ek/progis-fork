"""
Нарезка больших изображений BCSS на патчи 512×512 с шагом 256 (как в статье ProGIS)
Sliding window: patch_size=512, stride=256

Обрабатывает все folds и все 5 классов.

Использование:
  # Все фолды, все классы (рекомендуется):
  python create_patches.py --input_dir data/processed --output_dir data/patches

  # Один фолд, один класс:
  python create_patches.py --input_dir data/processed --output_dir data/patches \
      --folds 1 --classes tumor

Структура выхода:
  data/patches/
    fold_1/ ... fold_5/
      train/ val/
        tumor/ stroma/ inflammatory_infiltration/ necrosis/ others/
          image_npy/  mask_npy/  signal_all_line_npy/
"""

import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from tqdm import tqdm


ALL_CLASSES = [
    'tumor',
    'stroma',
    'inflammatory_infiltration',
    'necrosis',
    'others',
]


# ── signal generation ────────────────────────────────────────────────────────

def generate_guiding_signal(binary_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Distance-transform skeleton signal. Returns float32 [H, W]."""
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

    return skeletonize(skel_mask).astype(np.float32)


# ── patch extraction ─────────────────────────────────────────────────────────

def extract_patches(image: np.ndarray, mask: np.ndarray,
                    patch_size: int = 512, stride: int = 256,
                    min_fg_ratio: float = 0.05) -> list:
    """
    Sliding-window patch extraction.

    Keeps a patch if:
      - foreground ratio > min_fg_ratio  (foreground patch)
      - foreground ratio == 0            (pure background — kept for contrastive learning)
    """
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_p  = image[y:y+patch_size, x:x+patch_size]
            mask_p = mask[y:y+patch_size, x:x+patch_size]
            fg_ratio = mask_p.sum() / (patch_size * patch_size)

            if fg_ratio > min_fg_ratio or fg_ratio == 0.0:
                patches.append((img_p, mask_p, fg_ratio))

    return patches


# ── single class / single split processing ───────────────────────────────────

def process_split(input_split_cls: Path, output_split_cls: Path,
                  patch_size: int, stride: int, min_fg_ratio: float,
                  split_label: str) -> int:
    """
    Process all WSI in one (fold / split / class) directory.
    Returns total number of saved patches.
    """
    images_dir = input_split_cls / 'image_npy'
    masks_dir  = input_split_cls / 'mask_npy'

    if not images_dir.exists():
        print(f"    ⚠  not found: {images_dir}")
        return 0

    out_img = output_split_cls / 'image_npy'
    out_msk = output_split_cls / 'mask_npy'
    out_sig = output_split_cls / 'signal_all_line_npy'
    for d in (out_img, out_msk, out_sig):
        d.mkdir(parents=True, exist_ok=True)

    image_files  = sorted(images_dir.glob('*.npy'))
    total_patches = 0

    for img_file in tqdm(image_files, desc=f"    {split_label}", leave=False):
        image = np.load(img_file)           # [H, W, 3]
        mask  = np.load(masks_dir / img_file.name)  # [H, W]

        patches = extract_patches(image, mask, patch_size, stride, min_fg_ratio)
        base    = img_file.stem

        for p_idx, (img_p, mask_p, _) in enumerate(patches):
            seed      = total_patches
            fg_signal = generate_guiding_signal(mask_p,         seed=seed)
            bg_signal = generate_guiding_signal(1.0 - mask_p,   seed=seed + 1)
            signal    = np.stack([fg_signal, bg_signal], axis=0)  # [2, H, W]

            patch_name = f"{base}_patch{p_idx:04d}.npy"
            np.save(out_img / patch_name, img_p)
            np.save(out_msk / patch_name, mask_p)
            np.save(out_sig / patch_name, signal)

            total_patches += 1

    return total_patches


# ── main pipeline ─────────────────────────────────────────────────────────────

def create_patches(input_dir: str, output_dir: str,
                   patch_size: int, stride: int, min_fg_ratio: float,
                   folds: list, classes: list):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"CREATE PATCHES  {patch_size}×{patch_size}, stride={stride}")
    print(f"{'='*60}")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Folds : {folds}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Min foreground ratio: {min_fg_ratio}")
    print(f"{'='*60}\n")

    grand_total = 0

    for fold_num in folds:
        print(f"\n── fold_{fold_num} ──")

        for split in ('train', 'val'):
            fold_total = 0
            print(f"  {split}:")

            for cls in classes:
                in_dir  = input_dir  / f'fold_{fold_num}' / split / cls
                out_dir = output_dir / f'fold_{fold_num}' / split / cls

                n = process_split(in_dir, out_dir, patch_size, stride, min_fg_ratio,
                                  split_label=f"fold_{fold_num}/{split}/{cls}")

                print(f"    {cls:30s}: {n} patches")
                fold_total += n

            print(f"  {split} total: {fold_total} patches")
            grand_total += fold_total

    print(f"\n{'='*60}")
    print(f"✓ Done! Total patches saved: {grand_total}")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  python generate_superpixels.py --input_dir {output_dir} "
          f"--output_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create 512×512 patches from BCSS NPY files'
    )
    parser.add_argument('--input_dir',   default='data/processed',
                        help='Directory from convert_bcss_to_npy.py')
    parser.add_argument('--output_dir',  default='data/patches',
                        help='Output directory for patches')
    parser.add_argument('--patch_size',  type=int,   default=512)
    parser.add_argument('--stride',      type=int,   default=256)
    parser.add_argument('--min_foreground', type=float, default=0.05,
                        help='Min foreground ratio to keep patch (default 0.05 = 5%%)')
    parser.add_argument('--folds',  type=int, nargs='+', default=list(range(1, 6)),
                        help='Which folds to process (default: 1 2 3 4 5)')
    parser.add_argument('--classes', nargs='+', default=ALL_CLASSES,
                        choices=ALL_CLASSES,
                        help='Which classes to process (default: all 5)')
    args = parser.parse_args()

    create_patches(
        args.input_dir,
        args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fg_ratio=args.min_foreground,
        folds=args.folds,
        classes=args.classes,
    )


if __name__ == '__main__':
    main()
