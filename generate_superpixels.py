"""
Generate SLIC superpixels for ProGIS Feature Extractor training.
Also creates symlinks for Stage 1 (Contrast_learning) and Stage 2 (ROI_data/all_class).

Обрабатывает все folds и все 5 классов.

Использование:
  # Все фолды (рекомендуется):
  python generate_superpixels.py --input_dir data/patches --output_dir data/patches

  # Один фолд:
  python generate_superpixels.py --input_dir data/patches --output_dir data/patches \
      --folds 1

Структура выхода:
  data/patches/
    fold_k/
      train/ val/
        Contrast_learning/
          image_npy            → symlink → train/tumor/image_npy  (Stage 1)
          mask_npy             → symlink → train/tumor/mask_npy
          image_SLIC_500/      ← superpixels (generated here)
        ROI_data/all_class/
          image_npy            → symlink → train/tumor/image_npy  (Stage 2)
          mask_npy             → symlink → train/tumor/mask_npy
          signal_maxconnect_line_npy → symlink → train/tumor/signal_all_line_npy

Note: symlinks for Contrast_learning and ROI_data point to 'tumor' by default,
which is the primary class used in the paper for pre-training.
"""

import numpy as np
from pathlib import Path
from skimage.segmentation import slic
from tqdm import tqdm
import argparse
import os


ALL_CLASSES = [
    'tumor',
    'stroma',
    'inflammatory_infiltration',
    'necrosis',
    'others',
]


# ── SLIC generation ──────────────────────────────────────────────────────────

def compute_slic(image: np.ndarray, n_segments: int = 500,
                 compactness: float = 10.0) -> np.ndarray:
    """
    Compute SLIC superpixels for a [H, W, 3] image.
    Returns int32 label map [H, W].
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
    )
    return segments.astype(np.int32)


# ── symlink helpers ──────────────────────────────────────────────────────────

def make_symlink(src: Path, dst: Path):
    """Create symlink dst → src (absolute). Skips if already exists."""
    src_abs = src.resolve()
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src_abs, dst)


def create_symlinks(fold_dir: Path, split: str, primary_class: str = 'tumor',
                    n_segments: int = 500):
    """
    Create symlinks for Contrast_learning (Stage 1) and ROI_data/all_class (Stage 2).
    """
    cls_dir = fold_dir / split / primary_class

    # ── Stage 1: Contrast_learning ──────────────────────────────────────────
    cl_dir = fold_dir / split / 'Contrast_learning'
    cl_dir.mkdir(parents=True, exist_ok=True)

    make_symlink(cls_dir / 'image_npy', cl_dir / 'image_npy')
    make_symlink(cls_dir / 'mask_npy',  cl_dir / 'mask_npy')
    # image_SLIC_500 is created by generate_superpixels, no symlink needed

    # ── Stage 2: ROI_data/all_class ─────────────────────────────────────────
    roi_dir = fold_dir / split / 'ROI_data' / 'all_class'
    roi_dir.mkdir(parents=True, exist_ok=True)

    make_symlink(cls_dir / 'image_npy',              roi_dir / 'image_npy')
    make_symlink(cls_dir / 'mask_npy',               roi_dir / 'mask_npy')
    make_symlink(cls_dir / 'signal_all_line_npy',
                 roi_dir / 'signal_maxconnect_line_npy')


# ── superpixel generation for one split ─────────────────────────────────────

def generate_for_split(patch_dir: Path, output_dir: Path,
                       split: str, fold_num: int,
                       n_segments: int, compactness: float,
                       primary_class: str):
    """
    Generate SLIC superpixels for all patches in fold_k / split / primary_class.
    Writes to output_dir / fold_k / split / Contrast_learning / image_SLIC_{n}.
    """
    images_dir = patch_dir / f'fold_{fold_num}' / split / primary_class / 'image_npy'
    if not images_dir.exists():
        print(f"    ⚠  not found: {images_dir}")
        return 0

    out_slic = (output_dir / f'fold_{fold_num}' / split
                / 'Contrast_learning' / f'image_SLIC_{n_segments}')
    out_slic.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob('*.npy'))
    for img_file in tqdm(image_files, desc=f"    SLIC fold_{fold_num}/{split}", leave=False):
        dst = out_slic / img_file.name
        if dst.exists():
            continue
        image    = np.load(img_file)         # [H, W, 3]
        segments = compute_slic(image, n_segments, compactness)
        np.save(dst, segments)

    return len(image_files)


# ── main pipeline ─────────────────────────────────────────────────────────────

def generate_superpixels(input_dir: str, output_dir: str,
                         n_segments: int, compactness: float,
                         folds: list, primary_class: str = 'tumor'):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"GENERATE SLIC SUPERPIXELS  (n_segments={n_segments})")
    print(f"{'='*60}")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Folds : {folds}")
    print(f"Primary class for symlinks: {primary_class}")
    print(f"{'='*60}\n")

    for fold_num in folds:
        print(f"\n── fold_{fold_num} ──")
        fold_dir = output_dir / f'fold_{fold_num}'

        for split in ('train', 'val'):
            n = generate_for_split(
                input_dir, output_dir,
                split=split, fold_num=fold_num,
                n_segments=n_segments, compactness=compactness,
                primary_class=primary_class,
            )
            print(f"  {split}: {n} superpixel files saved")

            create_symlinks(fold_dir, split,
                            primary_class=primary_class,
                            n_segments=n_segments)
            print(f"  {split}: symlinks created (Contrast_learning + ROI_data/all_class)")

    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"{'='*60}")
    print(f"\nData is ready for training. Next steps:")
    print(f"  Stage 1 — Feature Extractor:")
    print(f"    cd models && python backbone_efficientunet_train.py")
    print(f"  Stage 2 — P-RoISeg:")
    print(f"    cd models && python train_roi_efficientunet_BCSS.py")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLIC superpixels + symlinks for ProGIS training'
    )
    parser.add_argument('--input_dir',   default='data/patches',
                        help='Directory with patches (from create_patches.py)')
    parser.add_argument('--output_dir',  default='data/patches',
                        help='Output directory (usually same as input)')
    parser.add_argument('--n_segments',  type=int,   default=500,
                        help='Number of SLIC superpixels (default: 500)')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness (default: 10.0)')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 6)),
                        help='Which folds to process (default: 1 2 3 4 5)')
    parser.add_argument('--primary_class', default='tumor',
                        choices=ALL_CLASSES,
                        help='Class used for symlinks and contrastive learning '
                             '(default: tumor)')
    args = parser.parse_args()

    generate_superpixels(
        args.input_dir,
        args.output_dir,
        n_segments=args.n_segments,
        compactness=args.compactness,
        folds=args.folds,
        primary_class=args.primary_class,
    )


if __name__ == '__main__':
    main()
