"""
Generate SLIC superpixels for ProGIS Feature Extractor (Stage 1).

Ключевые принципы:
  - SLIC генерируется ОДИН раз для каждого image-патча → all/slic_500/
  - Нет дублирования по фолдам или классам

Выходная структура:
  data/patches/
    all/
      image_npy/          ← image-патчи (из create_patches.py)
      slic_500/           ← SLIC superpixels [H, W], int32

Использование:
  python generate_superpixels.py --input_dir data/patches --output_dir data/patches
"""

import numpy as np
from pathlib import Path
from skimage.segmentation import slic
from tqdm import tqdm
import argparse


# ── SLIC ──────────────────────────────────────────────────────────────────────

def compute_slic(image: np.ndarray, n_segments: int = 500,
                 compactness: float = 10.0) -> np.ndarray:
    """
    Compute SLIC superpixels for a [H, W, 3] image patch.
    Returns int32 label map [H, W].
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)
    return segments.astype(np.int32)


# ── main pipeline ─────────────────────────────────────────────────────────────

def generate_superpixels(input_dir: str, output_dir: str,
                         n_segments: int, compactness: float):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    images_dir = input_dir / 'all' / 'image_npy'
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Not found: {images_dir}\n"
            f"Run create_patches.py first."
        )

    slic_dir = output_dir / 'all' / f'slic_{n_segments}'
    slic_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob('*.npy'))

    print(f"\n{'='*60}")
    print(f"GENERATE SLIC SUPERPIXELS  (n_segments={n_segments})")
    print(f"{'='*60}")
    print(f"Input : {images_dir}  ({len(image_files)} patches)")
    print(f"Output: {slic_dir}")
    print(f"Storage advantage: SLIC computed ONCE per patch (not per class/fold)")
    print(f"{'='*60}\n")

    skipped = 0
    for img_file in tqdm(image_files, desc="Generating SLIC"):
        dst = slic_dir / img_file.name
        if dst.exists():
            skipped += 1
            continue
        image    = np.load(img_file)          # [H, W, 3]
        segments = compute_slic(image, n_segments, compactness)
        np.save(dst, segments)

    total = len(image_files)
    computed = total - skipped
    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"  Total patches : {total}")
    print(f"  Computed      : {computed}  (skipped {skipped} already existing)")
    print(f"  Output dir    : {slic_dir}")
    print(f"{'='*60}")
    print(f"\nAll preprocessing complete. Ready for training:")
    print(f"  Stage 1: cd models && python backbone_efficientunet_train.py")
    print(f"  Stage 2: cd models && python train_roi_efficientunet_BCSS.py")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLIC superpixels once per image patch.'
    )
    parser.add_argument('--input_dir',   default='data/patches')
    parser.add_argument('--output_dir',  default='data/patches')
    parser.add_argument('--n_segments',  type=int,   default=500)
    parser.add_argument('--compactness', type=float, default=10.0)
    args = parser.parse_args()

    generate_superpixels(
        args.input_dir, args.output_dir,
        n_segments=args.n_segments,
        compactness=args.compactness,
    )


if __name__ == '__main__':
    main()
