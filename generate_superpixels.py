"""
Generate SLIC superpixels for ProGIS Feature Extractor training
"""

import numpy as np
from pathlib import Path
from skimage.segmentation import slic
from tqdm import tqdm
import argparse


def generate_superpixels_for_image(image, n_segments=500, compactness=10):
    """
    Generate SLIC superpixels for an image

    Args:
        image: RGB image [H, W, 3]
        n_segments: number of superpixels (default: 500)
        compactness: SLIC compactness parameter

    Returns:
        superpixels: [H, W] with superpixel labels
    """
    # Ensure image is in correct format [H, W, 3] with values in [0, 1] or [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Generate SLIC superpixels
    superpixels = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0
    )

    return superpixels.astype(np.int32)


def process_dataset_superpixels(
    input_dir,
    output_dir,
    n_segments=500,
    compactness=10,
    class_name='tumor'
):
    """
    Generate superpixels for all patches in dataset

    Args:
        input_dir: directory with patches (same as create_patches.py output)
        output_dir: output directory for superpixels
        n_segments: number of superpixels per image
        compactness: SLIC compactness parameter
        class_name: class name
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"GENERATING SUPERPIXELS (SLIC)")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"N segments: {n_segments}")
    print(f"Compactness: {compactness}")
    print(f"{'='*60}\n")

    # Process train and val
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()}")
        print(f"{'='*60}\n")

        # Input paths
        input_split_dir = input_dir / f'fold_1' / split / class_name
        images_dir = input_split_dir / 'image_npy'

        if not images_dir.exists():
            print(f"⚠️  Directory not found: {images_dir}")
            continue

        # Output path for superpixels
        output_split_dir = output_dir / 'fold_1' / split / 'Contrast_learning'
        output_superpixels_dir = output_split_dir / f'image_SLIC_{n_segments}'
        output_superpixels_dir.mkdir(parents=True, exist_ok=True)

        # Get list of images
        image_files = sorted(list(images_dir.glob('*.npy')))
        print(f"Found images: {len(image_files)}\n")

        # Generate superpixels for each image
        for img_file in tqdm(image_files, desc=f"Generating SLIC for {split}"):
            # Load image
            image = np.load(img_file)  # [H, W, 3]

            # Generate superpixels
            superpixels = generate_superpixels_for_image(
                image,
                n_segments=n_segments,
                compactness=compactness
            )

            # Save superpixels
            output_path = output_superpixels_dir / img_file.name
            np.save(output_path, superpixels)

        print(f"\n✓ {split.upper()}: {len(image_files)} superpixel files saved")

    print(f"\n{'='*60}")
    print(f"✓ SUPERPIXEL GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nStructure created:")
    print(f"  {output_dir}/fold_1/")
    print(f"    ├── train/Contrast_learning/image_SLIC_{n_segments}/")
    print(f"    └── val/Contrast_learning/image_SLIC_{n_segments}/")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLIC superpixels for ProGIS training'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/patches',
        help='Input directory with patches (from create_patches.py)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/patches',
        help='Output directory for superpixels (same as input)'
    )
    parser.add_argument(
        '--n_segments',
        type=int,
        default=500,
        help='Number of superpixels (default: 500)'
    )
    parser.add_argument(
        '--compactness',
        type=float,
        default=10.0,
        help='SLIC compactness parameter (default: 10.0)'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='tumor',
        help='Class name'
    )

    args = parser.parse_args()

    process_dataset_superpixels(
        args.input_dir,
        args.output_dir,
        n_segments=args.n_segments,
        compactness=args.compactness,
        class_name=args.class_name
    )


if __name__ == '__main__':
    main()
