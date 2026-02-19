"""
Нарезка BCSS изображений на патчи 512×512 (stride=256).

Ключевые принципы:
  - Изображения нарезаются ОДИН раз → all/image_npy/
  - Маски нарезаются отдельно для каждого класса → {class}/mask_npy/
  - Патч попадает в класс только если fg_ratio > min_fg_ratio
  - Нет дублирования по фолдам — фолды определяются в fold_splits.json

Выходная структура:
  data/patches/
    all/
      image_npy/           ← все image-патчи (sample_XXXX_patchYYYY.npy)
    tumor/
      mask_npy/            ← только патчи где tumor присутствует
      signal_all_line_npy/
    stroma/
      mask_npy/
      signal_all_line_npy/
    ...

Использование:
  python create_patches.py --input_dir data/processed --output_dir data/patches
  python create_patches.py --input_dir data/processed --output_dir data/patches \
      --classes tumor stroma   # только нужные классы
"""

import shutil
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


# ── patch extraction ─────────────────────────────────────────────────────────

def sliding_window_coords(h: int, w: int, patch_size: int, stride: int) -> list:
    """Returns list of (y, x) top-left corners for sliding window."""
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            coords.append((y, x))
    return coords


# ── main pipeline ─────────────────────────────────────────────────────────────

def create_patches(input_dir: str, output_dir: str,
                   patch_size: int, stride: int, min_fg_ratio: float,
                   classes: list):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"CREATE PATCHES  {patch_size}×{patch_size}, stride={stride}")
    print(f"{'='*60}")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Min foreground ratio: {min_fg_ratio}")
    print(f"\nStorage advantage: image patch saved ONCE, not {len(classes)}×")
    print(f"{'='*60}\n")

    # Директории выхода
    all_img_dir = output_dir / 'all' / 'image_npy'
    all_img_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        (output_dir / cls / 'mask_npy').mkdir(parents=True, exist_ok=True)
        (output_dir / cls / 'signal_all_line_npy').mkdir(parents=True, exist_ok=True)

    # Список всех WSI
    images_dir = input_dir / 'all' / 'image_npy'
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Not found: {images_dir}\n"
            f"Run convert_bcss_to_npy.py first."
        )

    wsi_files = sorted(images_dir.glob('*.npy'))
    print(f"Found {len(wsi_files)} WSI files\n")

    total_img_patches  = 0
    class_patch_counts = {cls: 0 for cls in classes}

    for wsi_file in tqdm(wsi_files, desc="Processing WSI"):
        stem  = wsi_file.stem          # sample_XXXX
        image = np.load(wsi_file)      # [H, W, 3], float32
        h, w  = image.shape[:2]

        coords = sliding_window_coords(h, w, patch_size, stride)

        # Загружаем маски для нужных классов (None если файл не существует)
        class_masks = {}
        for cls in classes:
            mask_file = input_dir / cls / 'mask_npy' / wsi_file.name
            class_masks[cls] = np.load(mask_file) if mask_file.exists() else None

        for p_idx, (y, x) in enumerate(coords):
            patch_name = f"{stem}_patch{p_idx:04d}.npy"

            # Сохраняем image-патч ОДИН раз
            img_patch = image[y:y+patch_size, x:x+patch_size]
            dst_img   = all_img_dir / patch_name
            if not dst_img.exists():
                np.save(dst_img, img_patch)
            total_img_patches += 1  # считаем для каждого WSI первый проход

            # Для каждого класса: сохраняем маску только если класс присутствует
            for cls in classes:
                mc = class_masks[cls]
                if mc is None:
                    continue

                mask_patch = mc[y:y+patch_size, x:x+patch_size]
                fg_ratio   = mask_patch.sum() / (patch_size * patch_size)

                if fg_ratio < min_fg_ratio:
                    continue  # класс не представлен в этом патче

                seed = class_patch_counts[cls]
                fg_signal = generate_guiding_signal(mask_patch,       seed=seed)
                bg_signal = generate_guiding_signal(1.0 - mask_patch, seed=seed + 1)
                signal    = np.stack([fg_signal, bg_signal], axis=0)

                np.save(output_dir / cls / 'mask_npy'           / patch_name, mask_patch)
                np.save(output_dir / cls / 'signal_all_line_npy'/ patch_name, signal)
                class_patch_counts[cls] += 1

    # Исправляем подсчёт: каждый (wsi × patch_coord) считается один раз
    # (цикл выше считал для каждого wsi каждый coord → правильно если WSI уникальны)

    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"  Image patches (all/image_npy): {len(list(all_img_dir.glob('*.npy')))}")
    for cls in classes:
        n = len(list((output_dir / cls / 'mask_npy').glob('*.npy')))
        print(f"  {cls:30s}: {n} foreground patches")
    print(f"{'='*60}")
    # Копируем fold_splits.json в папку патчей — processed/ можно удалить после этого
    src_splits = input_dir / 'fold_splits.json'
    dst_splits = output_dir / 'fold_splits.json'
    if src_splits.exists():
        shutil.copy2(src_splits, dst_splits)
        print(f"  fold_splits.json → {dst_splits}")

    print(f"\nNext step:")
    print(f"  python generate_superpixels.py --input_dir {output_dir} --output_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create 512×512 patches. Images stored once, masks per class.'
    )
    parser.add_argument('--input_dir',      default='data/processed')
    parser.add_argument('--output_dir',     default='data/patches')
    parser.add_argument('--patch_size',     type=int,   default=512)
    parser.add_argument('--stride',         type=int,   default=256)
    parser.add_argument('--min_foreground', type=float, default=0.05,
                        help='Min fg ratio for a patch to be saved for a class (default 0.05)')
    parser.add_argument('--classes', nargs='+', default=ALL_CLASSES,
                        choices=ALL_CLASSES)
    args = parser.parse_args()

    create_patches(
        args.input_dir, args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_fg_ratio=args.min_foreground,
        classes=args.classes,
    )


if __name__ == '__main__':
    main()
