"""
Нарезка больших изображений BCSS на патчи 512×512 с шагом 256 (как в статье ProGIS)
Sliding window operation: patch_size=512, stride=256
"""

import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from tqdm import tqdm


def generateGuidingSignal(mask, signal_type='Skeleton', seed=42):
    """Генерация направляющего сигнала из маски"""
    np.random.seed(seed)

    binary_mask = (mask > 0.5).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    dist_transform = distance_transform_edt(binary_mask)

    nonzero_dist = dist_transform[dist_transform > 0]
    if len(nonzero_dist) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    mean_dist = nonzero_dist.mean()
    std_dist = nonzero_dist.std()

    threshold = mean_dist + np.random.uniform(-std_dist, std_dist)
    threshold = max(0, threshold)

    skeleton_mask = (dist_transform > threshold).astype(np.uint8)

    if skeleton_mask.sum() > 0:
        skeleton = skeletonize(skeleton_mask)
    else:
        skeleton = skeleton_mask

    return skeleton.astype(np.float32)


def extract_patches(image, mask, patch_size=512, stride=256, min_foreground_ratio=0.05):
    """
    Извлечение патчей из большого изображения

    Args:
        image: RGB изображение [H, W, 3]
        mask: бинарная маска [H, W]
        patch_size: размер патча (512)
        stride: шаг окна (256)
        min_foreground_ratio: минимальная доля foreground пикселей в патче

    Returns:
        patches: список (image_patch, mask_patch, has_foreground)
    """
    h, w = image.shape[:2]
    patches = []

    # Sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Извлечение патча
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # Проверка наличия foreground
            foreground_ratio = mask_patch.sum() / (patch_size * patch_size)

            # Сохраняем патч если:
            # 1. Есть достаточно foreground (> min_ratio)
            # 2. Или полностью background (для обучения на негативных примерах)
            if foreground_ratio > min_foreground_ratio or foreground_ratio == 0:
                patches.append((img_patch, mask_patch, foreground_ratio))

    return patches


def process_dataset_to_patches(
    input_dir,
    output_dir,
    patch_size=512,
    stride=256,
    min_foreground_ratio=0.05,
    class_name='tumor'
):
    """
    Обработка всего датасета: нарезка на патчи

    Args:
        input_dir: директория с большими изображениями (.npy)
        output_dir: выходная директория для патчей
        patch_size: размер патча
        stride: шаг
        min_foreground_ratio: минимальная доля foreground
        class_name: название класса
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"НАРЕЗКА ИЗОБРАЖЕНИЙ НА ПАТЧИ")
    print(f"{'='*60}")
    print(f"Входная директория: {input_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Размер патча: {patch_size}×{patch_size}")
    print(f"Шаг (stride): {stride}")
    print(f"Min foreground ratio: {min_foreground_ratio}")
    print(f"{'='*60}\n")

    # Обработка train и val
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Обработка {split.upper()}")
        print(f"{'='*60}\n")

        # Пути к входным данным
        input_split_dir = input_dir / f'fold_1' / split / class_name
        images_dir = input_split_dir / 'image_npy'
        masks_dir = input_split_dir / 'mask_npy'

        # Проверка существования
        if not images_dir.exists():
            print(f"⚠️  Директория не найдена: {images_dir}")
            continue

        # Выходные директории
        output_split_dir = output_dir / 'fold_1' / split / class_name
        output_images_dir = output_split_dir / 'image_npy'
        output_masks_dir = output_split_dir / 'mask_npy'
        output_signals_dir = output_split_dir / 'signal_all_line_npy'

        for d in [output_images_dir, output_masks_dir, output_signals_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Получение списка изображений
        image_files = sorted(list(images_dir.glob('*.npy')))
        print(f"Найдено изображений: {len(image_files)}\n")

        total_patches = 0

        # Обработка каждого изображения
        for img_file in tqdm(image_files, desc=f"Нарезка {split}"):
            # Загрузка
            image = np.load(img_file)
            mask_file = masks_dir / img_file.name
            mask = np.load(mask_file)

            # Извлечение патчей
            patches = extract_patches(
                image, mask,
                patch_size=patch_size,
                stride=stride,
                min_foreground_ratio=min_foreground_ratio
            )

            # Сохранение патчей
            base_name = img_file.stem
            for patch_idx, (img_patch, mask_patch, fg_ratio) in enumerate(patches):
                # Генерация сигналов
                fg_signal = generateGuidingSignal(mask_patch, seed=total_patches)
                bg_mask = 1 - mask_patch
                bg_signal = generateGuidingSignal(bg_mask, seed=total_patches+1)
                combined_signal = np.stack([fg_signal, bg_signal], axis=0)

                # Имя файла
                patch_name = f"{base_name}_patch{patch_idx:04d}.npy"

                # Сохранение
                np.save(output_images_dir / patch_name, img_patch)
                np.save(output_masks_dir / patch_name, mask_patch)
                np.save(output_signals_dir / patch_name, combined_signal)

                total_patches += 1

            print(f"  {img_file.name}: {len(patches)} патчей")

        print(f"\n✓ {split.upper()}: всего {total_patches} патчей сохранено")

    print(f"\n{'='*60}")
    print(f"✓ НАРЕЗКА ЗАВЕРШЕНА!")
    print(f"{'='*60}")
    print(f"\nСтруктура создана:")
    print(f"  {output_dir}/fold_1/")
    print(f"    ├── train/{class_name}/")
    print(f"    │   ├── image_npy/")
    print(f"    │   ├── mask_npy/")
    print(f"    │   └── signal_all_line_npy/")
    print(f"    └── val/{class_name}/")
    print(f"        ├── image_npy/")
    print(f"        ├── mask_npy/")
    print(f"        └── signal_all_line_npy/")

    print(f"\n{'='*60}")
    print(f"СЛЕДУЮЩИЕ ШАГИ:")
    print(f"{'='*60}")
    print(f"1. Отредактируйте models/train_nuclick.py:")
    print(f"   path = '{output_dir.absolute()}'")
    print(f"   cls = '{class_name}'")
    print(f"   batch_size = 12  # Теперь можно использовать нормальный batch")
    print(f"\n2. Запустите обучение:")
    print(f"   cd models")
    print(f"   python3 train_nuclick.py")
    print(f"{'='*60}\n")


def get_statistics(data_dir, class_name='tumor'):
    """Статистика по патчам"""
    data_dir = Path(data_dir)

    print(f"\n{'='*60}")
    print(f"СТАТИСТИКА ПО ПАТЧАМ")
    print(f"{'='*60}\n")

    for split in ['train', 'val']:
        images_dir = data_dir / 'fold_1' / split / class_name / 'image_npy'

        if not images_dir.exists():
            continue

        files = list(images_dir.glob('*.npy'))
        print(f"{split.upper()}:")
        print(f"  Количество патчей: {len(files)}")

        if files:
            # Загрузка первого патча для проверки
            sample = np.load(files[0])
            print(f"  Размер патча: {sample.shape}")
            print(f"  Dtype: {sample.dtype}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Нарезка BCSS изображений на патчи 512×512'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/processed',
        help='Директория с большими изображениями'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/patches',
        help='Выходная директория для патчей'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=512,
        help='Размер патча (по умолчанию 512)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=256,
        help='Шаг sliding window (по умолчанию 256)'
    )
    parser.add_argument(
        '--min_foreground',
        type=float,
        default=0.05,
        help='Минимальная доля foreground в патче (по умолчанию 0.05 = 5%%)'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='tumor',
        help='Название класса'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Показать только статистику'
    )

    args = parser.parse_args()

    if args.stats:
        get_statistics(args.output_dir, args.class_name)
    else:
        process_dataset_to_patches(
            args.input_dir,
            args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            min_foreground_ratio=args.min_foreground,
            class_name=args.class_name
        )
        get_statistics(args.output_dir, args.class_name)


if __name__ == '__main__':
    main()
