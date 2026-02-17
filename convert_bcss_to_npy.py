"""
Конвертация BCSS данных из PNG в NPY формат для ProGIS
Создает бинарные маски для выбранного класса и генерирует guiding signals
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d


# Маппинг классов BCSS
CLASS_MAPPING = {
    'outside_roi': 0,
    'tumor': 1,
    'stroma': 2,
    'lymphocytic_infiltrate': 3,
    'necrosis_or_debris': 4,
    'glandular_secretions': 5,
    'blood': 6,
    'exclude': 7,
    'metaplasia_NOS': 8,
    'fat': 9,
    'plasma_cells': 10,
    'other_immune_infiltrate': 11,
    'mucoid_material': 12,
    'normal_acinus_or_duct': 13,
    'lymphatics': 14,
    'undetermined': 15,
    'nerve': 16,
    'skin_adnexa': 17,
    'blood_vessel': 18,
    'angioinvasion': 19,
    'dcis': 20,
    'other': 21
}


def generateGuidingSignal(mask, signal_type='Skeleton', seed=42):
    """
    Генерация направляющего сигнала из маски

    Args:
        mask: numpy array [H, W] с бинарной маской
        signal_type: тип сигнала ('Skeleton')
        seed: random seed

    Returns:
        skeleton: numpy array [H, W] со скелетным сигналом
    """
    np.random.seed(seed)

    # Преобразование в бинарный формат
    binary_mask = (mask > 0.5).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    # Distance transform
    dist_transform = distance_transform_edt(binary_mask)

    # Вычисление порога
    nonzero_dist = dist_transform[dist_transform > 0]
    if len(nonzero_dist) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    mean_dist = nonzero_dist.mean()
    std_dist = nonzero_dist.std()

    # Случайный порог
    threshold = mean_dist + np.random.uniform(-std_dist, std_dist)
    threshold = max(0, threshold)

    # Создание маски для скелетизации
    skeleton_mask = (dist_transform > threshold).astype(np.uint8)

    # Скелетизация
    if skeleton_mask.sum() > 0:
        skeleton = skeletonize_3d(skeleton_mask)
    else:
        skeleton = skeleton_mask

    return skeleton.astype(np.float32)


def create_binary_mask(multiclass_mask, target_class_id):
    """
    Создание бинарной маски для определенного класса

    Args:
        multiclass_mask: numpy array с мультиклассовой маской
        target_class_id: ID класса для извлечения

    Returns:
        binary_mask: бинарная маска [H, W] (0 или 1)
    """
    binary_mask = (multiclass_mask == target_class_id).astype(np.float32)
    return binary_mask


def resize_to_multiple_of_16(image, mask):
    """
    Изменение размера до кратного 16 (для U-Net архитектуры)

    Args:
        image: RGB изображение
        mask: маска

    Returns:
        resized_image, resized_mask
    """
    h, w = image.shape[:2]

    # Находим ближайший размер, кратный 16
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16

    # Если размер уже кратен 16, ничего не делаем
    if h == new_h and w == new_w:
        return image, mask

    # Resize с сохранением пропорций
    from PIL import Image as PILImage

    img_pil = PILImage.fromarray(image)
    mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))

    img_resized = img_pil.resize((new_w, new_h), PILImage.BILINEAR)
    mask_resized = mask_pil.resize((new_w, new_h), PILImage.NEAREST)

    return np.array(img_resized), (np.array(mask_resized) > 127).astype(np.float32)


def process_bcss_images(
    images_dir,
    masks_dir,
    output_dir,
    target_class='tumor',
    train_ratio=0.7,
    resize=True
):
    """
    Обработка BCSS изображений и конвертация в NPY формат

    Args:
        images_dir: директория с изображениями
        masks_dir: директория с масками
        output_dir: выходная директория
        target_class: целевой класс для бинарной сегментации
        train_ratio: доля для train (остальное в val)
        resize: изменять ли размер до кратного 16
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Получаем ID класса
    target_class_id = CLASS_MAPPING.get(target_class)
    if target_class_id is None:
        raise ValueError(f"Неизвестный класс: {target_class}")

    print(f"\n{'='*60}")
    print(f"КОНВЕРТАЦИЯ BCSS В NPY ФОРМАТ")
    print(f"{'='*60}")
    print(f"Входная директория изображений: {images_dir}")
    print(f"Входная директория масок: {masks_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Целевой класс: {target_class} (ID={target_class_id})")
    print(f"Train/Val split: {train_ratio:.0%}/{1-train_ratio:.0%}")
    print(f"Resize до кратного 16: {resize}")
    print(f"{'='*60}\n")

    # Получаем список изображений
    image_files = sorted(list(images_dir.glob('*.png')))
    print(f"Найдено изображений: {len(image_files)}\n")

    if len(image_files) == 0:
        print("⚠️  Изображения не найдены!")
        return

    # Разделение на train/val
    num_train = max(1, int(len(image_files) * train_ratio))
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    print(f"Train: {len(train_files)} изображений")
    print(f"Val: {len(val_files)} изображений\n")

    # Создание структуры директорий
    for split in ['train', 'val']:
        for data_type in ['image_npy', 'mask_npy', 'signal_all_line_npy']:
            dir_path = output_dir / 'fold_1' / split / target_class / data_type
            dir_path.mkdir(parents=True, exist_ok=True)

    # Обработка train данных
    print("Обработка TRAIN данных:")
    for i, img_file in enumerate(train_files):
        process_single_image(
            img_file,
            masks_dir,
            output_dir / 'fold_1' / 'train' / target_class,
            target_class_id,
            i,
            resize
        )

    # Обработка val данных
    print("\nОбработка VAL данных:")
    for i, img_file in enumerate(val_files):
        process_single_image(
            img_file,
            masks_dir,
            output_dir / 'fold_1' / 'val' / target_class,
            target_class_id,
            i,
            resize
        )

    print(f"\n{'='*60}")
    print(f"✓ КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
    print(f"{'='*60}")
    print(f"\nСтруктура создана:")
    print(f"  {output_dir}/fold_1/")
    print(f"    ├── train/{target_class}/")
    print(f"    │   ├── image_npy/")
    print(f"    │   ├── mask_npy/")
    print(f"    │   └── signal_all_line_npy/")
    print(f"    └── val/{target_class}/")
    print(f"        ├── image_npy/")
    print(f"        ├── mask_npy/")
    print(f"        └── signal_all_line_npy/")

    print(f"\n{'='*60}")
    print(f"СЛЕДУЮЩИЕ ШАГИ:")
    print(f"{'='*60}")
    print(f"1. Отредактируйте models/train_nuclick.py:")
    print(f"   path = '{output_dir.absolute()}'")
    print(f"   fold_num = 1")
    print(f"   cls = '{target_class}'")
    print(f"   epochs = 5  # Для быстрого теста")
    print(f"   batch_size = 2  # Изображения большие!")
    print(f"\n2. Запустите обучение:")
    print(f"   cd models")
    print(f"   python train_nuclick.py")
    print(f"{'='*60}\n")


def process_single_image(img_file, masks_dir, output_base, target_class_id, index, resize):
    """Обработка одного изображения"""
    # Загрузка изображения
    image = np.array(Image.open(img_file))

    # Загрузка соответствующей маски
    mask_file = masks_dir / img_file.name
    if not mask_file.exists():
        print(f"  ⚠️  Маска не найдена для {img_file.name}, пропускаем")
        return

    multiclass_mask = np.array(Image.open(mask_file))

    # Создание бинарной маски
    binary_mask = create_binary_mask(multiclass_mask, target_class_id)

    # Проверка наличия целевого класса
    if binary_mask.sum() == 0:
        print(f"  ⚠️  {img_file.name}: класс {target_class_id} отсутствует, пропускаем")
        return

    # Resize если нужно
    if resize:
        image, binary_mask = resize_to_multiple_of_16(image, binary_mask)

    # Генерация сигналов
    fg_signal = generateGuidingSignal(binary_mask, signal_type='Skeleton', seed=42+index)
    bg_mask = 1 - binary_mask
    bg_signal = generateGuidingSignal(bg_mask, signal_type='Skeleton', seed=43+index)
    combined_signal = np.stack([fg_signal, bg_signal], axis=0)

    # Сохранение
    filename = f'sample_{index:04d}.npy'

    np.save(output_base / 'image_npy' / filename, image)
    np.save(output_base / 'mask_npy' / filename, binary_mask)
    np.save(output_base / 'signal_all_line_npy' / filename, combined_signal)

    pixel_percentage = (binary_mask.sum() / binary_mask.size) * 100
    print(f"  ✓ {img_file.name}")
    print(f"    → {filename}")
    print(f"    → Размер: {image.shape}, Класс: {pixel_percentage:.2f}% пикселей")


def main():
    parser = argparse.ArgumentParser(
        description='Конвертация BCSS данных из PNG в NPY формат'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='data/images',
        help='Директория с изображениями PNG'
    )
    parser.add_argument(
        '--masks_dir',
        type=str,
        default='data/masks',
        help='Директория с масками PNG'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Выходная директория для NPY файлов'
    )
    parser.add_argument(
        '--target_class',
        type=str,
        default='tumor',
        choices=list(CLASS_MAPPING.keys()),
        help='Целевой класс для бинарной сегментации'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Доля данных для train (остальное в val)'
    )
    parser.add_argument(
        '--no_resize',
        action='store_true',
        help='Не изменять размер изображений'
    )

    args = parser.parse_args()

    process_bcss_images(
        args.images_dir,
        args.masks_dir,
        args.output_dir,
        args.target_class,
        args.train_ratio,
        resize=not args.no_resize
    )


if __name__ == '__main__':
    main()
