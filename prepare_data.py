"""
Скрипт для подготовки данных BCSS к обучению ProGIS
Генерирует направляющие сигналы (guiding signals) из масок
"""

import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d
from tqdm import tqdm
import argparse


def generateGuidingSignal(mask, signal_type='Skeleton', seed=42):
    """
    Генерация направляющего сигнала из маски

    Args:
        mask: numpy array [H, W] с бинарной маской
        signal_type: тип сигнала ('Skeleton')
        seed: random seed для воспроизводимости

    Returns:
        skeleton: numpy array [H, W] со скелетным сигналом
    """
    np.random.seed(seed)

    # Преобразование в бинарный формат
    binary_mask = (mask > 0.5).astype(np.uint8)

    if binary_mask.sum() == 0:
        # Пустая маска - возвращаем нули
        return np.zeros_like(mask, dtype=np.float32)

    # Distance transform
    dist_transform = distance_transform_edt(binary_mask)

    # Вычисление порога
    mean_dist = dist_transform[dist_transform > 0].mean()
    std_dist = dist_transform[dist_transform > 0].std()

    # Случайный порог
    threshold = mean_dist + np.random.uniform(-std_dist, std_dist)
    threshold = max(0, threshold)  # Не может быть отрицательным

    # Создание маски для скелетизации
    skeleton_mask = (dist_transform > threshold).astype(np.uint8)

    # Скелетизация
    if skeleton_mask.sum() > 0:
        skeleton = skeletonize_3d(skeleton_mask)
    else:
        skeleton = skeleton_mask

    return skeleton.astype(np.float32)


def process_dataset(images_dir, masks_dir, output_signal_dir, class_name='tumor'):
    """
    Обработка датасета: генерация сигналов для всех масок

    Args:
        images_dir: путь к директории с изображениями (.npy)
        masks_dir: путь к директории с масками (.npy)
        output_signal_dir: путь для сохранения сигналов
        class_name: название класса для отображения
    """

    # Создание выходной директории
    os.makedirs(output_signal_dir, exist_ok=True)

    # Получение списка файлов
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.npy')]

    print(f"\n{'='*60}")
    print(f"Обработка класса: {class_name}")
    print(f"Найдено масок: {len(mask_files)}")
    print(f"{'='*60}\n")

    # Обработка каждой маски
    for filename in tqdm(mask_files, desc=f"Генерация сигналов для {class_name}"):
        mask_path = os.path.join(masks_dir, filename)
        signal_path = os.path.join(output_signal_dir, filename)

        # Загрузка маски
        mask = np.load(mask_path)

        # Генерация сигнала для переднего плана (foreground)
        fg_signal = generateGuidingSignal(mask, signal_type='Skeleton')

        # Генерация сигнала для фона (background)
        bg_mask = 1 - mask
        bg_signal = generateGuidingSignal(bg_mask, signal_type='Skeleton')

        # Объединение в формат [2, H, W]
        # Канал 0: foreground signal
        # Канал 1: background signal
        combined_signal = np.stack([fg_signal, bg_signal], axis=0)

        # Сохранение
        np.save(signal_path, combined_signal)

    print(f"✓ Обработано {len(mask_files)} файлов для {class_name}")
    print(f"✓ Сигналы сохранены в: {output_signal_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Подготовка данных BCSS: генерация направляющих сигналов'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Корневая директория датасета (например, /path/to/BCSS)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Номер fold (1-3)'
    )
    parser.add_argument(
        '--class_name',
        type=str,
        default='tumor',
        choices=['tumor', 'stroma', 'inflammatory_infiltration', 'necrosis', 'others', 'all_class'],
        help='Класс тканей для обработки'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Какие splits обрабатывать (train, val, test)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"ПОДГОТОВКА ДАННЫХ BCSS")
    print(f"{'='*60}")
    print(f"Датасет: {args.data_root}")
    print(f"Fold: {args.fold}")
    print(f"Класс: {args.class_name}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"{'='*60}\n")

    # Обработка каждого split (train/val)
    for split in args.splits:
        print(f"\n>>> Обработка {split.upper()} набора <<<\n")

        # Пути к данным
        split_dir = os.path.join(args.data_root, f'fold_{args.fold}', split, args.class_name)
        images_dir = os.path.join(split_dir, 'image_npy')
        masks_dir = os.path.join(split_dir, 'mask_npy')
        output_signal_dir = os.path.join(split_dir, 'signal_all_line_npy')

        # Проверка существования директорий
        if not os.path.exists(images_dir):
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Директория изображений не найдена: {images_dir}")
            continue

        if not os.path.exists(masks_dir):
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Директория масок не найдена: {masks_dir}")
            continue

        # Обработка
        process_dataset(images_dir, masks_dir, output_signal_dir, f"{split}/{args.class_name}")

    print(f"\n{'='*60}")
    print(f"✓ ГОТОВО! Данные подготовлены для обучения")
    print(f"{'='*60}\n")


def prepare_example_structure():
    """
    Вспомогательная функция для отображения ожидаемой структуры
    """
    structure = """
    Ожидаемая структура данных:

    {data_root}/
    ├── fold_1/
    │   ├── train/
    │   │   └── {class_name}/
    │   │       ├── image_npy/          (существующие RGB изображения)
    │   │       ├── mask_npy/           (существующие бинарные маски)
    │   │       └── signal_all_line_npy/  (будет создано скриптом)
    │   └── val/
    │       └── {class_name}/
    │           ├── image_npy/
    │           ├── mask_npy/
    │           └── signal_all_line_npy/
    ├── fold_2/
    └── fold_3/

    Формат файлов:
    - image_npy/*.npy: [H, W, 3] uint8 или float32
    - mask_npy/*.npy: [H, W] бинарная маска (0 или 1)
    - signal_all_line_npy/*.npy: [2, H, W] будет сгенерировано
    """
    print(structure)


if __name__ == '__main__':
    # Раскомментируйте для отображения структуры
    # prepare_example_structure()

    main()
