"""
Скрипт для проверки окружения перед запуском обучения ProGIS
Проверяет установку всех зависимостей и доступность GPU
"""

import sys
import os


def check_python_version():
    """Проверка версии Python"""
    print("="*60)
    print("1. Проверка версии Python")
    print("="*60)

    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✓ Версия Python подходит (>= 3.8)")
        return True
    else:
        print("✗ Требуется Python 3.8 или выше!")
        return False


def check_pytorch():
    """Проверка установки PyTorch"""
    print("\n" + "="*60)
    print("2. Проверка PyTorch")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch установлен: {torch.__version__}")

        # Проверка CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA доступна: {torch.version.cuda}")
            print(f"✓ Количество GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

            # Тест выделения памяти
            try:
                x = torch.randn(100, 100).cuda()
                del x
                print("✓ Тест выделения памяти GPU пройден")
            except Exception as e:
                print(f"✗ Ошибка при выделении памяти GPU: {e}")
                return False
        else:
            print("⚠  CUDA не доступна - обучение будет на CPU (медленно!)")

        return True

    except ImportError:
        print("✗ PyTorch не установлен!")
        print("  Установите: pip install torch torchvision")
        return False


def check_dependencies():
    """Проверка остальных зависимостей"""
    print("\n" + "="*60)
    print("3. Проверка остальных зависимостей")
    print("="*60)

    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }

    all_ok = True

    for module, name in dependencies.items():
        try:
            if module == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif module == 'skimage':
                import skimage
                version = skimage.__version__
            elif module == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')

            print(f"✓ {name}: {version}")

        except ImportError:
            print(f"✗ {name} не установлен!")
            all_ok = False

    return all_ok


def check_project_structure():
    """Проверка структуры проекта"""
    print("\n" + "="*60)
    print("4. Проверка структуры проекта")
    print("="*60)

    required_files = [
        'models/train_nuclick.py',
        'models/train_roi_nuclick.py',
        'models/UNet.py',
        'models/efficientunet/efficientunet.py',
        'loss/loss.py',
        'requirements.txt'
    ]

    all_ok = True

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} не найден!")
            all_ok = False

    return all_ok


def check_data_preparation():
    """Проверка готовности данных"""
    print("\n" + "="*60)
    print("5. Проверка данных (опционально)")
    print("="*60)

    print("Для проверки данных укажите путь к датасету:")
    print("Пример: /path/to/BCSS")
    print("(Нажмите Enter, чтобы пропустить)")

    data_path = input("Путь к данным: ").strip()

    if not data_path:
        print("⊘ Пропущено")
        return True

    if not os.path.exists(data_path):
        print(f"✗ Путь не существует: {data_path}")
        return False

    # Проверка структуры
    fold_1 = os.path.join(data_path, 'fold_1')
    if not os.path.exists(fold_1):
        print(f"✗ Не найдена директория fold_1: {fold_1}")
        return False

    train_dir = os.path.join(fold_1, 'train')
    if not os.path.exists(train_dir):
        print(f"✗ Не найдена директория train: {train_dir}")
        return False

    print(f"✓ Базовая структура в порядке")

    # Поиск классов
    classes = []
    if os.path.exists(train_dir):
        classes = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]

    if classes:
        print(f"✓ Найдены классы: {', '.join(classes)}")

        # Проверка первого класса
        first_class = classes[0]
        class_dir = os.path.join(train_dir, first_class)

        required_subdirs = ['image_npy', 'mask_npy', 'signal_all_line_npy']
        for subdir in required_subdirs:
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.exists(subdir_path):
                num_files = len([f for f in os.listdir(subdir_path) if f.endswith('.npy')])
                print(f"  ✓ {subdir}: {num_files} файлов")
            else:
                print(f"  ✗ {subdir} не найдена")

                if subdir == 'signal_all_line_npy':
                    print("    → Запустите: python prepare_data.py --data_root {data_path} --fold 1 --class_name {first_class}")

    return True


def print_summary(results):
    """Вывод итоговой сводки"""
    print("\n" + "="*60)
    print("ИТОГОВАЯ СВОДКА")
    print("="*60)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    print("="*60)

    if all_passed:
        print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("\nВы можете начинать обучение:")
        print("  cd models")
        print("  python train_nuclick.py")
    else:
        print("\n⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        print("\nУстраните проблемы перед запуском обучения.")
        print("См. инструкции в README.md и QUICK_START.md")

    print()


def main():
    """Главная функция"""
    print("\n" + "="*60)
    print("ПРОВЕРКА ОКРУЖЕНИЯ ProGIS")
    print("="*60 + "\n")

    results = {
        'Python версия': check_python_version(),
        'PyTorch': check_pytorch(),
        'Зависимости': check_dependencies(),
        'Структура проекта': check_project_structure(),
        'Данные': check_data_preparation()
    }

    print_summary(results)


if __name__ == '__main__':
    main()
