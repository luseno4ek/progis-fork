# 🚀 Быстрый старт ProGIS

Пошаговая инструкция для запуска обучения на BCSS датасете.

## ✅ Шаг 1: Установка окружения

```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация (macOS/Linux)
source venv/bin/activate

# Активация (Windows)
# venv\Scripts\activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

**Проверка установки:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')"
```

---

## 📦 Шаг 2: Подготовка данных BCSS

### Вариант A: У вас уже есть данные в формате .npy

Убедитесь, что структура следующая:

```
/path/to/BCSS/
├── fold_1/
│   ├── train/
│   │   └── tumor/
│   │       ├── image_npy/      # RGB изображения [H, W, 3]
│   │       └── mask_npy/       # Бинарные маски [H, W]
│   └── val/
│       └── tumor/
│           ├── image_npy/
│           └── mask_npy/
```

**Генерация направляющих сигналов:**

```bash
python prepare_data.py \
    --data_root /path/to/BCSS \
    --fold 1 \
    --class_name tumor \
    --splits train val
```

После выполнения появятся директории `signal_all_line_npy/` с сигналами.

### Вариант B: Конвертация из изображений PNG/JPG

Если у вас изображения в формате PNG/JPG, создайте скрипт конвертации:

```python
import numpy as np
from PIL import Image
import os
from pathlib import Path

def convert_images_to_npy(image_dir, output_dir):
    """Конвертация PNG/JPG в .npy формат"""
    os.makedirs(output_dir, exist_ok=True)

    for img_file in Path(image_dir).glob('*.png'):  # или *.jpg
        # Загрузка изображения
        img = Image.open(img_file)
        img_array = np.array(img)

        # Для RGB изображений: [H, W, 3]
        # Для масок: [H, W] (бинарная)

        # Сохранение как .npy
        output_path = os.path.join(output_dir, img_file.stem + '.npy')
        np.save(output_path, img_array)

        print(f"Сконвертировано: {img_file.name} -> {img_file.stem}.npy")

# Использование:
# convert_images_to_npy('/path/to/images', '/path/to/BCSS/fold_1/train/tumor/image_npy')
# convert_images_to_npy('/path/to/masks', '/path/to/BCSS/fold_1/train/tumor/mask_npy')
```

### Вариант C: Тестовый мини-датасет

Для быстрого тестирования создайте минимальный датасет (5-10 изображений):

```bash
# Создание структуры
mkdir -p test_data/fold_1/train/tumor/{image_npy,mask_npy}
mkdir -p test_data/fold_1/val/tumor/{image_npy,mask_npy}

# Скопируйте туда несколько изображений и масок в формате .npy
```

---

## 🏋️ Шаг 3: Настройка скрипта обучения

Отредактируйте [models/train_nuclick.py](models/train_nuclick.py):

```python
# Найдите строки 464-474 и измените:

path = "/path/to/your/BCSS"  # ← ИЗМЕНИТЕ НА ВАШ ПУТЬ
fold_num = 1
cls = 'tumor'

# Для быстрого теста измените также:
epochs = 5  # строка ~502 (вместо 100)
batch_size = 4  # строка 490 (если мало памяти GPU)
device = 'cuda:0'  # строка 503 (или 'cpu' если нет GPU)
```

**Полный список изменяемых параметров:**

```python
# ПУТИ К ДАННЫМ (строки 464-474)
path = "/Users/olesyaindychko/Documents/phd/data/BCSS"
fold_num = 1
cls = 'tumor'

train_images_dir = f"{path}/fold_{fold_num}/train/{cls}/image_npy"
train_masks_dir = f"{path}/fold_{fold_num}/train/{cls}/mask_npy"
train_signal_dir = f"{path}/fold_{fold_num}/train/{cls}/signal_all_line_npy"

val_images_dir = f"{path}/fold_{fold_num}/val/{cls}/image_npy"
val_masks_dir = f"{path}/fold_{fold_num}/val/{cls}/mask_npy"
val_signal_dir = f"{path}/fold_{fold_num}/val/{cls}/signal_all_line_npy"

# ГИПЕРПАРАМЕТРЫ (строки 490-503)
batch_size = 12  # уменьшите до 4-6 для слабого GPU
num_workers = 4
learning_rate = 4e-4  # в optimizer, строка 497
epochs = 100  # в train_model(), строка 501
device = 'cuda:0'  # строка 503
```

---

## ▶️ Шаг 4: Запуск обучения

```bash
cd models
python train_nuclick.py
```

**Что вы увидите:**

```
Обучение модели...
Epoch 1/5
Train Loss: 0.3456 | Train Dice: 0.7234
Val Loss: 0.3123 | Val Dice: 0.7891
Модель сохранена! Новый лучший Dice: 0.7891

Epoch 2/5
...
```

**Проблемы и решения:**

### CUDA out of memory
```python
# В train_nuclick.py измените:
batch_size = 2  # или даже 1
```

### ModuleNotFoundError: No module named 'efficientunet'
```bash
# Убедитесь, что вы в директории models/
cd models
python train_nuclick.py

# Или добавьте путь:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### FileNotFoundError: efficientnet weights
```python
# В models/efficientunet/efficientnet.py (строка 189-192)
# Измените pretrained=False при создании модели:
model = MultiScaleResUnet(in_channels=5, num_classes=1)
# вместо EfficientUNet
```

---

## 📊 Шаг 5: Мониторинг обучения

### Сохранение моделей

По умолчанию лучшая модель сохраняется автоматически при улучшении Dice коэффициента.

Добавьте в код (после строки 563):

```python
# Сохранение лучшей модели
checkpoint_dir = f'{path}/checkpoints/fold_{i}'
os.makedirs(checkpoint_dir, exist_ok=True)

if val_dice_score > best_dice:
    best_dice = val_dice_score
    torch.save(
        model.state_dict(),
        f'{checkpoint_dir}/best_model_dice_{val_dice_score:.4f}.pth'
    )
    print(f'✓ Модель сохранена! Dice: {val_dice_score:.4f}')
```

### Логирование

Для детального логирования используйте tensorboard:

```python
from torch.utils.tensorboard import SummaryWriter

# В начале train_model()
writer = SummaryWriter(f'{path}/runs/fold_{i}')

# В цикле обучения
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Dice/train', train_dice_score, epoch)
writer.add_scalar('Dice/val', val_dice_score, epoch)
```

Просмотр:
```bash
tensorboard --logdir=/path/to/BCSS/runs
```

---

## 🎯 Шаг 6: Оценка результатов

После обучения проверьте метрики:

### Целевые значения

- **Dice Coefficient**:
  - 🟢 > 0.95 - отличный результат
  - 🟡 0.85-0.95 - хороший результат
  - 🔴 < 0.85 - нужно доработать

### Визуализация предсказаний

Добавьте код для сохранения примеров:

```python
import matplotlib.pyplot as plt

def save_predictions(images, masks, predictions, epoch, save_dir):
    """Сохранение примеров предсказаний"""
    os.makedirs(save_dir, exist_ok=True)

    # Берем первые 4 примера из batch
    num_samples = min(4, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    for i in range(num_samples):
        # Изображение
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        # Ground truth маска
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Предсказание
        axes[i, 2].imshow(predictions[i, 0].cpu() > 0.5, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_predictions.png', dpi=150)
    plt.close()

# Использование в валидационном цикле:
# save_predictions(images, masks, outputs, epoch, f'{path}/visualizations/fold_{i}')
```

---

## 🔄 Шаг 7: ROI-based обучение (Этап 2)

После успешного обучения базовой модели:

```bash
python train_roi_nuclick.py
```

**Изменения в train_roi_nuclick.py:**

```python
# Строки ~464-474: те же пути что и в train_nuclick.py
path = "/path/to/your/BCSS"
fold_num = 1
cls = 'tumor'

# Строка 718: загрузка весов из Этапа 1
model.load_state_dict(torch.load(
    f'{path}/checkpoints/fold_1/best_model_dice_*.pth'
))
```

---

## 📋 Чеклист готовности к запуску

- [ ] Виртуальное окружение создано и активировано
- [ ] Все зависимости установлены (`pip install -r requirements.txt`)
- [ ] CUDA доступна (если используете GPU)
- [ ] Данные подготовлены в формате .npy
- [ ] Сигналы сгенерированы (`prepare_data.py` выполнен)
- [ ] Пути в `train_nuclick.py` изменены на ваши
- [ ] Гиперпараметры настроены под вашу систему
- [ ] Директория для сохранения чекпоинтов создана

---

## 🐞 Частые ошибки

### 1. "No such file or directory: '...image_npy'"

**Решение:** Проверьте пути в скрипте. Используйте абсолютные пути.

```python
# Вместо относительных путей:
path = "/Users/your_name/Documents/BCSS"  # абсолютный путь
```

### 2. "ValueError: not enough values to unpack"

**Решение:** Проблема с размерностью данных. Проверьте формат .npy файлов:

```python
import numpy as np

# Проверка формата
img = np.load('test_image.npy')
mask = np.load('test_mask.npy')
signal = np.load('test_signal.npy')

print(f"Image shape: {img.shape}")  # должно быть [H, W, 3]
print(f"Mask shape: {mask.shape}")  # должно быть [H, W]
print(f"Signal shape: {signal.shape}")  # должно быть [2, H, W]
```

### 3. "RuntimeError: Input type and weight type should be the same"

**Решение:** Проблема с типами данных. Приведите к float32:

```python
# В CustomDataset.__getitem__():
image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
```

### 4. Очень медленное обучение

**Решения:**
- Уменьшите batch_size
- Уменьшите num_workers
- Используйте GPU вместо CPU
- Уменьшите размер изображений

---

## 📈 Следующие шаги

После успешного запуска базового обучения:

1. **Оптимизация гиперпараметров**
   - Попробуйте разные learning rates
   - Измените batch size
   - Добавьте learning rate scheduler

2. **Эксперименты с архитектурой**
   - Попробуйте EfficientUNet вместо MultiScaleResUnet
   - Измените количество каналов

3. **Расширение на другие классы**
   - Обучите на `stroma`, `necrosis` и т.д.
   - Используйте `all_class` для мультиклассовой сегментации

4. **Переход к полной pipeline**
   - Запустите ROI-based обучение
   - Используйте итеративную корректировку

---

## 💡 Полезные команды

```bash
# Проверка структуры данных
tree -L 4 /path/to/BCSS

# Подсчет количества файлов
find /path/to/BCSS/fold_1/train/tumor/image_npy -name "*.npy" | wc -l

# Проверка размера датасета
du -sh /path/to/BCSS/*

# Мониторинг GPU
watch -n 1 nvidia-smi

# Убить процесс если завис
ps aux | grep python
kill -9 <PID>
```

---

Готово! Теперь вы можете начать обучение ProGIS на BCSS датасете. Удачи! 🚀
