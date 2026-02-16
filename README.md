# ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images

Неофициальная версия репозитория для воспроизведения результатов статьи "ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images".

## 📋 Описание

ProGIS - это метод интерактивной сегментации патологических изображений, основанный на трех основных этапах:

1. **Prototype Initialization** - Инициализация прототипов с генерацией направляющих сигналов
2. **Prototype Navigation** - Навигация по прототипам с выделением ROI (256×256)
3. **Local Refinement** - Локальная доработка через итеративную корректировку

## 🏗️ Архитектура

Основная модель: **MultiScaleResUnet**
- Backbone: ResNet-based U-Net
- Входы: 5 каналов (3 RGB + 2 guiding signals)
- Выходы: 1 канал (бинарная маска сегментации)
- Многомасштабные сверточные блоки (kernels: 3×3, 5×5, 7×7)

## 📦 Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd ProGIS
```

### 2. Создание виртуального окружения

```bash
# Создание venv
python3 -m venv venv

# Активация
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### Требования к системе

- Python 3.8+
- CUDA-compatible GPU (рекомендуется для обучения)
- Минимум 16GB RAM
- 50GB свободного места на диске (для датасетов)

## 📊 Подготовка данных

### BCSS (Breast Cancer Semantic Segmentation)

#### Структура датасета

Данные должны быть в формате `.npy` и организованы следующим образом:

```
data/
├── BCSS/
│   ├── fold_1/
│   │   ├── train/
│   │   │   ├── tumor/              # или другой класс (stroma, inflammatory, etc.)
│   │   │   │   ├── image_npy/      # RGB изображения [H, W, 3]
│   │   │   │   ├── mask_npy/       # Бинарные маски [H, W]
│   │   │   │   └── signal_all_line_npy/  # Скелетные сигналы [2, H, W]
│   │   └── val/
│   │       └── tumor/
│   │           ├── image_npy/
│   │           ├── mask_npy/
│   │           └── signal_all_line_npy/
│   ├── fold_2/
│   └── fold_3/
```

#### Классы BCSS

- `tumor` - опухолевая ткань
- `stroma` - строма
- `inflammatory_infiltration` - воспалительная инфильтрация
- `necrosis` - некроз
- `others` - другие ткани
- `all_class` - все классы вместе

#### Формат данных

**image_npy**: Numpy array формата `[H, W, 3]` (uint8 или float32, значения 0-255 или 0-1)

**mask_npy**: Numpy array формата `[H, W]` (бинарная маска: 0=фон, 1=объект)

**signal_all_line_npy**: Numpy array формата `[2, H, W]`
- Канал 0: Скелетный сигнал переднего плана
- Канал 1: Скелетный сигнал фона

### Генерация направляющих сигналов

Если у вас есть только изображения и маски, вы можете сгенерировать сигналы с помощью функции `generateGuidingSignal()` из скриптов обучения:

```python
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import skeletonize_3d
import numpy as np

def generateGuidingSignal(mask, signal_type='Skeleton'):
    # Преобразование маски в бинарный формат
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Distance transform
    dist_transform = distance_transform_edt(binary_mask)

    # Порог на основе среднего ± std
    mean_dist = dist_transform.mean()
    std_dist = dist_transform.std()
    threshold = mean_dist + np.random.uniform(-std_dist, std_dist)

    # Скелетизация
    skeleton_mask = (dist_transform > threshold).astype(np.uint8)
    skeleton = skeletonize_3d(skeleton_mask)

    return skeleton.astype(np.float32)
```

## 🚀 Запуск обучения

### Этап 1: Prototype Initialization

Обучение базовой модели на полных изображениях:

```bash
cd models
python train_nuclick.py
```

**Важные параметры в скрипте:**

```python
# Путь к данным (отредактируйте в train_nuclick.py, строки 464-474)
path = "/path/to/your/BCSS"
fold_num = 1  # номер fold (1-3)
cls = 'tumor'  # класс для обучения

train_images_dir = f"{path}/fold_{fold_num}/train/{cls}/image_npy"
train_masks_dir = f"{path}/fold_{fold_num}/train/{cls}/mask_npy"
train_signal_dir = f"{path}/fold_{fold_num}/train/{cls}/signal_all_line_npy"

# Гиперпараметры
batch_size = 12
learning_rate = 4e-4
epochs = 100
device = 'cuda:0'  # или 'cpu' если нет GPU
```

### Этап 2: ROI-based Training

Обучение модели на ROI (регионах интереса):

```bash
python train_roi_nuclick.py
```

**Конфигурация:**

```python
# ROI размер: 256×256
# Модель: MultiScaleResUnet(in_channels=5, num_classes=1)
# Loss: BCELoss
# Optimizer: Adam (lr=4e-4)
```

### Этап 3: Inference с итеративной корректировкой

```bash
python inference_correction_new_BCSS_final.py
```

**Что происходит:**
1. Получение начальной сегментации
2. Анализ ошибок (False Positives / False Negatives)
3. Генерация новых guiding signals
4. Уточнение сегментации в ROI
5. Повторение до сходимости (Dice > 0.95)

## 📁 Структура кода

### Основные модули

```
ProGIS/
├── models/
│   ├── train_nuclick.py              # ЭТАП 1: Обучение базовой модели
│   ├── train_roi_nuclick.py          # ЭТАП 2: ROI-based обучение
│   ├── inference_correction_new_BCSS_final.py  # ЭТАП 3: Инференс
│   ├── UNet.py                       # U-Net архитектура
│   ├── efficientunet/                # EfficientUNet модели
│   │   ├── efficientunet.py
│   │   ├── efficientnet.py
│   │   └── layers.py
│   └── loss/
│       └── loss.py                   # Dice Loss
├── WSI_model/
│   └── WSI_model_ROI_5_Lung.py       # Для WSI (Whole Slide Images)
└── loss/
    └── loss.py                       # Функции потерь
```

### Ключевые функции по этапам

#### Этап 1: Prototype Initialization
- `generateGuidingSignal()` - Генерация скелетных сигналов
- `processMasks_signal()` - Пакетная обработка
- `CustomDataset` - Загрузка данных

#### Этап 2: Prototype Navigation
- `ROI_crop()` - Выделение ROI 256×256
- `ROI_crop_signal_line()` - Генерация сигналов в ROI
- `get_largest_connected_component()` - Фильтрация компонент

#### Этап 3: Local Refinement
- `processMasks()` - Анализ ошибок предсказания
- Итеративный цикл уточнения

## 📊 Метрики

Основные метрики для оценки:

- **Dice Coefficient**: > 0.95 (excellent), 0.85-0.95 (good)
- **IoU** (Intersection over Union)
- **Pixel Accuracy**

## ⚙️ Конфигурация для разных GPU

### Для слабого GPU (< 8GB VRAM):

```python
batch_size = 4
num_workers = 2
device = 'cuda:0'
# Используйте смешанную точность (mixed precision)
```

### Для мощного GPU (>= 16GB VRAM):

```python
batch_size = 12-16
num_workers = 4-8
device = 'cuda:0'
```

### Для CPU (не рекомендуется):

```python
batch_size = 1-2
num_workers = 2
device = 'cpu'
# Обучение будет очень медленным
```

## 🐛 Известные проблемы

1. **Отсутствие предобученных весов**: Авторы не предоставили чекпоинты, нужно обучать с нуля
2. **Hardcoded пути**: Многие пути к данным жестко заданы, нужно их редактировать
3. **Зависимость от EfficientNet весов**: Путь `/home/gjs/ISF_nuclick/checkpoints/Efficientnet/efficientnet-b0-355c32eb.pth` нужно заменить на загрузку из torchvision

## 📝 Быстрый старт для тестирования

1. **Подготовьте минимальный датасет** (10-20 изображений для быстрой проверки)
2. **Отредактируйте пути в `train_nuclick.py`** (строки 464-474)
3. **Уменьшите количество эпох** для тестового запуска:
   ```python
   epochs = 5  # вместо 100
   ```
4. **Запустите обучение**:
   ```bash
   cd models
   python train_nuclick.py
   ```

## 📚 Датасеты

### BCSS Dataset

**Источник**: [BCSS - Grand Challenge](https://bcsegmentation.grand-challenge.org/)

**Описание**:
- 151 WSI изображения рака молочной железы
- 5 классов тканей
- Разрешение: 0.25 µm/pixel

**Скачивание**: Требуется регистрация на Grand Challenge

### Другие поддерживаемые датасеты

- **Gastric**: Гастрические образцы
- **Lung**: Легочные ткани (WSI)

## 🔧 Troubleshooting

### Ошибка: "CUDA out of memory"

```python
# Уменьшите batch_size
batch_size = 4  # или даже 2

# Уменьшите размер изображений
# Используйте gradient checkpointing
```

### Ошибка: "FileNotFoundError: efficientnet weights"

```python
# В models/efficientunet/efficientnet.py (строка 191)
# Замените на:
from torch.hub import load_state_dict_from_url
pretrained_state_dict = load_state_dict_from_url(
    'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth'
)
```

### Ошибка импорта модулей

```python
# Убедитесь, что вы находитесь в директории models/
cd models
python train_nuclick.py

# Или добавьте путь к PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/ProGIS/models"
```

## 📖 Цитирование

Если вы используете этот код, пожалуйста, цитируйте оригинальную статью:

```bibtex
@article{progis2024,
  title={ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images},
  author={Authors},
  journal={Journal},
  year={2024}
}
```

## 📧 Контакты

Для вопросов и обсуждений:
- Original Paper: [ссылка на статью]
- Issues: GitHub Issues в этом репозитории

## 📄 Лицензия

См. LICENSE файл (если есть)

---

**Примечание**: Это неофициальная версия для воспроизведения результатов. Оригинальный код предоставлен авторами без предобученных весов и детальной документации.

**Статус**: 🚧 В разработке - тестирование на BCSS датасете
