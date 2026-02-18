# ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images

Неофициальная версия репозитория для воспроизведения результатов статьи:
> Ge et al., "ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images", IEEE Transactions on Medical Imaging, 2025. DOI: 10.1109/TMI.2025.3611123

## Описание

ProGIS - метод интерактивной сегментации патологических изображений, который идентифицирует **все связанные компоненты одного класса за одно взаимодействие** благодаря механизму прототипов.

Фреймворк состоит из трёх модулей:

1. **Prototype Initialization** — P-RoISeg сегментирует ROI вокруг первого интерактивного сигнала; выход используется как прототип категории
2. **Prototype Navigation** — Feature Extractor извлекает pixel-level features; по сходству с прототипом находятся все связанные компоненты того же класса
3. **Local Refinement** — тот же P-RoISeg уточняет ошибочные регионы по дополнительным корректирующим сигналам

## Архитектуры моделей

### ProGIS (основной метод)

| Модуль | Архитектура | Входные каналы | Файл обучения |
|--------|-------------|----------------|---------------|
| P-RoISeg (Init + Refinement) | EfficientUNet-B0, `backbone=False` | 6 (RGB + prev_mask + fg_signal + bg_signal) | `train_roi_efficientunet_BCSS.py` |
| Feature Extractor (Navigation) | EfficientUNet-B0, `backbone=True` | 3 (RGB) | `backbone_efficientunet_train.py` |

### NuClick (baseline для сравнения)

| Архитектура | Входные каналы | Файл обучения |
|-------------|----------------|---------------|
| MultiScaleResUnet | 5 (RGB + fg_signal + bg_signal) | `train_nuclick.py` |

> **Важно**: `train_nuclick.py` — это воспроизведение **baseline метода NuClick**, с которым сравнивается ProGIS в статье. Для воспроизведения ProGIS используйте файлы из раздела ниже.

## Результаты из статьи (BCSS dataset, 50 эпох, 5-fold CV)

| Метод | mDice@20 | mIoU@20 | mNoI@85 |
|-------|----------|---------|---------|
| NuClick | 83.16 | 70.20 | — |
| ProGIS+ResNet18 | **89.70** | **83.08** | **8.64** |
| ProGIS+EfficientNet-B0 | 89.27 | 82.32 | 9.02 |

## Установка

```bash
# Клонирование
git clone <repository-url>
cd ProGIS

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Зависимости
pip install -r requirements.txt
```

**Требования**: Python 3.8+, 16GB RAM, GPU рекомендуется (обучение на CPU возможно, но медленно).

---

## Подготовка данных

### 5-fold cross-validation

Статья использует 125 WSI (100 train + 25 val), однако полный BCSS содержит 151 WSI.
Мы используем все доступные изображения — больше данных улучшает обобщение.

При N WSI код автоматически разбивает на 5 равных групп (остаток уходит в fold_5):

| N WSI | val на фолд | train на фолд |
|-------|-------------|---------------|
| 125 (статья) | 25 | 100 |
| 151 (полный BCSS) | 30 (fold_5: 31) | 121 (fold_5: 120) |

Разбиение детерминированное: WSI сортируются по имени, затем нарезаются на 5 равных групп.

### 5 классов тканей

| Класс в ProGIS | BCSS label |
|----------------|-----------|
| `tumor` | 1 |
| `stroma` | 2 |
| `inflammatory_infiltration` | 3 (lymphocytic_infiltrate) |
| `necrosis` | 4 (necrosis_or_debris) |
| `others` | 5–21 (всё остальное) |

### Шаг 0. Скачать BCSS

Датасет доступен на [BCSS Grand Challenge](https://bcsegmentation.grand-challenge.org/). Скачайте изображения (`.png`) и маски:

```
data/raw/
├── images/   # RGB изображения WSI
└── masks/    # Маски с pixel-level аннотацией (те же имена файлов)
```

### Шаг 1. Конвертация PNG → NPY + 5-fold split

```bash
python3 convert_bcss_to_npy.py \
    --images_dir data/raw/images \
    --masks_dir  data/raw/masks \
    --output_dir data/processed
```

Создаёт `data/processed/fold_{1..5}/{train,val}/{5 классов}/{image_npy,mask_npy,signal_all_line_npy}/`.

### Шаг 2. Нарезка на патчи 512×512

```bash
python3 create_patches.py \
    --input_dir  data/processed \
    --output_dir data/patches
```

Для конкретного фолда или класса:
```bash
python3 create_patches.py \
    --input_dir  data/processed \
    --output_dir data/patches \
    --folds 1 2 3 \
    --classes tumor stroma
```

### Шаг 3. Генерация SLIC superpixels

> Требуется только для обучения Feature Extractor (ProGIS Stage 1).

```bash
python3 generate_superpixels.py \
    --input_dir  data/patches \
    --output_dir data/patches
```

Автоматически создаёт symlinks для Stage 1 и Stage 2.

### Итоговая структура данных

Каждый файл хранится **ровно один раз**. Разбиение на фолды — в JSON, не в папках.

```
data/processed/                        ← после convert_bcss_to_npy.py
  fold_splits.json                     ← {fold_1: {train:[...], val:[...]}, ...}
  all/
    image_npy/                         ← WSI изображения [H, W, 3] — один раз!
  tumor/
    mask_npy/                          ← маски только для класса tumor
    signal_all_line_npy/               ← guiding signals [2, H, W]
  stroma/  necrosis/  ...              ← аналогично для других классов

data/patches/                          ← после create_patches.py
  all/
    image_npy/                         ← image-патчи — один раз!
    slic_500/                          ← SLIC superpixels — один раз!
  tumor/
    mask_npy/                          ← только fg-патчи для tumor
    signal_all_line_npy/
  stroma/  necrosis/  ...
```

> Экономия: раньше изображение сохранялось 5× (по числу классов). Теперь — 1×.

---

## Обучение ProGIS

### Stage 1: Feature Extractor (Prototype Navigation)

Обучает backbone с contrastive learning на superpixel-level признаках.

**Настройки в `models/backbone_efficientunet_train.py`:**

```python
i = 1          # номер fold
path = "/path/to/data/patches"
# batch_size=2, num_workers=0  (для CPU/macOS)
# batch_size=16, num_workers=4 (для GPU сервера)
epochs = 50    # авторы обучали 50 эпох
```

**Запуск:**

```bash
cd models
python3 backbone_efficientunet_train.py
```

Чекпоинты сохраняются в `data/patches/fold_1/efficientUnet/`.

### Stage 2: P-RoISeg (Prototype Initialization + Local Refinement)

Обучает основную сегментационную сеть с CU-Training (2 forward pass → 1 backward pass).

**Настройки в `models/train_roi_efficientunet_BCSS.py`:**

```python
i = 1          # номер fold
path = "/path/to/data/patches"
# batch_size=2, num_workers=0  (для CPU/macOS)
# batch_size=42, num_workers=8 (для GPU сервера)
epochs = 200   # авторы обучали 200 эпох
```

**Запуск:**

```bash
cd models
python3 train_roi_efficientunet_BCSS.py
```

Чекпоинты сохраняются в `data/patches/fold_1/ROI_ckpt/`.

### Порядок обучения

```
Stage 1 (Feature Extractor) → Stage 2 (P-RoISeg) → Inference
```

Оба этапа независимы, Stage 1 не нужен для запуска Stage 2.

---

## Обучение NuClick (baseline)

> Только для воспроизведения baseline из статьи.

**Настройки в `models/train_nuclick.py`:**

```python
path = "/path/to/data/patches"
cls = 'tumor'
# for i in range(1, 4):  # 3 folds
batch_size = 12
device = 'cuda:1'  # или 'cpu'
epochs = 100
```

**Запуск:**

```bash
cd models
python3 train_nuclick.py
```

---

## Параметры для разных конфигураций

| Параметр | CPU (macOS) | GPU слабый (<8GB) | GPU мощный (≥16GB) |
|----------|-------------|-------------------|---------------------|
| batch_size | 2 | 4–8 | 16–42 |
| num_workers | 0 | 2 | 4–8 |
| device | `'cpu'` | `'cuda:0'` | `'cuda:0'` |
| epochs (ProGIS) | 2–5 (тест) | 50–100 | 50–200 |

---

## Структура кода

```
ProGIS/
├── models/
│   ├── backbone_efficientunet_train.py   # ProGIS Stage 1: Feature Extractor
│   ├── train_roi_efficientunet_BCSS.py   # ProGIS Stage 2: P-RoISeg (BCSS)
│   ├── train_roi_efficientunet_final.py  # ProGIS Stage 2: финальная версия
│   ├── train_nuclick.py                  # NuClick baseline
│   ├── inference_correction_new_BCSS_final.py  # Инференс ProGIS
│   ├── UNet.py                           # U-Net архитектура
│   └── efficientunet/                    # EfficientUNet-B0
│       ├── efficientunet.py              # backbone=True (3ch) / backbone=False (6ch)
│       ├── efficientnet.py
│       └── layers.py
├── convert_bcss_to_npy.py    # PNG → NPY конвертер
├── create_patches.py         # Нарезка на патчи 512×512
├── generate_superpixels.py   # SLIC superpixels для Stage 1
├── check_environment.py      # Проверка окружения
├── requirements.txt
└── data/
    ├── raw/                  # Исходные PNG (не в git)
    ├── processed/            # После convert_bcss_to_npy.py
    └── patches/              # После create_patches.py + generate_superpixels.py
```

---

## Troubleshooting

### Создание symlinks вручную

Если `generate_superpixels.py` не создал symlinks:

```bash
cd data/patches/fold_1
for split in train val; do
  # Stage 1
  mkdir -p $split/Contrast_learning
  ln -sf "$(pwd)/$split/tumor/image_npy" $split/Contrast_learning/image_npy
  ln -sf "$(pwd)/$split/tumor/mask_npy" $split/Contrast_learning/mask_npy

  # Stage 2
  mkdir -p $split/ROI_data/all_class
  ln -sf "$(pwd)/$split/tumor/image_npy" $split/ROI_data/all_class/image_npy
  ln -sf "$(pwd)/$split/tumor/mask_npy" $split/ROI_data/all_class/mask_npy
  ln -sf "$(pwd)/$split/tumor/signal_all_line_npy" \
         $split/ROI_data/all_class/signal_maxconnect_line_npy
done
```

### multiprocessing ошибка на macOS

```python
# В DataLoader используйте:
num_workers = 0
```

### backbone_efficientunet_train.py: 'float' has no attribute 'backward'

Контрастный loss возвращает float вместо tensor, когда все патчи в батче фоновые.
Решение: функция `get_fg_filenames()` уже добавлена в скрипт — фильтрует патчи без foreground.

### train_roi_efficientunet_BCSS.py: wrong number of channels

```python
# Убедитесь, что модель создана с backbone=False:
model = get_efficientunet_b0(out_channels=1, concat_input=True,
                              pretrained=False, backbone=False)
```

### Ошибка FileNotFoundError: efficientnet weights

```python
# В models/efficientunet/efficientnet.py замените путь на:
from torch.hub import load_state_dict_from_url
state_dict = load_state_dict_from_url(
    'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth'
)
```

---

## Известные проблемы

- **Нет предобученных весов**: авторы не предоставили чекпоинты, нужно обучать с нуля
- **Hardcoded пути**: многие пути к данным и чекпоинтам жёстко заданы, требуют правки
- **DeprecationWarning**: `scipy.ndimage.morphology` и `skeletonize_3d` устарели, но работают

---

## Цитирование

```bibtex
@article{ge2025progis,
  title={ProGIS: Prototype-Guided Interactive Segmentation for Pathological Images},
  author={Ge, Jiusong and Zhang, Di and Zhan, Yingkang and Liu, Jiashuai and
          Gong, Tieliang and Wu, Jialun and Crispin-Ortuzar, Mireia and
          Li, Chen and Gao, Zeyu},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  doi={10.1109/TMI.2025.3611123}
}
```

---

**Статус**: воспроизведение pipeline подтверждено на 3 BCSS изображениях (87 train + 348 val патчей).
Для воспроизведения метрик из статьи требуется полный BCSS датасет и GPU сервер.
