# Изменения в train_nuclick.py для тестового запуска

## Сделанные изменения:

### 1. Путь к данным (строка 465)
```python
# БЫЛО:
path = "/data_nas2/gjs/ISF_pixel_level_data/Gastric_new"

# СТАЛО:
path = "/Users/olesyaindychko/Documents/phd/code/ProGIS/data/patches"
```

### 2. Класс (строка 467)
```python
# БЫЛО:
cls = 'all_class'

# СТАЛО:
cls = 'tumor'
```

### 3. Цикл по folds (строка 467-470)
```python
# БЫЛО:
for i in range(1,4):  # обучение на 3 fold'ах

# СТАЛО:
if True:  # только fold_1 для теста
```

### 4. Batch size и num_workers (строки 493-494)
```python
# БЫЛО:
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=4)

# СТАЛО:
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
```

### 5. Device (строка 508)
```python
# БЫЛО:
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# СТАЛО:
device = 'cpu'  # для macOS без GPU
```

### 6. Количество эпох (строка 650)
```python
# БЫЛО:
train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100)

# СТАЛО:
train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=3)
```

## Датасет:

- **Train**: 87 патчей 512×512
- **Val**: 348 патчей 512×512
- **Класс**: tumor (бинарная сегментация)

## Запуск:

```bash
cd models
python3 train_nuclick.py
```

## Ожидаемое время выполнения:

- **На CPU (macOS)**: ~10-20 минут за эпоху (зависит от процессора)
- **На GPU**: ~1-3 минуты за эпоху
- **Всего 3 эпохи**: ~30-60 минут на CPU

## Что будет происходить:

1. Загрузка 87 train патчей и 348 val патчей
2. Создание модели MultiScaleResUnet (5 входных каналов, 1 выходной)
3. Обучение 3 эпохи с batch_size=4
4. Валидация после каждой эпохи
5. Сохранение лучшей модели (если Dice улучшается)

## Метрики:

- **Loss**: Binary Cross-Entropy
- **Metric**: Dice Coefficient
- **Целевой Dice**: > 0.85 (хороший результат для 3 эпох)

## После теста:

Если все работает, можно:
1. Увеличить epochs до 50-100
2. Увеличить batch_size до 8-12 (если есть память)
3. Добавить 'cuda:0' device (если есть GPU)
4. Скачать полный BCSS датасет
5. Обучить на всех классах

## Возврат к оригинальным настройкам:

Просто верните значения:
- `path = "/data_nas2/gjs/ISF_pixel_level_data/Gastric_new"`
- `cls = 'all_class'`
- `for i in range(1,4):`
- `batch_size=12`
- `device = 'cuda:1' if torch.cuda.is_available() else 'cpu'`
- `epochs=100`
