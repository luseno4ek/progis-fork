# iSegGym — подробный план разработки

Версия: draft v1  
Статус: согласованный архитектурный план  
Язык реализации: Python  
Планируемый GUI: Flet  
Целевое название проекта: **iSegGym** (*Interactive Segmentation Gym*)

---

## 1. Назначение документа

Этот документ задаёт архитектурный и организационный план разработки **iSegGym** — отдельного полигона для разработки, тестирования, ручного использования и сравнения методов интерактивной сегментации изображений со скриблами.

Документ предназначен для двух режимов разработки:

1. **классическая разработка** — по этапам, модулям, API-контрактам и acceptance criteria;
2. **агентная разработка** — через декомпозицию на независимые задачи, где агенты могут работать с чётко выделенными слоями, DTO, сериализацией, метриками, baseline-методами и GUI.

---

## 2. Цели проекта

### 2.1. Основная цель

Создать отдельную модульную среду **iSegGym** для интерактивной сегментации, которая позволяет:

- вручную взаимодействовать с методом интерактивной сегментации;
- подключать разные методы через единый интерфейс;
- работать с многоклассовой разметкой;
- сохранять и воспроизводить историю действий пользователя;
- считать метрики качества и метрики взаимодействия;
- экспортировать/импортировать скриблы и результаты;
- использовать систему как локально через GUI, так и позднее в headless/CLI-режиме.

### 2.2. Почему нужен отдельный проект

Существующие решения не удовлетворяют совокупности требований:

- требуемая модульность;
- контроль над протоколом эксперимента;
- ручная работа со скриблами;
- воспроизводимость сессий;
- возможность повторного прогона тех же действий на другом методе;
- многоклассовость;
- собственные метрики и формат журналов;
- дальнейшая интеграция в **PathScribe** (Flutter GUI + Python microservices + gRPC).

### 2.3. Основной сценарий этапа 1

На первом этапе iSegGym решает задачу класса **Research MVP**:

- открыть локальный датасет;
- выбрать изображение;
- нарисовать один или несколько скриблов;
- нажать `Update`;
- получить предсказанную маску;
- посмотреть GT и prediction;
- сохранить журнал действий;
- воспроизвести сессию без пользователя;
- повторно прогнать ту же сессию на другом методе;
- сохранить GIF и промежуточные PNG-маски.

---

## 3. Границы первой версии

### 3.1. Что входит в MVP

Входит:

- обычные 2D изображения;
- локальная работа на macOS и Windows;
- Python API между модулями;
- Flet GUI;
- батч-режим взаимодействия: пользователь может нанести несколько штрихов, затем вручную запускает инференс;
- многоклассовая сегментация в постановке **single-label per pixel**;
- хранение скриблов как векторных объектов;
- хранение prediction как raster label map;
- импорт GT из PNG-масок;
- экспорт скриблов в JSON;
- сохранение промежуточных prediction masks в PNG;
- replay и rerun сессий;
- dummy baseline и superpixel propagation baseline;
- базовые метрики: IoU, Dice, IoU@k, Dice@k, NoC и effort metrics.

### 3.2. Что сознательно не входит в MVP

Не входит:

- WSI и tile streaming;
- multilabel per pixel;
- редактирование уже нарисованного скрибла;
- полноценный gRPC между GUI и backend;
- удалённый inference service как обязательная часть архитектуры;
- полигоны как внутренний канонический формат;
- rich GIF/video overlays;
- обучение RL-компонентов;
- production-grade multi-user сценарии.

### 3.3. Что нужно предусмотреть архитектурно заранее

Нужно предусмотреть заранее:

- headless/CLI-режим;
- переход к отдельному process boundary для inference;
- расширение до высоких разрешений (до ~5k x 5k);
- подключение методов вроде CNN-assisted hybrid и ProGIS;
- журнал действий как будущий источник данных для realistic simulation / RL.

---

## 4. Ключевые архитектурные принципы

### 4.1. Ядро проекта — не GUI, а stateful session engine

Главный объект системы — **интерактивная сессия** над одним изображением.

Сессия содержит:

- изображение;
- схему классов;
- историю пользовательских действий;
- текущий набор активных скриблов;
- текущую prediction mask;
- историю шагов инференса;
- историю метрик;
- ссылки на артефакты.

GUI не должен напрямую управлять моделью. GUI работает через application/service layer и команды к session engine.

### 4.2. Метод сегментации должен быть подключаемым и по умолчанию stateless

На этапе 1 каждый метод интерактивной сегментации должен реализовывать единый интерфейс вида:

```python
result = predictor.predict(request: InferenceRequest) -> InferenceResult
```

По умолчанию метод:

- **не хранит** внутреннее состояние между вызовами;
- получает всё необходимое через `InferenceRequest`;
- при необходимости может возвращать optional diagnostics / method_state, но среда не обязана их использовать.

### 4.3. Event sourcing как основной принцип воспроизводимости

Главный источник истины для эксперимента — **журнал событий**.

Это даёт:

- replay без пользователя;
- rerun тех же пользовательских действий на другом методе;
- удобный headless-режим;
- источники данных для симуляции действий;
- трассировку ошибок;
- прозрачность и воспроизводимость.

### 4.4. Двойное представление интеракций

Скриблы должны существовать в двух формах:

1. **векторная форма** — каноническая для хранения и обмена;
2. **растровая форма** — каноническая для инференса и метрик.

### 4.5. Два уровня API

Архитектуру надо проектировать сразу на двух уровнях:

1. **внутренний Python API** — для MVP и скорости разработки;
2. **контрактный message layer** — DTO и структуры сообщений, близкие к будущему RPC/gRPC.

### 4.6. GUI и CLI обязаны использовать один application layer

Недопустимо дублировать логику между Flet GUI и headless runner.  
И GUI, и CLI должны ходить в одни и те же application services.

---

## 5. Концептуальная модель предметной области

### 5.1. Основные сущности

#### 5.1.1. ClassDefinition

Описание класса разметки.

Поля:

- `code: int`
- `label: str` — краткая аббревиатура, например `tum`
- `name: str` — полное имя, например `tumor epithelium`
- `description: str`
- `color: str` — цвет для GUI/overlay
- `is_background: bool = False`

Замечания:

- `unlabeled` не должен смешиваться с обычными классами предметной области;
- `background` — обычный класс схемы;
- в будущем можно добавить `ignore`, но в MVP он не обязателен.

#### 5.1.2. ClassSchema

Набор классов, валидируемый на уникальность `code` и `label`.

Поля:

- `classes: list[ClassDefinition]`
- `version: str`
- `dataset_name: str | None`

#### 5.1.3. ImageRecord

Описание одного изображения в датасете.

Поля:

- `image_id: str`
- `image_path: Path`
- `mask_path: Path | None`
- `width: int`
- `height: int`
- `channels: int`
- `metadata: dict`

#### 5.1.4. Scribble

Каноническая единица пользовательского ввода.

Поля:

- `id: str`
- `class_code: int`
- `points_norm: list[[x, y]]` — нормированные координаты в диапазоне `[0, 1]`
- `radius_view_px: float` — радиус в пикселях текущего масштаба view
- `radius_image_px: float` — радиус после пересчёта к масштабу изображения
- `created_at: datetime`
- `finished_at: datetime`
- `source: Literal["user", "simulator", "imported"]`
- `tool: Literal["brush", "erase"]`
- `batch_id: str`
- `meta: dict`

Замечания:

- точка — частный случай скрибла с одним сегментом или одной точкой;
- хранение идёт в координатах исходного изображения, нормированных по каждой оси;
- аналитическая форма скрибла сохраняется всегда;
- растеризация выполняется отдельно.

#### 5.1.5. ScribbleBatch

Набор скриблов, нарисованных между двумя последовательными инференсами.

Поля:

- `batch_id: str`
- `scribble_ids: list[str]`
- `created_at: datetime`
- `committed: bool`

#### 5.1.6. RenderedInteraction

Растровое представление пользовательских взаимодействий.

Поля:

- `per_class_mask: dict[int, np.ndarray]`
- `aggregate_label_map: np.ndarray`
- `background_mask: np.ndarray | None`
- `unlabeled_mask: np.ndarray | None`
- `conflict_policy: str`

#### 5.1.7. PredictionState

Текущее предсказание метода.

Поля:

- `label_map: np.ndarray`
- `step_index: int`
- `created_at: datetime`
- `source_method: str`
- `artifact_path: Path | None`

#### 5.1.8. SessionAction

Событие журнала.

Поля:

- `event_id: str`
- `session_id: str`
- `timestamp: datetime`
- `event_type: str`
- `payload: dict`

#### 5.1.9. SessionStep

Зафиксированный шаг инференса.

Поля:

- `step_index: int`
- `batch_id: str`
- `scribble_ids: list[str]`
- `prediction_path: Path`
- `metrics_snapshot: dict`
- `created_at: datetime`

#### 5.1.10. SessionState

Полное текущее состояние интерактивной сессии.

Поля:

- `session_id: str`
- `image_record: ImageRecord`
- `class_schema: ClassSchema`
- `status: SessionStatus`
- `all_scribbles: list[Scribble]`
- `committed_batches: list[ScribbleBatch]`
- `current_uncommitted_scribbles: list[Scribble]`
- `current_prediction: PredictionState | None`
- `step_history: list[SessionStep]`
- `action_counters: dict`
- `meta: dict`

---

## 6. Семантика интерактивной сессии

### 6.1. Базовый цикл работы

Для этапа 1 принимается режим **B**:

1. пользователь открывает изображение;
2. рисует один или несколько скриблов;
3. скриблы попадают в текущий uncommitted batch;
4. пользователь вручную нажимает `Update`;
5. среда:
   - коммитит batch,
   - растеризует скриблы,
   - вызывает predictor,
   - сохраняет prediction,
   - считает метрики,
   - создаёт `SessionStep`,
   - пишет события в журнал;
6. пользователь либо продолжает взаимодействие, либо завершает сессию.

### 6.2. Отдельные счётчики

Нужно **раздельно** считать:

- число пользовательских действий;
- число скриблов;
- число батчей;
- число шагов инференса.

### 6.3. Завершение сессии

Сессия может завершаться любым из способов:

- пользователь явно нажал `Finalize`;
- достигнут целевой порог метрики;
- исчерпан бюджет интеракций;
- сессия была остановлена вручную.

### 6.4. Undo/redo

Поддержать оба варианта отката:

1. откат последнего скрибла;
2. откат последнего committed batch.

Рекомендация для MVP:

- в UI поддержать undo для uncommitted scribbles;
- в application layer предусмотреть возможность undo committed batch;
- пересчёт prediction после undo выполняется **вручную** после следующего `Update`.

### 6.5. Правило конфликтов

Если скрибл нового класса накладывается на старую область другого класса, действует правило:

- **побеждает последний скрибл**.

Также нужен инструмент `erase`.

---

## 7. Канонические данные и форматы представления

### 7.1. Канон для вычислений

Для инференса, метрик и хранения промежуточных предсказаний канон — **raster label map**.

### 7.2. Канон для интеракций

Для хранения, импорта и экспорта интеракций канон — **vector scribbles**.

### 7.3. Канон для экспорта

Для MVP:

- raster prediction: PNG
- scribbles: JSON
- rendered scribble masks: PNG
- replay: GIF

В будущем добавить:

- polygons / contours
- richer media exports
- COCO/GeoJSON adapters

---

## 8. Архитектура системы

### 8.1. Предлагаемая модульная структура

```text
iseggym/
  domain/
  session/
  methods/
  metrics/
  data/
  rendering/
  storage/
  app/
  gui/
  cli/
  utils/
```

### 8.2. Описание модулей

#### `iseggym.domain`
Чистые доменные сущности и типы:

- классы разметки;
- скриблы;
- состояние сессии;
- изображения;
- артефакты;
- перечисления.

Не содержит UI, IO и логики инференса.

#### `iseggym.session`
Ядро stateful-среды:

- session engine;
- команды;
- события;
- state transitions;
- replay;
- rerun;
- batch commit logic;
- undo/redo semantics.

#### `iseggym.methods`
Интерфейсы и реализации методов:

- base predictor interface;
- dummy baseline;
- superpixel propagation baseline;
- позже: CNN-assisted hybrid;
- позже: ProGIS adapter.

#### `iseggym.metrics`
Метрики качества и усилия:

- IoU;
- Dice;
- IoU@k;
- Dice@k;
- NoC@τ;
- action counters;
- summary curves.

#### `iseggym.data`
Работа с датасетами:

- filesystem loader;
- image/mask reading;
- class schema loading;
- dataset validation.

#### `iseggym.rendering`
Растеризация и визуализация:

- vector scribble -> raster mask;
- overlay generation;
- mask colorization;
- GIF builder;
- optional superpixel overlay.

#### `iseggym.storage`
Сохранение и загрузка:

- session manifest;
- actions.jsonl;
- metrics.jsonl;
- PNG artifacts;
- import/export scribbles.

#### `iseggym.app`
Application services — общий слой для GUI и CLI:

- open dataset;
- open image session;
- add scribble;
- erase;
- update prediction;
- finalize session;
- replay session;
- rerun session;
- export gif.

#### `iseggym.gui`
Flet GUI:

- dataset browser;
- image canvas;
- class selection;
- overlay controls;
- undo/redo;
- update/finalize;
- current metrics panel.

#### `iseggym.cli`
Headless runner:

- run-from-scribbles;
- replay-session;
- rerun-session;
- export-gif.

---

## 9. Session engine: команды, события, переходы состояний

### 9.1. Команды

Минимальный набор команд:

- `CreateSession`
- `LoadSession`
- `AddScribble`
- `EraseRegion` / `AddEraseStroke`
- `UndoLastScribble`
- `UndoLastBatch`
- `Redo`
- `CommitBatchAndRunInference`
- `FinalizeSession`
- `ReplaySession`
- `RerunSessionWithMethod`
- `ExportSessionArtifacts`

### 9.2. События

Минимальный набор событий:

- `SessionCreated`
- `ScribbleAdded`
- `EraseStrokeAdded`
- `ScribbleUndone`
- `BatchCommitted`
- `InferenceStarted`
- `InferenceCompleted`
- `MetricsComputed`
- `SessionFinalized`
- `SessionReplayed`
- `SessionRerunCompleted`
- `ArtifactsExported`

### 9.3. Состояния сессии

```text
Created
  -> Active
  -> Finalized
  -> Replayed
  -> RerunCompleted
```

### 9.4. Рекомендуемая модель переходов

- пользовательские действия меняют только `current_uncommitted_scribbles`;
- prediction не меняется до `CommitBatchAndRunInference`;
- после инференса создаётся новый `SessionStep`;
- metrics snapshot привязывается к committed step.

---

## 10. Контракт между средой и методом сегментации

### 10.1. Основной интерфейс

```python
class InteractiveSegmentationPredictor(Protocol):
    def name(self) -> str: ...
    def capabilities(self) -> PredictorCapabilities: ...
    def predict(self, request: InferenceRequest) -> InferenceResult: ...
```

### 10.2. PredictorCapabilities

Поля:

- `supports_multiclass: bool`
- `supports_previous_mask: bool`
- `accepts_rendered_interactions: bool`
- `accepts_vector_scribbles: bool`
- `prefers_gpu: bool`
- `deterministic: bool | None`

### 10.3. InferenceRequest

Минимальные поля:

- `image_id: str`
- `image: np.ndarray`
- `image_size: tuple[int, int]`
- `class_schema: ClassSchema`
- `vector_scribbles: list[Scribble]`
- `rendered_interactions: RenderedInteraction`
- `previous_prediction: np.ndarray | None`
- `step_index: int`
- `session_id: str`
- `meta: dict`

Рекомендация для MVP:

- **канонически** среда рендерит скриблы и отдаёт методу готовые raster channels;
- дополнительно передаёт vector scribbles;
- метод может игнорировать то, что ему не нужно.

### 10.4. InferenceResult

Минимальные поля:

- `label_map: np.ndarray`
- `method_name: str`
- `runtime_ms: float`
- `diagnostics: dict`
- `optional_state: dict | None`

### 10.5. Требования к методам

Для всех методов:

- вход и выход должны быть детерминированы при фиксированном seed;
- метод обязан вернуть метку для каждого пикселя;
- shape результата должен совпадать с shape входного изображения;
- class codes должны принадлежать class schema.

---

## 11. Superpixel subsystem

### 11.1. Почему это отдельный subsystem

Superpixels нужны не только как baseline, но и как инфраструктурная ось для будущих методов и отладки. Поэтому superpixel logic нельзя жёстко зашивать в один predictor.

### 11.2. Компоненты subsystem

- `SuperpixelMethod` — абстракция генератора superpixels;
- `SuperpixelMap` — карта superpixel ids для изображения;
- `SuperpixelCache` — сохранение/загрузка с диска;
- `SuperpixelOverlayRenderer` — визуализация;
- `SuperpixelPropagationBaseline` — baseline predictor.

### 11.3. Базовые реализации методов

Для этапа 1 достаточно:

- `SLICSuperpixelMethod`

Опционально позже:

- `WatershedSuperpixelMethod`

### 11.4. Кэширование

Superpixel maps рекомендуется кэшировать на диск, потому что:

- они зависят только от изображения и параметров метода;
- не должны пересчитываться при каждом запуске сессии;
- нужны для replay/rerun;
- полезны в headless benchmark runs.

### 11.5. Baseline predictor на superpixels

Идея первой реализации:

1. создать superpixel map;
2. определить superpixels, пересечённые скриблами;
3. присвоить им классы;
4. распространить метку на связные/похожие суперпиксели по простому правилу;
5. собрать многоклассовую label map;
6. применить правило последнего скрибла при конфликтах.

Для MVP baseline может быть сравнительно простым. Главное — чтобы он проверял архитектуру целиком.

---

## 12. Метрики

### 12.1. Базовые pixel-wise метрики

Для MVP:

- IoU per class
- mIoU
- Dice per class
- mDice

### 12.2. Метрики вида @k

Надо отдельно считать по шагам инференса:

- `IoU@k`
- `Dice@k`

Где `k` — **номер committed inference step**, а не число отдельных скриблов.

### 12.3. NoC

Минимальный вариант:

- `NoC@0.85`
- `NoC@0.90`

Определение:
минимальное число шагов инференса, после которого метрика достигает заданного порога.

### 12.4. Effort metrics

Нужно хранить и считать:

- `num_user_actions`
- `num_scribbles`
- `num_batches`
- `num_inference_steps`

Опционально позже:

- total scribble length
- total painted pixels
- time to completion
- latency per step

### 12.5. Правила расчёта

- метрики считаются только если есть GT;
- должны поддерживать многоклассовость;
- результаты сохраняются после каждого committed step;
- нужен удобный summary для построения кривых.

---

## 13. Data layer и layout датасета

### 13.1. Канонический layout для MVP

```text
dataset_root/
  images/
    img_001.png
    img_002.png
  masks/
    img_001.png
    img_002.png
  meta.json
```

### 13.2. `meta.json`

Минимальное содержимое:

```json
{
  "dataset_name": "example_dataset",
  "version": "1.0",
  "classes": [
    {
      "code": 1,
      "label": "bg",
      "name": "background",
      "description": "Background class",
      "color": "#000000",
      "is_background": true
    },
    {
      "code": 2,
      "label": "tum",
      "name": "tumor epithelium",
      "description": "Tumor tissue",
      "color": "#ff0000",
      "is_background": false
    }
  ]
}
```

### 13.3. Маски

Для MVP:

- integer PNG;
- значение пикселя = `class_code`;
- маска совпадает по размеру с изображением.

### 13.4. Изображения без GT

Поддержать в architecture, даже если не использовать широко в MVP:

- если GT нет, сессия возможна;
- метрики не считаются;
- GUI отображает annotation-only режим.

---

## 14. Формат хранения сессии

### 14.1. Почему лучше не один файл, а директория

Сессия содержит:

- метаданные;
- журнал действий;
- метрики;
- промежуточные prediction masks;
- rendered interaction masks;
- финальные экспорты;
- GIF.

Поэтому естественный формат — **директория сессии**.

### 14.2. Рекомендуемая структура

```text
session_root/
  session.json
  actions.jsonl
  metrics.jsonl
  scribbles.json
  predictions/
    step_0001.png
    step_0002.png
  rendered_scribbles/
    step_0001_class_001.png
    step_0001_class_002.png
  exports/
    final_mask.png
    replay.gif
```

### 14.3. `session.json`

Содержит:

- session id;
- image id;
- dataset id;
- class schema version;
- method name;
- timestamps;
- status;
- counters;
- config;
- paths to artifacts.

### 14.4. `actions.jsonl`

Рекомендуется именно JSONL, потому что:

- удобно append-only;
- удобно читать построчно;
- подходит для replay;
- удобен для анализа и агентной обработки;
- устойчивее к частичным сбоям записи, чем один большой JSON.

Каждая строка — одно событие.

### 14.5. `metrics.jsonl`

Одна запись на committed step:

- step index;
- method name;
- IoU/Dice summary;
- per-class metrics;
- counters;
- runtime.

### 14.6. `scribbles.json`

Экспорт всех скриблов и батчей в каноническом формате.

### 14.7. Rendered scribble masks

Сохранять:

- в debug mode обязательно;
- в MVP можно включить по конфигу и/или для committed steps;
- эти маски полезны для GIF и для проверки корректности rasterization.

---

## 15. Импорт и экспорт

### 15.1. Импорт

Для MVP:

- dataset image + png mask
- session load
- scribbles from JSON

### 15.2. Экспорт

Для MVP:

- final prediction PNG
- full scribble log JSON
- replay GIF
- session bundle directory

### 15.3. Будущие расширения

Позже добавить:

- polygons/contours export
- COCO adapters
- GeoJSON-like export
- archive export as zip
- batch export of metrics and curves

---

## 16. GUI (Flet)

### 16.1. Основные требования

В MVP GUI должен уметь:

- открыть датасет;
- показывать список изображений;
- отображать изображение;
- накладывать GT и prediction;
- рисовать скриблы;
- выбирать класс;
- изменять opacity overlays;
- делать zoom/pan;
- делать undo/redo;
- запускать `Update`;
- переходить к следующему изображению;
- сохранять сессию.

### 16.2. Состав экранов

#### Экран 1. Dataset browser

- путь к датасету;
- список изображений;
- статус наличия GT;
- статус наличия сохранённых сессий.

#### Экран 2. Session screen

- image canvas;
- class palette;
- toolbar (`brush`, `erase`, `undo`, `redo`, `update`, `finalize`);
- overlay toggles (`show GT`, `show prediction`, `show scribbles`);
- opacity slider;
- info panel:
  - current step;
  - num scribbles;
  - num inference steps;
  - current metrics.

### 16.3. Принцип GUI

GUI должен быть максимально тонким:

- не считать метрики;
- не хранить бизнес-логику сессии;
- не звать predictor напрямую.

---

## 17. CLI / headless режим

### 17.1. Почему это нужно заложить заранее

Позже потребуется:

- запускать сессии по уже готовым скриблам;
- повторно прогонять действия на сервере с GPU;
- строить batch experiments без GUI.

### 17.2. Минимальный набор команд

```bash
iseggym run-from-scribbles ...
iseggym replay-session ...
iseggym rerun-session ...
iseggym export-gif ...
```

### 17.3. Что обязано быть общим с GUI

CLI и GUI обязаны использовать один и тот же application layer.

### 17.4. `run-from-scribbles`

Этот режим должен:

- загрузить изображение;
- загрузить JSON скриблов;
- воспроизвести batches;
- запустить predictor;
- сохранить prediction и metrics.

---

## 18. Replay и rerun

### 18.1. Replay

Replay — это воспроизведение уже записанной сессии с тем же методом и теми же артефактами, без участия пользователя.

Назначение:

- визуализация;
- дебаг;
- GIF;
- демонстрация.

### 18.2. Rerun

Rerun — это повторное применение **того же журнала пользовательских действий** к другому методу.

Назначение:

- честное сравнение методов;
- построение одинакового протокола эксперимента;
- offline benchmarks.

### 18.3. Требования

Нужно поддержать оба режима архитектурно уже на этапе 1.

### 18.4. Seed control

На уровне сессии и/или метода нужно предусмотреть фиксируемый seed для воспроизводимости.

---

## 19. Визуализация и GIF

### 19.1. Минимальный GIF для MVP

В MVP достаточно GIF с кадрами по committed steps:

- исходное изображение;
- prediction overlay;
- scribble overlay.

### 19.2. Что добавлять позже

Позже можно добавлять:

- GT overlay;
- captions with metrics;
- side-by-side methods;
- heatmaps;
- speed-up rendering.

---

## 20. Headless-ready архитектура для GPU-методов

### 20.1. Решение для MVP

На этапе 1 predictor может работать **в том же процессе**, что и session engine.

### 20.2. Но интерфейс надо проектировать с зазором

Интерфейс должен быть спроектирован так, чтобы predictor позднее можно было вынести:

- в отдельный worker process;
- в gRPC service;
- на отдельную GPU-машину.

### 20.3. Практическое правило

Нельзя допускать утечек GUI-объектов в predictor interface.  
Метод должен получать только сериализуемые DTO и массивы.

---

## 21. Этапы разработки

# Этап 1. Research MVP

## 21.1. Цели этапа

- получить рабочий полигон;
- обкатать architecture skeleton;
- проверить event log / replay / rerun;
- проверить baseline predictor;
- получить первый GUI.

## 21.2. Что реализовать

### Core
- domain entities
- session state
- commands/events
- actions.jsonl / metrics.jsonl
- session storage

### Data
- dataset loader
- png image + png mask
- meta.json parser

### Rendering
- rasterization of vector scribbles
- overlay generation
- PNG saving

### Methods
- predictor interface
- render-only dummy predictor
- superpixel propagation baseline

### Metrics
- IoU, Dice, IoU@k, Dice@k, NoC, counters

### GUI
- dataset browser
- canvas with scribble drawing
- zoom/pan
- GT/prediction overlays
- update/finalize
- undo/redo
- open next image

### CLI
- replay-session
- rerun-session
- export-gif
- run-from-scribbles

## 21.3. Результат этапа

По завершении этапа 1 можно:

- вручную провести интерактивную сессию;
- сохранить её;
- открыть позже;
- воспроизвести;
- перепрогнать другим методом;
- посчитать метрики;
- получить GIF.

---

# Этап 2. Method integration and benchmark extension

## 21.4. Цели этапа

- встроить ваши методы;
- усилить benchmark value системы;
- улучшить воспроизводимость и headless runs.

## 21.5. Что реализовать

- интеграция **CNN-assisted hybrid method**
- интеграция **ProGIS**
- улучшение capability flags
- richer session comparison
- конфигурации benchmark protocols
- дополнительные графики
- улучшенные GIF/exports
- superpixel overlays
- batch benchmark runner

## 21.6. Результат этапа

iSegGym становится не только GUI-средой, но и полноценным benchmark-framework для сравнения методов.

---

# Этап 3. Расширение и подготовка к интеграции в PathScribe

## 21.7. Цели этапа

- приблизить архитектуру к будущему microservice/RPC окружению;
- добавить подготовку к реалистичной симуляции и RL.

## 21.8. Что реализовать

- RPC/gRPC-friendly adapter layer
- вынос predictor execution boundary
- richer artifact bundles
- dataset adapters
- масштабирование к большим изображениям
- подготовка feature logs для realistic simulation / RL

---

## 22. Детальный технический backlog

### 22.1. Domain layer
1. Описать `ClassDefinition`, `ClassSchema`, `ImageRecord`.
2. Описать `Scribble`, `ScribbleBatch`, `SessionStep`.
3. Описать перечисления статусов и типов событий.
4. Ввести строгие валидаторы shape/code/range.

### 22.2. Session layer
1. Реализовать `SessionState`.
2. Реализовать command handlers.
3. Реализовать event emission.
4. Реализовать commit batch logic.
5. Реализовать replay engine.
6. Реализовать rerun engine.

### 22.3. Rendering layer
1. Реализовать rasterization polyline + radius.
2. Реализовать erase stroke semantics.
3. Реализовать conflict resolution by latest scribble.
4. Реализовать per-class rendered masks.
5. Реализовать overlay composer.
6. Реализовать GIF builder.

### 22.4. Storage layer
1. Реализовать `session.json` serializer.
2. Реализовать append-only `actions.jsonl`.
3. Реализовать `metrics.jsonl`.
4. Реализовать `scribbles.json`.
5. Реализовать PNG artifact storage.
6. Реализовать session load/restore.

### 22.5. Methods layer
1. Реализовать predictor protocol.
2. Реализовать `RenderOnlyPredictor`.
3. Реализовать `SuperpixelPropagationPredictor`.
4. Добавить capability flags.
5. Позже: adapters для CNN-assisted и ProGIS.

### 22.6. Metrics layer
1. Реализовать per-class IoU.
2. Реализовать per-class Dice.
3. Реализовать mIoU/mDice.
4. Реализовать `IoU@k`, `Dice@k`.
5. Реализовать `NoC@0.85`, `NoC@0.90`.
6. Реализовать effort counters.

### 22.7. Data layer
1. Реализовать filesystem dataset loader.
2. Реализовать meta.json parser.
3. Реализовать image/mask matching.
4. Реализовать dataset validator.
5. Поддержать image without GT.

### 22.8. GUI layer
1. Реализовать browser screen.
2. Реализовать drawing canvas.
3. Реализовать class palette.
4. Реализовать opacity toggles.
5. Реализовать GT/prediction overlays.
6. Реализовать update/finalize buttons.
7. Реализовать undo/redo.
8. Реализовать session save/load.

### 22.9. CLI layer
1. Реализовать `run-from-scribbles`.
2. Реализовать `replay-session`.
3. Реализовать `rerun-session`.
4. Реализовать `export-gif`.

---

## 23. Backlog задач в формате, удобном для агентной разработки

### Epic A. Domain and contracts
- A1. Define core dataclasses and enums.
- A2. Define predictor request/response DTO.
- A3. Define JSON schemas for session artifacts.
- A4. Define validation rules and invariants.

### Epic B. Session engine
- B1. Implement session lifecycle.
- B2. Implement event store.
- B3. Implement batch commit workflow.
- B4. Implement undo/redo semantics.
- B5. Implement replay.
- B6. Implement rerun with method substitution.

### Epic C. Rasterization and rendering
- C1. Polyline-to-mask rasterization.
- C2. Erase behavior.
- C3. Latest-scribble conflict policy.
- C4. Overlay renderer.
- C5. GIF exporter.

### Epic D. Predictors
- D1. Predictor protocol.
- D2. Render-only predictor.
- D3. Superpixel generator and cache.
- D4. Superpixel propagation predictor.
- D5. CNN-assisted adapter.
- D6. ProGIS adapter.

### Epic E. Metrics
- E1. IoU/Dice calculators.
- E2. NoC calculators.
- E3. Session curve summaries.
- E4. Metrics persistence.

### Epic F. Data and storage
- F1. Dataset loader.
- F2. Session bundle format.
- F3. Import/export scribbles.
- F4. Restore state from saved session.

### Epic G. GUI
- G1. Dataset browser.
- G2. Canvas interaction.
- G3. Overlays and opacity.
- G4. Toolbar and state sync.
- G5. Session screen metrics panel.

### Epic H. CLI
- H1. Run from scribbles.
- H2. Replay session.
- H3. Rerun session.
- H4. Batch runner.

---

## 24. Инварианты системы

Система должна гарантировать следующее:

1. Все координаты скриблов хранятся в нормированном виде `[0, 1]`.
2. Все предсказания совпадают по размеру с изображением.
3. Все class codes валидны относительно `ClassSchema`.
4. Все committed steps воспроизводимы через журнал.
5. Prediction не меняется без явного шага инференса.
6. Undo/redo не нарушают целостность журнала событий.
7. GUI и CLI используют один application layer.
8. Predictor не зависит от GUI.
9. Session bundle достаточно полон для replay.
10. Rerun одной и той же сессии на другом методе не требует ручного ввода пользователя.

---

## 25. Нефункциональные требования

### 25.1. Модульность
Каждый слой должен быть заменяемым без переписывания остальных.

### 25.2. Воспроизводимость
Все ключевые действия и результаты должны быть сериализуемы.

### 25.3. Кроссплатформенность
MVP должен запускаться на macOS и Windows.

### 25.4. Headless readiness
Архитектура должна позволять запуск без GUI.

### 25.5. GPU readiness
DL-методы должны интегрироваться без архитектурного слома.

---

## 26. Тестовая стратегия

### 26.1. Unit tests
- domain validation
- rasterization
- metrics
- session transitions
- serializers

### 26.2. Integration tests
- open dataset -> create session -> add scribbles -> update -> save session
- replay from saved session
- rerun with another predictor
- export gif

### 26.3. Golden tests
- фиксированные входы и ожидаемые PNG/JSON artifacts
- фиксированные curve summaries

### 26.4. GUI smoke tests
- open dataset
- draw scribble
- update prediction
- overlay toggle
- save session

---

## 27. Основные риски и способы снижения

### Риск 1. GUI быстро разрастается и тащит в себя бизнес-логику
**Снижение:** жёстко держать application layer и session engine отдельно.

### Риск 2. Формат сессии окажется неполным для replay/rerun
**Снижение:** проектировать event log как основной артефакт, а не как вспомогательный.

### Риск 3. Superpixel baseline окажется слишком примитивным
**Снижение:** использовать его как архитектурный baseline, а не как научный ориентир.

### Риск 4. В дальнейшем будет сложно вынести predictor в отдельный процесс
**Снижение:** уже сейчас использовать чистые DTO и сериализуемые структуры.

### Риск 5. Многоклассовость осложнит rasterization и conflict handling
**Снижение:** зафиксировать правила conflict resolution и семантику background/unlabeled заранее.

### Риск 6. Большие изображения позднее потребуют переработки data layer
**Снижение:** не смешивать image access, session logic и predictor interface.

---

## 28. Acceptance criteria по этапам

### Для этапа 1
- можно открыть датасет из папки;
- можно рисовать многоклассовые скриблы;
- можно использовать erase;
- можно вручную запускать `Update`;
- prediction сохраняется после каждого шага в PNG;
- журнал действий сохраняется;
- replay работает;
- rerun на другом методе работает;
- базовые метрики считаются;
- минимальный GIF экспортируется;
- headless запуск по scribbles JSON работает.

### Для этапа 2
- интегрированы не менее двух исследовательских методов;
- можно сравнить методы на одинаковом журнале действий;
- появляются richer summaries и benchmark reports.

### Для этапа 3
- архитектура готова к выносу predictor execution в RPC/process boundary;
- артефакты и логи пригодны для realistic interaction simulation / RL.

---

## 29. Рекомендуемый порядок реализации

### Порядок 1 — технически оптимальный
1. domain
2. session engine
3. storage
4. rendering
5. metrics
6. predictors
7. CLI
8. GUI

### Порядок 2 — психологически удобный, но рискованнее
1. GUI canvas
2. predictor
3. session storage
4. replay
5. metrics

**Рекомендация:** использовать **порядок 1**.

---

## 30. Предлагаемая дорожная карта по deliverables

### Milestone M1
- domain model
- session model
- dataset loader
- rasterizer
- session serialization

### Milestone M2
- dummy predictor
- superpixel baseline
- metrics
- CLI replay/rerun

### Milestone M3
- Flet GUI MVP
- overlays
- session save/load
- gif export

### Milestone M4
- CNN-assisted integration
- benchmark summaries
- improved visualizations

### Milestone M5
- ProGIS integration
- headless benchmark workflows
- prep for PathScribe integration

---

## 31. Дополнительные замечания по будущему развитию

### 31.1. RL и realistic simulation
Журнал действий пользователя нужно сразу проектировать так, чтобы позднее из него можно было извлекать:

- последовательности действий;
- время между действиями;
- контекст изменений маски;
- классы и области коррекции;
- локальные траектории редактирования.

### 31.2. Переход к большим изображениям
Нынешняя архитектура должна быть готова к будущему расширению через abstraction layer для image access, но без преждевременного усложнения MVP.

### 31.3. Интеграция с PathScribe
Для будущей интеграции особенно важны:

- чистые DTO;
- отсутствие GUI-зависимостей в predictor API;
- session engine как независимый backend-компонент;
- replay/rerun как headless workflows.

---

## 32. Краткое итоговое решение

Итоговая архитектурная ставка проекта:

- **iSegGym** строится вокруг **stateful Session Engine**;
- GUI и CLI — это только клиенты к общему application layer;
- методы интерактивной сегментации подключаются как **stateless predictors** через единый интерфейс;
- пользовательские действия хранятся как **vector scribbles + event log**;
- вычислительное состояние хранится как **raster masks + session steps**;
- воспроизводимость обеспечивается через **actions.jsonl + saved PNG states**;
- этап 1 ориентирован на локальный Research MVP;
- этап 2 — на интеграцию собственных методов и benchmark scenarios;
- этап 3 — на headless scaling и подготовку к PathScribe / RL / RPC.

---

## 33. Приложение A. Рекомендуемые JSON-ключи

### Пример `Scribble`
```json
{
  "id": "scr_000123",
  "class_code": 2,
  "points_norm": [[0.15, 0.22], [0.18, 0.25], [0.21, 0.28]],
  "radius_view_px": 5.0,
  "radius_image_px": 8.0,
  "created_at": "2026-04-08T10:15:00Z",
  "finished_at": "2026-04-08T10:15:01Z",
  "source": "user",
  "tool": "brush",
  "batch_id": "batch_0007",
  "meta": {}
}
```

### Пример строки в `actions.jsonl`
```json
{
  "event_id": "evt_00045",
  "session_id": "sess_001",
  "timestamp": "2026-04-08T10:15:05Z",
  "event_type": "InferenceCompleted",
  "payload": {
    "step_index": 3,
    "batch_id": "batch_0007",
    "prediction_path": "predictions/step_0003.png",
    "runtime_ms": 145.7
  }
}
```

### Пример строки в `metrics.jsonl`
```json
{
  "session_id": "sess_001",
  "step_index": 3,
  "method_name": "superpixel_propagation",
  "num_scribbles": 6,
  "num_batches": 3,
  "num_inference_steps": 3,
  "miou": 0.812,
  "mdice": 0.889,
  "per_class": {
    "tum": {"iou": 0.79, "dice": 0.88},
    "bg": {"iou": 0.83, "dice": 0.90}
  }
}
```

---

## 34. Приложение B. Источники архитектурных идей, учтённые при проектировании

При формировании плана были учтены следующие содержательные идеи из ваших материалов:

- представление скриблов как геометрических объектов с параметрами;
- отдельные сущности для superpixel-структур;
- хранение разметки как 2D массива целочисленных классов;
- итеративная логика улучшения prediction после новых скриблов;
- использование предыдущей prediction mask как части состояния следующего шага;
- пригодность гибкой backend-архитектуры для будущего удалённого GPU-execution;
- значимость superpixel-уровня как отдельной оси проектирования.
