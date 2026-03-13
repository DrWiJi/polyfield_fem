# GUI и модель данных: верхнеуровневый обзор

Этот документ помогает быстро сориентироваться в текущем GUI-приложении и структуре данных проекта.

## 1) Где находится код

- GUI-черновик: `fe_ui_draft.py`
- Модель данных проекта: `project_model.py`
- Основная симуляционная модель OpenCL: `diaphragm_opencl.py`

## 2) Назначение GUI

`fe_ui_draft.py` - desktop-интерфейс для подготовки проекта:

- импорт мешей;
- разметка мешей (роль, материал, свойства);
- настройка параметров симуляции;
- сохранение/загрузка состояния проекта;
- просмотр мешей во viewport.

GUI пока не запускает полноценный симуляционный pipeline. Основная цель текущего этапа - подготовка и управление данными проекта.

## 3) Структура окна GUI

## Центральная область

- Viewport на `pyvistaqt` (`QtInteractor`) при наличии зависимостей.
- Fallback на пустой placeholder, если `pyvistaqt` не установлен.
- Рендерятся импортированные меши:
  - обычные - серые;
  - выбранный - желтоватый.

## Левая панель: `Mesh List`

- поиск по имени/роли/материалу;
- список мешей из `Project.source_data.meshes`;
- кнопки:
  - `Add` (добавить пустой mesh-entity),
  - `Remove` (удалить выбранный),
  - `Isolate` (UI-заготовка).

## Правая панель: `Mesh Parameter Editor`

Вкладки:

- `Identity`: имя, роль, видимость;
- `Material`: пресет материала и базовые параметры;
- `Membrane`: толщина, преднатяжение, группа фиксации;
- `Boundary`: названия групп и заметки.

Если меш не выбран, редактор отключается.

## Нижняя панель: `Simulation`

- базовые параметры: dt, duration, air coupling gain, air grid step;
- параметры возбуждения: shape (impulse/uniform/sine/square/chirp), amplitude, freq;
- кнопки `Run Simulation`, `Stop`, `Export Case`;
- консоль вывода симуляции.

`Run Simulation` запускает debug test run через `diaphragm_opencl` с фиксированными аргументами (--no-plot, --dt 1e-7, --duration 0.001, --force-shape impulse, --air-inject-mode reduce, --debug и др.). Полный pipeline с проектной топологией пока не подключён.

## Верхнее меню `File`

- `New Project`
- `Import Mesh...`
- `Load...`
- `Save`
- `Save As...`
- `Exit`

Перед `New`, `Load` и `Exit` показывается вопрос о сохранении, если есть несохраненные изменения.

## 4) Импорт мешей

Используется `trimesh`:

- загрузка одного или нескольких файлов;
- поддержка одиночного меша и сцен с несколькими геометриями;
- для каждой геометрии создается `MeshEntity` в проекте;
- в `properties` пишутся метаданные:
  - `vertex_count`,
  - `face_count`,
  - `is_watertight`,
  - `trimesh_geom_name` (для повторной загрузки из сцены).

## 5) Модель данных `Project`

Центральная сущность: `Project`.

Ключевые поля:

- `model_version` - версия схемы данных;
- `name` - имя проекта;
- `created_at`, `updated_at` - таймстемпы;
- `source_data` - исходные данные симуляции;
- `simulation_runs` - список записей запусков (не ограничен по размеру).

### `source_data`

`SimulationSourceData` содержит:

- `meshes: list[MeshEntity]`
- `simulation_settings: SimulationSettings`
- `material_library`
- `metadata`

### `MeshEntity`

Основные поля:

- `mesh_id`
- `name`
- `source_path`
- `role` (`solid | membrane | boundary`)
- `material_key`
- `visible`
- `transform`
- `properties` (произвольные доп. параметры)
- `boundary_groups`

### `SimulationSettings`

Хранит настраиваемые параметры симуляции:

- `dt`, `duration` — шаг и длительность;
- `force_shape` — форма возбуждения (`impulse`|`uniform`|`sine`|`square`|`chirp`);
- `force_amplitude_pa`, `force_offset_pa`, `force_freq_hz`, `force_freq_end_hz`, `force_phase_deg`;
- `pre_tension_n_per_m`;
- `air_coupling_gain`, `air_grid_step_mm`, `air_boundary_damping`, `air_bulk_damping`, `air_pressure_clip_pa`;
- `notes`.

В GUI-форме (`fe_ui_draft.py`) отображаются не все поля: dt, duration, air_coupling_gain, air_grid_step_mm, force_shape, force_amplitude, force_freq. Остальные берутся из значений по умолчанию при запуске debug test run.

### `simulation_runs`

`SimulationRunRecord` для журналирования запусков:

- идентификатор/время;
- статус (`created/running/completed/failed/cancelled`);
- snapshot параметров;
- метрики;
- ссылки на артефакты;
- лог и сообщение ошибки.

## 6) Сериализация и файлы проекта

`Project` поддерживает:

- `to_dict()`, `to_json()`;
- `save_json(path)`;
- `from_dict()`, `from_json()`, `load_json(path)`.

Формат хранения: JSON.

## 7) Версионирование схемы и миграции

В `project_model.py` предусмотрены:

- `PROJECT_MODEL_VERSION`;
- `migrate_project_dict(...)`;
- миграции вида `_migrate_vX_to_vY(...)`.

Это точка расширения для будущих изменений структуры `Project`.

## 8) Логика dirty-state в GUI

- При ручном изменении полей GUI ставится флаг несохраненных изменений.
- Во время программной загрузки данных в контролы используется защита (`_is_loading_ui`) от ложных dirty-событий.
- В title окна добавляется `*`, если проект изменен.

## 9) Ограничения текущей версии

- Viewport: при наличии pyvistaqt — рендер мешей; picking по actor не реализован (выбор через список; fallback — mock-click).
- `Run Simulation` запускает тестовый pipeline `diaphragm_opencl` с фиксированной топологией (двухслойная мембрана+вата), а не проектную топологию из мешей.
- `Export Case` и `Stop` — заготовки.
- `Isolate` в списке мешей — заготовка UI.
- Нет отдельного менеджера undo/redo.

## 10) Ближайшие шаги развития

- Подключить actor-picking во viewport и двустороннюю синхронизацию выбора.
- Связать `Run` с реальным запуском симуляции и записью `SimulationRunRecord`.
- Добавить слой импорта/конвертации в FE-сетку (например, через `gmsh` API).
- Добавить валидацию проекта перед запуском.
