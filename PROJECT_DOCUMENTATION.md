# Документация проекта `diaphragm_opencl`

## 1) Назначение проекта

Проект моделирует динамику диафрагмы/мембраны на КЭ-сетке с OpenCL-вычислением, где:

- каждый КЭ имеет 6 степеней свободы (`x, y, z, rx, ry, rz`);
- вычисления сил и интегрирование выполняются на OpenCL (GPU);
- интегратор: RK2 (два kernel-stage за шаг времени);
- Python-слой отвечает за инициализацию, загрузку данных на устройство, управление шагами и постобработку.
- поддерживается как базовая прямоугольная топология (`nx x ny`), так и пользовательская топология через API.

Ключевые файлы:

- `diaphragm_opencl.py` - Python-модель, CLI, визуализация, валидация.
- `diaphragm_opencl_kernel.cl` - OpenCL-ядро (сила + интегрирование RK2).

## 2) Архитектура и поток данных

### Внешний путь (пользователь -> симуляция)

1. Запуск из CLI (`py diaphragm_opencl.py ...`).
2. Парсинг аргументов в `_parse_cli_args(...)`.
3. Создание профиля давления `pressure` по выбранной форме.
4. Создание объекта `PlanarDiaphragmOpenCL(...)`.
5. Вызов `model.simulate(pressure, dt, ...)`.
6. Опционально: графики/карта смещений/анимация.

### Внутренний путь (Python -> OpenCL -> Python)

На каждом шаге `step(...)`:

1. Python собирает `force_external` (давление -> сила по `z`).
2. Копирует массивы в OpenCL-буферы:
   - `position`, `velocity`, `force_external`,
   - `boundary_mask_elements`, `element_size_xyz`,
   - `material_index`, `material_props`,
   - `neighbors`, `laws`.
3. Обновляет 3D поле давления воздуха:
   - `air_step_3d` (волновое уравнение + демпфирующий слой + радиационное граничное условие открытого поля),
   - инжекция КЭ -> air:
     - `air_inject_membrane_velocity` (режим `reduce`, запись в промежуточный per-element буфер),
     - `air_inject_membrane_velocity_direct` (режим `direct`, прямая запись в `p_next`),
   - `add_air_pressure_to_force_external` (давление поля -> добавка к `force_external` по формуле `F = (p_lo - p_hi) * A * n`; реакция воздуха направлена от области с большим давлением).
   - При режиме `reduce` после kernel инжекции выполняется reduce-шаг в Python с аккумуляцией вкладов в ячейки air-grid.
4. Упаковывает структуру `Params` через `_pack_params(...)` (байтовый блок).
5. Запускает `diaphragm_rk2_stage1`.
6. Запускает `diaphragm_rk2_stage2`.
7. Копирует `position/velocity` обратно в Python.
8. Записывает историю `u_center(t)` и (опционально) историю смещений сенсорных КЭ (`MAT_SENSOR`) для визуализации.

## 3) Логическая модель (физика и численный шаг)

## 3.1 Сетка и топология

- Базовая сетка прямоугольная: `nx x ny`.
- Для механической модели: `n_layers_total = 1`, базовая топология генерируется helper-методом `generate_planar_membrane_topology(...)`.
- Плоскость построения мембраны задается аргументом `plane` (`xy` / `xz` / `yz`), также задаются толщина слоя и размеры по осям плоскости.
- Для сгенерированной плоской мембраны всегда выставляются граничные условия закрепления по краям.
- Доступен внешний API `set_custom_topology(...)` для задания:
  - координат всех КЭ,
  - размеров всех КЭ,
  - произвольной таблицы соседей,
  - материалов и граничной фиксации.
- Доступен API объединения двух топологий с приоритетом главной в зоне пересечения:
  - `merge_topologies(...)` (возвращает объединенную топологию),
  - `set_merged_topologies(...)` (объединяет и сразу применяет к модели).
- Для акустики: отдельная 3D декартова сетка `nx_air x ny_air x nz_air` (давление `p`), с отступом от мембраны по всем направлениям.
- Для связи КЭ с воздухом для каждого КЭ строятся:
  - `air_map_lo/air_map_hi` (пара ячеек в направлении `-n/+n`),
  - `air_elem_normal` (текущее направление coupling),
  - `air_elem_area` (эффективная площадь для силы давления).
- В актуальной версии `n` и `A_eff` вычисляются динамически по направлению движения КЭ (`velocity_delta` -> fallback `velocity`).

## 3.2 Типы сил в ядре

В `add_force_elastic(...)` рассматриваются связи по матрице `laws`:

- `LAW_SOLID_SPRING`:
  - нелинейная пружина с анизотропией по X/Y;
  - длина пружины и деформация считаются по вектору центр–центр (`center_len`, `rest_len`), а не face-to-face (важно для корректной упругой связи в покое);
  - преднатяжение `pre_tension * edge_length` применяется только для материалов `MAT_MEMBRANE`/`MAT_SENSOR` и только по направлениям `±X, ±Y`;
  - добавлено вязкое затухание связи через `eta_visc` материала.
- Иначе (границы):
  - вязкое сопротивление по нормали грани.
  - абсолютный атмосферный терм на свободной грани удален; давление задается через air-field.

Важно: при `E_parallel/E_perp ~ 0` fallback на мембранную жесткость отключен; жесткость связи становится 0, чтобы не получать численный разлет при малой массе.

## 3.3 Интегрирование

- RK2:
  - stage1 вычисляет промежуточные `position_mid`, `velocity_mid`;
  - stage2 вычисляет итог `x_new`, `v_new`.
- Интегрируются только поступательные DOF (`0..2`), моменты (`3..5`) принудительно обнуляются (`force_moments_zero`).
- Для границы (`boundary_mask_elements == 1`) обновление блокируется.

## 3.4 Масса и защита от деления на ноль

- Масса и инерции считаются из плотности и размеров КЭ (`get_mass_safe`).
- Есть нижние пороги (`+1e-18`) для устойчивости делений.

## 4) Инициализация `PlanarDiaphragmOpenCL`

### Геометрия и сетка

- `width_mm`, `height_mm`, `thickness_mm`
- `nx`, `ny`

Перевод в SI делается сразу (мм -> м).

### Материал/упругость

- `density_kg_m3`
- `E_parallel_gpa`, `E_perp_gpa`
- `poisson`
- `pre_tension_N_per_m`
- Параметры нелинейности:
  - `use_nonlinear_stiffness`
  - `stiffness_transition_center`
  - `stiffness_transition_width`
  - `stiffness_ratio`
  - `k_soft`, `k_stiff`, `strain_transition`, `strain_width`

### Воздух/сопротивление

- `rho_air`, `mu_air`, `Cd`
- Параметры 3D поля воздуха:
  - `air_sound_speed_m_s`
  - `air_padding_mm` (если `None`, используется рациональный отступ ~10 мм)
  - `air_grid_step_mm` (фиксированный шаг сетки воздуха; если `None`, шаг берется из размеров КЭ)
  - `air_boundary_damping`
  - `air_coupling_gain`
- `air_inject_mode` (`reduce`/`direct`)
  - `air_bulk_damping`
  - `air_pressure_clip_pa`

### OpenCL/отладка

- `platform_index`, `device_index`
- `kernel_debug`

## 5) Данные и их форматы

## 5.1 Главные массивы Python-слоя

- `position`: shape `[n_dof]`, `float64`.
- `velocity`: shape `[n_dof]`, `float64`.
- `force_external`: shape `[n_dof]`, `float64`.
- `velocity_delta`: shape `[n_dof]`, `float64`.
- `air_pressure_prev/curr/next`: shape `[n_air_cells]`, `float64`.
- `element_size_xyz`: shape `[n_elements, 3]`, `float64`.
- `material_index`: shape `[n_elements]`, `uint8`.
- `material_props`: shape `[n_materials, 7]`, `float64`.
- `neighbors`: shape `[n_elements, 6]`, `int32` (`-1` = нет соседа).
- `laws`: shape `[n_materials, n_materials]`, `uint8`.
- `boundary_mask_elements`: shape `[n_elements]`, `int32` (`0/1`).
- `air_map_lo/air_map_hi`: shape `[n_elements]`, `int32` (маппинг КЭ -> ячейки воздуха в направлении `-n/+n`).
- `air_elem_normal`: shape `[n_elements, 3]`, `float64`.
- `air_elem_area`: shape `[n_elements]`, `float64`.

## 5.2 Формат строки `material_props`

`[density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain]`

Встроенная стартовая библиотека материалов (индексы):

- `0` - `membrane` (параметры из конструктора модели)
- `1` - `foam_ve3015` (ориентировочные параметры memory-foam)
- `2` - `sheepskin_leather` (ориентировочные параметры овечьей кожи)
- `3` - `human_ear_avg` (усреднённые параметры уха человека без разделения тканей)
- `4` - `sensor` (временная модель микрофона; параметры скопированы из ПЭТ-мембраны)
- `5` - `cotton_wool` (хлопковая вата, приближённые эффективные параметры)

Примечание по совместимости:

- `set_material_library(...)` принимает форматы:
  - `[n, 7]` (актуальный полный формат),
  - `[n, 6]` (авто-добавляется `coupling_gain = 1`),
  - `[n, 5]` (авто-добавляются `eta_visc = 0`, `coupling_gain = 1`).

## 5.3 Параметры ядра `Params`

Упаковываются строго по `_PARAMS_FORMAT` и должны совпадать с C-struct в `diaphragm_opencl_kernel.cl`.
Критичен alignment/padding после `int use_nonlinear_stiffness`.

## 6) Логика приложения внешней силы

Функция `_build_force_external(pressure_pa)`:

- Находит элементы, лежащие в базовой плоскости `Z=0` (маска `_z0_elements_mask`).
- Поддерживает вход:
  - скаляр,
  - массив длины `1`,
  - массив длины `n_z0_elements`,
  - массив длины `n_elements` (берутся только `Z=0` индексы).
- Сила прикладывается как `Fz = p * area` для нефиксированных элементов.

Это означает, что возбуждение не завязано на конкретный material ID.

## 7) Публичные методы модели

- `simulate(pressure_profile, dt, record_history=False, check_air_resistance=False, validate_steps=True, reset_state=True, show_progress=True, progress_every_pct=5.0)`
  - основной цикл шагов.
- `step(dt, pressure_pa, ...)`
  - один временной шаг (2 OpenCL-ядра).
- `set_material_library(material_props)`
  - полная замена библиотеки материалов.
- `set_element_material_index(material_index)`
  - назначение материала каждому КЭ.
- `set_material_laws(laws)`
  - матрица законов взаимодействия материалов.
- `set_neighbors_topology(neighbors)`
  - ручная топология соседей.
- `set_boundary_mask(boundary_mask_elements)`
  - ручная маска фиксации.
- `set_custom_topology(element_position_xyz, element_size_xyz, neighbors, ...)`
  - полная внешняя установка топологии всех КЭ (координаты/размеры/соседи/материалы/границы),
  - опционально можно передать `visual_shape=(ny, nx)` для 2D-визуализации,
  - в актуальной версии `ny*nx` должно совпадать с числом `MAT_SENSOR` (а не с `n_elements`),
  - опционально сразу перестроить поле воздуха с нужными параметрами.
- `generate_planar_membrane_topology(plane, thickness_m, size_u_m, size_v_m)`
  - генерация однослойной плоской мембраны на регулярной сетке `nx x ny`,
  - автоматическое закрепление элементов по краям.
- `rebuild_air_field(air_grid_step_mm=None, air_padding_mm=None)`
  - перестройка 3D поля воздуха по текущей геометрии КЭ.
- `merge_topologies(primary, secondary, primary_is_main=True, overlap_tol_m=...)`
  - объединение двух топологий КЭ с разрешением пересечений в пользу главной топологии.
- `set_merged_topologies(...)`
  - объединение двух топологий + немедленное применение результата в текущую модель.
- `get_numerical_natural_frequency(...)`
  - численная оценка собственной частоты через FFT.
- `plot_time_and_spectrum(...)`, `plot_displacement_map(...)`, `animate(...)`
  - визуализация.

## 8) CLI-аргументы

### Общие

- `--no-plot` - не строить графики.
- `--debug` - включить режим kernel debug (`ENABLE_DEBUG=1`).
- `--validate` - запуск валидации собственных частот.
- `--pre-tension` (`--pre_tension`) - преднатяжение, Н/м.
- `--dt` - шаг времени, с.
- `--duration` - длительность моделирования, с.
- `--air-grid-step-mm` (`--air_grid_step_mm`) - шаг сетки поля воздуха, мм.
- `--air-inject-mode` - режим инжекции КЭ -> air (`reduce`/`direct`).
- `--uniform` - legacy-флаг (принудительно включает форму `uniform`).

### Форма внешней силы

- `--force-shape`:
  - `impulse` (по умолчанию),
  - `uniform`,
  - `sine`,
  - `square`,
  - `chirp`.
- `--force-amplitude` - амплитуда, Па.
- `--force-offset` - постоянная составляющая, Па.
- `--force-freq` - частота `f0`, Гц (или старт для chirp).
- `--force-freq-end` - конечная частота `f1` для chirp.
- `--force-phase-deg` - начальная фаза, градусы.

## Формирование `pressure` в CLI

- `impulse`: на первом шаге `offset + amplitude`, далее `offset`.
- `uniform`: константа `offset + amplitude`.
- `sine`: `offset + amplitude * sin(2*pi*f0*t + phase)`.
- `square`: `offset + amplitude * sign(sin(...))`.
- `chirp`: линейный свип частоты от `f0` до `f1`.

## 9) Валидация и критерии качества

Глобальные пороги в Python:

- `MAX_UZ_UM_OK = 500.0` - критерий "взрыв" по max|uz|.
- `MIN_FREQ_HZ_OK = 1.0` - слишком низкий пик -> "0 Гц".
- `MIN_PEAK_PROMINENCE = 2.0` - выделенность пика над фоном.

Метрика "пик/фон":

- `_spectrum_peak_prominence = max(spec) / mean(spec)` в рабочем частотном диапазоне.

При `kernel_debug=True` в kernel stage2 есть:

- запись детального debug-трейса в буфер;
- проверка NaN/Inf и запись первого проблемного элемента (`first_bad_elem`).

В трассировке упругости dir0 выводится `center_len` (длина центр–центр), а не `link_len` (face-to-face).

## 9.1 Визуализация и поведение при несовместимой топологии

- В модели есть флаг `visualization_enabled`.
- Если топология не соответствует 2D представлению (например, не задан `visual_shape` для custom-топологии), визуализация автоматически отключается.
- Для custom-топологии 2D-визуализация строится по подмножеству `MAT_SENSOR`, заданному через `visual_shape`.
- Методы визуализации в этом случае работают в "мягком" режиме:
  - не выбрасывают исключение по этой причине;
  - печатают понятное сообщение и завершаются (`return` / `return None`).

## 9.2 Акустические границы и открытое поле

- В `air_step_3d` применяется демпфирование объёма (`bulk_damping`) и демпфирующий слой у границ (`boundary_damping`).
- Дополнительно на внешней границе используется радиационное условие Sommerfeld в устойчивой upwind-форме, чтобы волна уходила из расчетной области (режим открытого поля), а не зеркально отражалась.
- Инжекция в воздух ограничивается по амплитуде через `pressure_clip` (ограничение на `p_drive` в ядре связи), что повышает устойчивость двусторонней связи.

### Связь КЭ ↔ воздух (направления)

- **Инжекция (мембрана → воздух):** при `v_n > 0` (движение в +n) в `idx_hi` добавляется `+p_drive`, в `idx_lo` — `-p_drive` (сжатие в +n, разрежение в -n).
- **Сила (воздух → мембрана):** `F = (p_lo - p_hi) * A * n` — реакция воздуха направлена от области с большим давлением (при `p_hi > p_lo` сила в -n, тормозит движение).

## 10) Единицы измерения и соглашения

- Геометрия во входе конструктора: мм; внутри модели: м.
- Давление: Па.
- Плотность: кг/м^3.
- Модуль Юнга: в конструкторе ГПа, внутри Па.
- Время: секунды.
- История смещения центра (`history_disp_center`) в метрах.

## 11) Ограничения и текущие нюансы

- Вращательные DOF в интегрировании не развиваются (моменты обнуляются).
- Модель сейчас плоская по слоям (`n_layers_total=1`), без генерации воздушных объемных КЭ.
- Проверка NaN/Inf на шаге ядра доступна только в debug-сборке (`--debug`).
- Все элементы по умолчанию инициализируются как сенсорные (`MAT_SENSOR`) с параметрами ПЭТ-мембраны.
- API `set_custom_topology(...)` не меняет количество КЭ (`n_elements` остается фиксированным после `__init__`), а переопределяет геометрию/связи существующих элементов.
- API `set_merged_topologies(...)` также требует, чтобы итоговое число КЭ после merge совпадало с `self.n_elements`.
- Связь КЭ->воздух использует приращение скорости (`velocity_delta`) и требует подбора `air_coupling_gain` под шаг времени и масштаб задачи.
- `simulate(reset_state=True)` в текущей версии обнуляет `position`; для custom-топологий обычно нужен `reset_state=False`, чтобы не терять заданную геометрию.
- В `__main__` после создания модели применяется тестовая двухслойная топология (`MAT_SENSOR` + `MAT_COTTON_WOOL`) через `set_custom_topology(...)`.

## 12) Минимальные примеры запуска

- Импульс (по умолчанию):
  - `py -3 diaphragm_opencl.py --dt 1e-6 --duration 0.05`
- Синус:
  - `py -3 diaphragm_opencl.py --force-shape sine --force-amplitude 3 --force-freq 800 --duration 0.05`
- Свип:
  - `py -3 diaphragm_opencl.py --force-shape chirp --force-amplitude 1 --force-freq 200 --force-freq-end 5000`
- Валидация:
  - `py -3 diaphragm_opencl.py --validate --pre-tension 10`
- Заданный шаг air-grid:
  - `py -3 diaphragm_opencl.py --air-grid-step-mm 0.25 --dt 1e-6 --duration 0.01`
