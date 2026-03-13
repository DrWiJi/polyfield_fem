# Polyfield

Мультидоменная симуляция полей: механика, акустика, с возможностью расширения на магнитный и электрический домены.

## Возможности

- **Механика:** конечно-элементная модель мембраны/диафрагмы с нелинейной упругостью, преднатяжением и граничными условиями
- **Акустика:** 3D поле давления воздуха со связанной мембраной, радиационные границы (Sommerfeld), демпфирование
- **OpenCL:** вычисления на GPU, интегратор RK2
- **Расширяемость:** структура готова к добавлению магнитного и электрического доменов

## Требования

- Python 3.10+
- PyOpenCL
- OpenCL 1.2+ с поддержкой `cl_khr_fp64` (double)
- NumPy, Matplotlib (для визуализации)

## Установка

```bash
pip install pyopencl numpy matplotlib
```

## Быстрый старт

```bash
# Импульс (по умолчанию)
py diaphragm_opencl.py --dt 1e-6 --duration 0.05

# Синусоидальное возбуждение
py diaphragm_opencl.py --force-shape sine --force-amplitude 3 --force-freq 800 --duration 0.05

# Валидация собственных частот
py diaphragm_opencl.py --validate --pre-tension 10

# Режим отладки
py diaphragm_opencl.py --debug --dt 1e-7 --duration 0.001 --force-shape impulse
```

## Структура проекта

| Файл | Назначение |
|------|------------|
| `diaphragm_opencl.py` | Python-модель, CLI, визуализация |
| `diaphragm_opencl_kernel.cl` | OpenCL-ядро (силы, RK2, воздух) |
| `analytical_diaphragm.py` | Аналитические решения для валидации |
| `fe_ui/` | GUI для подготовки проектов (модульный) |
| `project_model.py` | Модель данных проекта |
| `PROJECT_DOCUMENTATION.md` | Подробная техническая документация |

## Документация

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) — архитектура, физика, CLI, API
- [fe_ui/FE_UI_STRUCTURE.md](fe_ui/FE_UI_STRUCTURE.md) — структура GUI

## CLI-аргументы

- `--force-shape` — форма давления: `impulse`, `uniform`, `sine`, `square`, `chirp`
- `--force-amplitude`, `--force-freq`, `--force-freq-end` — параметры возбуждения
- `--dt`, `--duration` — шаг времени и длительность
- `--pre-tension` — преднатяжение, Н/м
- `--air-inject-mode` — режим инжекции: `reduce` или `direct`
- `--air-grid-step-mm` — шаг сетки воздуха
- `--debug` — трассировка ядра, проверка NaN/Inf
- `--validate` — валидация собственных частот
- `--no-plot` — без графиков
