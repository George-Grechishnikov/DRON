# Handoff: Адриадна / DRON

Этот файл нужен, чтобы быстро продолжить работу на другом устройстве или в новом Codex-контексте.

## Текущее направление

Сейчас основная работа идет по визуалу dashboard. Цель: сверстать первую страницу максимально близко к референсу `photo_2026-06-28_04-23-49.jpg`, но не статично, а поверх реальных данных pipeline.

## Что уже сделано

- Добавлен C++ backend для ускорения корреляции:
  - `terrain_nav_core/`
  - `correlation_fallback.py`
  - `scripts/build_cpp_backend.ps1`
  - `correlator.py` умеет использовать C++ fast path и fallback на Python.
- Добавлен стандартный запуск демо:
  - `run_standard_replay.ps1`
  - по умолчанию запускает replay + realtime dashboard + demo dashboard.
- Добавлен режим под технический чекпоинт экспертов:
  - `checkpoint_runner.py`
  - `CHECKPOINT_INPUT_GUIDE.md`
  - принимает `heights.txt`, `start-x/start-y`, `heading`, `speed`, `freq`, `dem.tif`.
  - сохраняет `output/checkpoint/trajectory_estimated.csv` и `trajectory_visualization.html`.
- В `main.py` добавлен `initial_heading_deg` / CLI `--initial-heading`, чтобы курс из входных данных реально попадал в расчет.
- В `visualizer.py` начата верстка первой страницы:
  - верхняя статусная панель;
  - большая DEM-карта слева;
  - управление симуляцией справа;
  - кнопки `GNSS ВКЛ`, `GNSS ВЫКЛ`, `СТАРТ / ЗАНОВО`;
  - скорость повтора и шкала времени;
  - текущие метрики;
  - профиль высоты;
  - корреляция рельефа;
  - временная шкала режимов;
  - журнал событий;
  - карта показывает точки + линию траектории, старт, потерю GNSS, TERRAIN_NAV и точку `ИСКАТЬ ЗДЕСЬ`.

## Как запускать демо

```powershell
cd "C:\Users\ceisi\OneDrive\Рабочий стол\DRON"
.\run_standard_replay.ps1
```

Открыть:

```text
http://127.0.0.1:8050
```

Быстрая проверка без UI:

```powershell
.\run_standard_replay.ps1 -NoVisualizer
```

## Как запускать формат техчека

```powershell
python .\checkpoint_runner.py `
  --dem .\data\checkpoint\dem.tif `
  --heights .\data\checkpoint\heights.txt `
  --start-x 120 `
  --start-y 240 `
  --xy-mode auto `
  --heading 84.3 `
  --speed 50 `
  --freq 10
```

Если эксперты уточнят тип `(x,y)`, лучше явно ставить:

- `--xy-mode pixel`
- `--xy-mode crs`
- `--xy-mode local-m`

## Проверки

Последние успешные проверки:

```powershell
python -m pytest test_visualizer.py test_main.py -q
# 36 passed
```

Также ранее полный набор проходил:

```powershell
python -m pytest -q
# 123 passed
```

После любых больших правок UI лучше повторить:

```powershell
python -m pytest test_visualizer.py test_main.py -q
```

## Что важно не сломать

- Все callback ID в `visualizer.py` должны остаться:
  - `gnss-on-button`
  - `gnss-off-button`
  - `route-restart-button`
  - `control-status`
  - `metrics-panel`
  - `history-store`
  - `dashboard-interval`
  - `terrain-map`
  - `correlation-heatmap`
  - `profiles-graph`
  - `telemetry-graph`
- `telemetry-graph` сейчас может быть скрытым, но его нельзя удалять без изменения callbacks/tests.
- Большой raw dataset не коммитить:
  - `data/uav_3m_dataset/uav_3m_dataset/`
  - `data/uav_3m_dataset/quick_5000/`
- C++ build artifacts не коммитить:
  - `terrain_nav_core/build/`
  - `terrain_nav_core/Release/`
  - `terrain_nav_core/*.pyd`

## Ближайшие задачи

1. Дожать первую страницу пиксельно под референс:
   - размеры карточек;
   - высоты блоков;
   - плотность отступов;
   - стиль карты;
   - правый блок управления;
   - журнал событий.
2. Сделать журнал событий динамическим от текущего состояния, а не статичным списком.
3. Дожать подписи и русские строки, если где-то в браузере появятся проблемы с кодировкой.
4. После подтверждения первой страницы перейти ко второй странице из референсов.

## Git policy

Перед переносом на другое устройство:

```powershell
git status --short
python -m pytest test_visualizer.py test_main.py -q
git add <нужные файлы>
git commit -m "George"
git push origin main
```

Не добавлять большие данные и build-артефакты.
