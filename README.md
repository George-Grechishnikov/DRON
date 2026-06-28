# TERRAIN NAVIGATOR

Статус связи аудита с текущей версией: см. `PROBLEM_SOLUTION_MAP.md`.

Черновая реализация проекта автономной навигации БПЛА без GNSS по рельефу.

Сейчас в репозитории готовы первые модули:
- `sim_generator.py` — генератор синтетических NMEA-данных радиовысотомера
- `nmea_parser.py` — парсер потока NMEA-0183
- `dem_loader.py` — загрузчик DEM/ЦМР с выборкой высот и профилей
- `profile_extractor.py` — построитель эталонных профилей рельефа
- `correlator.py` — корреляционный движок для поиска лучшего азимута и смещения
- `position_solver.py` — преобразование корреляционного результата в геодезический fix
- `imm_filter.py` — сглаживание решения через режимы hover/cruise/turn
- `visualizer.py` — Plotly Dash дашборд реального времени
- `main.py` — главный оркестратор pipeline

Этот `README` будем постепенно дополнять по мере реализации следующих задач.

## Нужны ли тесты

Для самого запуска проекта тесты не обязательны.

То есть:
- если цель просто запустить модуль вручную, тесты не нужны
- если цель безопасно развивать проект дальше, тесты очень полезны

Зачем они нам здесь:
- быстро проверяют, что новые изменения не сломали уже готовые модули
- помогают стыковать части проекта поэтапно
- упрощают отладку перед сборкой общего pipeline

## Требования

- Python `3.12+`
- `numpy`
- `rasterio`
- `pyproj`
- `scipy`
- `pytest`
- `dash`
- `plotly`
- `pymavlink` for `ArduPilot SITL` bridge

Установка зависимостей:

```powershell
python -m pip install --user numpy rasterio pyproj scipy pytest dash plotly pymavlink
```

## Структура проекта

```text
DRON/
  sim_generator.py
  nmea_parser.py
  dem_loader.py
  profile_extractor.py
  correlator.py
  position_solver.py
  imm_filter.py
  visualizer.py
  sitl_bridge.py
  main.py
  test_sim_generator.py
  test_nmea_parser.py
  test_dem_loader.py
  test_profile_extractor.py
  test_correlator.py
  test_position_solver.py
  test_imm_filter.py
  test_visualizer.py
  test_sitl_bridge.py
  integration_test.py
  test_main.py
```

## Case-aligned input path

The competition case requires:

- radar-altimeter data in `NMEA-0183 v3`
- message type compatible with `GPGGA`
- altitude value in meters in the GGA altitude field
- barometric altitude fixed at `1500 m`
- message frequency in the `1-10 Hz` range

The project now follows that contract directly:

- the core algorithm consumes only parsed `NMEAFrame` radar-altimeter samples
- the terrain profile is reconstructed as `terrain_height_m = 1500.0 - radar_alt_m`
- `--live` mode is the most case-faithful runtime path because it reads `NMEA GPGGA` from UDP
- if `--lat` and `--lon` are omitted, the initial search point defaults to the center of the DEM, which matches the case wording about starting from the map center

Recommended strict demo path:

1. Use `sitl_bridge.py` to synthesize radar-altimeter `GPGGA` messages from the simulator.
2. Stream those messages over UDP at `1-10 Hz`.
3. Run `main.py --live` so the navigation pipeline receives only `NMEA` input.

Example strict-case pipeline:

```powershell
python .\sitl_bridge.py --dem .\data\fabdem_canberra.tif --connect udp:127.0.0.1:14550 --stream-nmea-udp --udp-host 127.0.0.1 --udp-port 10110 --stream-rate-hz 5 --count 0
```

In another terminal:

```powershell
python .\main.py --live --dem .\data\fabdem_canberra.tif --udp-host 127.0.0.1 --udp-port 10110
```

For an even stricter file-based case demo where `main.py` receives an explicit `NMEA GPGGA` log together with the `DEM`, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_case_nmea_demo.ps1
```

This script does two steps automatically:

1. generates a real `GPGGA` radar-altimeter `.nmea` file;
2. starts `main.py --replay --dem ... --nmea ... --gt ...`.

Recommended real DEM for Canberra/SITL:

- `data/fabdem_canberra_wide.tif` - merged real `FABDEM` cutout around Canberra and nearby mountains
- bounds: `148.15 .. 149.45 lon`, `-35.90 .. -35.00 lat`
- elevation span: about `216 .. 1906 m`

## SITL bridge

`sitl_bridge.py` is the dedicated bridge layer for `ArduPilot SITL`.

It is responsible for:
- connecting to SITL over `MAVLink`
- reading telemetry from `GLOBAL_POSITION_INT` and related messages
- deriving a unified sample stream:
  - `timestamp`
  - `lat`
  - `lon`
  - `alt_msl`
  - `heading_deg`
  - `ground_speed_mps`
  - `radar_alt_m`
  - `gnss_available`
- emulating `GNSS ON/OFF` by timer
- converting samples back into `GPGGA`-compatible radar-altimeter NMEA for pipeline compatibility
- optional streaming of synthesized `GPGGA` messages over UDP for strict case-aligned testing

This keeps the existing terrain-navigation pipeline intact while we add SITL integration step by step.

Recent navigation hardening:

- FFT-backed normalized cross-correlation for the fast path
- ambiguity metrics (`PSLR`, peak count, peak isolation)
- sub-step offset interpolation
- predictive fallback when terrain update is flat, ambiguous, or low-confidence
- optional top-k terrain-feature refinement for stronger candidates

Quick smoke test after SITL is running:

```powershell
python .\sitl_bridge.py --dem .\data\dem.tif --connect udp:127.0.0.1:14550 --count 5
```

To force the bridge into demo mode with GNSS unavailable:

```powershell
python .\sitl_bridge.py --dem .\data\dem.tif --connect udp:127.0.0.1:14550 --count 5 --gnss-off
```

Full pipeline SITL mode:

```powershell
python .\main.py --sitl --dem .\data\dem.tif --sitl-connect udp:127.0.0.1:14550 --lat 60.5 --lon 90.3
```

SITL mode with demo GNSS loss:

```powershell
python .\main.py --sitl --dem .\data\dem.tif --sitl-connect udp:127.0.0.1:14550 --sitl-gnss-drop-after 20 --lat 60.5 --lon 90.3
```

In `--sitl` mode the dashboard now shows:

- live `GNSS AVAILABLE / GNSS LOST` state
- estimated trajectory from the TERRAIN NAVIGATOR pipeline
- truth position from SITL when available
- adaptive window size, PSLR, CRLB-like observability, and terrain fallback mode

Manual GNSS control for live demo:

- start the full `--sitl` pipeline command
- open the Dash dashboard in the browser
- use the `GNSS ON` and `GNSS OFF` buttons in the control panel
- the command is delivered into the SITL bridge immediately through the internal control queue
- after `GNSS OFF`, the pipeline continues in terrain-navigation / prediction-assisted mode
- after `GNSS ON`, the demo can show recovery of satellite availability and comparison against SITL truth again

Recommended jury demo flow:

1. Start with `GNSS ON` and show that truth telemetry is visible.
2. Press `GNSS OFF` during motion and point to the live `GNSS LOST` status.
3. Show that the estimated trajectory and terrain correlation diagnostics continue updating.
4. Press `GNSS ON` again and show reappearance of GNSS-assisted comparison against SITL truth.

Correlation benchmark for desktop / Jetson / Raspberry Pi:

```powershell
python .\benchmark_correlator.py --iterations 20
```

Optional C++ correlation backend:

- the project runs without C++; Python/NumPy/SciPy fallback is always available
- C++ backend is used automatically when `terrain_nav_core\_terrain_nav_core*.pyd` is built
- on Windows install Microsoft C++ Build Tools first, then run:

```powershell
.\scripts\build_cpp_backend.ps1
python -c "import correlation_fallback; print(correlation_fallback.cpp_backend_available())"
```

Standard replay launch with C++ correlation check and 10 Hz point-by-point dashboard trajectory:

```powershell
.\run_standard_replay.ps1
```

This opens the dashboard, plays NMEA at `--freq 10`, and draws one trajectory point for each sample while the correlation window remains `64` frames for accuracy.
The demo uses fixed case parameters: barometric altitude `1500 m` and display speed `50 m/s`.
When the replay reaches the end of the route, it stays on the final point and waits for the dashboard button `СТАРТ / ЗАНОВО` before clearing the path and starting again.

For fast metrics without the browser:

```powershell
.\run_standard_replay.ps1 -NoVisualizer
```

Current MVP limitations and honest claims:

- validated well on synthetic and local integration scenarios, not yet on flight-grade real datasets
- current SITL integration is intended for GNSS-loss continuation demos, not full cold-start navigation
- best performance is expected on terrain with informative relief; flat terrain correctly triggers fallback more often
- offline DEM coverage still assumes mission-region data is preloaded before launch
- cost, Arctic universality, and final field accuracy should be presented as hypotheses or next-stage validation items, not as proven production claims

## Что уже умеет проект

### 1. Генерация синтетического NMEA

`sim_generator.py` умеет:
- читать `GeoTIFF` с рельефом
- строить маршрут по заданной траектории
- вычислять высоту над рельефом
- добавлять шум радиовысотомера
- формировать валидные строки `GPGGA`
- сохранять `.nmea` и `CSV` с ground truth

Пример запуска:

```powershell
python .\sim_generator.py `
  --dem .\data\dem.tif `
  --lat 60.5 `
  --lon 90.3 `
  --trajectory 1 `
  --noise 2.0 `
  --freq 5 `
  --output file `
  --out-nmea .\output\traj1.nmea `
  --out-csv .\output\traj1_ground_truth.csv
```

Поддерживаются траектории:
- `1` — прямолинейная
- `2` — с поворотом
- `3` — с набором высоты

### 2. Парсинг NMEA

`nmea_parser.py` умеет:
- разбирать `GPGGA` и `GNGGA`
- проверять checksum
- возвращать объекты `NMEAFrame`
- читать поток из файла или UDP
- собирать профиль высот из последовательности фреймов

Пример использования:

```python
from nmea_parser import NMEAReader, frames_to_profile

reader = NMEAReader.from_file("output/traj1.nmea")
frames = reader.read_window(50)
profile = frames_to_profile(frames, speed_mps=50.0, freq_hz=5.0)
print(profile.shape)
reader.close()
```

### 3. Работа с DEM

`dem_loader.py` умеет:
- открывать `GeoTIFF`
- при необходимости приводить его к `EPSG:4326`
- получать высоту по координатам
- вырезать патч рельефа вокруг точки
- строить профиль рельефа вдоль азимута
- кэшировать патчи

Пример использования:

```python
from dem_loader import DEMLoader

with DEMLoader("data/dem.tif") as dem:
    elevation = dem.get_elevation(60.5, 90.3)
    patch, transform = dem.get_patch(60.5, 90.3, radius_m=5000)
    profile = dem.get_profile_along_azimuth(
        lat=60.5,
        lon=90.3,
        azimuth_deg=45.0,
        distance_m=5000.0,
        step_m=30.0,
    )
```

### 4. Построение эталонных профилей

`profile_extractor.py` умеет:
- строить матрицу эталонных профилей для набора азимутов
- собирать профили параллельно
- переиспользовать кэш, если центр окна сместился меньше чем на `200 м`
- нормализовать профили перед корреляцией
- определять плоский рельеф, где корреляция ненадёжна

Пример использования:

```python
import numpy as np

from dem_loader import DEMLoader
from profile_extractor import ProfileExtractor, is_flat_terrain, normalize_profile

with DEMLoader("data/dem.tif") as dem:
    extractor = ProfileExtractor(dem, profile_length_m=5000.0, step_m=30.0)
    ref_matrix = extractor.build_reference_matrix(
        center_lat=60.5,
        center_lon=90.3,
        azimuths=np.arange(0, 360, 1.0),
    )
    normalized = normalize_profile(ref_matrix[45])
    flat = is_flat_terrain(ref_matrix[45])
```

### 5. Корреляционный движок

`correlator.py` умеет:
- сравнивать измеренный профиль `H_meas` с матрицей эталонных профилей
- искать лучший `azimuth + offset`
- считать `heatmap` корреляции для визуализации
- оценивать `confidence` и флаг `is_reliable`
- принимать буфер `NMEAFrame` напрямую через `sliding_window_compute()`

Пример использования:

```python
import numpy as np

from correlator import Correlator, build_heatmap

correlator = Correlator(
    profile_length_m=5000.0,
    step_m=30.0,
    max_offset_m=2000.0,
)

result = correlator.compute(h_meas, ref_matrix, azimuths_deg=np.arange(0, 360, 1.0))
heatmap = build_heatmap(result)

print(result.best_azimuth_deg)
print(result.best_offset_m)
print(result.peak_correlation)
```

### 6. Решатель позиции и скорости

`position_solver.py` умеет:
- превращать `best_azimuth + best_offset` в `lat/lon`
- вычислять путевую скорость из длины окна
- строить простую ковариацию позиции
- оценивать скорость и азимут между двумя fix
- хранить последние `10` навигационных состояний

Пример использования:

```python
from position_solver import PositionSolver

solver = PositionSolver()
fix = solver.solve(
    result=result,
    start_lat=60.5,
    start_lon=90.3,
    window_duration_s=10.0,
)

print(fix.lat, fix.lon)
print(fix.speed_mps, fix.azimuth_deg)
```

### 7. IMM-фильтр

`imm_filter.py` умеет:
- смешивать три модели движения `hover / cruise / turn`
- выполнять цикл `mixing -> predict -> update -> fusion`
- сглаживать координаты и скорость относительно сырых fix
- возвращать веса режимов и доминирующую модель
- считать `HDOP`-подобную оценку горизонтальной неопределённости

Пример использования:

```python
from imm_filter import IMMFilter

imm = IMMFilter()
imm_result = imm.update(
    position_fix=fix,
    dt=2.0,
    is_flat=False,
)

print(imm_result.lat, imm_result.lon)
print(imm_result.model_weights)
print(imm_result.dominant_mode)
```

### 8. Визуализация

`visualizer.py` умеет:
- поднимать `Plotly Dash` дашборд
- неблокирующе читать состояние из `queue.Queue`
- показывать 4 панели: heatmap, карту, профили и IMM telemetry
- хранить историю траектории между тиками
- экспортировать HTML-отчёт полёта

Пример использования:

```python
import queue

from visualizer import TerrainNavigatorDash

state_queue = queue.Queue(maxsize=100)
dashboard = TerrainNavigatorDash(state_queue=state_queue)
dashboard.run(host="127.0.0.1", port=8050, debug=False)
```

### 9. Главный оркестратор

`main.py` умеет:
- запускать проект в режимах `sim`, `live`, `replay`
- поднимать producer/pipeline/dashboard потоки
- крутить скользящее окно обработки
- при включении adaptive mode подбирать размер окна между `min_window_size` и `max_window_size`
- собирать состояние для `visualizer.py`
- считать replay-метрики по `ground truth`
- экспортировать HTML-отчёт по завершении

Примеры запуска:

```powershell
python .\main.py --sim --dem .\data\dem.tif --trajectory 1 --lat 60.5 --lon 90.3
```

```powershell
python .\main.py --sim --dem .\data\dem.tif --trajectory 1 --adaptive-window --min-window-size 20 --max-window-size 50 --window-growth-step 10
```

```powershell
python .\main.py --live --dem .\data\dem.tif --udp-host 127.0.0.1 --udp-port 10110 --lat 60.5 --lon 90.3
```

```powershell
python .\main.py --replay --dem .\data\dem.tif --nmea .\logs\flight.nmea --gt .\logs\ground_truth.csv --lat 60.5 --lon 90.3
```

Если нужно прогонять вычисления без UI:

```powershell
python .\main.py --sim --dem .\data\dem.tif --trajectory 1 --no-visualizer
```

## Как запускать то, что уже есть

### Проверка модулей тестами

Запуск всех текущих тестов:

```powershell
python -m pytest .\test_sim_generator.py .\test_nmea_parser.py .\test_dem_loader.py .\test_profile_extractor.py .\test_correlator.py .\test_position_solver.py .\test_imm_filter.py .\test_visualizer.py .\test_main.py -q
```

Финальный интеграционный прогон:

```powershell
python -m pytest .\integration_test.py -q
```

Этот тест сейчас подтверждает, что полный pipeline:
- запускается end-to-end
- не падает на многопоточном сценарии
- выдаёт историю результатов и метрики
- сохраняет HTML-отчёт

### Минимальный сценарий работы

1. Подготовить `DEM` в формате `GeoTIFF`
2. Сгенерировать `.nmea` и `ground truth` через `sim_generator.py`
3. Прочитать `.nmea` через `nmea_parser.py`
4. Использовать `dem_loader.py` для получения профилей рельефа
5. Построить эталонную матрицу через `profile_extractor.py`
6. Найти лучший азимут и смещение через `correlator.py`
7. Перевести корреляционный результат в координаты через `position_solver.py`
8. Сгладить навигационное решение через `imm_filter.py`
9. Показать потоковое состояние в `visualizer.py`
10. Запустить всё вместе через `main.py`

## Полный pipeline

Текущая схема работы:

1. `sim_generator.py` генерирует поток NMEA или готовит ground truth
2. `nmea_parser.py` превращает поток в `NMEAFrame`
3. `dem_loader.py` читает рельеф и отдаёт высоты/патчи
4. `profile_extractor.py` строит эталонные профили по азимутам
5. `correlator.py` ищет лучший `azimuth + offset`
6. `position_solver.py` строит геодезический fix
7. `imm_filter.py` сглаживает решение по режимам движения
8. `visualizer.py` показывает текущее состояние и историю
9. `main.py` связывает всё в один рабочий pipeline

Дополнительная логика устойчивости:
- `correlator.py` считает `PSLR`, ambiguity и sub-step offset
- `main.py` считает CRLB-like observability и может уйти в predictive fallback
- adaptive window выбирает минимальное окно, которое уже даёт достаточно информативный terrain signal

## Что осталось дальше

Следующий шаг после этого — уже не сборка модулей, а улучшение точности навигационного решения:
- калибровка корреляции и стратегии смещения окна
- усиление логики начальной привязки
- доведение replay-метрик до целевых порогов из задания
