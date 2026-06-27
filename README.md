# TERRAIN NAVIGATOR

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

- Python `3.10+`
- `numpy`
- `rasterio`
- `pyproj`
- `scipy`
- `pytest`
- `dash`
- `plotly`

Установка зависимостей:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Быстрый показ результата

Если нужен сценарий для обычного пользователя, запускай конвейер с флагом `--open-report`.
Тогда после завершения прогона HTML-отчет откроется в браузере автоматически.

Для кейсового набора из `config.yaml` достаточно одной команды:

```bash
.venv/bin/python main.py --config input/incoming/config.yaml --open-report
```

Или через готовый пользовательский launcher:

```bash
.venv/bin/python case_runner.py --config input/incoming/config.yaml
```

Если нужен локальный веб-интерфейс с backend API и replay-управлением:

```bash
.venv/bin/python -m uvicorn web_backend:app --host 127.0.0.1 --port 8000
```

После запуска открывай:

```text
http://127.0.0.1:8000
```

Важно:

- этот backend рассчитан на локальный инженерный запуск на `127.0.0.1`
- state-changing API защищены локальной browser-origin проверкой и session token
- не нужно публиковать этот backend наружу без отдельного production auth/reverse proxy слоя

Пример:

```bash
.venv/bin/python main.py \
  --replay \
  --dem input/incoming/dem/terrain.tif \
  --nmea input/incoming/radar_data.nmea \
  --lat 60.56274504 \
  --lon 90.37800000 \
  --no-visualizer \
  --report-path output/full_50k_report.html \
  --open-report
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
  main.py
  test_sim_generator.py
  test_nmea_parser.py
  test_dem_loader.py
  test_profile_extractor.py
  test_correlator.py
  test_position_solver.py
  test_imm_filter.py
  test_visualizer.py
  integration_test.py
  test_main.py
```

## Куда класть входные данные

Для внешних тестовых или реальных данных используй готовую структуру:

```text
input/
  incoming/
    config.yaml
    radar_data.nmea
    truth.csv
    barometer.csv
    dem/
      README.txt
      terrain.tif
```

- `input/incoming/radar_data.nmea` — входной поток `GPGGA`
- `input/incoming/truth.csv` — truth-траектория для сравнения
- `input/incoming/barometer.csv` — барометрическая высота
- `input/incoming/dem/` — DEM-файл
- `input/incoming/config.yaml` — все пути и параметры

Готовый шаблон уже добавлен в репозиторий.

## Что уже умеет проект

### Локальный web backend

`web_backend.py` умеет:

- загружать case-датасет из `config.yaml`
- валидировать DEM / NMEA / truth / barometer входы
- запускать replay через тот же pipeline
- отдавать trajectory / heatmap / profiles / metrics / logs
- защищать mutating API локальной session-защитой для UI

Что важно понимать:

- это локальный инженерный backend, а не интернет-facing production service
- для UI достаточно открыть `/`, токен сессии встраивается в HTML автоматически
- для внешних браузерных запросов mutating endpoints требуют и допустимый local origin, и session token

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

### 7. Unified sample stream

`main.py` умеет принимать единый поток samples, который потом можно напрямую
использовать для стыковки с внешним источником телеметрии.

Формат одного sample:

```python
{
    "timestamp_s": 0.0,
    "lat": 60.5,
    "lon": 90.3,
    "alt_msl": 1500.0,
    "radar_alt_m": 1200.0,
    "terrain_h": 300.0,
    "heading_deg": 45.0,
    "speed_mps": 50.0,
    "gnss_available": True,
    "nav_mode": "GNSS",
    "truth_lat": 60.5,
    "truth_lon": 90.3,
    "estimated_lat": null,
    "estimated_lon": null,
    "correlation_score": null,
    "correlation_heatmap": null
}
```

Для проверки такого потока без live-источника можно сохранить samples в JSONL:
одна строка — один JSON object.

Пример запуска:

```powershell
python .\main.py `
  --samples-jsonl .\output\samples.jsonl `
  --dem .\data\dem.tif `
  --gnss-drop-after 30
```

После `--gnss-drop-after` dashboard показывает `GNSS OFF` и
`NAV MODE: TERRAIN_NAV`, а траектории truth/estimated продолжают
отображаться вместе с correlation heatmap.

### 8. IMM-фильтр

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

### 9. Визуализация

`visualizer.py` умеет:
- поднимать `Plotly Dash` дашборд
- неблокирующе читать состояние из `queue.Queue`
- показывать 4 панели: heatmap, карту, профили и IMM telemetry
- хранить историю траектории между тиками
- экспортировать HTML-отчёт полёта
- не падать без truth trajectory, DEM или incoming correlation heatmap

Пример использования:

```python
import queue

from visualizer import TerrainNavigatorDash

state_queue = queue.Queue(maxsize=100)
dashboard = TerrainNavigatorDash(state_queue=state_queue)
dashboard.run(host="127.0.0.1", port=8050, debug=False)
```

### 10. Главный оркестратор

`main.py` умеет:
- запускать проект в режимах `sim`, `live`, `replay`
- запускать unified stream режим через `--samples-jsonl` / `--unified-stream`
- поднимать producer/pipeline/dashboard потоки
- крутить скользящее окно обработки
- собирать состояние для `visualizer.py`
- считать replay-метрики по `ground truth`
- экспортировать HTML-отчёт по завершении

Примеры запуска:

```powershell
python .\main.py --sim --dem .\data\dem.tif --trajectory 1 --lat 60.5 --lon 90.3
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

### Боевой preflight

Перед демонстрацией или деплоем запустить полный preflight:

```powershell
.\.venv\Scripts\python .\scripts\preflight.py
```

macOS/Linux:

```bash
.venv/bin/python scripts/preflight.py
```

Preflight делает:
- компиляцию основных Python-модулей
- полный `pytest`
- end-to-end smoke-run через `--samples-jsonl`
- проверку создания `output/terrain_navigator_report.html`

Если нужно быстро проверить только запуск без полного тестового набора:

```powershell
.\.venv\Scripts\python .\scripts\preflight.py --skip-tests
```

### Чистка кэша и артефактов

Быстрая чистка проектных кэшей и build-артефактов:

```powershell
.\.venv\Scripts\python .\scripts\clean_cache.py
```

Посмотреть, что будет удалено, без фактической очистки:

```powershell
.\.venv\Scripts\python .\scripts\clean_cache.py --dry-run
```

### Валидация unified stream

Проверка incoming JSONL перед запуском pipeline:

```powershell
.\.venv\Scripts\python .\sample_validator.py .\output\samples.jsonl
```

### Нормализация случайных входных данных

Если дадут `.csv`, `.json` или `.jsonl` с другими именами полей, их можно привести
к нашему unified sample формату так:

```powershell
.\.venv\Scripts\python .\sample_ingest.py .\input\random_samples.csv .\output\samples.jsonl
```

После этого уже можно:

```powershell
.\.venv\Scripts\python .\sample_validator.py .\output\samples.jsonl
.\.venv\Scripts\python .\main.py --samples-jsonl .\output\samples.jsonl --dem .\data\dem.tif
```

### Проверка модулей тестами

Запуск всех текущих тестов:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Точечные тесты на unified stream и визуализацию:

```powershell
.\.venv\Scripts\python -m pytest .\test_main.py .\test_visualizer.py .\test_correlation_fallback.py -q
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

## Боевой сценарий демо

### Вариант A: unified JSONL replay

Подходит для проверки будущего `sitl_bridge.py` без live-подключения.

```powershell
.\.venv\Scripts\python .\main.py `
  --samples-jsonl .\output\samples.jsonl `
  --dem .\data\dem.tif `
  --report-path .\output\demo_report.html `
  --dashboard-host 127.0.0.1 `
  --dashboard-port 8050
```

Открыть dashboard:

```text
http://127.0.0.1:8050
```

### Вариант B: simulation demo с GNSS loss

```powershell
.\.venv\Scripts\python .\main.py `
  --sim `
  --dem .\data\dem.tif `
  --trajectory 1 `
  --lat 60.5 `
  --lon 90.3 `
  --gnss-drop-after 30 `
  --report-path .\output\demo_gnss_loss.html `
  --dashboard-host 127.0.0.1 `
  --dashboard-port 8050
```

На dashboard должны быть видны:
- `GNSS AVAILABLE` до события потери
- `GNSS OFF` после события потери
- `NAV MODE: TERRAIN_NAV`
- truth trajectory и estimated trajectory
- correlation heatmap с пиком
- measured/reference terrain profiles

## Сборка C++ ядра

Опциональное C++ ядро лежит в `terrain_nav_core/` и не требуется для Python-only запуска.
Если модуль не собран, проект автоматически использует `correlation_fallback.py`.

Пример сборки:

```bash
python3 -m pip install pybind11
cmake -S terrain_nav_core -B terrain_nav_core/build -DPython_EXECUTABLE=$(which python3)
cmake --build terrain_nav_core/build --config Release
```

После сборки Python backend продолжит использовать тот же интерфейс:

```python
from correlation_fallback import find_best_match
```

## Чеклист перед показом

1. Установлены зависимости из `requirements.txt`.
2. `scripts/preflight.py` проходит без ошибок.
3. DEM покрывает стартовую точку и весь маршрут.
4. `--window-size`, `--step-size`, `--max-offset` соответствуют скорости и частоте samples.
5. Dashboard открывается на `http://127.0.0.1:8050`.
6. При GNSS loss отображаются `GNSS OFF` и `NAV MODE: TERRAIN_NAV`.
7. В `output/terrain_navigator_report.html` или указанном `--report-path` создаётся отчёт после завершения прогона.

## Что осталось дальше

Следующий шаг после этого — уже не сборка модулей, а улучшение точности навигационного решения:
- калибровка корреляции и стратегии смещения окна
- усиление логики начальной привязки
- доведение replay-метрик до целевых порогов из задания
