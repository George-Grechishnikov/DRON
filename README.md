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

Установка зависимостей:

```powershell
python -m pip install --user numpy rasterio pyproj scipy pytest
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
  test_sim_generator.py
  test_nmea_parser.py
  test_dem_loader.py
  test_profile_extractor.py
  test_correlator.py
  test_position_solver.py
  test_imm_filter.py
```

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

## Как запускать то, что уже есть

### Проверка модулей тестами

Запуск всех текущих тестов:

```powershell
python -m pytest .\test_sim_generator.py .\test_nmea_parser.py .\test_dem_loader.py .\test_profile_extractor.py .\test_correlator.py .\test_position_solver.py .\test_imm_filter.py -q
```

### Минимальный сценарий работы

1. Подготовить `DEM` в формате `GeoTIFF`
2. Сгенерировать `.nmea` и `ground truth` через `sim_generator.py`
3. Прочитать `.nmea` через `nmea_parser.py`
4. Использовать `dem_loader.py` для получения профилей рельефа
5. Построить эталонную матрицу через `profile_extractor.py`
6. Найти лучший азимут и смещение через `correlator.py`
7. Перевести корреляционный результат в координаты через `position_solver.py`
8. Сгладить навигационное решение через `imm_filter.py`

## Что будет добавлено дальше

Следующие модули, которые будут появляться в проекте:
- `position_solver.py`
- `visualizer.py`
- `main.py`

Когда они будут готовы, я буду дополнять этот `README`:
- новыми командами запуска
- схемой общего pipeline
- описанием входных и выходных данных
- инструкцией для полного запуска проекта end-to-end
