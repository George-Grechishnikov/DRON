# TERRAIN NAVIGATOR

Черновая реализация проекта автономной навигации БПЛА без GNSS по рельефу.

Сейчас в репозитории готовы первые модули:
- `sim_generator.py` — генератор синтетических NMEA-данных радиовысотомера
- `nmea_parser.py` — парсер потока NMEA-0183
- `dem_loader.py` — загрузчик DEM/ЦМР с выборкой высот и профилей

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
  test_sim_generator.py
  test_nmea_parser.py
  test_dem_loader.py
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

## Как запускать то, что уже есть

### Проверка модулей тестами

Запуск всех текущих тестов:

```powershell
python -m pytest .\test_sim_generator.py .\test_nmea_parser.py .\test_dem_loader.py -q
```

### Минимальный сценарий работы

1. Подготовить `DEM` в формате `GeoTIFF`
2. Сгенерировать `.nmea` и `ground truth` через `sim_generator.py`
3. Прочитать `.nmea` через `nmea_parser.py`
4. Использовать `dem_loader.py` для получения профилей рельефа

## Что будет добавлено дальше

Следующие модули, которые будут появляться в проекте:
- `profile_extractor.py`
- `correlator.py`
- `position_solver.py`
- `imm_filter.py`
- `visualizer.py`
- `main.py`

Когда они будут готовы, я буду дополнять этот `README`:
- новыми командами запуска
- схемой общего pipeline
- описанием входных и выходных данных
- инструкцией для полного запуска проекта end-to-end
