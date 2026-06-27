## Куда складывать входные данные

Используй эту структуру без изменения кода:

```text
input/
  incoming/
    config.yaml
    radar_data.nmea
    truth.csv
    barometer.csv
    dem/
      README.txt
      your_dem_file.tif
```

### Что куда класть

- `input/incoming/config.yaml` — основной конфиг запуска.
- `input/incoming/radar_data.nmea` — поток NMEA-0183 `GPGGA`.
- `input/incoming/truth.csv` — ground truth для проверки качества.
- `input/incoming/barometer.csv` — барометрическая высота.
- `input/incoming/dem/` — DEM-файлы (`GeoTIFF` или `SRTM`).

### Вводные из кейса

- Источник DEM: `Copernicus GLO-30`, `SRTM`, `ALOS AW3D30`.
- Радиовысотомер: только `NMEA-0183 v3` в виде строк `GPGGA`.
- Частота `radar_data.nmea`: от `1` до `10 Гц`.
- Барометрическая высота: около `1500 м MSL` как абсолютная высота полета.

Это значит:
- `radar_data.nmea` должен менять высоту по рельефу под дроном;
- `barometer.csv` должен держаться около `1500 м`;
- `truth.csv` должен описывать тот же полет и ту же временную шкалу.

### Быстрый порядок работы

1. Положить полученные файлы в `input/incoming/`.
2. Убедиться, что пути в `config.yaml` совпадают с реальными именами файлов.
3. Запускать конвейер уже с этими путями.

### Замечание

Если DEM-файл называется не `terrain.tif`, просто поправь `dem_path` в `config.yaml`.
