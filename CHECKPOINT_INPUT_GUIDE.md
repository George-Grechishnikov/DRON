# Подача данных по кейсу

Основной вход для `checkpoint_runner.py` соответствует формулировке хакатона:

- `--dem` — цифровая модель рельефа GeoTIFF.
- `--nmea` — поток радиовысотомера в NMEA-0183 GGA (`GPGGA` или `GNGGA`), высота в метрах.
- `--speed` — начальная путевая скорость ЛА в м/с.
- `--freq` — частота радиовысотомера, 1-10 Гц. Если в NMEA есть корректные UTC-метки, частота может быть выведена автоматически.

Стартовая точка по умолчанию берется из центра DEM. Это соответствует требованию строить эталонные профили из произвольной стартовой точки в центре карты.

## Команда запуска

```powershell
cd "C:\Users\ceisi\OneDrive\Рабочий стол\DRON"

python .\checkpoint_runner.py `
  --dem .\data\checkpoint\dem.tif `
  --nmea .\data\checkpoint\radar_altimeter.nmea `
  --speed 50 `
  --freq 10
```

Для визуализации:

```powershell
python .\checkpoint_runner.py `
  --dem .\data\checkpoint\dem.tif `
  --nmea .\data\checkpoint\radar_altimeter.nmea `
  --speed 50 `
  --freq 10 `
  --dashboard `
  --open-browser
```

## Результат

После запуска файлы будут лежать в `output/checkpoint/`:

- `trajectory_estimated.csv` — итоговая траектория: `local_x_m`, `local_y_m`, `lat`, `lon`, скорость и азимут.
- `trajectory_visualization.html` — визуализация DEM + найденной траектории + финальная точка.

## Legacy-режим

Если вместо NMEA есть только текстовый файл высот, можно использовать legacy-вход:

```powershell
python .\checkpoint_runner.py `
  --dem .\data\checkpoint\dem.tif `
  --heights .\data\checkpoint\heights.txt `
  --input-kind radar `
  --speed 50 `
  --freq 10
```

`--heights` автоматически конвертируется во временный GPGGA-поток. Для уже восстановленного профиля рельефа используйте `--input-kind terrain`.
