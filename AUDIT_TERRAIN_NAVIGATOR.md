# TERRAIN NAVIGATOR: глубокий технический аудит

Дата аудита: 2026-06-27  
Приоритет: максимальная точность координат, затем задержка без ухудшения точности.

## 0. Короткий вывод

Проект уже имеет рабочий end-to-end pipeline: NMEA GPGGA радиовысотомер -> абсолютный профиль рельефа -> DEM reference profiles -> корреляция по азимуту/смещению -> позиция/скорость -> IMM/визуализация/SITL bridge. Тестовая база неплохая: `108 passed`.

Но текущая точность еще не соответствует уровню надежной GNSS-denied навигации. Главные причины:

1. Модель поиска почти всегда предполагает один прямолинейный профиль. На повороте это ломается: `scenario_turn` дает `mean=2855.60 m`, `rmse=3043.63 m`.
2. Глобальная неоднозначность рельефа не удерживается как несколько гипотез. Есть PF/ESKF, но они не являются главным устойчивым режимом legacy pipeline.
3. Cold start из неизвестной позиции по карте не решен как полноценная глобальная задача.
4. Надежность корреляционного максимума оценивается эвристически; PSLR/ambiguity считаются по flatten heatmap, что может давать ложные решения.
5. Валидация пока в основном синтетическая и self-generated, поэтому может переоценивать качество.

## 1. Проверенное состояние проекта

Команды, выполненные во время аудита:

```powershell
python -m pytest -q
python .\benchmark_correlator.py --iterations 5 --azimuths 360 --length 50 --offset-steps 67
python .\main.py --replay --dem .\data\fabdem_canberra_wide.tif --nmea .\output\scenario_straight.nmea --gt .\output\scenario_straight_gt.csv --lat -35.36 --lon 149.05 --no-visualizer
python .\main.py --replay --dem .\data\fabdem_canberra_wide.tif --nmea .\output\scenario_noisy.nmea --gt .\output\scenario_noisy_gt.csv --lat -35.36 --lon 149.05 --no-visualizer
python .\main.py --replay --dem .\data\fabdem_canberra_wide.tif --nmea .\output\scenario_flat.nmea --gt .\output\scenario_flat_gt.csv --lat -35.36 --lon 149.05 --no-visualizer
python .\main.py --replay --dem .\data\fabdem_canberra_wide.tif --nmea .\output\scenario_turn.nmea --gt .\output\scenario_turn_gt.csv --lat -35.36 --lon 149.05 --no-visualizer
```

Результаты:

| Проверка | Результат |
|---|---:|
| Unit/integration tests | 108 passed |
| Correlator benchmark mean | 0.084216 s |
| Correlator benchmark p95 | 0.090344 s |
| Correlator benchmark ambiguity | ambiguous=True |
| Straight replay mean/RMSE | 360.71 m / 398.80 m |
| Noisy replay mean/RMSE | 337.43 m / 371.09 m |
| Flat replay mean/RMSE | 14034.12 m / 14034.88 m |
| Turn replay mean/RMSE | 2855.60 m / 3043.63 m |

Важно: `speed=0.00 m/s` в строках replay metrics означает ошибку скорости относительно ground truth, а не фактическую скорость БПЛА.

## 2. Современный контекст

Актуальная TRN/TAN литература подтверждает, что надежный путь для GNSS-denied terrain navigation - это не одиночный максимум корреляции, а байесовское сопровождение нескольких гипотез, обычно через particle/point-mass/marginalized particle filtering, с INS/ESKF и корректной моделью ошибки измерителя.

Полезные источники:

- Terrain-Aided Navigation Using a Point Cloud Measurement Sensor, arXiv 2025: point-cloud/terrain model, marginalized particle filters, сравнение radar altimeter vs point cloud. https://arxiv.org/pdf/2510.06470
- Terrain Referenced Navigation with Path Optimization, 2022: обзор state-of-the-art TRN и методов оценивания. https://www.diva-portal.org/smash/get/diva2%3A1705455/FULLTEXT01.pdf
- Enhanced Terrain-Referenced Navigation Through Adaptive Radar Altimetry Error Modelling, 2024: state/terrain-dependent radar-altimeter error model. https://link.springer.com/article/10.1007/s42405-024-00881-8
- Marginalized Particle Filter for Accurate and Reliable Terrain-Aided Navigation: совместное INS/TAP оценивание. https://people.isy.liu.se/rt/fredrik/reports/09TAES_nav.pdf

Примечание: две дополнительные PMC-страницы по ESKF/vision TRN были найдены, но при проверке открылись только через browser check, поэтому не включены как подтвержденные источники.

## 3. Цепочка локализации

### 3.1 Вход

Вход: NMEA-0183 GPGGA, поле высоты используется как радиовысотомер AGL. Барометрическая высота фиксирована: 1500 м MSL.

Выход: `NMEAFrame(timestamp_utc, radar_alt_m, valid)`.

Источники ошибки:

- checksum mismatch не удаляет значение высоты из `values_m`;
- джиттер частоты не используется в основном `measurement_step_m`;
- GPGGA используется нестандартно: поле altitude обычно MSL, но по кейсу это допустимая договоренность.

### 3.2 Measurement layer

Формула:

```python
h_terrain = 1500.0 - radar_agl - terrain_bias
```

Это соответствует кейсу. Основной риск - invalid frame может иметь числовую высоту и пройти в профиль как число, потому что `valid_mask` строится, но сами `values_m` не заменяются на `nan`.

### 3.3 DEM sampling

DEM читается целиком в `float64`, затем точки профиля берутся через `scipy.ndimage.map_coordinates`. Геодезические лучи строятся через `pyproj.Geod.fwd`.

Риски:

- вся карта в памяти;
- reference matrix перестраивается часто;
- profile sampling по 360 азимутам выполняется как множество отдельных вызовов;
- нет DEM uncertainty layer.

### 3.4 Correlation

Поиск: азимуты 0..359, смещения до `max_offset_m`, score = hybrid NCC/MSD.

Риски:

- один максимум не является надежным доказательством позиции;
- ambiguity flatten logic не учитывает 2D геометрию heatmap;
- нет полноценного phase correlation/DTW/MI fallback;
- score смешивает разные шкалы NCC и MSD через эвристики `alpha/beta/msd_scale_m2`.

### 3.5 Position solver

Берет `best_azimuth + offset`, проходит геодезически от start, затем до конца окна. Скорость после первого fix оценивает по двум последовательным fix.

Риск: если terrain fix ошибочный, скорость становится производной от ошибочных координат.

### 3.6 Fusion

Legacy path использует IMM после terrain decision. ESKF/PF реализованы, но не являются основным надежным контуром. PF есть в `--engine eskf`, но этот engine не является зрелой заменой legacy.

### 3.7 Visualization/SITL

Dashboard показывает карту, trajectory, probability ellipse, GNSS ON/OFF, latency/integrity. Это хорошо для демонстрации, но не улучшает навигационную точность.

## 4. Аудит по файлам

| Файл | Что делает | Качество | Главные риски |
|---|---|---|---|
| `constants.py` | Фиксированная баро-высота 1500 м | Нормально | Нет конфигурации/валидации сценария |
| `nmea_parser.py` | Парсит GPGGA из file/UDP | Средне | `float(radar_alt_token)` может упасть; invalid checksum не исключается из terrain values |
| `measurement_layer.py` | Преобразует AGL в MSL terrain profile | Хорошо по кейсу | `valid_mask` не применяется к `values_m`; timestamp fallback не связан с реальной частотой |
| `dem_loader.py` | DEM loading, bilinear sampling, patches | Хорошо для demo | Читает весь DEM; patch transform у краев может не совпасть с clipped patch; нет DEM uncertainty |
| `profile_extractor.py` | Строит reference profiles по азимутам | Рабоче | Перестройка 360 профилей дорогая; cache threshold слишком мал; нет route corridor/global precompute |
| `correlator.py` | NCC/MSD/feature correlation, ambiguity, CRLB | Главный модуль, неплохая база | Heuristic scoring; ambiguity flatten; weak reliability; нет multi-hypothesis output |
| `position_solver.py` | Переводит peak в координату и скорость | Аккуратно | Одиночная гипотеза; covariance эвристическая; speed зависит от качества fix |
| `imm_filter.py` | IMM hover/cruise/turn | Полезная заготовка | Модель линейная и принимает уже решенный fix; не исправляет мультимодальность |
| `eskf.py` | ESKF-like inertial core | Учебно-практичный | Без реального IMU/velocity updates быстро теряет смысл; нет строгой error-state математики |
| `terrain_pf.py` | Particle filter по DEM | Важная заготовка | Не default; медленный lat/lon loop; likelihood слишком прост; MAP/cov inconsistency |
| `sensor_fusion.py` | Adaptive R, gate, federated fusion | Черновой полезный слой | Не интегрирован в main; lidar/radar/camera не соответствуют текущему кейсу |
| `local_frame.py` | ENU/WGS84 helper | Хорошо для малых областей | Геодезический ENU не полноценный ECEF/ENU; на больших расстояниях линейность ограничена |
| `sim_generator.py` | Генерирует NMEA/GT/IMU, noise | Хорошо для demo | Синтетика может совпадать с моделью алгоритма; turn резкий и нереалистичный |
| `sitl_bridge.py` | MAVLink/SITL -> unified sample/NMEA | Хорошо для demo | Радиовысотомер синтезируется из DEM, не из SITL terrain/rangefinder; GNSS OFF только флаг |
| `main.py` | Оркестратор pipeline | Функционально, но перегружено | 2000+ строк; смешаны CLI, producer, correlator, filters, gating, UI payload |
| `visualizer.py` | Dash dashboard | Хорошо для защиты | UI не влияет на качество; locale частично русифицирован, внутренние modes еще английские |
| `benchmark_correlator.py` | Benchmark synthetic correlator | Полезно | Synthetic random benchmark не отражает DEM realism |
| `integration_test.py` | End-to-end smoke tests | Полезно | Искусственный DEM слишком благоприятен |
| `test_*.py` | Unit tests | Хорошее покрытие модулей | Мало property/adversarial tests для ambiguity, turns, route deviation |
| `README.md` | Инструкции запуска | Полезно | Нужно отделить claims от доказанных метрик |
| `IMPLEMENTATION_GUIDE.md` | Handoff guide | Полезно | Может устаревать относительно кода |
| `scripts/run_case_nmea_demo.ps1` | Demo script | Полезно | Нужно зафиксировать expected metrics |

Камерного пайплайна в проекте нет. Поэтому пункты CV-аудита (детекторы, дескрипторы, RANSAC, PnP, BA, optical flow, rolling shutter) неприменимы к текущему коду. Если камеру добавлять, она должна входить как отдельный измерительный канал: visual odometry / visual terrain matching -> covariance -> ESKF/PF update.

GPU/CUDA/OpenCL/TensorRT в проекте нет. Производительность сейчас CPU/NumPy/SciPy/rasterio.

## 5. Детальные критичные проблемы

### P01. Поворот ломает модель прямого профиля

Где: `main.py`, `select_processing_window`, `choose_navigation_fix`, `position_solver.py`.

Почему: окно из 50 кадров при 5 Гц и 50 м/с содержит около 490 м пути. Если внутри окна поворот, measured profile не лежит на одном DEM ray, но reference profile строится как один прямой азимут.

Влияние: Critical. Доказано: `scenario_turn mean=2855.60 m`, `rmse=3043.63 m`.

Точность: резко падает. Задержка: попытки reacquisition добавляют задержку. Вероятность: высокая при маневрах.

Исправление: перейти от ray-profile matching к polyline/path hypothesis matching. Для каждого candidate state строить эталон вдоль предполагаемой траектории за окно, а не вдоль одного азимута.

Пример:

```python
def sample_reference_along_path(dem, start_lat, start_lon, headings_deg, steps_m, geod):
    lat, lon = start_lat, start_lon
    values = []
    for heading, step_m in zip(headings_deg, steps_m):
        values.append(dem.get_elevation(lat, lon))
        lon, lat, _ = geod.fwd(lon, lat, heading, step_m)
    return np.asarray(values, dtype=float)
```

Ожидаемый прирост: на поворотах потенциально километры -> сотни/десятки метров при хорошей DEM-информативности. Задержка: может уменьшиться, потому что не нужен долгий reacquisition. Риск: нужен надежный heading prior.

### P02. Нет полноценной глобальной мультимодальной локализации

Где: `main.py`, `terrain_pf.py`.

Почему: legacy pipeline выбирает один максимум и вокруг него двигает window_start. PF есть, но не является default и не замкнут на correlation heatmap.

Влияние: Critical. В плоском/повторяющемся рельефе один максимум может быть ложным.

Исправление: сделать PF/point-mass filter главным источником позиции. Correlation heatmap должна обновлять веса гипотез, а не сразу отдавать одну координату.

```python
weights *= np.exp((heatmap_likelihood - heatmap_likelihood.max()) / temperature)
weights /= weights.sum()
fix = weighted_mean_or_map_cluster(particles, weights)
```

Ожидаемый прирост: высокая устойчивость при ambiguity; задержка может снизиться за счет локального поиска вокруг частиц. Риск: настройка числа частиц/температуры.

### P03. Cold start фактически не решен

Где: `main.py`, `resolve_initial_coordinates`, `pipeline_worker`.

Почему: старт берется из CLI lat/lon или center DEM, но нет настоящего поиска стартовой позиции по всей карте/коридору.

Влияние: Critical. Без GNSS и без начальной привязки система может уверенно идти от неверного места.

Исправление: acquisition mode: coarse grid по карте + azimuth search + voting по нескольким окнам.

```python
for cell in coarse_grid:
    for az in range(0, 360, 5):
        score = correlate_window(cell.lat, cell.lon, az)
        push_topk(cell, az, score)
```

Ожидаемый прирост: появится реальный GNSS-denied cold start. Задержка: выше на acquisition, ниже после lock. Риск: нужно ограничивать offline coverage.

### P04. Invalid NMEA может попадать в профиль как валидное число

Где: `measurement_layer.py`, `frames_to_terrain_profile`.

Почему: `valid_mask` считается, но `values_m` не маскируется.

Влияние: High. Ошибочный checksum/единицы могут исказить корреляцию.

Исправление:

```python
values_m = agl_to_terrain(...)
values_m = np.where(valid_mask, values_m, np.nan)
```

Прирост: устойчивость к плохому NMEA. Задержка: без изменения. Риск: если много invalid, нужно fallback.

### P05. Ambiguity metric считает пики по flattened heatmap

Где: `correlator.py`, `compute_ambiguity`.

Почему: индекс в flatten не является метрическим расстоянием в пространстве `(azimuth, offset)`. `distance=2` не означает ни 2 градуса, ни 2 offset.

Влияние: High. Может принять ложный peak как надежный или наоборот.

Исправление: 2D non-maximum suppression с метрикой по азимуту и смещению.

```python
from scipy.ndimage import maximum_filter
local_max = heatmap == maximum_filter(heatmap, size=(7, 7), mode="wrap")
peaks = np.argwhere(local_max & (heatmap >= peak * 0.8))
```

Прирост: меньше ложных terrain fixes. Задержка: малый рост. Риск: подобрать размер окна NMS.

### P06. Reliability thresholds эвристические и не калиброваны

Где: `correlator.py`: `PSR_THRESHOLD=1.3`, `peak>=0.5`, PSLR thresholds; `main.py`: confidence/offset thresholds.

Влияние: High. Порог может работать на текущей синтетике и ломаться на другой DEM.

Исправление: offline calibration на сценариях + ROC/PR curve + auto threshold по terrain observability.

Прирост: меньше ложных acceptance. Задержка: без изменения. Риск: нужен датасет.

### P07. `msd_scale_m2=100` слишком жестко связывает абсолютную высоту

Где: `correlator.py`.

Почему: 100 м2 = sigma около 10 м. При сезонном bias, лесах, DEM vertical error, баро bias score деградирует.

Влияние: High.

Исправление: robust Huber/Tukey loss + bias marginalization.

```python
residual = ref_window - h_meas
bias = np.nanmedian(residual)
loss = huber(residual - bias, delta=5.0)
score = np.exp(-np.mean(loss) / sigma2)
```

Прирост: устойчивость к bias/лесу/снегу. Задержка: малый рост. Риск: можно потерять абсолютную привязку на плоском рельефе.

### P08. Reference matrix строится вокруг текущей гипотезы и может закреплять ошибку

Где: `main.py`, `ProfileExtractor.build_reference_matrix`.

Почему: если `window_start_lat/lon` ушли, DEM reference строится уже вокруг неверного центра.

Влияние: High. Ошибка самоподдерживается.

Исправление: локальный spatial search всегда вокруг prediction covariance, а не одной точки; хранить несколько centers.

Прирост: лучше reacquisition. Задержка: рост без оптимизации; можно компенсировать vectorized DEM sampling.

### P09. PF не использует весь потенциал и не default

Где: `terrain_pf.py`, `main.py --engine eskf`.

Почему: particles обновляются по простому MSE, lat/lon conversion циклом, no corridor constraints, no multi-cluster output to UI/main.

Влияние: High.

Исправление: RBPF/PMF default for ambiguity; vectorize geodetic conversion или перейти на local raster coordinates.

Прирост: сильно лучше ambiguity/flat-to-informative transitions. Задержка: может снизиться при grid/vectorization. Риск: сложнее отладка.

### P10. ESKF engine не имеет реального IMU-motion prediction

Где: `main.py`, `make_level_imu_sample`, `eskf.py`.

Почему: synthetic IMU фактически не задает горизонтальное ускорение. Без velocity correction ESKF не несет реальной инерциальной информации.

Влияние: High.

Исправление: либо убрать как production path, либо подать реальные IMU/airspeed/Doppler и timestamp sync.

Прирост: честность архитектуры; с реальным IMU точность выше при потере terrain. Риск: нужны данные.

### P11. Flat terrain fallback честно деградирует, но не решает координаты

Где: `choose_navigation_fix`, `is_flat_terrain`.

Влияние: High. `scenario_flat mean=14034 m`.

Исправление: показывать "не локализуемо по рельефу" + удерживать covariance growth + использовать route prior/INS/airspeed, а не заявлять точную позицию.

### P12. Синтетика совпадает с моделью алгоритма

Где: `sim_generator.py`, tests.

Почему: измерения генерируются из той же DEM и похожей геометрии, что используется для локализации.

Влияние: High. Риск переоценки качества.

Исправление: DEM mismatch, vertical datum offset, resampling mismatch, forest/snow bias, independent DEM source.

### P13. Скорость зависит от качества последовательных fix

Где: `position_solver.py`.

Влияние: Medium/High. При ложной координате скорость может стать физически невозможной.

Исправление: velocity gating + bounded acceleration + smoothing from motion model.

### P14. `except Exception` скрывает классы отказов

Где: `main.py`, window processing.

Влияние: Medium/High. Можно молча уйти в coasting.

Исправление: ловить конкретные `ValueError`, `RasterioIOError`, numerical errors; писать reason в state.

### P15. Нет строгой синхронизации timestamps

Где: `main.py`, `measurement_layer.py`.

Влияние: Medium/High для live/SITL.

Исправление: вычислять `dt` и `step_m` по NMEA timestamps/monotonic ingest, а не только `freq_hz`.

## 6. ТОП-50 критичных проблем по влиянию на точность

1. Single-ray model на поворотах.
2. Нет полноценного global acquisition/cold start.
3. Нет default multi-hypothesis PF/PMF.
4. Flat terrain не локализуем, но pipeline может продолжать выдавать координату.
5. Ambiguity считается по flattened heatmap.
6. Reliability thresholds не калиброваны.
7. Invalid NMEA values не маскируются в `values_m`.
8. Reference center drift закрепляет ошибку.
9. Reacquisition search слишком локальный и эвристический.
10. PF не интегрирован в legacy acceptance.
11. ESKF path не имеет реальной инерциальной информации.
12. Correlation score смешивает NCC/MSD без статистической калибровки.
13. Нет robust loss для выбросов радиовысотомера.
14. Нет DEM vertical uncertainty.
15. Нет учета леса/снега/воды в measurement model.
16. Нет timestamp-based step length.
17. Скорость зависит от ошибочных terrain-fix.
18. Нет physical plausibility gate по ускорению/turn rate.
19. Нет route corridor prior.
20. Нет multi-window consistency voting для acquisition.
21. Нет delayed acceptance после ambiguity с кластерной проверкой.
22. `max_offset_m=2000` допускает далекие ложные offset.
23. `flat_terrain_threshold_m=15` фиксированный и не адаптирован к noise/length.
24. CRLB-like metric использует 1D gradient, не 2D terrain information.
25. `confidence = peak - max_sidelobe` плохо масштабируется для hybrid score.
26. `PSR_THRESHOLD=1.3` не имеет доказанной связи с error probability.
27. Не используются covariance ellipses для gating acceptance.
28. Нет learning/calibration на реальных полетах.
29. Синтетический turn мгновенный, но алгоритм не имеет continuous turn model.
30. DEM sampling не проверяет vertical datum consistency.
31. DEM nodata/water handling частично, но нет semantic mask в main likelihood.
32. SITL radar_alt генерируется из той же DEM, снижая независимость validation.
33. GNSS OFF в demo - флаг, не полноценный отказ источника позиции автопилота.
34. IMM принимает уже выбранный fix и не предотвращает ложный fix заранее.
35. IMM covariance может быть слишком оптимистичной при ложной корреляции.
36. Particle filter возвращает MAP, covariance считает вокруг mean.
37. Particle filter likelihood штрафует nodata константой, но не учитывает spatial structure.
38. No covariance inflation on repeated ambiguous windows beyond simple fallback.
39. No RTS smoother for replay/offline analysis.
40. Нет phase correlation fast global offset.
41. Нет DTW для speed variation/maneuver windows.
42. Нет mutual information fallback для nonlinear terrain relation.
43. Нет adaptive window objective на основе posterior entropy.
44. No map tiling/preloaded corridor strategy.
45. `LocalFrame` не ECEF/ENU; точность ограничена при больших расстояниях.
46. Визуализация может внушить точность, которой нет, если covariance плохо калибрована.
47. `except Exception` может скрыть DEM/out-of-bounds как обычный fallback.
48. Нет логов причин reject/accept в structured CSV.
49. Нет adversarial tests на периодический рельеф.
50. Нет hardware-in-loop latency profiling на целевом железе.

## 7. ТОП-30 быстрых улучшений

1. Маскировать invalid NMEA как `nan` в `frames_to_terrain_profile`.
2. Добавить `nmea_valid_fraction` в dashboard/metrics.
3. Заменить flattened ambiguity на 2D local maxima.
4. Добавить physical gate: max offset per update, max acceleration, max turn rate.
5. Логировать reason для каждого fallback/accept.
6. Добавить `accepted_fix_distance_from_prediction_m`.
7. Ввести robust Huber loss в MSD.
8. Сделать `msd_scale_m2` адаптивным от noise/terrain_bias.
9. Разделить `confidence_ncc`, `confidence_msd`, `confidence_final`.
10. Добавить route corridor prior для demo.
11. В `PositionSolver` reject speed jumps.
12. В `choose_navigation_fix` учитывать covariance growth.
13. Добавить replay CSV с per-window diagnostics.
14. Добавить tests на invalid checksum with numeric altitude.
15. Добавить tests на periodic terrain ambiguity.
16. Добавить tests на timestamp jitter.
17. Добавить flat terrain explicit "unobservable" status.
18. Добавить `DEM vertical datum warning`.
19. Добавить command `--expected-max-error` для demo validation.
20. Сохранить heatmap snapshots для turn windows.
21. Уменьшить misleading claims в README.
22. Перевести internal mode names для UI.
23. Добавить benchmark по real DEM profiles.
24. Добавить `--profile-step-m` отдельным параметром.
25. Добавить `--azimuth-step-deg` для coarse/fine.
26. Добавить entropy metric для heatmap.
27. Добавить top-k peaks output.
28. Добавить consistency vote по 3 последним окнам.
29. Сделать DEM patch out-of-bounds graceful.
30. Добавить sanity check: if flat + GNSS off -> no precise claim.

## 8. ТОП-20 архитектурных улучшений

1. Перейти на PF/PMF как главный localization state.
2. Разделить main.py на `pipeline`, `acquisition`, `tracking`, `dashboard_payload`.
3. Ввести state machine: `ACQUIRE`, `TRACK`, `AMBIGUOUS`, `COAST`, `REACQUIRE`.
4. Сделать correlation output multi-peak, не single result.
5. Path-based reference вместо single-ray reference.
6. Интегрировать ESKF с реальными IMU/timestamp или убрать из default.
7. Добавить route corridor prior как optional layer.
8. Добавить global coarse-to-fine map pyramid.
9. Добавить calibrated measurement model for radar altimeter.
10. Ввести DEM metadata/uncertainty layer.
11. Сделать sensor fusion measurement bus с timestamps.
12. Поддержать offline RTS smoother.
13. Добавить replay diagnostics artifact.
14. Добавить real/SITL independent radar/rangefinder input.
15. Отделить synthetic generator от validation benchmark.
16. Добавить batch/vectorized DEM sampler для many hypotheses.
17. Ввести plugin-like metric strategies: NCC, phase, Huber, DTW, MI.
18. Сделать configuration profiles: demo, strict, embedded.
19. Добавить structured logging JSONL.
20. Добавить acceptance calibration pipeline.

## 9. ТОП-20 оптимизаций производительности без ухудшения точности

1. Vectorize reference profiles for all azimuths in one DEM sampling batch.
2. Cache DEM raster-coordinate rays instead of lat/lon geodesic rays.
3. Coarse-to-fine azimuth search: 10° -> 3° -> 1°.
4. Batch FFT/NCC per azimuth.
5. Use local raster coordinates for PF instead of per-particle pyproj loop.
6. Precompute route corridor DEM tiles.
7. Use `sliding_window_view` carefully with memory bounds.
8. Keep top-k candidates and refine only top-k with features/DTW.
9. Use Numba/C++ for inner correlation/PF likelihood.
10. Avoid copying cached reference matrix on every reuse when immutable.
11. Use float32 for heatmaps if validated, keep geodesy/covariance float64.
12. Reuse arrays in correlator to reduce allocations.
13. Precompute Hann windows/features for reference profiles.
14. Build map pyramid for acquisition.
15. Reduce Dash payload size by downsampling DEM patch.
16. Make dashboard optional/lower priority thread.
17. Use producer timestamps and drop UI states, not sensor frames.
18. Preload DEM route patches.
19. Use process pool only if raster sampling releases GIL poorly; otherwise vectorize first.
20. Add hardware benchmark script for CM4/Jetson.

## 10. Roadmap

### Этап 1: быстрые победы

Сложность: 0.5-1 день.  
Точность: +10-30% устойчивости на шуме/ошибках ввода.  
Задержка: примерно без изменений или немного меньше за счет fewer false reacquisitions.  
Риски: низкие.

Задачи:

- invalid NMEA -> `nan`;
- 2D ambiguity NMS;
- physical gate;
- structured diagnostics CSV;
- UI status "локализация невозможна по рельефу";
- replay tests для invalid, flat, periodic, turn.

### Этап 2: средние улучшения

Сложность: 2-4 дня.  
Точность: сильный прирост на turn/reacquisition.  
Задержка: может снизиться после vectorization.  
Риски: средние.

Задачи:

- path-based reference для поворотов;
- top-k multi-peak correlation result;
- consistency voting;
- robust Huber/MSD with bias marginalization;
- adaptive window по entropy/observability;
- vectorized DEM profile batch.

### Этап 3: крупный рефакторинг

Сложность: 1-2 недели.  
Точность: переход от сотен/километровых провалов к устойчивому tracking при информативном DEM.  
Задержка: зависит от PF/grid оптимизации.  
Риски: высокие, нужен regression benchmark.

Задачи:

- state machine `ACQUIRE/TRACK/AMBIGUOUS/COAST/REACQUIRE`;
- PF/PMF как default state;
- ESKF только с реальным IMU/velocity source;
- route corridor prior;
- map pyramid acquisition.

### Этап 4: фундаментальная архитектура

Сложность: 2-6 недель.  
Точность: максимальная для соревнования и дальнейшего продукта.  
Задержка: можно довести до embedded real-time.  
Риски: требуется dataset и дисциплина экспериментов.

Задачи:

- marginalized PF / RBPF INS+TAN;
- calibrated radar altimeter error model;
- independent validation DEM/source;
- optional vision/point-cloud TRN;
- C++/Numba/CUDA acceleration only после фикса математики.

## 11. Минимальный следующий порядок исправлений

1. Исправить invalid NMEA masking.
2. Заменить ambiguity на 2D NMS.
3. Добавить physical gate для offset/speed/turn jumps.
4. Сделать per-window diagnostics CSV.
5. Переписать turn windows на path-based reference.
6. Вынести multi-hypothesis top-k.
7. Интегрировать PF/PMF в основной pipeline.

Именно этот порядок лучше всего соответствует принципу: сначала не принимать ложные координаты, затем повышать точность, и только потом агрессивно ускорять.
