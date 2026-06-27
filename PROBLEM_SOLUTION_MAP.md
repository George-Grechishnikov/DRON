# TERRAIN NAVIGATOR: связь проблем аудита с текущей версией

Версия кода: `main`, после пакета `Harden terrain fix gating` + top-k/timestamp hardening.

Цель файла: быстро объяснить, какие проблемы из аудита уже связаны с нашей реализацией, где это находится в коде, что можно честно говорить на защите и какие ограничения еще остаются.

## Короткий вывод

Текущая версия не превращает проект в полностью завершенную промышленную TRN-систему, но закрывает важный слой надежности вокруг текущего pipeline:

- входные NMEA-данные теперь безопаснее: invalid frame не портит профиль рельефа;
- ambiguity/PSLR стали ближе к реальной геометрии heatmap;
- terrain-fix теперь проходит физическую проверку перед принятием;
- prediction-state больше не расходится с heading, который уже выбран fallback-решением;
- коррелятор возвращает `top_candidates`, то есть несколько сильных гипотез вместо одного peak;
- physical gate может принять физически правдоподобную top-k альтернативу, если лучший peak невозможен;
- длительность окна теперь берется из NMEA timestamps, а `--freq` используется как fallback;
- fallback больше не строится от случайного local reacquisition candidate, если этот candidate не принят как terrain-fix;
- ambiguous heading candidate с большим offset игнорируется, чтобы не раскачивать прямой полет;
- straight/noisy replay снижены с сотен метров до сантиметрового уровня на текущей синтетике с известной стартовой точкой;
- turn replay улучшен примерно с `mean=2855.60 m` до `mean=1078.86 m`;
- тестовая база после текущего пакета: `117 passed`.

Правильная формулировка для жюри:

> Мы реализовали case-compliant pipeline по NMEA GPGGA радиовысотомера и DEM, добавили диагностику надежности, ambiguity gating, физическую проверку terrain-fix и визуализацию состояния. Последний commit закрывает критичные ошибки принятия ложных измерений, но глобальная мультимодальная локализация и полноценный cold start остаются направлением дальнейшего развития.

## Связка проблем с текущей версией

| ID | Проблема из аудита | Статус в текущей версии | Где связано в коде | Что изменилось |
|---|---|---|---|---|
| P01 | Поворот ломает модель прямого профиля | Частично закрыто | `main.py`, `update_motion_state_after_decision`, `maybe_update_heading_from_correlation`, turn gating/reacquisition logic | Убрано рассогласование курса и добавлена мягкая heading-подсказка из top-k при сильном маневренном сигнале. `scenario_turn mean` снижен примерно до `608 m`, но модель одного прямого профиля все еще ограничивает точность на маневрах. |
| P02 | Нет полноценной глобальной мультимодальной локализации | Частично подготовлено | `correlator.py`, `CorrelationCandidate`, `CorrelationResult.top_candidates`, `terrain_pf.py` | Коррелятор теперь возвращает top-k гипотезы, но PF/point-mass filter еще не стал основным default-контуром. |
| P03 | Cold start из неизвестной позиции не решен | Не закрыто | `main.py`, `resolve_initial_coordinates` | Система требует стартовую гипотезу через `--lat/--lon` или центр DEM. Это честно GNSS-assisted continuation / known-start режим, а не полный blind cold start. |
| P04 | Invalid NMEA может попасть в профиль как валидное число | Закрыто | `measurement_layer.py`, `frames_to_terrain_profile`; `test_measurement_layer.py` | После вычисления `valid_mask` invalid samples заменяются на `NaN`, поэтому коррелятор видит пропуск, а не ложную высоту. |
| P05 | Ambiguity metric считала пики по flattened heatmap | Закрыто для текущего heatmap | `correlator.py`, `compute_ambiguity`; `test_correlator.py` | Пики теперь ищутся в 2D-пространстве азимут-смещение через local maximum, с учетом цикличности азимута и исключением окрестности главного пика для PSLR. |
| P06 | Reliability thresholds эвристические | Частично закрыто | `correlator.py`, `main.py`, tests | Добавлены более строгие ambiguity/physical gates, но пороги все еще требуют калибровки на разных DEM и реальных сценариях. |
| P07 | MSD scale жестко связан с абсолютной высотой | Не закрыто полностью | `correlator.py` | Текущий hybrid NCC/MSD сохранен. Для следующей версии нужны адаптивный scale, phase correlation / MI / DTW fallback или калибровка по шуму радиовысотомера. |
| P08 | Reference matrix вокруг текущей гипотезы закрепляет ошибку | Частично смягчено | `main.py`, `correlator.py`, top-k + physical gates | Добавлена защита от физически невозможных фиксов и возможность выбрать сильную top-k альтернативу. Fallback больше не использует случайный local reacquisition center как старт prediction. Но фундаментально нужен 2D/multi-hypothesis search, а не одна стартовая точка. |
| P09 | PF есть, но не default | Не закрыто | `terrain_pf.py`, `main.py --engine eskf` | PF пока не основной рабочий режим. На текущих replay он не готов заменить legacy path. |
| P10 | ESKF без реального IMU-motion prediction | Не закрыто | `eskf.py`, `main.py --engine eskf` | ESKF остается экспериментальным. Для защиты лучше показывать legacy pipeline, а ESKF/PF описывать как следующий слой архитектуры. |
| P11 | Flat terrain fallback не решает координаты | Частично закрыто диагностически | `profile_extractor.py`, `main.py`, dashboard metrics | Система умеет сказать, что рельеф неинформативен, и уйти в prediction/fallback. Но точную координату в степи/равнине без информативного рельефа она физически не восстановит только по радиовысотомеру. |
| P12 | Валидация в основном на синтетике | Частично закрыто | `data/fabdem_canberra_wide.tif`, `sim_generator.py`, replay scripts | Используется реальный DEM FABDEM и NMEA replay, но радиовысотомер все еще синтетический. Нужен SITL/полевой лог/более реалистичный noise model. |
| P13 | Скорость зависит от качества последовательных fix | Частично закрыто | `main.py`, `terrain_fix_plausibility_reason`; `test_main.py` | Перед принятием fix проверяются speed jump, acceleration jump и turn jump. Если лучший peak невозможен, проверяются сильные top-k альтернативы. Ложные heading candidates с большим offset теперь не управляют prediction. |
| P14 | `except Exception` скрывает классы отказов | Частично закрыто по последствиям | `main.py`, fallback modes, runtime stats/dashboard | Ошибки еще не типизированы полностью, но навигация не принимает плохой fix молча: есть degraded/fallback modes и dashboard diagnostics. |
| P15 | Нет строгой синхронизации timestamps | Частично закрыто | `measurement_layer.py`, `main.py`, `estimate_window_duration_s`; `test_main.py` | Длительность окна берется из NMEA timestamps с fallback на `--freq`. Полностью переменный spatial sampling reference-профилей еще не внедрен. |

## Что именно добавили последние hardening-пакеты

### 1. Безопасность NMEA-профиля

Файл: `measurement_layer.py`.

Проблема: раньше invalid `NMEAFrame` мог иметь числовую высоту и попадать в `values_m`.

Решение:

```python
values_m = np.where(valid_mask, values_m, np.nan)
```

Эффект: поврежденная строка NMEA не становится ложным участком рельефа.

### 2. Геометрически корректная ambiguity-диагностика

Файл: `correlator.py`.

Проблема: heatmap `[azimuth, offset]` раньше фактически превращалась в 1D-массив. Это ломало смысл расстояния между пиками.

Решение:

- local maxima ищутся в 2D;
- азимут считается цикличным;
- sidelobe считается вне окрестности главного пика;
- `peak_isolation_m` учитывает offset и азимутальную дистанцию.

Эффект: меньше риск принять ложный максимум как надежный.

### 3. Physical gate перед принятием terrain-fix

Файл: `main.py`.

Проблема: ложный offset мог дать физически невозможный скачок координаты, а скорость потом считалась из двух fix и тоже становилась ложной.

Решение: перед `solver.solve(...)` строится preview-candidate через `solve_with_velocity(...)`, затем проверяется:

- максимальная допустимая скорость;
- максимальное ускорение;
- максимальный скачок курса.

Если кандидат невозможен, система не принимает terrain update, а уходит в prediction fallback.

### 4. Согласование heading-state после fallback

Файл: `main.py`.

Проблема: fallback-fix мог показывать один heading, а внутренний `current_azimuth` продолжал жить старым курсом. На `scenario_turn` это давало километровый уход.

Решение: добавлен `update_motion_state_after_decision(...)`. Теперь следующий prediction использует heading из принятого fallback/fix.

Эффект: turn replay улучшился:

| Сценарий | До аудита | После `ed304c6` |
|---|---:|---:|
| `scenario_turn mean` | `2855.60 m` | `1078.86 m` |
| `scenario_turn RMSE` | `3043.63 m` | `1466.88 m` |

После дополнительного bugfix fallback/reacquisition:

| Сценарий | Текущее значение |
|---|---:|
| `scenario_straight mean/RMSE` | `0.02 m / 0.02 m` |
| `scenario_noisy mean/RMSE` | `0.02 m / 0.02 m` |
| `scenario_turn mean/RMSE` | `608.02 m / 846.01 m` |

### 5. Top-k correlation candidates

Файлы: `correlator.py`, `main.py`.

Проблема: один максимум heatmap может быть ложным или физически невозможным.

Решение:

- `CorrelationResult.top_candidates` хранит несколько локальных максимумов;
- `choose_navigation_fix(...)` может проверить top-k альтернативы physical gate;
- ambiguous heading fallback может использовать сильную low-offset top-k подсказку только как мягкий курс, не как координату.

Эффект: система подготовлена к полноценной multi-hypothesis локализации и меньше зависит от одного peak.

### 6. Timestamp-aware window duration

Файл: `main.py`.

Проблема: pipeline считал длительность окна только как `(N-1)/freq`, хотя по кейсу поток NMEA может идти 1-10 Гц и иметь jitter.

Решение:

```python
window_duration = estimate_window_duration_s(active_frame_packets, config.freq_hz)
measurement_span_m = max(current_speed, 0.0) * window_duration
```

Эффект: скорость/длина окна стали ближе к реальному NMEA timing. `--freq` теперь fallback, а не единственный источник времени.

## Что говорить экспертам простым языком

Мы не обещаем, что система магически всегда дает точность <50 м на любой равнине. По физике задачи это невозможно, если рельеф не содержит информации: радиовысотомер видит только высоту под аппаратом, а на плоской местности одинаковые профили могут соответствовать множеству точек.

Наша версия делает три честные вещи:

1. Если рельеф информативен, ищет совпадение профиля радиовысотомера с DEM по азимуту и смещению.
2. Если совпадение неоднозначно, не выдает его как надежную координату, а показывает fallback/degraded состояние.
3. Если найденный fix физически невозможен, он отбрасывается и не ломает скорость/курс.

## Что осталось следующим этапом

Самый важный следующий шаг по точности:

1. Сделать PF/point-mass filter основным режимом, где heatmap/top-k обновляет веса гипотез.
2. Для поворотов заменить прямой profile ray на polyline/path-profile matching.
3. Для cold start добавить acquisition по coarse grid вокруг DEM/маршрутного коридора.
4. Для live/SITL сделать полностью variable-step reference sampling, а не только timestamp-aware duration.
5. Калибровать reliability/PSLR/MSD thresholds на нескольких DEM и шумовых моделях.

Эта дорожная карта напрямую продолжает текущую версию, а не противоречит ей: последний commit укрепил вход, диагностику и gates, чтобы следующий multi-hypothesis слой работал не на грязных данных.
