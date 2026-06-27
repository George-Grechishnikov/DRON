# Commit Review Errors

## Scope

Reviewed commits:

- `82a76a8` — `Integrate unified stream, visualization, and correlation backends`
- `c93310b` — `Merge George and local terrain navigation updates`
- `f73940a` — `Add human2 unified stream and reporting workflow`

Primary findings are in the latest case/unified-stream integration path introduced by `f73940a`.

## Findings

### 1. Fixed-speed correlation sampling breaks replay/case runs with variable speed

Severity: High

Files:

- [main.py](/Users/fosspor/DRON/main.py:806)
- [main.py](/Users/fosspor/DRON/main.py:810)
- [main.py](/Users/fosspor/DRON/main.py:820)

Problem:

The replay/case pipeline computes `measurement_step_m` once from `config.speed_mps / config.freq_hz`, and then uses that constant value for:

- measured profile length
- reference profile sampling step
- correlation step

But in case mode `config.speed_mps` is seeded only from the first truth sample, while the actual incoming samples can have changing `speed_mps`.

Impact:

- the measured profile is mapped to the wrong spatial step whenever speed changes;
- offset and azimuth matching become physically inconsistent;
- long real replay runs can drift or show incorrect best-match peaks even when the raw data is valid.

Why this matters:

This directly affects the core algorithm result, not just the UI or reporting layer.

Recommended fix:

- derive profile step from the actual window samples, or
- resample the measured profile to metric distance using per-sample speed / dt, or
- constrain case mode explicitly to constant-speed input and validate that assumption.

### 2. Case reader reports `TERRAIN_NAV` immediately after GNSS loss, before the algorithm actually transitions

Severity: High

Files:

- [case_reader.py](/Users/fosspor/DRON/case_reader.py:127)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:139)
- [main.py](/Users/fosspor/DRON/main.py:866)
- [main.py](/Users/fosspor/DRON/main.py:868)

Problem:

`iter_case_unified_samples()` sets:

- `gnss_available = False`
- `nav_mode = "TERRAIN_NAV"`

immediately after `gnss_drop_after_s`.

At the same time, the pipeline itself has its own transition logic:

- if GNSS is unavailable
- and mode is still `INIT`
- and enough cold-start windows passed
- then it switches to `TERRAIN_NAV`

Because the sample already arrives labeled as `TERRAIN_NAV`, the pipeline’s own transition gating is bypassed.

Impact:

- reports can claim terrain navigation started earlier than it really did;
- event logs and timelines become semantically wrong;
- future SITL/live adapters can inherit misleading mode semantics from the file adapter.

Recommended fix:

- case reader should only describe source availability, not final navigation mode;
- after GNSS drop it should emit `INIT` or a neutral source mode;
- actual `TERRAIN_NAV` status should be decided inside the pipeline only.

### 3. Radar stream alignment ignores NMEA timestamps entirely and only zips by row order

Severity: Medium

Files:

- [case_reader.py](/Users/fosspor/DRON/case_reader.py:111)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:117)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:121)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:127)

Problem:

The case reader validates timestamp alignment only between:

- `truth.csv`
- `barometer.csv`

But for radar input it does this:

- reads all valid `NMEAFrame`s
- drops invalid ones
- checks only that the count matches truth rows
- zips the three sources by index

The parsed NMEA timestamp is never used for alignment.

Impact:

- one dropped or duplicated radar frame silently shifts all following samples;
- the run still “loads” if counts happen to match after filtering;
- error metrics and correlation can degrade for reasons that are very hard to diagnose.

Recommended fix:

- validate radar timestamps against truth/barometer timestamps;
- surface dropped invalid frames explicitly;
- fail fast when sequence timing diverges beyond tolerance.

### 4. `case_runner --validate-only` validates existence, not actual parseability

Severity: Medium

Files:

- [case_runner.py](/Users/fosspor/DRON/case_runner.py:22)
- [case_runner.py](/Users/fosspor/DRON/case_runner.py:30)
- [case_runner.py](/Users/fosspor/DRON/case_runner.py:53)

Problem:

The validation path checks only whether required files exist. It does not:

- parse `truth.csv`
- parse `barometer.csv`
- parse `radar_data.nmea`
- validate row counts
- validate alignment

So `--validate-only` can print success even though the actual run will fail immediately once parsing starts.

Impact:

- misleading operator feedback;
- weak preflight for external datasets;
- extra friction during handoff and demo use.

Recommended fix:

- make `--validate-only` call the same parsing/alignment layer as real case loading;
- return validation errors before the user starts the run.

## Summary

The most important issues are not cosmetic:

1. spatial sampling assumes fixed speed even for variable-speed replay data;
2. terrain-navigation mode is declared too early by the file adapter;
3. radar data is aligned by row count instead of timestamps.

These three points can materially distort algorithm quality, reported transitions, and replay trustworthiness even when the rest of the pipeline remains stable.
