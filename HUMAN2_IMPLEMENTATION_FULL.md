# Human 2 Implementation Report

## Scope

This document covers only the work assigned to "person 2" from `IMPLEMENTATION_GUIDE.md`:

- integration of a unified sample stream into `main.py`
- visualization in `visualizer.py`
- truth vs estimated trajectory
- correlation heatmap
- GNSS ON/OFF statuses
- transition status into terrain navigation mode
- implementation prepared so later integration with `sitl_bridge.py` is straightforward

It does not describe person 1 tasks as owned work.

## What Was Implemented

### 1. Unified sample stream integrated into `main.py`

Added config-driven case ingestion and a single pipeline entry path so the main processing loop can consume case data in the same shape as replay/unified samples.

Implemented in:

- `main.py`
- `case_reader.py`
- `case_runner.py`

Key additions:

- `--config` support in `main.py`
- YAML config loading
- path resolution for DEM / NMEA / truth / barometer inputs
- `case` mode support in the main pipeline
- `case_config_producer(...)` to build unified samples from external files
- replay/case truth loading compatible with both old and new truth formats

This keeps the downstream pipeline stable and makes later connection to `sitl_bridge.py` easier because the pipeline now depends on a normalized sample interface instead of one-off file logic.

### 2. Infrastructure for external case inputs

Implemented a dedicated input reader layer that prepares a normalized sample stream from operator-provided files.

Implemented in:

- `case_reader.py`
- `case_runner.py`
- `input/README.md`
- `input/incoming/config.yaml`

Supported inputs:

- `radar_data.nmea`
- `truth.csv`
- `barometer.csv`
- DEM from `dem/`
- `config.yaml`

Provided:

- dataclasses for input records
- format validation
- alignment checks
- descriptive error handling
- unified per-sample interface

The resulting flow is:

1. user fills `config.yaml`
2. files are loaded and validated
3. the system yields normalized samples into the main pipeline

### 3. Visualization upgrades in `visualizer.py`

Expanded visualization/export so the output is understandable to both engineers and non-technical users.

Implemented in:

- `visualizer.py`

Added:

- operator-facing HTML report export
- truth vs estimated trajectory view
- correlation heatmap rendering
- correlation peak summary
- GNSS status display
- terrain navigation mode status display
- output summaries in text and JSON
- records export in CSV

Output artifacts now include:

- HTML report
- summary TXT
- summary JSON
- records CSV

### 4. Truth vs estimated trajectory

Added support for plotting and exporting comparison between:

- ground-truth trajectory
- estimated trajectory from terrain navigation / IMM result

Used in:

- report generation
- quality summary generation
- replay/case metric calculations

Metrics included in summaries:

- average trajectory error
- maximum trajectory error
- final trajectory error

### 5. Correlation heatmap

Added export and visualization of the correlation surface so the user can inspect:

- how the azimuth search behaves
- where the peak correlation is found
- whether the solution is sharp or ambiguous

This is now embedded in the HTML report and available from the latest processed windows.

### 6. GNSS ON/OFF status handling

Added operator-visible GNSS state reporting based on incoming samples.

Included in:

- pipeline record generation
- report summary
- final operator status outputs

The report now explicitly shows whether:

- GNSS loss was detected
- final GNSS state is ON or OFF

### 7. Terrain navigation mode transition status

Added explicit handling and export of transition into terrain-navigation operation.

Included in:

- pipeline state derivation
- report summary
- final operator outputs

The report now shows whether:

- terrain navigation mode was entered
- final system mode is `INIT` or `TERRAIN_NAV`

### 8. Stability fix for DEM boundary processing

During a long real-case run, a crash was found at the DEM boundary due to geodesic sampling touching the exact raster edge.

Fixed in:

- `dem_loader.py`

What changed:

- small tolerant coercion of points lying within a tiny epsilon around DEM bounds
- retained strict failure for truly out-of-bounds coordinates
- added a focused regression test

This fix is important for real input streams and future SITL-style integration because boundary jitter is normal in long trajectories.

## Files Added

- `case_reader.py`
- `case_runner.py`
- `input/README.md`
- `input/incoming/config.yaml`
- `scripts/stress_test_million_samples.py`
- `test_case_reader.py`
- `test_case_runner.py`
- `test_unified_stream_pipeline.py`
- `HUMAN2_IMPLEMENTATION_FULL.md`

## Files Updated

- `main.py`
- `visualizer.py`
- `dem_loader.py`
- `README.md`
- `requirements.txt`
- `scripts/preflight.py`
- `integration_test.py`
- `test_main.py`
- `test_visualizer.py`
- `test_dem_loader.py`

## Validation Performed

### Automated tests

Validated with the focused test set covering the new work:

- `test_dem_loader.py`
- `test_main.py`
- `test_case_reader.py`
- `test_case_runner.py`
- `test_visualizer.py`
- `test_unified_stream_pipeline.py`
- `integration_test.py`

Result:

- `30 passed`

### Case validation

Validated config-driven case input parsing using:

- `case_runner.py --validate-only`

### Real-case run

Executed a real long case run on the provided `input/incoming` dataset.

Observed:

- pipeline runs correctly
- report artifacts are generated correctly
- GNSS loss is detected
- transition to terrain navigation mode is detected
- report summary is produced for operator use

Also observed:

- the exhaustive correlation mode is still computationally heavy on the 50k input set
- current implementation is functionally ready, but full-length "combat" processing of very large inputs should be optimized further

## What the User Gets Now

The user can now provide a case folder with:

- DEM
- radar NMEA stream
- optional truth
- barometer
- config

and run:

```bash
.venv/bin/python case_runner.py --config input/incoming/config.yaml
```

The user receives:

- a visual HTML report
- a text summary
- a machine-readable JSON summary
- a CSV with processed records

## How This Was Kept Compatible with Future `sitl_bridge.py`

The implementation was intentionally structured around a normalized sample stream:

- file-specific parsing is isolated in `case_reader.py`
- main pipeline consumes unified samples
- reporting consumes generic processed records
- state/status logic is derived from sample fields, not hardcoded to a single source

Because of this, `sitl_bridge.py` can later provide the same normalized sample shape and reuse the existing pipeline/report stack with minimal glue code.

## Remaining Practical Next Steps

These are follow-up improvements, not blockers for the delivered scope:

1. optimize exhaustive correlation for long runs
2. add coarse-to-fine azimuth/offset search
3. cache/reference-matrix reuse improvements for large trajectories
4. add a streaming adapter from `sitl_bridge.py` to the unified sample interface
5. optionally expose a lighter operator UI mode for very large runs

## Bottom Line

The person 2 scope is implemented:

- unified stream integrated into `main.py`
- visualization implemented in `visualizer.py`
- truth vs estimated trajectory added
- correlation heatmap added
- GNSS ON/OFF statuses added
- terrain navigation mode transition status added
- case input infrastructure added
- future compatibility with `sitl_bridge.py` improved through a unified sample interface
