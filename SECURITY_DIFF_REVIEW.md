# Security Diff Review

## Scope

Reviewed current working-tree changes in:

- `README.md`
- `case_reader.py`
- `case_runner.py`
- `main.py`
- `requirements.txt`
- `test_case_reader.py`
- `test_case_runner.py`
- `test_main.py`
- pending local web/backend additions centered around `web_backend.py` and `backend/`

Base revision:

- `f73940a065e163c2fb195932a3918bd326eec3f9`

Working-tree content digest:

- `de9a57efdaf7a105ace79dd6f9f710431bea472aac807fcf08c3178231dfe350`

Codex Security preflight:

- helper executed successfully with Python 3.11
- profile: `security_diff_scan`
- status: `ready`
- MCP workspace persistence hook was unavailable in this host, so the review below is a manual diff scan following the plugin workflow

## Findings

### 1. Variable-speed correlation geometry now depends on truth labels after GNSS loss

Severity: High
Status: Fixed

Files:

- [main.py](/Users/fosspor/DRON/main.py:791)
- [main.py](/Users/fosspor/DRON/main.py:837)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:213)

Why it is plausible:

- `_window_sampling_geometry()` derives travelled distance from `_sample_reference_latlon()`
- `_sample_reference_latlon()` prefers `truth_lat` and `truth_lon`
- in case mode, `iter_case_unified_samples()` keeps injecting truth coordinates even after GNSS is unavailable

Impact:

- post-dropout correlation geometry is being sized from oracle truth labels rather than only from information that a real deployed pipeline would still have
- this can materially overstate replay quality after GNSS loss and make validation results look better than a live or SITL-connected system could achieve

Recommended fix:

- after GNSS loss, derive metric distance from sensor-available motion only, such as speed and timestamps, or from estimated state, but not from `truth_*`

Resolution:

- `_window_sampling_geometry()` now uses only live-motion inputs, not `truth_lat` / `truth_lon`

### 2. Web backend allows config-directed arbitrary local file read/write within process privileges

Severity: High
Status: Mitigated

Files:

- [web_backend.py](/Users/fosspor/DRON/web_backend.py:401)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:426)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1028)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1035)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1066)

Why it is plausible:

- `POST /api/dataset/load` accepts a caller-supplied `config_path`
- that path is stored as the active dataset config
- `GET /api/settings` later reads that file
- `POST /api/settings` writes back to that same path with no workspace restriction

Security impact:

- any caller that can reach the local backend can coerce it into reading and overwriting arbitrary YAML files that the backend process can access
- the documented default bind is localhost, which reduces scope, but there is no auth layer and no path allowlist enforcing that inputs stay under the project or `input/`

Recommended fix:

- restrict `config_path` and all dataset file paths to an allowed project-local root
- reject absolute paths outside that root
- separate read-only dataset selection from writable config persistence

Resolution:

- `/api/settings` no longer follows arbitrary loaded dataset paths
- settings persistence is pinned to the project input config path and report-path writes are restricted to the project `output/` tree

Residual risk:

- `POST /api/dataset/load` still accepts a user-supplied `config_path` for dataset loading itself, so this remains local-tooling functionality that should stay bound to trusted local use

### 3. Localhost web backend exposes state-changing endpoints without authentication or CSRF defenses

Severity: Medium
Status: Mitigated

Files:

- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1086)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1122)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:1346)

Why it is plausible:

- the backend exposes dataset loading, replay controls, GNSS overrides, and config writes with no authentication
- CORS explicitly trusts `localhost` and `127.0.0.1` origins and allows credentials, all methods, and all headers

Impact:

- any page or tool running on those trusted local origins can drive backend state changes
- the implementation effectively treats localhost origin as a trust boundary, which is weaker than actual authentication

Recommended fix:

- require an explicit local auth token or session secret for state-changing endpoints
- if this is intentionally single-user local tooling, document that boundary clearly and avoid treating browser origin alone as authorization

Resolution:

- state-changing POST endpoints now enforce a same-origin local browser check
- CORS is reduced to the local UI origin set and credentials are disabled

Residual risk:

- this is still local engineering tooling without a full auth system, so it should remain bound to localhost-only deployment

### 4. Terrain-navigation duration metric is now inconsistent with actual pipeline mode transitions

Severity: Medium
Status: Fixed

Files:

- [case_reader.py](/Users/fosspor/DRON/case_reader.py:212)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:946)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:947)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:962)

Why it is plausible:

- the reader now correctly emits `INIT` after GNSS loss
- `/api/metrics` still computes `time_in_terrain_nav_s` from raw `dataset.unified_samples`
- those raw samples no longer represent the final pipeline transition into `TERRAIN_NAV`

Impact:

- UI and API metrics can report zero or understated terrain-navigation dwell time even when the algorithm did enter terrain navigation
- this weakens operator trust and can mislead acceptance testing

Recommended fix:

- compute mode-duration metrics from `pipeline_artifacts.report_records`, using final `mode` or `terrain_active`

Resolution:

- backend metrics now derive terrain-navigation dwell time from final pipeline records

### 5. Correlation heatmap offset axis still uses fixed first-sample speed and can mislabel variable-speed runs

Severity: Medium
Status: Fixed

Files:

- [web_backend.py](/Users/fosspor/DRON/web_backend.py:901)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:908)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:913)

Why it is plausible:

- the main pipeline now computes per-window metric sampling geometry
- the heatmap API still derives offset labels from `truth_rows[0].speed_mps / sample_rate_hz`

Impact:

- the returned heatmap can show visually incorrect offset axes for the very replay/case scenarios that now support variable speed
- the best peak may be numerically correct while the displayed offset scale is misleading

Recommended fix:

- persist the actual window `step_m` or full offset axis in pipeline artifacts and return that through the API

Resolution:

- pipeline now persists `correlation_step_m` and the backend uses it to label heatmap offsets

### 6. Radar timestamp validation rejects valid datasets that cross midnight

Severity: Medium
Status: Fixed

Files:

- [case_reader.py](/Users/fosspor/DRON/case_reader.py:96)
- [case_reader.py](/Users/fosspor/DRON/case_reader.py:180)

Why it is plausible:

- radar timestamps are compared as seconds-of-day relative to the first sample
- the validation does not account for `23:59:59 -> 00:00:00` rollover

Impact:

- valid overnight datasets can be rejected as having extreme timestamp drift
- this is a false-negative validation failure that will surprise operators during long or late-running captures

Recommended fix:

- normalize radar timestamps with explicit day-rollover handling before alignment checks

Resolution:

- radar timestamp alignment now handles midnight rollover before comparing relative timing

### 7. Report fields labeled as truth are not consistently true ground-truth values

Severity: Medium
Status: Fixed

Files:

- [main.py](/Users/fosspor/DRON/main.py:983)
- [main.py](/Users/fosspor/DRON/main.py:984)
- [main.py](/Users/fosspor/DRON/main.py:985)

Why it is plausible:

- `truth_alt_msl`, `truth_heading_deg`, and `truth_speed_mps` are populated from `latest_sample.alt_msl`, `effective_heading_deg`, and `effective_speed_mps`
- those fields are not guaranteed to be oracle truth values in every mode

Impact:

- downstream reports and UI can present measured or inferred values as truth
- that can skew operator interpretation and any follow-on analysis that trusts those labels literally

Recommended fix:

- populate `truth_*` fields only from explicit ground-truth sources, otherwise use neutral labels such as `source_*` or `measured_*`

Resolution:

- ambiguous `truth_*` report fields are no longer filled from non-truth values

### 8. Variable-speed fix introduces a performance regression by disabling reference-matrix cache reuse

Severity: Medium
Status: Fixed

Files:

- [main.py](/Users/fosspor/DRON/main.py:888)
- [main.py](/Users/fosspor/DRON/main.py:893)
- [main.py](/Users/fosspor/DRON/main.py:898)
- [profile_extractor.py](/Users/fosspor/DRON/profile_extractor.py:67)
- [profile_extractor.py](/Users/fosspor/DRON/profile_extractor.py:79)

Why it is plausible:

- `ProfileExtractor` caches reference matrices per instance
- the new code creates a fresh extractor and correlator on every window
- this prevents reuse of the existing cache even when the center moves only slightly

Impact:

- longer replays and low-power targets pay the full reference-build cost on each window
- this is especially relevant because performance optimization for constrained onboard compute is one of the stated goals

Recommended fix:

- retain extractor/correlator across windows and only rebuild when step or profile geometry materially changes

Resolution:

- extractor/correlator reuse is restored behind a geometry signature check

### 9. Pre-processing replay state can report `TERRAIN_NAV` before the pipeline has actually switched modes

Severity: Low
Status: Fixed

Files:

- [web_backend.py](/Users/fosspor/DRON/web_backend.py:783)
- [web_backend.py](/Users/fosspor/DRON/web_backend.py:794)

Why it is plausible:

- before processed records exist, `state()` derives `nav_mode` directly from `gnss_available`
- after GNSS loss that produces `TERRAIN_NAV` even though the pipeline transition logic has not run yet

Impact:

- transient UI inconsistency during processing startup
- less severe than the metrics issue above, but still semantically wrong

Recommended fix:

- use `INIT` while processing has not yet produced the first pipeline record

Resolution:

- pre-processing state now reports `INIT` instead of pretending terrain navigation is already active

## Notes

- No direct `eval`, shell injection, pickle deserialization, or obvious remote-code-execution sink was found in the reviewed Python diff.
- CORS is restricted to localhost origins, which helps, but it does not mitigate the path/write issue for any caller that can already reach the local service.
