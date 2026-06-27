# Frontend Handoff

## Scope

This note is for the frontend team that will build the practical UI on top of the current DRON terrain-navigation backend.

Backend source of truth:

- [web_backend.py](/Users/fosspor/DRON/web_backend.py)
- [frontend/src/types.ts](/Users/fosspor/DRON/frontend/src/types.ts)
- [web_ui.html](/Users/fosspor/DRON/web_ui.html)

## Product Shape

This is not a marketing site.

The first screen should be the actual operator/engineering workspace:

- dataset load and validation
- replay controls
- truth vs estimated trajectory
- correlation heatmap
- altitude profiles
- metrics
- logs
- GNSS ON/OFF and terrain-navigation mode status

The UI should feel like a compact engineering console, not a landing page.

## Backend Base

Local default:

- `http://127.0.0.1:8000`

Swagger:

- `/docs`

Health:

- `GET /api/health`

## Important Security Contract

The backend is a local engineering service.

For mutating `POST` endpoints the browser client must send:

- a valid local browser `Origin` / `Referer`
- header `X-DRON-Session-Token`

Current local HTML UI gets the token from:

- `<meta name="dron-session-token" content="...">`

For the React frontend, use the same pattern:

1. load the shell page from the same backend origin, or
2. inject the token into the page/template before bootstrapping, or
3. if you proxy through the same origin, make sure the token is still available client-side for mutating calls

Without the token, `POST` calls will return `403`.

## Main Screens / Blocks

### 1. Dataset Panel

Inputs:

- `config_path`
- `dem_path`
- `radar_data_path`
- `truth_path`
- `barometer_path`

Actions:

- validate dataset
- load dataset

Endpoints:

- `GET /api/dataset/validate`
- `POST /api/dataset/load`

Good UX:

- show counts after load: radar / truth / barometer
- show sample rate and duration
- show validation errors inline, not only in console/logs

### 2. Replay Controls

Actions:

- start
- pause
- stop
- restart
- set speed
- step forward
- step backward
- optional GNSS force OFF / ON for debug

Endpoints:

- `POST /api/replay/start`
- `POST /api/replay/pause`
- `POST /api/replay/stop`
- `POST /api/replay/restart`
- `POST /api/replay/set_speed`
- `POST /api/replay/step_forward`
- `POST /api/replay/step_backward`
- `POST /api/gnss/force_off`
- `POST /api/gnss/force_on`

### 3. Current State Header

Endpoint:

- `GET /api/state`

Must surface:

- current timestamp
- sample index / total samples
- `gnss_available`
- `nav_mode`
- `sensors_status`
- current correlation score
- current error
- radar altitude
- barometric altitude
- terrain height
- estimate position
- truth position when available

Important mode semantics:

- `IDLE` = no dataset / inactive
- `READY` = dataset loaded but replay not processed yet
- `INIT` = pipeline not yet switched into terrain navigation
- `GNSS` = GNSS-backed mode
- `TERRAIN_NAV` = terrain navigation active

Do not collapse `INIT` and `TERRAIN_NAV` into one label.

### 4. Truth vs Estimated Trajectory

Endpoint:

- `GET /api/trajectory`

Response shape:

- `truth[]`
- `estimated[]`
- `events[]`

Events currently relevant:

- `GNSS_LOST`
- `TERRAIN_NAV_START`

UX expectations:

- truth track and estimated track must be visually distinct
- mark GNSS loss and terrain-nav start directly on the path
- if truth is absent, the view must still render estimated trajectory cleanly

### 5. Correlation Heatmap

Endpoint:

- `GET /api/correlation/heatmap`

Response:

- `azimuths`
- `offsets`
- `values`
- `best_azimuth`
- `best_offset`
- `best_score`

Important:

- offsets are already returned in meters
- use backend offsets as-is
- do not recompute axis spacing on the frontend

### 6. Profiles

Endpoint:

- `GET /api/profiles`

Series:

- `time`
- `baro_alt_m`
- `dem_height_m`
- `radar_alt_m`
- `reconstructed_profile_m`

UX expectation:

- this should be a multi-series time plot
- toggling series visibility is useful
- null gaps should not crash rendering

### 7. Timeline

Endpoint:

- `GET /api/timeline`

Response:

- `duration_s`
- `current_time_s`
- `segments[]`

Segment modes:

- `GNSS_ON`
- `GNSS_LOST`
- `TERRAIN_NAV`

This is a good place for a segmented horizontal strip or scrubber-style mode bar.

### 8. Metrics

Endpoint:

- `GET /api/metrics`

Key values:

- total flight time
- total distance
- average / max speed
- mean / max / RMSE / CEP errors
- average / min correlation
- time in GNSS
- time in terrain navigation
- time lost

Good UX:

- top-line KPI blocks
- expandable raw JSON only as secondary detail

### 9. Logs

Endpoint:

- `GET /api/logs`

Response:

- `logs[]` with `time`, `level`, `event`, `details`

Use a fixed-height scrollable log panel.

## Settings

Endpoints:

- `GET /api/settings`
- `POST /api/settings`

Safe editable fields currently expected:

- `sample_rate_hz`
- `gnss_drop_after_s`
- `dashboard_host`
- `dashboard_port`
- `report_path`
- `correlation.*`
- `visualization.*`

Frontend should treat settings editing as secondary, not as the primary operator workflow.

## Polling / Refresh

Recommended simple strategy:

- poll `GET /api/state` every `500-1000 ms` while replay is active
- poll `trajectory`, `heatmap`, `profiles`, `metrics`, `timeline`, `logs` every `1-2 s`
- after any replay control action, force one immediate refresh

## Error Handling

Backend error shape for failures is usually:

- `{"detail": "..."}`

Expected failure cases:

- `400` bad dataset/config input
- `403` missing origin or session token on mutating calls
- `404` heatmap not ready yet

Do not render empty white panels on these states.
Prefer explicit empty/loading/error states:

- `No dataset loaded`
- `Processing replay`
- `Heatmap not available yet`
- `Session token missing`

## Existing TypeScript Types

Start from:

- [frontend/src/types.ts](/Users/fosspor/DRON/frontend/src/types.ts)

But update them carefully if the UI needs:

- `position_error_m`
- `baro_alt_m`
- `playing`
- `processing`
- settings response typing

The backend contract is the authority.

## Visual Priorities

The first viewport should clearly show:

1. dataset/replay controls
2. live GNSS + mode status
3. trajectory plot
4. heatmap

The operator must be able to answer these questions quickly:

- Is the dataset loaded?
- Is GNSS currently ON or OFF?
- Has terrain navigation started?
- Does estimated trajectory follow truth?
- Is the correlation peak clean or ambiguous?

## Suggested Page Layout

One practical layout:

1. top status bar
   - backend health
   - dataset status
   - GNSS
   - nav mode
   - replay state
2. left upper control block
   - dataset paths
   - validate/load
   - replay controls
3. main content
   - trajectory
   - heatmap
   - profiles
4. right rail or lower band
   - metrics
   - timeline
   - logs

## Frontend Notes

- Keep the interface dense and calm.
- Avoid decorative cards-inside-cards.
- This is an engineering console, not a promo site.
- Preserve exact mode names from backend.
- Do not infer terrain-nav activation from GNSS loss; use backend `nav_mode` / timeline / events.

## Minimal Done Definition

Frontend handoff can be considered usable when:

- dataset can be loaded and validated from the UI
- replay can be controlled from the UI
- state badge area updates live
- truth vs estimated trajectory is visible
- heatmap is visible with correct offset axis
- GNSS loss and terrain-nav transition are visible
- logs and metrics are readable
- mutating requests include the session token correctly
