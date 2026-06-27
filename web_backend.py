"""FastAPI backend and practical local web UI for TERRAIN NAVIGATOR."""

from __future__ import annotations

import csv
import json
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from pyproj import Geod

from case_reader import CaseInputConfig, iter_case_unified_samples, load_barometer_samples, load_truth_samples
from main import (
    Config,
    PipelineArtifacts,
    configure_logging,
    load_yaml_config,
    run_pipeline_capture,
)
from nmea_parser import NMEAReader


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "input" / "incoming" / "config.yaml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "web_ui_report.html"
WGS84_GEOD = Geod(ellps="WGS84")
LOG_PATH = PROJECT_ROOT / "terrain_navigator_web.log"
LOCAL_UI_ORIGINS = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}
SESSION_TOKEN_PLACEHOLDER = "__DRON_SESSION_TOKEN__"

API_TAGS = [
    {"name": "system", "description": "Service health, runtime state, and operator logs."},
    {"name": "dataset", "description": "Dataset loading and validation for DEM, radar NMEA, truth, and barometer inputs."},
    {"name": "replay", "description": "Replay lifecycle controls for start, pause, stop, and frame stepping."},
    {"name": "analytics", "description": "Trajectory, correlation, profile, and metric outputs produced by the terrain-navigation pipeline."},
    {"name": "settings", "description": "Read and persist safe configuration values backed by `config.yaml`."},
    {"name": "ui", "description": "Local engineering web interface."},
]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    backend: str = Field(examples=["running"])


class DatasetLoadRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dem_path": "input/incoming/dem/terrain.tif",
                "radar_data_path": "input/incoming/radar_data.nmea",
                "truth_path": "input/incoming/truth.csv",
                "barometer_path": "input/incoming/barometer.csv",
                "config_path": "input/incoming/config.yaml",
            }
        }
    )
    dem_path: str | None = Field(default=None, description="Path to a DEM file or DEM directory.")
    radar_data_path: str | None = Field(default=None, description="Path to the NMEA-0183 radar-altimeter stream.")
    truth_path: str | None = Field(default=None, description="Path to truth.csv used for quality evaluation.")
    barometer_path: str | None = Field(default=None, description="Path to barometer.csv with barometric altitude.")
    config_path: str | None = Field(default=None, description="Path to config.yaml. Relative paths are resolved from this file.")


class DatasetLoadResponse(BaseModel):
    loaded: bool
    radar_samples: int
    truth_samples: int
    barometer_samples: int
    sample_rate_hz: float
    duration_s: float
    errors: list[str]


class DatasetValidateResponse(BaseModel):
    valid: bool
    dem: str
    radar: str
    truth: str
    barometer: str
    errors: list[str]


class ReplayStartRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"speed": 1.0}})
    speed: float = Field(default=1.0, gt=0.0, description="Replay speed multiplier.")


class ReplayActionResponse(BaseModel):
    started: bool | None = None
    paused: bool | None = None
    stopped: bool | None = None
    restarted: bool | None = None
    updated: bool | None = None
    processing: bool | None = None
    sample_index: int | None = None
    speed: float | None = None


class PositionState(BaseModel):
    lat: float | None = None
    lon: float | None = None
    alt_msl: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None


class ReplayStateResponse(BaseModel):
    timestamp: float
    elapsed_time: str | None = None
    sample_index: int
    total_samples: int
    gnss_available: bool
    nav_mode: str
    sensors_status: str
    data_rate_hz: float | None = None
    sample_rate_hz: float | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    alt_msl: float | None = None
    truth: PositionState | None = None
    estimate: PositionState | None = None
    radar_alt_m: float | None = None
    baro_alt_m: float | None = None
    terrain_h: float | None = None
    correlation_score: float | None = None
    position_error_m: float | None = None
    position_error_2d_m: float | None = None
    position_error_3d_m: float | None = None
    distance_km: float | None = None
    processing: bool | None = None
    playing: bool | None = None
    error: str | None = None


class TrajectoryPointResponse(BaseModel):
    timestamp: float
    lat: float | None = None
    lon: float | None = None


class ReplayEventResponse(BaseModel):
    timestamp: float
    type: str


class TrajectoryResponse(BaseModel):
    truth: list[TrajectoryPointResponse]
    estimated: list[TrajectoryPointResponse]
    events: list[ReplayEventResponse]


class HeatmapResponse(BaseModel):
    azimuths: list[int]
    offsets: list[float]
    values: list[list[float]]
    best_azimuth: float | None = None
    best_offset: float | None = None
    best_score: float | None = None


class ProfilesResponse(BaseModel):
    time: list[float | None]
    baro_alt_m: list[float | None]
    dem_height_m: list[float | None]
    radar_alt_m: list[float | None]
    reconstructed_profile_m: list[float | None]


class MetricsResponse(BaseModel):
    total_flight_time_s: float
    total_distance_m: float
    average_speed_mps: float
    max_speed_mps: float
    mean_position_error_m: float | None = None
    max_position_error_m: float | None = None
    rmse_m: float | None = None
    cep50_m: float | None = None
    cep95_m: float | None = None
    average_correlation_score: float | None = None
    min_correlation_score: float | None = None
    time_in_gnss_s: float
    time_in_terrain_nav_s: float
    time_lost_s: float


class LogEntryResponse(BaseModel):
    time: str
    level: str
    event: str
    details: str
    message: str | None = None


class LogsResponse(BaseModel):
    logs: list[LogEntryResponse]


class TimelineSegmentResponse(BaseModel):
    start: float
    end: float
    mode: str


class TimelineResponse(BaseModel):
    duration_s: float
    current_time_s: float
    segments: list[TimelineSegmentResponse]


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_rate_hz": 5.0,
                "gnss_drop_after_s": 10.0,
                "dashboard_host": "127.0.0.1",
                "dashboard_port": 8050,
                "report_path": "output/terrain_navigator_report.html",
                "correlation": {
                    "max_offset_m": 2000.0,
                    "flat_terrain_threshold_m": 15.0,
                    "cold_start_windows": 3,
                },
                "visualization": {
                    "enabled": False,
                    "export_report_path": "output/terrain_navigator_report.html",
                },
            }
        }
    )
    sample_rate_hz: float | None = None
    gnss_drop_after_s: float | None = None
    dashboard_host: str | None = None
    dashboard_port: int | None = None
    report_path: str | None = None
    correlation: dict[str, Any] | None = None
    visualization: dict[str, Any] | None = None


class SettingsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    config_path: str | None = None


class SettingsSaveResponse(BaseModel):
    saved: bool
    config_path: str


@dataclass
class LoadedDataset:
    config_path: Path
    case_config: CaseInputConfig
    config_payload: dict[str, Any]
    dem_path: Path
    radar_data_path: Path
    truth_path: Path
    barometer_path: Path
    truth_rows: list[Any]
    barometer_rows: list[Any]
    radar_samples: list[float]
    unified_samples: list[dict[str, Any]]
    duration_s: float
    demo_mode: bool = False


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_series(values: list[Any], limit: int | None = None) -> list[float | None]:
    if limit is not None and limit > 0 and len(values) > limit:
        step = max(1, math.ceil(len(values) / limit))
        values = values[::step]
    return [_safe_float(value) for value in values]


def _coerce_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve() if path.is_absolute() else path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_settings_report_path(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    candidate = Path(value).expanduser()
    resolved = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    output_root = (PROJECT_ROOT / "output").resolve()
    if not _is_relative_to(resolved, output_root):
        raise HTTPException(status_code=400, detail="report_path must stay under the project output directory")
    return str(resolved)


def _is_project_settings_path(path: Path) -> bool:
    return _is_relative_to(path, PROJECT_ROOT / "input")


def _verify_browser_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin:
        if origin not in LOCAL_UI_ORIGINS:
            raise HTTPException(status_code=403, detail="Cross-origin browser access is not allowed")
        return
    referer = request.headers.get("referer")
    if referer:
        parsed = urlparse(referer)
        referer_origin = f"{parsed.scheme}://{parsed.netloc}"
        if referer_origin not in LOCAL_UI_ORIGINS:
            raise HTTPException(status_code=403, detail="Cross-origin browser access is not allowed")


def _verify_session_token(request: Request, expected_token: str) -> None:
    provided_token = request.headers.get("x-dron-session-token")
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=403, detail="Missing or invalid session token")


def _first_dem_in(path: Path) -> Path:
    if path.is_file():
        return path
    for pattern in ("*.tif", "*.tiff", "*.hgt"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No DEM file found in {path}")


def _read_valid_radar_samples(path: Path) -> list[float]:
    reader = NMEAReader.from_file(path)
    try:
        return [float(frame.radar_alt_m) for frame in reader if frame.valid]
    finally:
        reader.close()


def _load_records_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if value in ("", None):
                normalized[key] = None
                continue
            if value in {"True", "False"}:
                normalized[key] = value == "True"
                continue
            try:
                number = float(value)
            except ValueError:
                normalized[key] = value
            else:
                normalized[key] = int(number) if number.is_integer() else number
        parsed.append(normalized)
    return parsed


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, distance_m = WGS84_GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance_m)


def _compute_distance_and_speed(dataset: LoadedDataset) -> tuple[float, float, float]:
    total_distance_m = 0.0
    speeds = [float(item.speed_mps) for item in dataset.truth_rows]
    for previous, current in zip(dataset.truth_rows, dataset.truth_rows[1:]):
        total_distance_m += _distance_m(previous.lat, previous.lon, current.lat, current.lon)
    average_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    return total_distance_m, average_speed, max_speed


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TerrainNavigationController:
    """Owns loaded dataset state, replay state, and web-facing artifacts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.loaded_dataset: LoadedDataset | None = None
        self.pipeline_artifacts: PipelineArtifacts | None = None
        self.report_summary: dict[str, Any] | None = None
        self.processing_thread: threading.Thread | None = None
        self.processing_error: str | None = None
        self.processing: bool = False
        self.playing: bool = False
        self.playback_speed: float = 1.0
        self.playback_index: int = 0
        self.playback_time_s: float = 0.0
        self._last_tick_monotonic: float | None = None
        self.force_gnss: bool | None = None
        self.logs: deque[dict[str, str]] = deque(maxlen=400)
        self.settings_config_path: Path = DEFAULT_CONFIG_PATH
        self.session_token = secrets.token_urlsafe(24)
        configure_logging("INFO", LOG_PATH)

    def log(self, level: str, message: str, details: str = "") -> None:
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "level": level.upper(),
            "event": message,
            "details": details,
            "message": message,
        }
        with self._lock:
            self.logs.append(entry)

    def health(self) -> dict[str, str]:
        return {"status": "ok", "backend": "running"}

    def _resolve_dataset_request(self, request: DatasetLoadRequest) -> tuple[Path, dict[str, Any]]:
        config_path = _coerce_path(request.config_path) or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        payload = load_yaml_config(config_path)
        base_dir = config_path.parent

        def choose(key: str, explicit_value: str | None) -> Path:
            raw_value = explicit_value if explicit_value not in (None, "") else payload.get(key)
            path = _coerce_path(None if raw_value is None else str(raw_value), base_dir)
            if path is None:
                raise FileNotFoundError(f"Missing required path for {key}")
            return path

        dem_path = _first_dem_in(choose("dem_path", request.dem_path))
        radar_data_path = choose("radar_data_path", request.radar_data_path)
        truth_path = choose("truth_path", request.truth_path)
        barometer_path = choose("barometer_path", request.barometer_path)

        payload["dem_path"] = str(dem_path)
        payload["radar_data_path"] = str(radar_data_path)
        payload["truth_path"] = str(truth_path)
        payload["barometer_path"] = str(barometer_path)
        return config_path, payload

    def load_dataset(self, request: DatasetLoadRequest) -> dict[str, Any]:
        config_path, payload = self._resolve_dataset_request(request)
        sample_rate_hz = float(payload.get("sample_rate_hz", 5.0))
        gnss_drop_after_s = payload.get("gnss_drop_after_s")
        case_config = CaseInputConfig(
            dem_path=Path(payload["dem_path"]),
            radar_data_path=Path(payload["radar_data_path"]),
            truth_path=Path(payload["truth_path"]),
            barometer_path=Path(payload["barometer_path"]),
            sample_rate_hz=sample_rate_hz,
            gnss_drop_after_s=None if gnss_drop_after_s is None else float(gnss_drop_after_s),
        )

        truth_rows = load_truth_samples(case_config.truth_path)
        barometer_rows = load_barometer_samples(case_config.barometer_path)
        radar_samples = _read_valid_radar_samples(case_config.radar_data_path)
        unified_samples = [{**sample, "total_samples": len(radar_samples)} for sample in iter_case_unified_samples(case_config)]
        if len(unified_samples) != len(radar_samples):
            raise ValueError("Unified sample count does not match radar sample count")
        duration_s = (
            float(unified_samples[-1]["timestamp_s"]) - float(unified_samples[0]["timestamp_s"])
            if len(unified_samples) >= 2
            else 0.0
        )

        loaded = LoadedDataset(
            config_path=config_path,
            case_config=case_config,
            config_payload=payload,
            dem_path=case_config.dem_path,
            radar_data_path=case_config.radar_data_path,
            truth_path=case_config.truth_path,
            barometer_path=case_config.barometer_path,
            truth_rows=truth_rows,
            barometer_rows=barometer_rows,
            radar_samples=radar_samples,
            unified_samples=unified_samples,
            duration_s=duration_s,
        )

        with self._lock:
            self.loaded_dataset = loaded
            self.pipeline_artifacts = None
            self.report_summary = None
            self.processing_error = None
            self.processing = False
            self.playing = False
            self.playback_index = 0
            self.playback_time_s = 0.0
            self._last_tick_monotonic = None
            self.force_gnss = None
            if _is_project_settings_path(config_path):
                self.settings_config_path = config_path

        self.log("INFO", "Dataset loaded")
        return {
            "loaded": True,
            "radar_samples": len(radar_samples),
            "truth_samples": len(truth_rows),
            "barometer_samples": len(barometer_rows),
            "sample_rate_hz": sample_rate_hz,
            "duration_s": duration_s,
            "errors": [],
        }

    def validate_dataset(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            request = DatasetLoadRequest(config_path=str(DEFAULT_CONFIG_PATH))
            try:
                self.load_dataset(request)
            except Exception as exc:
                return {
                    "valid": False,
                    "dem": "missing",
                    "radar": "missing",
                    "truth": "missing",
                    "barometer": "missing",
                    "errors": [str(exc)],
                }
            dataset = self.loaded_dataset
        assert dataset is not None
        errors: list[str] = []
        for path in (dataset.dem_path, dataset.radar_data_path, dataset.truth_path, dataset.barometer_path):
            if not path.exists():
                errors.append(f"Missing file: {path}")
        valid = not errors
        if valid:
            self.log("INFO", "Dataset validated")
        return {
            "valid": valid,
            "dem": "loaded" if dataset.dem_path.exists() else "missing",
            "radar": "ok" if dataset.radar_data_path.exists() else "missing",
            "truth": "ok" if dataset.truth_path.exists() else "missing",
            "barometer": "ok" if dataset.barometer_path.exists() else "missing",
            "errors": errors,
        }

    def _build_pipeline_config(self, dataset: LoadedDataset) -> Config:
        payload = dataset.config_payload
        correlation = payload.get("correlation", {}) if isinstance(payload.get("correlation"), dict) else {}
        visualization = payload.get("visualization", {}) if isinstance(payload.get("visualization"), dict) else {}
        return Config(
            mode="case",
            dem_path=dataset.dem_path,
            start_lat=dataset.truth_rows[0].lat if dataset.truth_rows else 60.5,
            start_lon=dataset.truth_rows[0].lon if dataset.truth_rows else 90.3,
            trajectory=1,
            config_path=dataset.config_path,
            nmea_path=dataset.radar_data_path,
            samples_path=None,
            gt_path=dataset.truth_path,
            barometer_path=dataset.barometer_path,
            udp_host="127.0.0.1",
            udp_port=10110,
            dashboard_host=str(visualization.get("dashboard_host", "127.0.0.1")),
            dashboard_port=int(visualization.get("dashboard_port", 8050)),
            enable_visualizer=False,
            seed=42,
            speed_mps=float(dataset.truth_rows[0].speed_mps if dataset.truth_rows else 50.0),
            altitude_msl_m=float(dataset.barometer_rows[0].baro_alt_m if dataset.barometer_rows else 1500.0),
            noise_sigma=0.0,
            window_size=50,
            step_size=10,
            freq_hz=float(dataset.case_config.sample_rate_hz),
            dem_patch_radius_m=5000.0,
            max_offset_m=float(correlation.get("max_offset_m", 2000.0)),
            flat_terrain_threshold_m=float(correlation.get("flat_terrain_threshold_m", 15.0)),
            cold_start_windows=int(correlation.get("cold_start_windows", 3)),
            gnss_drop_after_s=dataset.case_config.gnss_drop_after_s,
            report_path=Path(str(visualization.get("export_report_path", DEFAULT_REPORT_PATH))),
            auto_open_report=False,
            log_level="INFO",
        )

    def _process_dataset(self) -> None:
        dataset = self.loaded_dataset
        if dataset is None:
            with self._lock:
                self.processing = False
                self.processing_error = "Dataset not loaded"
            return
        try:
            self.log("INFO", "Replay processing started")
            config = self._build_pipeline_config(dataset)
            artifacts = run_pipeline_capture(config)
            summary_path = config.report_path.with_name(f"{config.report_path.stem}.summary.json")
            report_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            records_csv = config.report_path.with_name(f"{config.report_path.stem}.records.csv")
            records = _load_records_csv(records_csv)
            if records:
                artifacts = PipelineArtifacts(
                    history=artifacts.history,
                    metrics=artifacts.metrics,
                    report_records=records,
                    report_context=artifacts.report_context,
                )
            with self._lock:
                self.pipeline_artifacts = artifacts
                self.report_summary = report_summary
                self.processing = False
                self.processing_error = None
                self.playback_index = 0
                self.playback_time_s = float(artifacts.report_records[0]["timestamp"]) if artifacts.report_records else 0.0
                self._last_tick_monotonic = time.monotonic()
                self.playing = True
            self._register_runtime_events()
            self.log("INFO", "Replay processing completed")
        except Exception as exc:
            with self._lock:
                self.processing = False
                self.processing_error = str(exc)
                self.playing = False
            self.log("ERROR", f"Replay processing failed: {exc}")

    def _ensure_processed_async(self) -> None:
        with self._lock:
            if self.processing:
                return
            if self.pipeline_artifacts is not None:
                return
            self.processing = True
            self.processing_error = None
            self.processing_thread = threading.Thread(target=self._process_dataset, name="web-replay", daemon=True)
            self.processing_thread.start()

    def start_replay(self, speed: float) -> dict[str, Any]:
        if speed <= 0:
            raise HTTPException(status_code=400, detail="speed must be positive")
        if self.loaded_dataset is None:
            raise HTTPException(status_code=400, detail="Dataset is not loaded")
        with self._lock:
            self.playback_speed = float(speed)
            self._last_tick_monotonic = time.monotonic()
            if self.pipeline_artifacts is not None:
                self.playing = True
                self.log("INFO", f"Replay started at x{speed:.2f}")
                return {"started": True, "processing": False}
        self._ensure_processed_async()
        self.log("INFO", f"Replay queued at x{speed:.2f}")
        return {"started": True, "processing": True}

    def pause_replay(self) -> dict[str, Any]:
        with self._lock:
            self._advance_playback_locked()
            self.playing = False
            self._last_tick_monotonic = None
        self.log("INFO", "Replay paused")
        return {"paused": True}

    def stop_replay(self) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            self.playback_index = 0
            self._last_tick_monotonic = None
            if self.pipeline_artifacts and self.pipeline_artifacts.report_records:
                self.playback_time_s = float(self.pipeline_artifacts.report_records[0]["timestamp"])
            else:
                self.playback_time_s = 0.0
        self.log("INFO", "Replay stopped")
        return {"stopped": True}

    def restart_replay(self) -> dict[str, Any]:
        with self._lock:
            self.playback_index = 0
            self._last_tick_monotonic = time.monotonic()
            self.playing = True
            self.force_gnss = None
            if self.pipeline_artifacts and self.pipeline_artifacts.report_records:
                self.playback_time_s = float(self.pipeline_artifacts.report_records[0]["timestamp"])
            else:
                self.playback_time_s = 0.0
        self.log("INFO", "Replay restarted")
        return {"restarted": True, "processing": self.pipeline_artifacts is None}

    def set_speed(self, speed: float) -> dict[str, Any]:
        if speed <= 0:
            raise HTTPException(status_code=400, detail="speed must be positive")
        with self._lock:
            self.playback_speed = float(speed)
            self._last_tick_monotonic = time.monotonic()
        self.log("INFO", f"Replay speed updated to x{speed:.2f}")
        return {"updated": True, "speed": float(speed)}

    def force_gnss_off(self) -> dict[str, Any]:
        with self._lock:
            self.force_gnss = False
        self.log("WARN", "GNSS signal lost", "Forced from UI control")
        return {"updated": True}

    def force_gnss_on(self) -> dict[str, Any]:
        with self._lock:
            self.force_gnss = True
        self.log("INFO", "GNSS restored", "Forced from UI control")
        return {"updated": True}

    def step_forward(self) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            self._last_tick_monotonic = None
            self._shift_index_locked(1)
            return {"sample_index": self._current_record_locked().get("index", 0) if self._current_record_locked() else 0}

    def step_backward(self) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            self._last_tick_monotonic = None
            self._shift_index_locked(-1)
            return {"sample_index": self._current_record_locked().get("index", 0) if self._current_record_locked() else 0}

    def _shift_index_locked(self, delta: int) -> None:
        if self.pipeline_artifacts is None or not self.pipeline_artifacts.report_records:
            return
        last_index = len(self.pipeline_artifacts.report_records) - 1
        self.playback_index = min(max(self.playback_index + delta, 0), last_index)
        self.playback_time_s = float(self.pipeline_artifacts.report_records[self.playback_index]["timestamp"])

    def _advance_playback_locked(self) -> None:
        if not self.playing or self.pipeline_artifacts is None or not self.pipeline_artifacts.report_records:
            return
        now = time.monotonic()
        if self._last_tick_monotonic is None:
            self._last_tick_monotonic = now
            return
        elapsed = now - self._last_tick_monotonic
        self._last_tick_monotonic = now
        self.playback_time_s += elapsed * self.playback_speed
        records = self.pipeline_artifacts.report_records
        while self.playback_index + 1 < len(records) and float(records[self.playback_index + 1]["timestamp"]) <= self.playback_time_s:
            self.playback_index += 1
        if self.playback_index >= len(records) - 1:
            self.playback_index = len(records) - 1
            self.playback_time_s = float(records[self.playback_index]["timestamp"])
            self.playing = False

    def _current_record_locked(self) -> dict[str, Any] | None:
        if self.pipeline_artifacts is None or not self.pipeline_artifacts.report_records:
            return None
        self._advance_playback_locked()
        self.playback_index = min(self.playback_index, len(self.pipeline_artifacts.report_records) - 1)
        return self.pipeline_artifacts.report_records[self.playback_index]

    def _raw_sample_for_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        dataset = self.loaded_dataset
        if dataset is None:
            return None
        raw_index = int(record.get("index", 0))
        if 0 <= raw_index < len(dataset.unified_samples):
            return dataset.unified_samples[raw_index]
        return None

    def _register_runtime_events(self) -> None:
        artifacts = self.pipeline_artifacts
        if artifacts is None:
            return
        records = artifacts.report_records
        gnss_event = next((row for row in records if not bool(row.get("gnss_available", True))), None)
        terrain_event = next((row for row in records if bool(row.get("terrain_active", False))), None)
        if gnss_event is not None:
            self.log("WARN", f"GNSS signal lost at t={float(gnss_event['timestamp']):.1f}s")
        if terrain_event is not None:
            self.log("INFO", f"Switching to TERRAIN_NAV mode at t={float(terrain_event['timestamp']):.1f}s")

    def state(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            return {
                "timestamp": 0.0,
                "elapsed_time": "00:00:00",
                "sample_index": 0,
                "total_samples": 0,
                "gnss_available": True,
                "nav_mode": "IDLE",
                "sensors_status": "NO_DATASET",
                "data_rate_hz": 0.0,
                "sample_rate_hz": 0.0,
                "speed_mps": None,
                "heading_deg": None,
                "alt_msl": None,
                "truth": None,
                "estimate": None,
                "radar_alt_m": None,
                "baro_alt_m": None,
                "terrain_h": None,
                "correlation_score": None,
                "position_error_m": None,
                "position_error_2d_m": None,
                "position_error_3d_m": None,
                "distance_km": None,
                "processing": self.processing,
                "error": self.processing_error,
            }
        with self._lock:
            record = self._current_record_locked()
            processing = self.processing
            processing_error = self.processing_error
            playing = self.playing
            force_gnss = self.force_gnss
        total_distance_m, _, _ = _compute_distance_and_speed(dataset)
        if record is None:
            first_sample = dataset.unified_samples[0] if dataset.unified_samples else None
            gnss_available = bool(first_sample["gnss_available"]) if first_sample else True
            if force_gnss is not None:
                gnss_available = force_gnss
            return {
                "timestamp": float(first_sample["timestamp_s"]) if first_sample else 0.0,
                "elapsed_time": _format_elapsed(float(first_sample["timestamp_s"]) if first_sample else 0.0),
                "sample_index": 0,
                "total_samples": len(dataset.unified_samples),
                "gnss_available": gnss_available,
                "nav_mode": "GNSS" if gnss_available else "INIT",
                "sensors_status": "PROCESSING" if processing else "READY",
                "data_rate_hz": dataset.case_config.sample_rate_hz,
                "sample_rate_hz": dataset.case_config.sample_rate_hz,
                "speed_mps": _safe_float(first_sample.get("speed_mps")) if first_sample else None,
                "heading_deg": _safe_float(first_sample.get("heading_deg")) if first_sample else None,
                "alt_msl": float(first_sample["alt_msl"]) if first_sample else None,
                "truth": None,
                "estimate": None,
                "radar_alt_m": float(first_sample["radar_alt_m"]) if first_sample else None,
                "baro_alt_m": float(first_sample["alt_msl"]) if first_sample else None,
                "terrain_h": float(first_sample["terrain_h"]) if first_sample and first_sample["terrain_h"] is not None else None,
                "correlation_score": None,
                "position_error_m": None,
                "position_error_2d_m": None,
                "position_error_3d_m": None,
                "distance_km": 0.0,
                "processing": processing,
                "playing": playing,
                "error": processing_error,
            }
        sample = self._raw_sample_for_record(record)
        gnss_available = bool(record.get("gnss_available", True))
        if force_gnss is not None:
            gnss_available = force_gnss
        nav_mode = str(record.get("mode", "INIT"))
        if force_gnss is True:
            nav_mode = "GNSS"
        elif force_gnss is False and nav_mode == "GNSS":
            nav_mode = "TERRAIN_NAV"
        truth = {
            "lat": _safe_float(record.get("truth_lat")),
            "lon": _safe_float(record.get("truth_lon")),
            "alt_msl": _safe_float(record.get("truth_alt_msl")),
            "heading_deg": _safe_float(record.get("truth_heading_deg")),
            "speed_mps": _safe_float(record.get("truth_speed_mps")),
        }
        estimate = {
            "lat": _safe_float(record.get("estimated_lat")),
            "lon": _safe_float(record.get("estimated_lon")),
            "heading_deg": _safe_float(record.get("estimated_heading_deg")),
            "speed_mps": _safe_float(record.get("estimated_speed_mps")),
        }
        current_distance_m = 0.0
        if dataset.truth_rows:
            target_index = min(int(record.get("index", 0)), len(dataset.truth_rows) - 1)
            for previous, current in zip(dataset.truth_rows[:target_index], dataset.truth_rows[1 : target_index + 1]):
                current_distance_m += _distance_m(previous.lat, previous.lon, current.lat, current.lon)
        position_error_3d_m = _safe_float(record.get("truth_error_m"))
        position_error_2d_m = position_error_3d_m
        return {
            "timestamp": float(record.get("timestamp", 0.0)),
            "elapsed_time": _format_elapsed(float(record.get("timestamp", 0.0))),
            "sample_index": int(record.get("index", 0)),
            "total_samples": len(dataset.unified_samples),
            "gnss_available": gnss_available,
            "nav_mode": nav_mode,
            "sensors_status": "OK" if not processing else "PROCESSING",
            "data_rate_hz": dataset.case_config.sample_rate_hz,
            "sample_rate_hz": dataset.case_config.sample_rate_hz,
            "speed_mps": _safe_float(record.get("estimated_speed_mps")),
            "heading_deg": _safe_float(record.get("estimated_heading_deg")),
            "alt_msl": _safe_float(record.get("baro_alt_m")) if sample is None else _safe_float(sample.get("alt_msl")),
            "truth": truth,
            "estimate": estimate,
            "radar_alt_m": _safe_float(record.get("radar_alt_m")) if sample is None else _safe_float(sample.get("radar_alt_m")),
            "baro_alt_m": _safe_float(record.get("baro_alt_m")) if sample is None else _safe_float(sample.get("alt_msl")),
            "terrain_h": _safe_float(record.get("terrain_h")) if sample is None else _safe_float(sample.get("terrain_h")),
            "correlation_score": _safe_float(record.get("correlation_peak")),
            "position_error_m": position_error_3d_m,
            "position_error_2d_m": position_error_2d_m,
            "position_error_3d_m": position_error_3d_m,
            "distance_km": current_distance_m / 1000.0 if total_distance_m >= 0.0 else None,
            "processing": processing,
            "playing": playing,
            "error": processing_error,
        }

    def trajectory(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="Dataset is not loaded")
        estimated = []
        if self.pipeline_artifacts is not None:
            estimated = [
                {
                    "timestamp": float(row["timestamp"]),
                    "lat": _safe_float(row.get("estimated_lat")),
                    "lon": _safe_float(row.get("estimated_lon")),
                }
                for row in self.pipeline_artifacts.report_records
            ]
        truth = [
            {"timestamp": float(sample["timestamp_s"]), "lat": float(sample["truth_lat"]), "lon": float(sample["truth_lon"])}
            for sample in dataset.unified_samples
            if sample.get("truth_lat") is not None and sample.get("truth_lon") is not None
        ]
        events: list[dict[str, Any]] = []
        if self.pipeline_artifacts is not None:
            gnss_event = next((row for row in self.pipeline_artifacts.report_records if not bool(row.get("gnss_available", True))), None)
            terrain_event = next((row for row in self.pipeline_artifacts.report_records if bool(row.get("terrain_active", False))), None)
            if gnss_event is not None:
                events.append({"timestamp": float(gnss_event["timestamp"]), "type": "GNSS_LOST"})
            if terrain_event is not None:
                events.append({"timestamp": float(terrain_event["timestamp"]), "type": "TERRAIN_NAV_START"})
        return {"truth": truth, "estimated": estimated, "events": events}

    def correlation_heatmap(self) -> dict[str, Any]:
        if self.pipeline_artifacts is None:
            raise HTTPException(status_code=400, detail="Replay results are not ready")
        heatmap = np.asarray(self.pipeline_artifacts.report_context.get("correlation_heatmap", []), dtype=float)
        if heatmap.size == 0:
            raise HTTPException(status_code=404, detail="Correlation heatmap is unavailable")
        record = self.pipeline_artifacts.report_records[min(self.playback_index, len(self.pipeline_artifacts.report_records) - 1)]
        step_m = _safe_float(self.pipeline_artifacts.report_context.get("correlation_step_m")) or 1.0
        offsets = [round(index * step_m, 3) for index in range(heatmap.shape[1])]
        return {
            "azimuths": list(range(heatmap.shape[0])),
            "offsets": offsets,
            "values": heatmap.tolist(),
            "best_azimuth": _safe_float(record.get("best_azimuth_deg")),
            "best_offset": _safe_float(record.get("best_offset_m")),
            "best_score": _safe_float(record.get("correlation_peak")),
        }

    def profiles(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="Dataset is not loaded")
        return {
            "time": _finite_series([sample["timestamp_s"] for sample in dataset.unified_samples], limit=5000),
            "baro_alt_m": _finite_series([sample["alt_msl"] for sample in dataset.unified_samples], limit=5000),
            "dem_height_m": _finite_series([sample["terrain_h"] for sample in dataset.unified_samples], limit=5000),
            "radar_alt_m": _finite_series([sample["radar_alt_m"] for sample in dataset.unified_samples], limit=5000),
            "reconstructed_profile_m": _finite_series(
                [float(sample["alt_msl"]) - float(sample["radar_alt_m"]) for sample in dataset.unified_samples],
                limit=5000,
            ),
        }

    def metrics(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="Dataset is not loaded")
        records = self.pipeline_artifacts.report_records if self.pipeline_artifacts is not None else []
        total_distance_m, average_speed_mps, max_speed_mps = _compute_distance_and_speed(dataset)
        errors = np.asarray([_safe_float(row.get("truth_error_m")) for row in records if _safe_float(row.get("truth_error_m")) is not None], dtype=float)
        correlations = np.asarray([_safe_float(row.get("correlation_peak")) for row in records if _safe_float(row.get("correlation_peak")) is not None], dtype=float)
        if records:
            gnss_count = sum(1 for row in records if bool(row.get("gnss_available", True)))
            terrain_count = sum(1 for row in records if bool(row.get("terrain_active", False)))
        else:
            gnss_count = sum(1 for sample in dataset.unified_samples if bool(sample.get("gnss_available", True)))
            terrain_count = sum(1 for sample in dataset.unified_samples if str(sample.get("nav_mode", "INIT")).upper() == "TERRAIN_NAV")
        sample_rate = max(dataset.case_config.sample_rate_hz, 1e-6)
        return {
            "total_flight_time_s": dataset.duration_s,
            "total_distance_m": total_distance_m,
            "average_speed_mps": average_speed_mps,
            "max_speed_mps": max_speed_mps,
            "mean_position_error_m": float(np.mean(errors)) if errors.size else None,
            "max_position_error_m": float(np.max(errors)) if errors.size else None,
            "rmse_m": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "cep50_m": float(np.percentile(errors, 50)) if errors.size else None,
            "cep95_m": float(np.percentile(errors, 95)) if errors.size else None,
            "average_correlation_score": float(np.mean(correlations)) if correlations.size else None,
            "min_correlation_score": float(np.min(correlations)) if correlations.size else None,
            "time_in_gnss_s": gnss_count / sample_rate,
            "time_in_terrain_nav_s": terrain_count / sample_rate,
            "time_lost_s": max(dataset.duration_s - (gnss_count / sample_rate) - (terrain_count / sample_rate), 0.0),
        }

    def logs_payload(self) -> dict[str, Any]:
        with self._lock:
            return {"logs": list(self.logs)}

    def timeline(self) -> dict[str, Any]:
        dataset = self.loaded_dataset
        if dataset is None:
            raise HTTPException(status_code=400, detail="Dataset is not loaded")
        if self.pipeline_artifacts is not None and self.pipeline_artifacts.report_records:
            rows = self.pipeline_artifacts.report_records
            segments: list[dict[str, Any]] = []
            current_mode = None
            segment_start = None
            for row in rows:
                timestamp = float(row["timestamp"])
                if bool(row.get("terrain_active", False)):
                    mode = "TERRAIN_NAV"
                elif bool(row.get("gnss_available", True)):
                    mode = "GNSS_ON"
                else:
                    mode = "GNSS_LOST"
                if current_mode is None:
                    current_mode = mode
                    segment_start = timestamp
                    continue
                if mode != current_mode:
                    segments.append({"start": float(segment_start), "end": timestamp, "mode": current_mode})
                    current_mode = mode
                    segment_start = timestamp
            if current_mode is not None and segment_start is not None:
                segments.append(
                    {
                        "start": float(segment_start),
                        "end": float(rows[-1]["timestamp"]),
                        "mode": current_mode,
                    }
                )
            current_time_s = self.playback_time_s
            duration_s = float(rows[-1]["timestamp"]) if rows else dataset.duration_s
        else:
            segments = []
            last_mode = None
            segment_start = None
            for sample in dataset.unified_samples:
                timestamp = float(sample["timestamp_s"])
                mode = "GNSS_ON" if bool(sample.get("gnss_available", True)) else "GNSS_LOST"
                if last_mode is None:
                    last_mode = mode
                    segment_start = timestamp
                    continue
                if mode != last_mode:
                    segments.append({"start": float(segment_start), "end": timestamp, "mode": last_mode})
                    last_mode = mode
                    segment_start = timestamp
            if last_mode is not None and segment_start is not None:
                segments.append({"start": float(segment_start), "end": dataset.duration_s, "mode": last_mode})
            current_time_s = self.playback_time_s
            duration_s = dataset.duration_s
        return {"duration_s": duration_s, "current_time_s": current_time_s, "segments": segments}

    def get_settings(self) -> dict[str, Any]:
        config_path = self.settings_config_path
        payload = load_yaml_config(config_path)
        payload["config_path"] = str(config_path)
        return payload

    def update_settings(self, request: SettingsUpdateRequest) -> dict[str, Any]:
        config_path = self.settings_config_path
        payload = load_yaml_config(config_path)
        if request.sample_rate_hz is not None:
            payload["sample_rate_hz"] = float(request.sample_rate_hz)
        if request.gnss_drop_after_s is not None:
            payload["gnss_drop_after_s"] = float(request.gnss_drop_after_s)
        if request.correlation is not None:
            correlation = payload.get("correlation", {})
            if not isinstance(correlation, dict):
                correlation = {}
            correlation.update(request.correlation)
            payload["correlation"] = correlation
        if request.visualization is not None:
            visualization = payload.get("visualization", {})
            if not isinstance(visualization, dict):
                visualization = {}
            visualization.update(request.visualization)
            payload["visualization"] = visualization
        if request.dashboard_host is not None or request.dashboard_port is not None or request.report_path is not None:
            visualization = payload.get("visualization", {})
            if not isinstance(visualization, dict):
                visualization = {}
            if request.dashboard_host is not None:
                visualization["dashboard_host"] = request.dashboard_host
            if request.dashboard_port is not None:
                visualization["dashboard_port"] = int(request.dashboard_port)
            if request.report_path is not None:
                visualization["export_report_path"] = _resolve_settings_report_path(request.report_path)
            payload["visualization"] = visualization
        import yaml

        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.log("INFO", "Settings updated")
        return {"saved": True, "config_path": str(config_path)}


controller = TerrainNavigationController()
app = FastAPI(
    title="DRON / Terrain Navigation System API",
    summary="Local engineering backend for terrain-navigation replay, validation, metrics, and visualization.",
    description=(
        "REST backend for a local engineering demonstration of UAV terrain navigation. "
        "It loads DEM, radar NMEA, truth, and barometer datasets; runs replay processing through "
        "the existing pipeline; and exposes state, trajectories, correlation heatmaps, profiles, "
        "metrics, logs, and editable configuration through documented endpoints."
    ),
    version="1.0.0",
    openapi_tags=API_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["ui"],
    summary="Open the local engineering web interface",
    description="Returns the practical local HTML UI used to load datasets, drive replay, and inspect results.",
)
def index() -> str:
    template = (PROJECT_ROOT / "web_ui.html").read_text(encoding="utf-8")
    return template.replace(SESSION_TOKEN_PLACEHOLDER, controller.session_token)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Confirms that the backend process is running and ready to accept requests.",
)
def api_health() -> HealthResponse:
    return controller.health()


@app.post(
    "/api/dataset/load",
    response_model=DatasetLoadResponse,
    tags=["dataset"],
    summary="Load a terrain-navigation dataset",
    description=(
        "Loads DEM, radar NMEA, truth, barometer, and config inputs. "
        "If `config_path` is provided, relative paths are resolved from it."
    ),
)
def api_dataset_load(request: DatasetLoadRequest, http_request: Request) -> DatasetLoadResponse:
    _verify_browser_origin(http_request)
    _verify_session_token(http_request, controller.session_token)
    try:
        return controller.load_dataset(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/dataset/validate",
    response_model=DatasetValidateResponse,
    tags=["dataset"],
    summary="Validate currently configured dataset files",
    description="Checks that DEM, radar, truth, and barometer inputs exist and can be resolved.",
)
def api_dataset_validate() -> DatasetValidateResponse:
    return controller.validate_dataset()


@app.post(
    "/api/replay/start",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Start replay processing and playback",
    description=(
        "Starts replay at the requested speed. "
        "If the pipeline artifacts are not ready yet, processing is launched asynchronously."
    ),
)
def api_replay_start(request: ReplayStartRequest, http_request: Request) -> ReplayActionResponse:
    _verify_browser_origin(http_request)
    _verify_session_token(http_request, controller.session_token)
    return controller.start_replay(request.speed)


@app.post(
    "/api/replay/pause",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Pause replay playback",
    description="Pauses the replay cursor without discarding already processed results.",
)
def api_replay_pause(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.pause_replay()


@app.post(
    "/api/replay/restart",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Restart replay from the beginning",
    description="Rewinds replay to the beginning and starts playback again using the current replay speed.",
)
def api_replay_restart(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.restart_replay()


@app.post(
    "/api/replay/set_speed",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Update replay speed",
    description="Changes the active replay speed multiplier without resetting the loaded dataset.",
)
def api_replay_set_speed(payload: ReplayStartRequest, request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.set_speed(payload.speed)


@app.post(
    "/api/replay/stop",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Stop replay playback",
    description="Stops playback and rewinds the current replay cursor to the beginning.",
)
def api_replay_stop(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.stop_replay()


@app.post(
    "/api/replay/step_forward",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Step replay forward by one processed sample",
    description="Advances the replay cursor by one processed record and pauses continuous playback.",
)
def api_replay_step_forward(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.step_forward()


@app.post(
    "/api/replay/step_backward",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Step replay backward by one processed sample",
    description="Moves the replay cursor backward by one processed record and pauses continuous playback.",
)
def api_replay_step_backward(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.step_backward()


@app.post(
    "/api/gnss/force_off",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Force GNSS OFF in the active UI session",
    description="Applies a temporary GNSS OFF override for the current replay view without rewriting the dataset.",
)
def api_gnss_force_off(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.force_gnss_off()


@app.post(
    "/api/gnss/force_on",
    response_model=ReplayActionResponse,
    tags=["replay"],
    summary="Force GNSS ON in the active UI session",
    description="Applies a temporary GNSS ON override for the current replay view without rewriting the dataset.",
)
def api_gnss_force_on(request: Request) -> ReplayActionResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.force_gnss_on()


@app.get(
    "/api/state",
    response_model=ReplayStateResponse,
    tags=["system"],
    summary="Get current replay and navigation state",
    description=(
        "Returns the current replay cursor state including GNSS availability, navigation mode, "
        "truth and estimate positions, terrain heights, correlation score, and current error."
    ),
)
def api_state() -> ReplayStateResponse:
    return controller.state()


@app.get(
    "/api/trajectory",
    response_model=TrajectoryResponse,
    tags=["analytics"],
    summary="Get truth and estimated trajectories",
    description="Returns truth trajectory, estimated trajectory, and important replay events such as GNSS loss and TERRAIN_NAV entry.",
)
def api_trajectory() -> TrajectoryResponse:
    return controller.trajectory()


@app.get(
    "/api/correlation/heatmap",
    response_model=HeatmapResponse,
    tags=["analytics"],
    summary="Get the correlation heatmap",
    description="Returns the latest available azimuth/offset correlation surface together with the best peak location.",
)
def api_correlation_heatmap() -> HeatmapResponse:
    return controller.correlation_heatmap()


@app.get(
    "/api/profiles",
    response_model=ProfilesResponse,
    tags=["analytics"],
    summary="Get altitude and terrain profiles",
    description=(
        "Returns synchronized barometer altitude, radar altitude, DEM terrain profile, "
        "and reconstructed terrain profile where reconstructed terrain is `baro_alt_m - radar_alt_m`."
    ),
)
def api_profiles() -> ProfilesResponse:
    return controller.profiles()


@app.get(
    "/api/timeline",
    response_model=TimelineResponse,
    tags=["analytics"],
    summary="Get replay mode timeline",
    description="Returns timeline segments for GNSS ON, GNSS LOST, and TERRAIN_NAV together with the current replay time cursor.",
)
def api_timeline() -> TimelineResponse:
    return controller.timeline()


@app.get(
    "/api/metrics",
    response_model=MetricsResponse,
    tags=["analytics"],
    summary="Get aggregate replay metrics",
    description="Returns aggregate duration, distance, speed, error, CEP, correlation, and time-in-mode metrics for the loaded case.",
)
def api_metrics() -> MetricsResponse:
    return controller.metrics()


@app.get(
    "/api/logs",
    response_model=LogsResponse,
    tags=["system"],
    summary="Get operator-facing runtime logs",
    description="Returns a rolling list of informational, warning, and error log lines collected by the backend controller.",
)
def api_logs() -> LogsResponse:
    return controller.logs_payload()


@app.get(
    "/api/settings",
    response_model=SettingsResponse,
    tags=["settings"],
    summary="Read current settings from config.yaml",
    description="Returns the current persisted configuration payload used by the local case workflow.",
)
def api_settings() -> SettingsResponse:
    return controller.get_settings()


@app.post(
    "/api/settings",
    response_model=SettingsSaveResponse,
    tags=["settings"],
    summary="Persist safe settings back to config.yaml",
    description="Updates safe runtime settings in config.yaml without altering the core processing code path.",
)
def api_settings_update(payload: SettingsUpdateRequest, request: Request) -> SettingsSaveResponse:
    _verify_browser_origin(request)
    _verify_session_token(request, controller.session_token)
    return controller.update_settings(payload)
