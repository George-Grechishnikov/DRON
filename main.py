"""Main orchestrator for TERRAIN NAVIGATOR."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import signal
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pyproj

from case_reader import CaseInputConfig, iter_case_unified_samples
from correlator import Correlator, CorrelationResult
from dem_loader import DEMLoader
from imm_filter import IMMFilter, IMMResult
from nmea_parser import NMEAFrame, NMEAReader
from position_solver import PositionEstimate, PositionSolver
from profile_extractor import ProfileExtractor, is_flat_terrain
from sim_generator import SimulationConfig, TrajectoryPoint, generate_points
from visualizer import TerrainNavigatorDash, export_demo_report, export_flight_report, export_operator_outputs


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the full TERRAIN NAVIGATOR pipeline."""

    mode: str
    dem_path: Path
    start_lat: float
    start_lon: float
    trajectory: int
    config_path: Path | None
    nmea_path: Path | None
    samples_path: Path | None
    gt_path: Path | None
    barometer_path: Path | None
    udp_host: str
    udp_port: int
    dashboard_host: str
    dashboard_port: int
    enable_visualizer: bool
    seed: int
    speed_mps: float
    altitude_msl_m: float
    noise_sigma: float
    window_size: int = 50
    step_size: int = 10
    freq_hz: float = 5.0
    dem_patch_radius_m: float = 5000.0
    max_offset_m: float = 2000.0
    flat_terrain_threshold_m: float = 15.0
    cold_start_windows: int = 3
    gnss_drop_after_s: float | None = None
    report_path: Path = Path("output") / "terrain_navigator_report.html"
    auto_open_report: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class UnifiedSample:
    """Sample contract shared by simulation, replay, live adapters, and SITL."""

    timestamp_s: float
    lat: float | None
    lon: float | None
    alt_msl: float
    radar_alt_m: float
    terrain_h: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None
    gnss_available: bool = True
    nav_mode: str = "INIT"
    truth_lat: float | None = None
    truth_lon: float | None = None
    estimated_lat: float | None = None
    estimated_lon: float | None = None
    correlation_score: float | None = None
    correlation_heatmap: object | None = None
    best_azimuth_deg: float | None = None
    best_offset_m: float | None = None

    @property
    def timestamp(self) -> float:
        return float(self.timestamp_s)

    @property
    def ground_speed_mps(self) -> float | None:
        return self.speed_mps

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def has_truth_position(self) -> bool:
        return self.truth_lat is not None and self.truth_lon is not None

    @property
    def effective_truth_lat(self) -> float | None:
        return self.truth_lat if self.truth_lat is not None else self.lat

    @property
    def effective_truth_lon(self) -> float | None:
        return self.truth_lon if self.truth_lon is not None else self.lon

    @property
    def effective_heading_deg(self) -> float:
        return float(self.heading_deg if self.heading_deg is not None else 0.0)

    @property
    def effective_speed_mps(self) -> float:
        return float(self.speed_mps if self.speed_mps is not None else 0.0)

    def with_updates(self, **changes: Any) -> "UnifiedSample":
        return replace(self, **changes)


@dataclass(frozen=True)
class SamplePacket:
    """Unified sample packet moving through producer/pipeline queues."""

    index: int
    sample: UnifiedSample


@dataclass(frozen=True)
class GroundTruthPoint:
    """Ground-truth sample for replay/simulation evaluation."""

    index: int
    timestamp_s: float
    lat: float
    lon: float
    speed_mps: float
    azimuth_deg: float


@dataclass(frozen=True)
class ReplayMetrics:
    """Aggregate replay evaluation metrics."""

    mean_error_m: float
    max_error_m: float
    rmse_m: float
    speed_error_mps: float
    azimuth_error_deg: float


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the main orchestrator."""

    parser = argparse.ArgumentParser(description="Run the TERRAIN NAVIGATOR pipeline")
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--sim", action="store_true", help="Simulation mode")
    mode_group.add_argument("--live", action="store_true", help="Live UDP mode")
    mode_group.add_argument("--replay", action="store_true", help="Replay NMEA log mode")
    mode_group.add_argument("--samples-jsonl", "--unified-stream", type=Path, help="Replay unified sample JSONL stream")
    parser.add_argument("--config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--dem", type=Path, help="Path to DEM GeoTIFF")
    parser.add_argument("--lat", type=float, default=60.5, help="Initial latitude")
    parser.add_argument("--lon", type=float, default=90.3, help="Initial longitude")
    parser.add_argument("--trajectory", type=int, default=1, choices=(1, 2, 3), help="Simulation trajectory id")
    parser.add_argument("--nmea", type=Path, help="Replay NMEA file path")
    parser.add_argument("--gt", type=Path, help="Ground-truth CSV file path")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP host for live mode")
    parser.add_argument("--udp-port", type=int, default=10110, help="UDP port for live mode")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--no-visualizer", action="store_true", help="Disable Dash UI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    parser.add_argument("--speed", type=float, default=50.0, help="Nominal speed in m/s")
    parser.add_argument("--altitude-msl", type=float, default=1500.0, help="Absolute altitude MSL")
    parser.add_argument("--noise", type=float, default=2.0, help="Radar-altimeter noise sigma")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size in frames")
    parser.add_argument("--step-size", type=int, default=10, help="Sliding step in frames")
    parser.add_argument("--freq", type=float, default=5.0, help="NMEA stream frequency in Hz")
    parser.add_argument("--dem-patch-radius", type=float, default=5000.0, help="DEM patch radius in meters")
    parser.add_argument("--max-offset", type=float, default=2000.0, help="Maximum correlation offset in meters")
    parser.add_argument("--flat-threshold", type=float, default=15.0, help="Flat-terrain threshold in meters")
    parser.add_argument("--cold-start-windows", type=int, default=3, help="Windows before GNSS-like binding starts")
    parser.add_argument(
        "--gnss-drop-after",
        type=float,
        help="Set unified samples to GNSS unavailable after this many seconds",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("output") / "terrain_navigator_report.html",
        help="HTML report output path",
    )
    parser.add_argument("--open-report", action="store_true", help="Open the HTML report in a browser after the run")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into a Config object."""

    parser = build_argument_parser()
    raw_argv = list(argv) if argv is not None else []
    args = parser.parse_args(argv)
    if args.config is None and not any((args.sim, args.live, args.replay, args.samples_jsonl)):
        parser.error("choose a mode flag or pass --config")
    if args.replay and args.nmea is None and args.config is None:
        parser.error("--nmea is required in replay mode")
    config_payload = load_yaml_config(args.config) if args.config is not None else {}
    visualization_cfg = config_payload.get("visualization", {}) if isinstance(config_payload.get("visualization"), dict) else {}
    correlation_cfg = config_payload.get("correlation", {}) if isinstance(config_payload.get("correlation"), dict) else {}

    def explicit(option: str) -> bool:
        return option in raw_argv

    mode = "sim" if args.sim else "live" if args.live else "samples" if args.samples_jsonl else "replay" if args.replay else "case"
    dem_path = args.dem if explicit("--dem") else _resolve_dem_path(_path_from_config(config_payload, "dem_path", args.config) or args.dem)
    if dem_path is None:
        parser.error("DEM path is required via --dem or config.yaml")
    nmea_path = args.nmea if explicit("--nmea") else _path_from_config(config_payload, "radar_data_path", args.config) or args.nmea
    gt_path = args.gt if explicit("--gt") else _path_from_config(config_payload, "truth_path", args.config) or args.gt
    barometer_path = _path_from_config(config_payload, "barometer_path", args.config)
    if mode == "case" and (nmea_path is None or gt_path is None or barometer_path is None):
        parser.error("case mode requires radar_data_path, truth_path, and barometer_path in config.yaml")
    if mode == "replay" and nmea_path is None:
        parser.error("--nmea is required in replay mode")
    return Config(
        mode=mode,
        dem_path=dem_path,
        start_lat=args.lat,
        start_lon=args.lon,
        trajectory=args.trajectory,
        config_path=args.config,
        nmea_path=nmea_path,
        samples_path=args.samples_jsonl,
        gt_path=gt_path,
        barometer_path=barometer_path,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        dashboard_host=args.dashboard_host if explicit("--dashboard-host") else str(visualization_cfg.get("dashboard_host", args.dashboard_host)),
        dashboard_port=args.dashboard_port if explicit("--dashboard-port") else int(visualization_cfg.get("dashboard_port", args.dashboard_port)),
        enable_visualizer=not args.no_visualizer if explicit("--no-visualizer") else bool(visualization_cfg.get("enabled", True)),
        seed=args.seed,
        speed_mps=args.speed,
        altitude_msl_m=args.altitude_msl,
        noise_sigma=args.noise,
        window_size=args.window_size,
        step_size=args.step_size,
        freq_hz=args.freq if explicit("--freq") else float(config_payload.get("sample_rate_hz", args.freq)),
        dem_patch_radius_m=args.dem_patch_radius,
        max_offset_m=args.max_offset if explicit("--max-offset") else float(correlation_cfg.get("max_offset_m", args.max_offset)),
        flat_terrain_threshold_m=args.flat_threshold if explicit("--flat-threshold") else float(correlation_cfg.get("flat_terrain_threshold_m", args.flat_threshold)),
        cold_start_windows=args.cold_start_windows if explicit("--cold-start-windows") else int(correlation_cfg.get("cold_start_windows", args.cold_start_windows)),
        gnss_drop_after_s=args.gnss_drop_after if explicit("--gnss-drop-after") else _optional_float(config_payload.get("gnss_drop_after_s")),
        report_path=args.report_path if explicit("--report-path") else Path(str(visualization_cfg.get("export_report_path", args.report_path))),
        auto_open_report=args.open_report,
        log_level=args.log_level,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load config.yaml for case-driven runs."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config support") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _path_from_config(payload: dict[str, Any], key: str, config_path: Path | None) -> Path | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute() or config_path is None:
        return path
    if path.exists():
        return path
    return (config_path.parent / path).resolve()


def _resolve_dem_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        for pattern in ("*.tif", "*.tiff", "*.hgt"):
            matches = sorted(path.glob(pattern))
            if matches:
                return matches[0]
    return path


def open_report_in_browser(report_path: Path) -> bool:
    """Open the generated HTML report in the default browser."""

    if not report_path.exists():
        return False
    try:
        return bool(webbrowser.open(report_path.resolve().as_uri()))
    except Exception:
        LOGGER.warning("Failed to open report in browser: %s", report_path)
        return False


def configure_logging(level: str, log_path: Path) -> None:
    """Configure console and file logging."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def load_ground_truth_csv(path: Path, geod: pyproj.Geod | None = None) -> list[GroundTruthPoint]:
    """Load and augment ground-truth CSV with speed and azimuth."""

    geod = geod or pyproj.Geod(ellps="WGS84")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        use_case_header = {"timestamp", "lat", "lon"}.issubset(header) and "timestamp_s" not in header
        for row in reader:
            if use_case_header:
                rows.append(
                    {
                        "index": float(len(rows)),
                        "timestamp_s": float(row["timestamp"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "speed_mps": float(row["speed_mps"]) if row.get("speed_mps") else float("nan"),
                        "azimuth_deg": float(row["heading_deg"]) if row.get("heading_deg") else float("nan"),
                    }
                )
            else:
                rows.append(
                    {
                        "index": float(row["index"]),
                        "timestamp_s": float(row["timestamp_s"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                    }
                )

    gt_points: list[GroundTruthPoint] = []
    for idx, row in enumerate(rows):
        if idx == 0:
            speed_mps = float(row.get("speed_mps", 0.0))
            azimuth_deg = float(row.get("azimuth_deg", 0.0))
        else:
            prev = rows[idx - 1]
            if np.isfinite(row.get("speed_mps", float("nan"))) and np.isfinite(row.get("azimuth_deg", float("nan"))):
                speed_mps = float(row["speed_mps"])
                azimuth_deg = float(row["azimuth_deg"] % 360.0)
            else:
                azimuth_deg, _, distance_m = geod.inv(prev["lon"], prev["lat"], row["lon"], row["lat"])
                dt = max(row["timestamp_s"] - prev["timestamp_s"], 1e-6)
                speed_mps = distance_m / dt
                azimuth_deg = float(azimuth_deg % 360.0)
        gt_points.append(
            GroundTruthPoint(
                index=int(row["index"]),
                timestamp_s=row["timestamp_s"],
                lat=row["lat"],
                lon=row["lon"],
                speed_mps=speed_mps,
                azimuth_deg=azimuth_deg,
            )
        )
    return gt_points


def compute_replay_metrics(
    history: list[tuple[int, IMMResult]],
    ground_truth: list[GroundTruthPoint],
    geod: pyproj.Geod | None = None,
) -> ReplayMetrics:
    """Compare IMM history against ground truth."""

    geod = geod or pyproj.Geod(ellps="WGS84")
    gt_by_index = {point.index: point for point in ground_truth}
    pos_errors: list[float] = []
    speed_errors: list[float] = []
    azimuth_errors: list[float] = []

    for frame_index, result in history:
        gt = gt_by_index.get(frame_index)
        if gt is None:
            continue
        _, _, distance_m = geod.inv(result.lon, result.lat, gt.lon, gt.lat)
        pos_errors.append(float(distance_m))
        speed_errors.append(abs(result.speed_mps - gt.speed_mps))
        delta_azimuth = abs(((result.azimuth_deg - gt.azimuth_deg + 180.0) % 360.0) - 180.0)
        azimuth_errors.append(float(delta_azimuth))

    if not pos_errors:
        return ReplayMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    pos_array = np.asarray(pos_errors, dtype=float)
    return ReplayMetrics(
        mean_error_m=float(np.mean(pos_array)),
        max_error_m=float(np.max(pos_array)),
        rmse_m=float(np.sqrt(np.mean(pos_array**2))),
        speed_error_mps=float(np.mean(np.asarray(speed_errors, dtype=float))),
        azimuth_error_deg=float(np.mean(np.asarray(azimuth_errors, dtype=float))),
    )


def make_simulation_points(config: Config) -> list[TrajectoryPoint]:
    """Generate simulation trajectory points from the current config."""

    sim_config = SimulationConfig(
        dem_path=config.dem_path,
        start_lat=config.start_lat,
        start_lon=config.start_lon,
        frequency_hz=config.freq_hz,
        noise_sigma_m=config.noise_sigma,
        output_mode="file",
        out_nmea=None,
        out_csv=None,
        udp_host=config.udp_host,
        udp_port=config.udp_port,
        random_seed=config.seed,
        altitude_msl_m=config.altitude_msl_m,
        speed_mps=config.speed_mps,
        azimuth_deg=45.0,
        duration_s=None,
        length_km=None,
        trajectory_id=config.trajectory,
    )
    return generate_points(sim_config)


def build_sim_ground_truth(points: list[TrajectoryPoint]) -> list[GroundTruthPoint]:
    """Convert simulation points to ground-truth points."""

    geod = pyproj.Geod(ellps="WGS84")
    gt_points: list[GroundTruthPoint] = []
    for idx, point in enumerate(points):
        if idx == 0:
            speed_mps = 0.0
            azimuth_deg = 0.0
        else:
            prev = points[idx - 1]
            azimuth_deg, _, distance_m = geod.inv(prev.lon, prev.lat, point.lon, point.lat)
            dt = max(point.timestamp_s - prev.timestamp_s, 1e-6)
            speed_mps = distance_m / dt
            azimuth_deg = float(azimuth_deg % 360.0)
        gt_points.append(
            GroundTruthPoint(
                index=point.index,
                timestamp_s=point.timestamp_s,
                lat=point.lat,
                lon=point.lon,
                speed_mps=speed_mps,
                azimuth_deg=azimuth_deg,
            )
        )
    return gt_points


def _gnss_available_at(timestamp_s: float, config: Config) -> bool:
    """Return the sample-level GNSS availability flag for demo playback."""

    return config.gnss_drop_after_s is None or timestamp_s < config.gnss_drop_after_s


def _sample_from_nmea_frame(
    frame: NMEAFrame,
    index: int,
    config: Config,
    gt_by_index: dict[int, GroundTruthPoint] | None = None,
) -> UnifiedSample:
    """Adapt legacy radar-only NMEA frames into the unified sample contract."""

    gt = gt_by_index.get(index) if gt_by_index is not None else None
    timestamp_s = gt.timestamp_s if gt is not None else index / config.freq_hz
    has_trusted_position = gt is not None
    return UnifiedSample(
        timestamp_s=float(timestamp_s),
        lat=float(gt.lat if gt is not None else config.start_lat),
        lon=float(gt.lon if gt is not None else config.start_lon),
        alt_msl=float(config.altitude_msl_m),
        terrain_h=None,
        heading_deg=float(gt.azimuth_deg if gt is not None else 45.0),
        speed_mps=float(gt.speed_mps if gt is not None and gt.speed_mps > 0.0 else config.speed_mps),
        radar_alt_m=float(frame.radar_alt_m),
        nav_mode="GNSS" if has_trusted_position and _gnss_available_at(float(timestamp_s), config) else "INIT",
        gnss_available=has_trusted_position and _gnss_available_at(float(timestamp_s), config),
        truth_lat=float(gt.lat) if gt is not None else None,
        truth_lon=float(gt.lon) if gt is not None else None,
    )


def _samples_from_sim_points(points: list[TrajectoryPoint], config: Config) -> list[UnifiedSample]:
    """Convert simulated trajectory points into unified samples."""

    geod = pyproj.Geod(ellps="WGS84")
    samples: list[UnifiedSample] = []
    for idx, point in enumerate(points):
        if idx == 0:
            heading_deg = 45.0
            speed_mps = config.speed_mps
        else:
            prev = points[idx - 1]
            heading_deg, _, distance_m = geod.inv(prev.lon, prev.lat, point.lon, point.lat)
            dt = max(point.timestamp_s - prev.timestamp_s, 1e-6)
            speed_mps = distance_m / dt
            heading_deg = float(heading_deg % 360.0)
        samples.append(
            UnifiedSample(
                timestamp_s=float(point.timestamp_s),
                lat=float(point.lat),
                lon=float(point.lon),
                alt_msl=float(point.alt_msl),
                terrain_h=float(point.terrain_h),
                heading_deg=float(heading_deg),
                speed_mps=float(speed_mps),
                radar_alt_m=float(point.radar_alt_measured),
                gnss_available=_gnss_available_at(float(point.timestamp_s), config),
                nav_mode="GNSS" if _gnss_available_at(float(point.timestamp_s), config) else "TERRAIN_NAV",
                truth_lat=float(point.lat),
                truth_lon=float(point.lon),
            )
        )
    return samples


def coerce_unified_sample(sample: UnifiedSample | dict[str, Any]) -> UnifiedSample:
    """Normalize bridge-emitted dicts into the internal unified sample dataclass."""

    if isinstance(sample, UnifiedSample):
        return sample
    timestamp_s = sample.get("timestamp_s", sample.get("timestamp"))
    speed_mps = sample.get("speed_mps", sample.get("ground_speed_mps"))
    return UnifiedSample(
        timestamp_s=float(timestamp_s),
        lat=None if sample.get("lat") is None else float(sample["lat"]),
        lon=None if sample.get("lon") is None else float(sample["lon"]),
        alt_msl=float(sample["alt_msl"]),
        terrain_h=None if sample.get("terrain_h") is None else float(sample["terrain_h"]),
        heading_deg=None if sample.get("heading_deg") is None else float(sample["heading_deg"]),
        speed_mps=None if speed_mps is None else float(speed_mps),
        radar_alt_m=float(sample["radar_alt_m"]),
        gnss_available=bool(sample["gnss_available"]),
        nav_mode=str(sample.get("nav_mode", "GNSS" if sample.get("gnss_available", True) else "TERRAIN_NAV")),
        truth_lat=None if sample.get("truth_lat") is None else float(sample["truth_lat"]),
        truth_lon=None if sample.get("truth_lon") is None else float(sample["truth_lon"]),
        estimated_lat=None if sample.get("estimated_lat") is None else float(sample["estimated_lat"]),
        estimated_lon=None if sample.get("estimated_lon") is None else float(sample["estimated_lon"]),
        correlation_score=None
        if sample.get("correlation_score") is None
        else float(sample["correlation_score"]),
        correlation_heatmap=sample.get("correlation_heatmap"),
        best_azimuth_deg=None
        if sample.get("best_azimuth_deg") is None
        else float(sample["best_azimuth_deg"]),
        best_offset_m=None if sample.get("best_offset_m") is None else float(sample["best_offset_m"]),
    )


def enqueue_sample(frame_queue: queue.Queue, packet: SamplePacket, stop_event: threading.Event) -> None:
    """Push one unified sample packet into the queue with graceful shutdown support."""

    while not stop_event.is_set():
        try:
            frame_queue.put(packet, timeout=0.1)
            return
        except queue.Full:
            continue


def unified_sample_producer(
    samples: Iterable[UnifiedSample | dict[str, Any]],
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Feed any unified sample stream into the main pipeline queue."""

    for index, sample in enumerate(samples):
        if stop_event.is_set():
            break
        enqueue_sample(frame_queue, SamplePacket(index=index, sample=coerce_unified_sample(sample)), stop_event)


def load_unified_samples_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield unified sample dicts from a JSONL stream capture."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on unified sample line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Unified sample line {line_number} must be a JSON object")
            yield payload


def samples_jsonl_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Produce unified samples from a JSONL stream capture."""

    assert config.samples_path is not None
    unified_sample_producer(load_unified_samples_jsonl(config.samples_path), frame_queue, stop_event)


def case_config_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Produce unified samples from case-style config inputs."""

    assert config.nmea_path is not None
    assert config.gt_path is not None
    assert config.barometer_path is not None
    case_config = CaseInputConfig(
        dem_path=config.dem_path,
        radar_data_path=config.nmea_path,
        truth_path=config.gt_path,
        barometer_path=config.barometer_path,
        sample_rate_hz=config.freq_hz,
        gnss_drop_after_s=config.gnss_drop_after_s,
    )
    unified_sample_producer(iter_case_unified_samples(case_config), frame_queue, stop_event)


def simulation_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> list[GroundTruthPoint]:
    """Produce unified samples from the simulation generator."""

    points = make_simulation_points(config)
    for point, sample in zip(points, _samples_from_sim_points(points, config)):
        if stop_event.is_set():
            break
        enqueue_sample(frame_queue, SamplePacket(index=point.index, sample=sample), stop_event)
    return build_sim_ground_truth(points)


def replay_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Produce unified samples from a recorded NMEA log."""

    assert config.nmea_path is not None
    gt_by_index: dict[int, GroundTruthPoint] | None = None
    if config.gt_path is not None:
        gt_by_index = {point.index: point for point in load_ground_truth_csv(config.gt_path)}
    reader = NMEAReader.from_file(config.nmea_path)
    try:
        valid_index = 0
        for frame in reader:
            if stop_event.is_set():
                break
            if frame.valid:
                sample = _sample_from_nmea_frame(frame, valid_index, config, gt_by_index)
                enqueue_sample(frame_queue, SamplePacket(index=valid_index, sample=sample), stop_event)
                valid_index += 1
    finally:
        reader.close()


def live_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Produce unified samples from a legacy live UDP NMEA stream."""

    reader = NMEAReader.from_udp(config.udp_host, config.udp_port)
    valid_index = 0
    try:
        while not stop_event.is_set():
            got_frame = False
            for frame in reader:
                got_frame = True
                if frame.valid:
                    sample = _sample_from_nmea_frame(frame, valid_index, config)
                    enqueue_sample(frame_queue, SamplePacket(index=valid_index, sample=sample), stop_event)
                    valid_index += 1
            if not got_frame:
                time.sleep(0.05)
    finally:
        reader.close()


def predict_fix(
    current_lat: float,
    current_lon: float,
    speed_mps: float,
    azimuth_deg: float,
    dt: float,
    confidence: float,
) -> PositionEstimate:
    """Create a simple predicted fix for the cold-start phase."""

    geod = pyproj.Geod(ellps="WGS84")
    lon, lat, _ = geod.fwd(current_lon, current_lat, azimuth_deg, speed_mps * dt)
    return PositionEstimate(
        lat=float(lat),
        lon=float(lon),
        speed_mps=float(speed_mps),
        azimuth_deg=float(azimuth_deg),
        timestamp_s=float(time.time()),
        confidence=float(confidence),
        is_reliable=False,
        cov_matrix=np.diag([400.0, 400.0]).astype(float),
    )


def normalize_nav_mode(sample: UnifiedSample) -> str:
    """Normalize navigation mode labels for UI and future bridge compatibility."""

    mode = sample.nav_mode.strip().upper() if sample.nav_mode else ""
    if mode == "TERRAIN":
        mode = "TERRAIN_NAV"
    if mode:
        return mode
    if sample.gnss_available and sample.has_position:
        return "GNSS"
    if sample.has_position or sample.effective_truth_lat is not None:
        return "TERRAIN_NAV"
    return "INIT"


def sample_position_fix(sample: UnifiedSample, confidence: float = 1.0) -> PositionEstimate:
    """Create a primary GNSS fix from a unified sample."""

    return PositionEstimate(
        lat=float(sample.lat if sample.lat is not None else 0.0),
        lon=float(sample.lon if sample.lon is not None else 0.0),
        speed_mps=float(sample.effective_speed_mps),
        azimuth_deg=float(sample.effective_heading_deg % 360.0),
        timestamp_s=float(sample.timestamp),
        confidence=float(confidence),
        is_reliable=True,
        cov_matrix=np.diag([9.0, 9.0]).astype(float),
    )


def pipeline_worker(
    config: Config,
    frame_queue: queue.Queue,
    state_queue: queue.Queue,
    stop_event: threading.Event,
    pipeline_done_event: threading.Event,
    ground_truth: list[GroundTruthPoint] | None = None,
    report_records: list[dict[str, Any]] | None = None,
    report_context: dict[str, Any] | None = None,
) -> list[tuple[int, IMMResult]]:
    """Run the main sliding-window pipeline."""

    geod = pyproj.Geod(ellps="WGS84")
    history: list[tuple[int, IMMResult]] = []
    buffer: deque[SamplePacket] = deque(maxlen=config.window_size)
    window_start_lat = config.start_lat
    window_start_lon = config.start_lon
    current_azimuth = 45.0
    current_speed = config.speed_mps
    step_dt = config.step_size / config.freq_hz
    window_duration = config.window_size / config.freq_hz
    window_counter = 0
    measurement_step_m = config.speed_mps / config.freq_hz
    measured_profile_length_m = config.window_size * measurement_step_m
    reference_profile_length_m = measured_profile_length_m + config.max_offset_m

    with DEMLoader(config.dem_path) as dem:
        extractor = ProfileExtractor(
            dem,
            profile_length_m=reference_profile_length_m,
            step_m=measurement_step_m,
        )
        correlator = Correlator(
            profile_length_m=measured_profile_length_m,
            step_m=measurement_step_m,
            max_offset_m=config.max_offset_m,
        )
        solver = PositionSolver()
        imm = IMMFilter()

        while not stop_event.is_set():
            try:
                packet = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if pipeline_done_event.is_set():
                    break
                continue

            buffer.append(packet)
            if len(buffer) < config.window_size:
                continue

            samples_window = [item.sample for item in buffer]
            latest_sample = samples_window[-1]
            center_frame_index = buffer[-1].index
            h_meas = np.array(
                [sample.alt_msl - sample.radar_alt_m for sample in samples_window],
                dtype=float,
            )
            flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
            ref_matrix = extractor.build_reference_matrix(window_start_lat, window_start_lon)
            corr_result = correlator.compute(
                h_meas=h_meas,
                ref_matrix=ref_matrix,
                azimuths_deg=np.arange(0.0, ref_matrix.shape[0], 1.0),
            )

            if latest_sample.correlation_heatmap is None:
                latest_sample = latest_sample.with_updates(
                    correlation_heatmap=corr_result.heatmap,
                    correlation_score=corr_result.peak_correlation,
                    best_azimuth_deg=corr_result.best_azimuth_deg,
                    best_offset_m=corr_result.best_offset_m,
                )

            truth_lat = latest_sample.effective_truth_lat
            truth_lon = latest_sample.effective_truth_lon
            gnss_available = bool(latest_sample.gnss_available)
            nav_mode = normalize_nav_mode(latest_sample)
            trusted_gnss = gnss_available and latest_sample.has_position
            if not trusted_gnss and nav_mode == "INIT" and window_counter >= config.cold_start_windows:
                nav_mode = "TERRAIN_NAV"
            primary_mode = nav_mode

            if trusted_gnss:
                position_fix = sample_position_fix(latest_sample, confidence=max(corr_result.confidence, 0.5))
            elif window_counter < config.cold_start_windows:
                position_fix = predict_fix(
                    current_lat=window_start_lat,
                    current_lon=window_start_lon,
                    speed_mps=current_speed,
                    azimuth_deg=current_azimuth,
                    dt=window_duration,
                    confidence=corr_result.confidence,
                )
            else:
                position_fix = solver.solve(
                    result=corr_result,
                    start_lat=window_start_lat,
                    start_lon=window_start_lon,
                    window_duration_s=window_duration,
                )

            imm_result = imm.update(position_fix, dt=step_dt, is_flat=flat)
            latest_sample = latest_sample.with_updates(
                estimated_lat=float(imm_result.lat),
                estimated_lon=float(imm_result.lon),
            )
            current_azimuth = imm_result.azimuth_deg
            current_speed = imm_result.speed_mps
            dem_patch, dem_transform = dem.get_patch(
                imm_result.lat,
                imm_result.lon,
                radius_m=config.dem_patch_radius_m,
            )
            rows, cols = dem_patch.shape
            left, top = dem_transform * (0, 0)
            right, bottom = dem_transform * (cols, rows)
            if truth_lat is not None and truth_lon is not None:
                _, _, truth_error_m = geod.inv(imm_result.lon, imm_result.lat, truth_lon, truth_lat)
                truth_error_value = float(truth_error_m)
            else:
                truth_error_value = float("nan")

            history.append((center_frame_index, imm_result))
            if report_records is not None:
                report_records.append(
                    {
                        "index": int(center_frame_index),
                        "timestamp": float(latest_sample.timestamp),
                        "estimated_lat": float(imm_result.lat),
                        "estimated_lon": float(imm_result.lon),
                        "truth_lat": None if truth_lat is None else float(truth_lat),
                        "truth_lon": None if truth_lon is None else float(truth_lon),
                        "estimated_speed_mps": float(imm_result.speed_mps),
                        "estimated_heading_deg": float(imm_result.azimuth_deg),
                        "gnss_available": gnss_available,
                        "mode": primary_mode,
                        "terrain_active": primary_mode == "TERRAIN_NAV",
                        "truth_error_m": truth_error_value,
                        "best_azimuth_deg": float(corr_result.best_azimuth_deg),
                        "best_offset_m": float(corr_result.best_offset_m),
                        "correlation_peak": float(latest_sample.correlation_score or corr_result.peak_correlation),
                    }
                )
            if report_context is not None:
                report_context["correlation_heatmap"] = np.asarray(
                    latest_sample.correlation_heatmap if latest_sample.correlation_heatmap is not None else corr_result.heatmap,
                    dtype=float,
                ).tolist()
                report_context["dem_patch"] = np.asarray(dem_patch, dtype=float).tolist()
                report_context["dem_extent"] = {
                    "left": float(min(left, right)),
                    "right": float(max(left, right)),
                    "bottom": float(min(bottom, top)),
                    "top": float(max(bottom, top)),
                }
            _enqueue_state(
                state_queue,
                {
                    "corr": corr_result,
                    "fix": imm_result,
                    "sample": latest_sample,
                    "truth": None
                    if truth_lat is None or truth_lon is None
                    else {"lat": float(truth_lat), "lon": float(truth_lon)},
                    "estimated": {"lat": float(imm_result.lat), "lon": float(imm_result.lon)},
                    "h_meas": h_meas,
                    "ref": corr_result.best_reference_profile,
                    "dem_patch": dem_patch,
                    "dem_extent": {
                        "left": float(min(left, right)),
                        "right": float(max(left, right)),
                        "bottom": float(min(bottom, top)),
                        "top": float(max(bottom, top)),
                    },
                    "hdop": imm.get_hdop(),
                    "gnss_available": gnss_available,
                    "mode": primary_mode,
                    "terrain_active": primary_mode == "TERRAIN_NAV",
                    "truth_error_m": truth_error_value,
                    "correlation_score": float(latest_sample.correlation_score or corr_result.peak_correlation),
                    "correlation_heatmap": latest_sample.correlation_heatmap,
                    "best_azimuth_deg": float(latest_sample.best_azimuth_deg or corr_result.best_azimuth_deg),
                    "best_offset_m": float(latest_sample.best_offset_m or corr_result.best_offset_m),
                },
                stop_event,
            )

            for _ in range(min(config.step_size, len(buffer))):
                if buffer:
                    buffer.popleft()
            if trusted_gnss:
                window_start_lat = float(latest_sample.lat if latest_sample.lat is not None else window_start_lat)
                window_start_lon = float(latest_sample.lon if latest_sample.lon is not None else window_start_lon)
            else:
                next_start_lon, next_start_lat, _ = geod.fwd(
                    window_start_lon,
                    window_start_lat,
                    current_azimuth,
                    current_speed * step_dt,
                )
                window_start_lat = float(next_start_lat)
                window_start_lon = float(next_start_lon)
            window_counter += 1

    return history


def _enqueue_state(state_queue: queue.Queue, payload: dict[str, Any], stop_event: threading.Event) -> None:
    """Push dashboard state into the queue."""

    while not stop_event.is_set():
        try:
            state_queue.put(payload, timeout=0.1)
            return
        except queue.Full:
            try:
                state_queue.get_nowait()
            except queue.Empty:
                pass


def run_pipeline(config: Config) -> tuple[list[tuple[int, IMMResult]], ReplayMetrics | None]:
    """Run the configured pipeline end-to-end."""

    frame_queue: queue.Queue = queue.Queue(maxsize=1000)
    state_queue: queue.Queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    producer_done_event = threading.Event()
    pipeline_done_event = threading.Event()

    ground_truth: list[GroundTruthPoint] | None = None
    if config.mode == "sim":
        ground_truth = build_sim_ground_truth(make_simulation_points(config))
    elif config.mode in {"replay", "case"} and config.gt_path is not None:
        ground_truth = load_ground_truth_csv(config.gt_path)

    pipeline_history: list[tuple[int, IMMResult]] = []
    report_records: list[dict[str, Any]] = []
    report_context: dict[str, Any] = {}
    pipeline_metrics: ReplayMetrics | None = None

    def producer_target() -> None:
        try:
            if config.mode == "sim":
                simulation_producer(config, frame_queue, stop_event)
            elif config.mode == "replay":
                replay_producer(config, frame_queue, stop_event)
            elif config.mode == "case":
                case_config_producer(config, frame_queue, stop_event)
            elif config.mode == "samples":
                samples_jsonl_producer(config, frame_queue, stop_event)
            else:
                live_producer(config, frame_queue, stop_event)
        finally:
            producer_done_event.set()

    def pipeline_target() -> None:
        nonlocal pipeline_history
        pipeline_history = pipeline_worker(
            config=config,
            frame_queue=frame_queue,
            state_queue=state_queue,
            stop_event=stop_event,
            pipeline_done_event=producer_done_event,
            ground_truth=ground_truth,
            report_records=report_records,
            report_context=report_context,
        )
        pipeline_done_event.set()

    producer_thread = threading.Thread(target=producer_target, name="producer", daemon=True)
    pipeline_thread = threading.Thread(target=pipeline_target, name="pipeline", daemon=True)
    threads = [producer_thread, pipeline_thread]

    dashboard_history: list[IMMResult] = []
    dashboard_thread: threading.Thread | None = None
    if config.enable_visualizer:
        dashboard = TerrainNavigatorDash(state_queue=state_queue)

        def dashboard_target() -> None:
            dashboard.run(host=config.dashboard_host, port=config.dashboard_port, debug=False)

        dashboard_thread = threading.Thread(target=dashboard_target, name="visualizer", daemon=True)
        threads.append(dashboard_thread)

    def _signal_handler(signum: int, frame: Any) -> None:
        del signum, frame
        LOGGER.info("Shutdown requested, stopping threads...")
        stop_event.set()

    previous_handler = signal.signal(signal.SIGINT, _signal_handler)
    try:
        producer_thread.start()
        pipeline_thread.start()
        if dashboard_thread is not None:
            dashboard_thread.start()

        producer_thread.join()
        pipeline_thread.join()
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, previous_handler)

    if config.mode in {"sim", "replay", "samples", "case"} and ground_truth is not None:
        pipeline_metrics = compute_replay_metrics(pipeline_history, ground_truth)
        LOGGER.info(
            "Replay metrics | mean=%.2f m | max=%.2f m | rmse=%.2f m | speed=%.2f m/s | azimuth=%.2f deg",
            pipeline_metrics.mean_error_m,
            pipeline_metrics.max_error_m,
            pipeline_metrics.rmse_m,
            pipeline_metrics.speed_error_mps,
            pipeline_metrics.azimuth_error_deg,
        )

    dashboard_history = [item for _, item in pipeline_history]
    if report_records:
        export_demo_report(report_records, str(config.report_path), context=report_context)
    elif dashboard_history:
        export_flight_report(dashboard_history, str(config.report_path))
        export_operator_outputs(
            [
                {
                    "timestamp": float(index),
                    "estimated_lat": float(item.lat),
                    "estimated_lon": float(item.lon),
                    "truth_lat": None,
                    "truth_lon": None,
                    "estimated_speed_mps": float(item.speed_mps),
                    "estimated_heading_deg": float(item.azimuth_deg),
                    "gnss_available": False,
                    "mode": str(item.dominant_mode).upper(),
                    "terrain_active": False,
                    "truth_error_m": float("nan"),
                    "best_azimuth_deg": float(item.azimuth_deg),
                    "best_offset_m": 0.0,
                    "correlation_peak": float("nan"),
                }
                for index, item in enumerate(dashboard_history, start=1)
            ],
            str(config.report_path),
        )
    if config.auto_open_report:
        open_report_in_browser(config.report_path)
    return pipeline_history, pipeline_metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    config = parse_args(argv)
    configure_logging(config.log_level, Path("terrain_navigator.log"))
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
