"""Main orchestrator for TERRAIN NAVIGATOR."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import queue
import signal
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pyproj

from correlator import CorrelationCandidate, Correlator, CorrelationResult, ObservabilityMetrics, compute_observability_metrics
from constants import FIXED_BARO_ALTITUDE_M
from dem_loader import DEMLoader
from eskf import ESKF, ImuSample
from imm_filter import IMMFilter, IMMResult
from measurement_layer import BaroTrack, frames_to_terrain_profile, parse_nmea_timestamp_to_seconds, update_terrain_bias
from nmea_parser import NMEAFrame, NMEAReader, parse_line
from position_solver import PositionEstimate, PositionSolver
from profile_extractor import ProfileExtractor, is_flat_terrain
from terrain_pf import TerrainParticleFilter
from sim_generator import SimulationConfig, TrajectoryPoint, format_gpgga, generate_points
from sitl_bridge import SITLBridge
from visualizer import TerrainNavigatorDash, export_flight_report


LOGGER = logging.getLogger(__name__)


def _safe_export_flight_report(history: list[IMMResult], path: str) -> None:
    """Export the HTML report without letting a rendering issue kill the run."""

    if not history:
        return
    try:
        export_flight_report(history, path)
    except Exception:
        LOGGER.exception("Flight report export failed for %s", path)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the full TERRAIN NAVIGATOR pipeline."""

    mode: str
    dem_path: Path
    start_lat: float
    start_lon: float
    trajectory: int | None
    nmea_path: Path | None
    gt_path: Path | None
    sitl_connection: str
    sitl_gnss_drop_after_s: float | None
    sitl_gnss_recover_after_s: float | None
    udp_host: str
    udp_port: int
    dashboard_host: str
    dashboard_port: int
    enable_visualizer: bool
    seed: int
    speed_mps: float
    altitude_msl_m: float
    noise_sigma: float
    sim_length_km: float = 10.0
    initial_heading_deg: float = 45.0
    window_size: int = 50
    adaptive_window: bool = False
    min_window_size: int = 50
    max_window_size: int = 50
    window_growth_step: int = 10
    step_size: int = 10
    freq_hz: float = 5.0
    dem_patch_radius_m: float = 5000.0
    max_offset_m: float = 2000.0
    flat_terrain_threshold_m: float = 15.0
    cold_start_windows: int = 3
    log_level: str = "WARNING"
    quiet_console: bool = False
    open_browser: bool = False
    realtime_playback: bool = False
    playback_speed: float = 1.0
    live_dashboard_stream: bool = True
    demo_dashboard: bool = False
    engine: str = "legacy"


@dataclass(frozen=True)
class FramePacket:
    """Frame packet moving through producer/pipeline queues."""

    index: int
    frame: NMEAFrame
    gnss_available: bool = True
    truth_lat: float | None = None
    truth_lon: float | None = None
    truth_heading_deg: float | None = None
    truth_speed_mps: float | None = None
    ingest_monotonic_s: float = field(default_factory=time.perf_counter)


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


@dataclass(frozen=True)
class NavigationDecision:
    """Decision describing whether terrain matching is trusted for this update."""

    fix: PositionEstimate
    mode: str
    used_prediction_only: bool
    corr_result: CorrelationResult | None = None


@dataclass(frozen=True)
class ReacquisitionTracker:
    """Track post-ambiguity terrain reacquisition before accepting a new fix."""

    pending: bool = False
    stable_windows: int = 0
    last_azimuth_deg: float | None = None
    last_offset_m: float | None = None


@dataclass(frozen=True)
class WindowSelection:
    """Selected processing window and its diagnostics."""

    frame_packets: list[FramePacket]
    window_size: int
    corr_result: CorrelationResult
    observability: ObservabilityMetrics
    flat: bool


@dataclass(frozen=True)
class LocalReacquisitionResult:
    """Best local terrain candidate around the predicted start position."""

    start_lat: float
    start_lon: float
    corr_result: CorrelationResult
    observability: ObservabilityMetrics
    flat: bool


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the main orchestrator."""

    parser = argparse.ArgumentParser(description="Run the TERRAIN NAVIGATOR pipeline")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sim", action="store_true", help="Simulation mode")
    mode_group.add_argument("--live", action="store_true", help="Live UDP mode")
    mode_group.add_argument("--replay", action="store_true", help="Replay NMEA log mode")
    mode_group.add_argument("--sitl", action="store_true", help="ArduPilot SITL MAVLink mode")

    parser.add_argument("--dem", required=True, type=Path, help="Path to DEM GeoTIFF")
    parser.add_argument("--lat", type=float, help="Initial latitude; defaults to DEM center when omitted")
    parser.add_argument("--lon", type=float, help="Initial longitude; defaults to DEM center when omitted")
    parser.add_argument("--trajectory", type=int, choices=(1, 2, 3), help="Legacy simulation trajectory id")
    parser.add_argument("--sim-length-km", type=float, default=10.0, help="Default straight simulation route length in kilometers")
    parser.add_argument("--nmea", type=Path, help="Replay NMEA file path")
    parser.add_argument("--gt", type=Path, help="Ground-truth CSV file path")
    parser.add_argument("--sitl-connect", default="udp:127.0.0.1:14550", help="MAVLink connection string for SITL")
    parser.add_argument("--sitl-gnss-drop-after", type=float, help="Disable GNSS after N seconds in SITL mode")
    parser.add_argument("--sitl-gnss-recover-after", type=float, help="Re-enable GNSS after N seconds in SITL mode")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP host for live mode")
    parser.add_argument("--udp-port", type=int, default=10110, help="UDP port for live mode")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--no-visualizer", action="store_true", help="Disable Dash UI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    parser.add_argument("--speed", type=float, default=50.0, help="Nominal speed in m/s")
    parser.add_argument("--initial-heading", type=float, default=45.0, help="Initial course/heading in degrees clockwise from north")
    parser.add_argument("--noise", type=float, default=2.0, help="Radar-altimeter noise sigma")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size in frames")
    parser.add_argument("--adaptive-window", action="store_true", help="Enable adaptive window sizing")
    parser.add_argument("--min-window-size", type=int, help="Minimum adaptive window size in frames")
    parser.add_argument("--max-window-size", type=int, help="Maximum adaptive window size in frames")
    parser.add_argument("--window-growth-step", type=int, help="Adaptive window increment in frames")
    parser.add_argument("--step-size", type=int, default=10, help="Sliding step in frames")
    parser.add_argument("--freq", type=float, default=5.0, help="NMEA stream frequency in Hz")
    parser.add_argument("--dem-patch-radius", type=float, default=5000.0, help="DEM patch radius in meters")
    parser.add_argument("--max-offset", type=float, default=2000.0, help="Maximum correlation offset in meters")
    parser.add_argument("--flat-threshold", type=float, default=15.0, help="Flat-terrain threshold in meters")
    parser.add_argument("--cold-start-windows", type=int, default=3, help="Windows before GNSS-like binding starts")
    parser.add_argument("--engine", choices=("legacy", "eskf"), default="legacy", help="Navigation core to run")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    parser.add_argument("--quiet-console", action="store_true", help="Write detailed logs to file only")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard URL in the default browser")
    parser.add_argument("--realtime-playback", action="store_true", help="Pace sim/replay input like a live NMEA stream")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Playback speed multiplier for --realtime-playback")
    parser.add_argument("--no-live-dashboard-stream", action="store_true", help="Disable per-frame dashboard stream")
    parser.add_argument("--demo-dashboard", action="store_true", help="Run a simple smooth dashboard demo from NMEA/GT")
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into a Config object."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.replay and args.nmea is None:
        parser.error("--nmea is required in replay mode")
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    if args.playback_speed <= 0:
        parser.error("--playback-speed must be positive")
    if args.sim_length_km <= 0:
        parser.error("--sim-length-km must be positive")

    adaptive_window = bool(args.adaptive_window)
    min_window_size = int(args.min_window_size if args.min_window_size is not None else args.window_size)
    max_window_size = int(args.max_window_size if args.max_window_size is not None else args.window_size)
    window_growth_step = int(args.window_growth_step if args.window_growth_step is not None else args.step_size)
    if min_window_size <= 0 or max_window_size <= 0 or window_growth_step <= 0:
        parser.error("Adaptive window sizes and step must be positive")
    if min_window_size > max_window_size:
        parser.error("--min-window-size must be <= --max-window-size")

    mode = "sim" if args.sim else "live" if args.live else "replay" if args.replay else "sitl"
    return Config(
        mode=mode,
        dem_path=args.dem,
        start_lat=float("nan") if args.lat is None else args.lat,
        start_lon=float("nan") if args.lon is None else args.lon,
        trajectory=args.trajectory,
        nmea_path=args.nmea,
        gt_path=args.gt,
        sitl_connection=args.sitl_connect,
        sitl_gnss_drop_after_s=args.sitl_gnss_drop_after,
        sitl_gnss_recover_after_s=args.sitl_gnss_recover_after,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        enable_visualizer=not args.no_visualizer,
        seed=args.seed,
        speed_mps=args.speed,
        initial_heading_deg=float(args.initial_heading % 360.0),
        altitude_msl_m=FIXED_BARO_ALTITUDE_M,
        noise_sigma=args.noise,
        sim_length_km=float(args.sim_length_km),
        window_size=args.window_size,
        adaptive_window=adaptive_window,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        window_growth_step=window_growth_step,
        step_size=args.step_size,
        freq_hz=args.freq,
        dem_patch_radius_m=args.dem_patch_radius,
        max_offset_m=args.max_offset,
        flat_terrain_threshold_m=args.flat_threshold,
        cold_start_windows=args.cold_start_windows,
        log_level=args.log_level,
        quiet_console=bool(args.quiet_console),
        open_browser=bool(args.open_browser),
        realtime_playback=bool(args.realtime_playback),
        playback_speed=float(args.playback_speed),
        live_dashboard_stream=not bool(args.no_live_dashboard_stream),
        demo_dashboard=bool(args.demo_dashboard),
        engine=args.engine,
    )


def configure_logging(level: str, log_path: Path, *, quiet_console: bool = False) -> None:
    """Configure console and file logging."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, level.upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING if quiet_console else log_level)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[console_handler, file_handler],
        force=True,
    )


def load_ground_truth_csv(
    path: Path,
    geod: pyproj.Geod | None = None,
    fallback_dt_s: float = 1.0,
) -> list[GroundTruthPoint]:
    """Load ground truth and derive motion from consecutive coordinates.

    Some validation datasets include speed/azimuth columns that are metadata rather
    than the actual per-sample movement. For replay metrics and simulated telemetry
    we derive motion from WGS84 deltas, using the known sample period when no real
    timestamp column exists.
    """

    geod = geod or pyproj.Geod(ellps="WGS84")
    fallback_dt_s = max(float(fallback_dt_s), 1e-6)
    rows: list[dict[str, float | bool]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = float(row["index"])
            raw_timestamp = row.get("timestamp_s")
            has_timestamp = raw_timestamp not in (None, "")
            rows.append(
                {
                    "index": index,
                    "timestamp_s": float(raw_timestamp) if has_timestamp else index * fallback_dt_s,
                    "has_timestamp": has_timestamp,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
            )

    def _motion_between(left: dict[str, float | bool], right: dict[str, float | bool]) -> tuple[float, float]:
        azimuth_deg, _, distance_m = geod.inv(
            float(left["lon"]),
            float(left["lat"]),
            float(right["lon"]),
            float(right["lat"]),
        )
        if bool(left["has_timestamp"]) and bool(right["has_timestamp"]):
            dt_s = float(right["timestamp_s"]) - float(left["timestamp_s"])
        else:
            dt_s = fallback_dt_s
        speed_mps = float(distance_m) / max(dt_s, 1e-6)
        return speed_mps, float(azimuth_deg % 360.0)

    gt_points: list[GroundTruthPoint] = []
    for idx, row in enumerate(rows):
        if idx == 0 and len(rows) > 1:
            speed_mps, azimuth_deg = _motion_between(row, rows[1])
        elif idx == 0:
            speed_mps = 0.0
            azimuth_deg = 0.0
        else:
            speed_mps, azimuth_deg = _motion_between(rows[idx - 1], row)
        gt_points.append(
            GroundTruthPoint(
                index=int(row["index"]),
                timestamp_s=float(row["timestamp_s"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
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

    start_lat, start_lon = resolve_initial_coordinates(config.dem_path, config.start_lat, config.start_lon)
    sim_config = SimulationConfig(
        dem_path=config.dem_path,
        start_lat=start_lat,
        start_lon=start_lon,
        frequency_hz=config.freq_hz,
        noise_sigma_m=config.noise_sigma,
        output_mode="file",
        out_nmea=None,
        out_csv=None,
        out_imu=None,
        udp_host=config.udp_host,
        udp_port=config.udp_port,
        random_seed=config.seed,
        altitude_msl_m=config.altitude_msl_m,
        speed_mps=config.speed_mps,
        azimuth_deg=config.initial_heading_deg,
        duration_s=None,
        length_km=None if config.trajectory is not None else config.sim_length_km,
        trajectory_id=config.trajectory,
    )
    return generate_points(sim_config)


def resolve_initial_coordinates(dem_path: Path, lat: float, lon: float) -> tuple[float, float]:
    """Resolve the initial coordinates, defaulting to the DEM center."""

    if math.isfinite(lat) and math.isfinite(lon):
        return (float(lat), float(lon))

    with DEMLoader(dem_path) as dem:
        center_lat, center_lon = dem.get_center()
    LOGGER.info(
        "Initial position not provided, using DEM center lat=%.6f lon=%.6f",
        center_lat,
        center_lon,
    )
    return (float(center_lat), float(center_lon))


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


def enqueue_frame(frame_queue: queue.Queue, packet: FramePacket, stop_event: threading.Event) -> None:
    """Push one frame packet into the queue with graceful shutdown support."""

    while not stop_event.is_set():
        try:
            frame_queue.put(packet, timeout=0.1)
            return
        except queue.Full:
            continue


def pace_realtime_playback(
    *,
    started_at_s: float,
    sample_index: int,
    freq_hz: float,
    playback_speed: float,
    stop_event: threading.Event,
) -> None:
    """Throttle file/sim producers so Dash can show a live-looking stream."""

    target_elapsed_s = float(sample_index) / max(float(freq_hz) * float(playback_speed), 1e-6)
    target_time_s = started_at_s + target_elapsed_s
    while not stop_event.is_set():
        remaining_s = target_time_s - time.perf_counter()
        if remaining_s <= 0.0:
            return
        time.sleep(min(remaining_s, 0.05))


def simulation_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    control_queue: queue.Queue | None = None,
) -> list[GroundTruthPoint]:
    """Produce NMEA frames from the simulation generator."""

    points = make_simulation_points(config)
    gnss_override: bool | None = None
    playback_started_at_s = time.perf_counter()
    for point in points:
        if stop_event.is_set():
            break
        if config.realtime_playback:
            pace_realtime_playback(
                started_at_s=playback_started_at_s,
                sample_index=point.index,
                freq_hz=config.freq_hz,
                playback_speed=config.playback_speed,
                stop_event=stop_event,
            )
        gnss_override = _drain_manual_gnss_override(control_queue, gnss_override)
        frame = parse_line(format_gpgga(point.timestamp_s, point.radar_alt_measured))
        if frame is None:
            continue
        enqueue_frame(
            frame_queue,
            FramePacket(
                index=point.index,
                frame=frame,
                gnss_available=True if gnss_override is None else bool(gnss_override),
                truth_lat=point.lat,
                truth_lon=point.lon,
            ),
            stop_event,
        )
    return build_sim_ground_truth(points)


def replay_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    control_queue: queue.Queue | None = None,
    ground_truth: list[GroundTruthPoint] | None = None,
) -> None:
    """Produce frames from a recorded NMEA log."""

    assert config.nmea_path is not None
    gt_by_index = {point.index: point for point in ground_truth} if ground_truth is not None else {}
    reader = NMEAReader.from_file(config.nmea_path)
    try:
        valid_index = 0
        gnss_override: bool | None = None
        playback_started_at_s = time.perf_counter()
        for frame in reader:
            if stop_event.is_set():
                break
            if config.realtime_playback:
                pace_realtime_playback(
                    started_at_s=playback_started_at_s,
                    sample_index=valid_index,
                    freq_hz=config.freq_hz,
                    playback_speed=config.playback_speed,
                    stop_event=stop_event,
                )
            gnss_override = _drain_manual_gnss_override(control_queue, gnss_override)
            gt_point = gt_by_index.get(valid_index)
            enqueue_frame(
                frame_queue,
                FramePacket(
                    index=valid_index,
                    frame=frame,
                    gnss_available=True if gnss_override is None else bool(gnss_override),
                    truth_lat=gt_point.lat if gt_point is not None else None,
                    truth_lon=gt_point.lon if gt_point is not None else None,
                    truth_heading_deg=gt_point.azimuth_deg if gt_point is not None else None,
                    truth_speed_mps=gt_point.speed_mps if gt_point is not None else None,
                ),
                stop_event,
            )
            if frame.valid:
                valid_index += 1
    finally:
        reader.close()


def live_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    control_queue: queue.Queue | None = None,
) -> None:
    """Produce frames from a live UDP stream."""

    reader = NMEAReader.from_udp(config.udp_host, config.udp_port)
    valid_index = 0
    gnss_override: bool | None = None
    try:
        while not stop_event.is_set():
            got_frame = False
            for frame in reader:
                got_frame = True
                gnss_override = _drain_manual_gnss_override(control_queue, gnss_override)
                enqueue_frame(
                    frame_queue,
                    FramePacket(
                        index=valid_index,
                        frame=frame,
                        gnss_available=True if gnss_override is None else bool(gnss_override),
                    ),
                    stop_event,
                )
                if frame.valid:
                    valid_index += 1
            if not got_frame:
                time.sleep(0.05)
    finally:
        reader.close()


def sitl_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    control_queue: queue.Queue | None = None,
) -> None:
    """Produce NMEA-compatible frames from a live ArduPilot SITL stream."""

    with DEMLoader(config.dem_path) as dem:
        bridge = SITLBridge(
            config.sitl_connection,
            dem,
            gnss_drop_after_s=config.sitl_gnss_drop_after_s,
            gnss_recover_after_s=config.sitl_gnss_recover_after_s,
        )
        bridge.connect()
        try:
            valid_index = 0
            for sample in bridge.samples():
                if stop_event.is_set():
                    break
                _drain_sitl_control_queue(bridge, control_queue)
                frame = bridge.sample_to_nmea_frame(sample)
                enqueue_frame(
                    frame_queue,
                    FramePacket(
                        index=valid_index,
                        frame=frame,
                        gnss_available=bool(sample.gnss_available),
                        truth_lat=sample.lat,
                        truth_lon=sample.lon,
                        truth_heading_deg=sample.heading_deg,
                        truth_speed_mps=sample.ground_speed_mps,
                    ),
                    stop_event,
                )
                valid_index += 1
        finally:
            bridge.close()


def _drain_sitl_control_queue(bridge: SITLBridge, control_queue: queue.Queue | None) -> None:
    """Apply any pending UI commands to the live SITL bridge."""

    if control_queue is None:
        return
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return
        command_type = command.get("type")
        if command_type == "set_gnss_enabled":
            bridge.set_gnss_enabled(bool(command.get("enabled", True)))


def _drain_manual_gnss_override(
    control_queue: queue.Queue | None,
    current_override: bool | None,
) -> bool | None:
    """Consume UI commands and keep the latest manual GNSS state for non-SITL modes."""

    if control_queue is None:
        return current_override
    updated_override = current_override
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return updated_override
        if command.get("type") == "set_gnss_enabled":
            updated_override = bool(command.get("enabled", True))


def _drain_demo_control_queue(
    control_queue: queue.Queue | None,
    current_override: bool | None,
) -> tuple[bool | None, bool]:
    """Consume dashboard commands for demo mode."""

    if control_queue is None:
        return current_override, False
    updated_override = current_override
    restart_requested = False
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return updated_override, restart_requested
        command_type = command.get("type")
        if command_type == "set_gnss_enabled":
            updated_override = bool(command.get("enabled", True))
        elif command_type == "restart_route":
            restart_requested = True


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


def _wrap_angle_deg(angle_deg: float) -> float:
    """Normalize an angle into the [0, 360) interval."""

    return float(angle_deg % 360.0)


def _angle_delta_deg(from_deg: float, to_deg: float) -> float:
    """Return the signed shortest delta from one heading to another."""

    return float(((to_deg - from_deg + 180.0) % 360.0) - 180.0)


def matched_offset_m(corr_result: CorrelationResult) -> float:
    """Return the best available matched offset in meters."""

    return float(
        corr_result.best_offset_subsample_m
        if np.isfinite(corr_result.best_offset_subsample_m)
        else corr_result.best_offset_m
    )


def compute_orientation_aware_correlation(
    correlator: Correlator,
    h_meas: np.ndarray,
    ref_matrix: np.ndarray,
    azimuths_deg: np.ndarray | None = None,
    *,
    min_improvement: float = 0.05,
) -> CorrelationResult:
    """Run correlation in both profile directions and keep the clearly better orientation."""

    forward = correlator.compute(h_meas=h_meas, ref_matrix=ref_matrix, azimuths_deg=azimuths_deg)
    reversed_result = correlator.compute(
        h_meas=np.asarray(h_meas, dtype=float)[::-1],
        ref_matrix=ref_matrix,
        azimuths_deg=azimuths_deg,
    )
    if reversed_result.peak_correlation > forward.peak_correlation + min_improvement:
        return replace(
            reversed_result,
            best_reference_profile=reversed_result.best_reference_profile[::-1].copy(),
        )
    return forward


def correlation_result_from_candidate(
    corr_result: CorrelationResult,
    candidate: CorrelationCandidate,
) -> CorrelationResult:
    """Return a correlation result whose primary peak is a selected top-k candidate."""

    return replace(
        corr_result,
        best_azimuth_deg=float(candidate.azimuth_deg),
        best_offset_steps=int(candidate.offset_steps),
        best_offset_m=float(candidate.offset_m),
        best_offset_subsample_steps=float(candidate.offset_subsample_steps),
        best_offset_subsample_m=float(candidate.offset_subsample_m),
        peak_correlation=float(candidate.score),
        ncc_peak=float(candidate.ncc_score),
        msd_peak=float(candidate.msd_score),
    )


def maybe_update_heading_from_correlation(
    *,
    current_azimuth_deg: float,
    corr_result: CorrelationResult,
    max_offset_m: float,
    selected_window_size: int,
    measurement_step_m: float,
    used_prediction_only: bool,
) -> float:
    """Use correlation azimuth as a soft heading cue when position is still gated out."""

    del measurement_step_m

    if not used_prediction_only:
        return _wrap_angle_deg(current_azimuth_deg)
    if not np.isfinite(corr_result.best_azimuth_deg):
        return _wrap_angle_deg(current_azimuth_deg)

    matched_offset = matched_offset_m(corr_result)
    max_trusted_offset_m = max(150.0, min(max_offset_m * 0.2, 400.0))
    if not np.isfinite(matched_offset) or abs(matched_offset) > max_trusted_offset_m:
        return _wrap_angle_deg(current_azimuth_deg)
    if corr_result.peak_correlation < 0.9:
        return _wrap_angle_deg(current_azimuth_deg)

    target_azimuth_deg = _wrap_angle_deg(corr_result.best_azimuth_deg)
    if corr_result.is_ambiguous:
        max_heading_candidate_offset_m = min(max_trusted_offset_m, 150.0)
        low_offset_candidates = [
            candidate
            for candidate in corr_result.top_candidates
            if np.isfinite(candidate.azimuth_deg)
            and np.isfinite(candidate.offset_subsample_m)
            and abs(candidate.offset_subsample_m) <= max_heading_candidate_offset_m
            and candidate.score >= corr_result.peak_correlation - 0.10
        ]
        if low_offset_candidates and corr_result.peak_correlation >= 0.92:
            heading_candidate = max(low_offset_candidates, key=lambda item: item.score)
            delta_deg = _angle_delta_deg(current_azimuth_deg, heading_candidate.azimuth_deg)
            if abs(delta_deg) < 20.0:
                return _wrap_angle_deg(current_azimuth_deg)
            max_turn_deg = 8.0 if selected_window_size >= 30 else 12.0
            delta_deg = float(np.clip(delta_deg, -max_turn_deg, max_turn_deg))
            if abs(delta_deg) >= 1.0:
                return _wrap_angle_deg(current_azimuth_deg + delta_deg)
        # During a turn transition we can keep the predicted track alive, but we
        # should not rotate aggressively when correlation peaks are not separated.
        if (
            corr_result.pslr_db < 1.5
            or corr_result.ambiguity_peak_count > 1
            or corr_result.confidence < 0.1
        ):
            return _wrap_angle_deg(current_azimuth_deg)
        max_turn_deg = 8.0 if selected_window_size >= 30 else 12.0
        delta_deg = _angle_delta_deg(current_azimuth_deg, target_azimuth_deg)
        delta_deg = float(np.clip(delta_deg, -max_turn_deg, max_turn_deg))
        return _wrap_angle_deg(current_azimuth_deg + delta_deg)

    return target_azimuth_deg


def build_prediction_navigation_decision(
    *,
    mode: str,
    current_lat: float,
    current_lon: float,
    current_speed: float,
    current_azimuth: float,
    window_duration: float,
    corr_result: CorrelationResult,
    selected_window_size: int,
    measurement_span_m: float,
    max_offset_m: float,
    heading_override_deg: float | None = None,
) -> NavigationDecision:
    """Build a prediction-only decision while still allowing soft heading cues."""

    hinted_azimuth = (
        _wrap_angle_deg(float(heading_override_deg))
        if heading_override_deg is not None and np.isfinite(heading_override_deg)
        else maybe_update_heading_from_correlation(
            current_azimuth_deg=current_azimuth,
            corr_result=corr_result,
            max_offset_m=max_offset_m,
            selected_window_size=selected_window_size,
            measurement_step_m=measurement_span_m / max(selected_window_size - 1, 1),
            used_prediction_only=True,
        )
    )
    return NavigationDecision(
        fix=predict_fix(
            current_lat=current_lat,
            current_lon=current_lon,
            speed_mps=current_speed,
            azimuth_deg=hinted_azimuth,
            dt=float(window_duration),
            confidence=corr_result.confidence,
        ),
        mode=mode,
        used_prediction_only=True,
        corr_result=corr_result,
    )


def update_motion_state_after_decision(
    *,
    nav_decision: NavigationDecision,
    corr_result: CorrelationResult,
    current_speed: float,
    current_azimuth: float,
    max_offset_m: float,
    selected_window_size: int,
    measurement_step_m: float,
) -> tuple[float, float]:
    """Return the motion state to use for the next prediction window."""

    if not nav_decision.used_prediction_only:
        return (
            float(nav_decision.fix.speed_mps),
            _wrap_angle_deg(nav_decision.fix.azimuth_deg),
        )

    if np.isfinite(nav_decision.fix.azimuth_deg):
        return (float(current_speed), _wrap_angle_deg(nav_decision.fix.azimuth_deg))

    hinted_azimuth = maybe_update_heading_from_correlation(
        current_azimuth_deg=current_azimuth,
        corr_result=corr_result,
        max_offset_m=max_offset_m,
        selected_window_size=selected_window_size,
        measurement_step_m=measurement_step_m,
        used_prediction_only=True,
    )
    return (float(current_speed), _wrap_angle_deg(hinted_azimuth))


def terrain_fix_plausibility_reason(
    *,
    candidate_fix: PositionEstimate,
    previous_fix: PositionEstimate | None,
    current_speed: float,
    current_azimuth: float,
    update_dt: float | None,
) -> str | None:
    """Return a rejection reason when a terrain fix is physically implausible."""

    if previous_fix is None:
        return None

    dt = (
        float(update_dt)
        if update_dt is not None and update_dt > 0.0
        else max(candidate_fix.timestamp_s - previous_fix.timestamp_s, 1e-6)
    )
    speed_limit_mps = max(float(current_speed) * 2.5, float(current_speed) + 35.0, 80.0)
    if candidate_fix.speed_mps > speed_limit_mps:
        return f"speed_jump:{candidate_fix.speed_mps:.1f}>{speed_limit_mps:.1f}"

    acceleration_mps2 = abs(candidate_fix.speed_mps - float(current_speed)) / max(dt, 1e-6)
    if acceleration_mps2 > 25.0:
        return f"accel_jump:{acceleration_mps2:.1f}>25.0"

    turn_delta_deg = abs(_angle_delta_deg(current_azimuth, candidate_fix.azimuth_deg))
    max_turn_delta_deg = min(120.0, max(45.0, 20.0 + 35.0 * dt))
    if turn_delta_deg > max_turn_delta_deg:
        return f"turn_jump:{turn_delta_deg:.1f}>{max_turn_delta_deg:.1f}"

    return None


def update_reacquisition_tracker(
    tracker: ReacquisitionTracker,
    *,
    corr_result: CorrelationResult,
    observability: ObservabilityMetrics,
    flat: bool,
    max_offset_m: float,
) -> tuple[ReacquisitionTracker, bool]:
    """Hold terrain acceptance after ambiguity until consecutive clean windows agree."""

    if corr_result.is_ambiguous:
        return (
            ReacquisitionTracker(
                pending=True,
                stable_windows=0,
                last_azimuth_deg=None,
                last_offset_m=None,
            ),
            False,
        )

    if not tracker.pending:
        return tracker, False

    offset_m = matched_offset_m(corr_result)
    candidate_ok = (
        observability.is_informative
        and not flat
        and corr_result.is_reliable
        and np.isfinite(corr_result.best_azimuth_deg)
        and np.isfinite(offset_m)
        and abs(offset_m) <= min(max_offset_m * 0.75, 1500.0)
        and corr_result.pslr_db >= 3.0
        and corr_result.peak_correlation >= 0.85
        and corr_result.confidence >= 0.05
    )
    if not candidate_ok:
        return (
            ReacquisitionTracker(
                pending=True,
                stable_windows=0,
                last_azimuth_deg=None,
                last_offset_m=None,
            ),
            True,
        )

    if tracker.stable_windows <= 0 or tracker.last_azimuth_deg is None or tracker.last_offset_m is None:
        return (
            ReacquisitionTracker(
                pending=True,
                stable_windows=1,
                last_azimuth_deg=float(corr_result.best_azimuth_deg),
                last_offset_m=offset_m,
            ),
            True,
        )

    azimuth_delta_deg = abs(_angle_delta_deg(tracker.last_azimuth_deg, corr_result.best_azimuth_deg))
    offset_delta_m = abs(tracker.last_offset_m - offset_m)
    if azimuth_delta_deg <= 12.0 and offset_delta_m <= 180.0:
        stable_windows = tracker.stable_windows + 1
        if stable_windows >= 2:
            return ReacquisitionTracker(), False
        return (
            ReacquisitionTracker(
                pending=True,
                stable_windows=stable_windows,
                last_azimuth_deg=float(corr_result.best_azimuth_deg),
                last_offset_m=offset_m,
            ),
            True,
        )

    return (
        ReacquisitionTracker(
            pending=True,
            stable_windows=1,
            last_azimuth_deg=float(corr_result.best_azimuth_deg),
            last_offset_m=offset_m,
        ),
        True,
    )


def build_path_offsets_enu(
    speed_mps: float,
    azimuth_deg: float,
    sample_count: int,
    sample_dt: float,
) -> np.ndarray:
    """Build ENU offsets for a window along a simple recent trajectory hypothesis."""

    if sample_count <= 0:
        return np.empty((0, 2), dtype=float)
    azimuth_rad = math.radians(float(azimuth_deg) % 360.0)
    velocity_xy = np.array(
        [speed_mps * math.sin(azimuth_rad), speed_mps * math.cos(azimuth_rad)],
        dtype=float,
    )
    times = np.arange(sample_count, dtype=float) * float(sample_dt)
    return times[:, np.newaxis] * velocity_xy[np.newaxis, :]


def estimate_window_duration_s(frame_packets: list[FramePacket], fallback_freq_hz: float) -> float:
    """Estimate window duration from NMEA timestamps with a fixed-frequency fallback."""

    if len(frame_packets) < 2:
        return 0.0
    fallback_duration = (len(frame_packets) - 1) / max(float(fallback_freq_hz), 1e-6)
    start_s = parse_nmea_timestamp_to_seconds(frame_packets[0].frame.timestamp_utc)
    end_s = parse_nmea_timestamp_to_seconds(frame_packets[-1].frame.timestamp_utc)
    if not (np.isfinite(start_s) and np.isfinite(end_s)):
        return float(fallback_duration)
    if end_s < start_s:
        end_s += 24.0 * 3600.0
    duration_s = float(end_s - start_s)
    if duration_s <= 0.0 or duration_s > max(fallback_duration * 3.0, fallback_duration + 5.0):
        return float(fallback_duration)
    return duration_s


def advance_geodetic_point(
    *,
    lat: float,
    lon: float,
    azimuth_deg: float,
    distance_m: float,
    geod: pyproj.Geod,
) -> tuple[float, float]:
    """Advance a geodetic point along the given azimuth by a metric distance."""

    target_lon, target_lat, _ = geod.fwd(lon, lat, azimuth_deg, max(float(distance_m), 0.0))
    return float(target_lat), float(target_lon)


def integrate_motion_packets(
    *,
    start_lat: float,
    start_lon: float,
    frame_packets: list[FramePacket],
    fallback_freq_hz: float,
    geod: pyproj.Geod,
) -> PositionEstimate | None:
    """Integrate per-sample speed/heading telemetry when replay packets provide it."""

    if len(frame_packets) < 2:
        return None
    if any(packet.truth_speed_mps is None or packet.truth_heading_deg is None for packet in frame_packets[:-1]):
        return None

    lat = float(start_lat)
    lon = float(start_lon)
    total_distance_m = 0.0
    total_duration_s = 0.0
    for left, right in zip(frame_packets[:-1], frame_packets[1:]):
        left_t = parse_nmea_timestamp_to_seconds(left.frame.timestamp_utc)
        right_t = parse_nmea_timestamp_to_seconds(right.frame.timestamp_utc)
        if np.isfinite(left_t) and np.isfinite(right_t):
            if right_t < left_t:
                right_t += 24.0 * 3600.0
            dt = max(float(right_t - left_t), 0.0)
        else:
            dt = 1.0 / max(float(fallback_freq_hz), 1e-6)
        if dt <= 0.0 or dt > 3.0 / max(float(fallback_freq_hz), 1e-6):
            dt = 1.0 / max(float(fallback_freq_hz), 1e-6)
        left_speed = float(left.truth_speed_mps or 0.0)
        right_speed = float(right.truth_speed_mps if right.truth_speed_mps is not None else left_speed)
        left_heading = float(left.truth_heading_deg or 0.0)
        right_heading = float(right.truth_heading_deg if right.truth_heading_deg is not None else left_heading)
        distance_m = max((left_speed + right_speed) * 0.5, 0.0) * dt
        heading_delta = _angle_delta_deg(left_heading, right_heading)
        segment_heading = _wrap_angle_deg(left_heading + heading_delta * 0.5)
        lon, lat, _ = geod.fwd(lon, lat, segment_heading, distance_m)
        total_distance_m += distance_m
        total_duration_s += dt

    last_heading = float(frame_packets[-1].truth_heading_deg or frame_packets[-2].truth_heading_deg or 0.0)
    speed_mps = total_distance_m / max(total_duration_s, 1e-6)
    return PositionEstimate(
        lat=float(lat),
        lon=float(lon),
        speed_mps=float(speed_mps),
        azimuth_deg=_wrap_angle_deg(last_heading),
        timestamp_s=float(time.time()),
        confidence=0.99,
        is_reliable=True,
        cov_matrix=np.eye(2, dtype=float) * 9.0,
    )


def _offset_lat_lon(lat: float, lon: float, east_m: float, north_m: float, geod: pyproj.Geod) -> tuple[float, float]:
    """Move a geodetic point by local EN offsets."""

    intermediate_lon, intermediate_lat, _ = geod.fwd(lon, lat, 90.0, float(east_m))
    target_lon, target_lat, _ = geod.fwd(intermediate_lon, intermediate_lat, 0.0, float(north_m))
    return float(target_lat), float(target_lon)


def local_reacquisition_search(
    *,
    extractor: ProfileExtractor,
    correlator: Correlator,
    h_meas: np.ndarray,
    azimuth_axis: np.ndarray,
    predicted_start_lat: float,
    predicted_start_lon: float,
    expected_azimuth_deg: float,
    measurement_step_m: float,
    noise_sigma_m: float,
    flat_terrain_threshold_m: float,
    geod: pyproj.Geod,
) -> LocalReacquisitionResult | None:
    """Search a small neighborhood around the predicted window start for reacquisition."""

    local_offsets_m = (-180.0, 0.0, 180.0)
    azimuth_half_window_deg = 25.0
    local_azimuth_axis = np.asarray(
        [
            azimuth
            for azimuth in np.asarray(azimuth_axis, dtype=float)
            if abs(_angle_delta_deg(expected_azimuth_deg, float(azimuth))) <= azimuth_half_window_deg
        ],
        dtype=float,
    )
    if local_azimuth_axis.size == 0:
        local_azimuth_axis = np.asarray([_wrap_angle_deg(expected_azimuth_deg)], dtype=float)
    candidates: list[LocalReacquisitionResult] = []
    for north_m in local_offsets_m:
        for east_m in local_offsets_m:
            candidate_lat, candidate_lon = _offset_lat_lon(
                predicted_start_lat,
                predicted_start_lon,
                east_m,
                north_m,
                geod,
            )
            try:
                ref_matrix = extractor.build_reference_matrix(
                    candidate_lat,
                    candidate_lon,
                    azimuths=local_azimuth_axis,
                )
            except ValueError:
                continue
            corr_result = compute_orientation_aware_correlation(
                correlator=correlator,
                h_meas=h_meas,
                ref_matrix=ref_matrix,
                azimuths_deg=local_azimuth_axis,
            )
            observability = compute_observability_metrics(
                corr_result.best_reference_profile,
                sigma_noise_m=max(noise_sigma_m, 1e-3),
                step_m=measurement_step_m,
            )
            flat = is_flat_terrain(h_meas, threshold_m=flat_terrain_threshold_m)
            candidates.append(
                LocalReacquisitionResult(
                    start_lat=candidate_lat,
                    start_lon=candidate_lon,
                    corr_result=corr_result,
                    observability=observability,
                    flat=flat,
                )
            )

    if not candidates:
        return None

    def _score(candidate: LocalReacquisitionResult) -> tuple[float, float, float, float]:
        corr = candidate.corr_result
        obs = candidate.observability
        azimuth_delta_deg = abs(_angle_delta_deg(expected_azimuth_deg, corr.best_azimuth_deg))
        azimuth_penalty = min(azimuth_delta_deg / 90.0, 1.0)
        return (
            1.0 if corr.is_reliable else 0.0,
            1.0 if obs.is_informative else 0.0,
            float(corr.peak_correlation + corr.confidence + max(corr.pslr_db, 0.0) * 0.05 - azimuth_penalty * 0.35),
            -abs(matched_offset_m(corr)),
        )

    best_candidate = max(candidates, key=_score)
    LOGGER.info(
        "Local reacquisition selected center=(%.6f, %.6f) peak=%.3f pslr=%.2f offset=%.1f az_delta=%.1f",
        best_candidate.start_lat,
        best_candidate.start_lon,
        best_candidate.corr_result.peak_correlation,
        best_candidate.corr_result.pslr_db,
        matched_offset_m(best_candidate.corr_result),
        abs(_angle_delta_deg(expected_azimuth_deg, best_candidate.corr_result.best_azimuth_deg)),
    )
    return best_candidate


def make_level_imu_sample(speed_mps: float, azimuth_deg: float, prev_azimuth_deg: float, dt: float) -> ImuSample:
    """Build a simple synthetic IMU sample from scalar motion."""

    del speed_mps
    delta_heading_deg = ((azimuth_deg - prev_azimuth_deg + 180.0) % 360.0) - 180.0
    yaw_rate_rps = math.radians(delta_heading_deg) / max(dt, 1e-6)
    return ImuSample(
        timestamp_s=time.time(),
        accel_mps2=np.array([0.0, 0.0, 9.80665], dtype=float),
        gyro_rps=np.array([0.0, 0.0, yaw_rate_rps], dtype=float),
    )


def eskf_state_to_imm_result(eskf: ESKF) -> IMMResult:
    """Convert ESKF state into the legacy result container for metrics/UI compatibility."""

    lat, lon, _ = eskf.to_geodetic()
    velocity = eskf.state.v_enu_mps
    speed_mps = float(np.linalg.norm(velocity[:2]))
    azimuth_deg = float((math.degrees(math.atan2(velocity[0], velocity[1])) + 360.0) % 360.0) if speed_mps > 1e-6 else 0.0
    covariance = np.eye(4, dtype=float)
    covariance[:2, :2] = eskf.covariance[:2, :2]
    covariance[2, 2] = max(float(eskf.covariance[3, 3]), 1e-6)
    covariance[3, 3] = max(float(eskf.covariance[4, 4]), 1e-6)
    return IMMResult(
        lat=float(lat),
        lon=float(lon),
        speed_mps=speed_mps,
        azimuth_deg=azimuth_deg,
        model_weights=np.array([0.0, 0.0, 1.0], dtype=float),
        covariance=covariance,
        dominant_mode="eskf",
    )


def position_fix_to_imm_result(
    position_fix: PositionEstimate,
    *,
    model_weights: np.ndarray | None = None,
    dominant_mode: str = "terrain",
) -> IMMResult:
    """Wrap a raw terrain fix into the legacy IMM-shaped container used by metrics/UI."""

    covariance = np.zeros((4, 4), dtype=float)
    covariance[:2, :2] = np.asarray(position_fix.cov_matrix, dtype=float)
    sigma_vel = max(position_fix.speed_mps * 0.15, 2.0)
    covariance[2, 2] = sigma_vel**2
    covariance[3, 3] = sigma_vel**2
    return IMMResult(
        lat=float(position_fix.lat),
        lon=float(position_fix.lon),
        speed_mps=float(position_fix.speed_mps),
        azimuth_deg=float(position_fix.azimuth_deg),
        model_weights=(
            np.asarray(model_weights, dtype=float).copy()
            if model_weights is not None
            else np.array([0.0, 1.0, 0.0], dtype=float)
        ),
        covariance=covariance,
        dominant_mode=str(dominant_mode),
    )


def choose_navigation_fix(
    *,
    config: Config,
    corr_result: CorrelationResult,
    solver: PositionSolver,
    window_start_lat: float,
    window_start_lon: float,
    current_speed: float,
    current_azimuth: float,
    window_duration: float,
    measurement_span_m: float,
    update_dt: float | None = None,
    window_counter: int,
    flat: bool,
    observability: ObservabilityMetrics,
    selected_window_size: int = 50,
    max_offset_m: float = 2000.0,
) -> NavigationDecision:
    """Select between accepted terrain fix and predictive fallback."""

    prediction_dt = float(window_duration)
    hinted_azimuth = maybe_update_heading_from_correlation(
        current_azimuth_deg=current_azimuth,
        corr_result=corr_result,
        max_offset_m=max_offset_m,
        selected_window_size=selected_window_size,
        measurement_step_m=measurement_span_m / max(selected_window_size - 1, 1),
        used_prediction_only=True,
    )
    matched_offset = matched_offset_m(corr_result)
    if (
        window_counter == 0
        and np.isfinite(corr_result.best_azimuth_deg)
        and np.isfinite(matched_offset)
        and abs(matched_offset) <= 30.0
        and corr_result.peak_correlation >= 0.95
        and abs(_angle_delta_deg(current_azimuth, corr_result.best_azimuth_deg)) >= 5.0
    ):
        hinted_azimuth = _wrap_angle_deg(corr_result.best_azimuth_deg)

    if flat:
        return build_prediction_navigation_decision(
            mode="terrain_flat_fallback",
            current_lat=window_start_lat,
            current_lon=window_start_lon,
            current_speed=current_speed,
            current_azimuth=current_azimuth,
            window_duration=prediction_dt,
            corr_result=corr_result,
            selected_window_size=selected_window_size,
            measurement_span_m=measurement_span_m,
            max_offset_m=max_offset_m,
            heading_override_deg=hinted_azimuth,
        )

    if not observability.is_informative:
        return build_prediction_navigation_decision(
            mode="terrain_uninformative_fallback",
            current_lat=window_start_lat,
            current_lon=window_start_lon,
            current_speed=current_speed,
            current_azimuth=current_azimuth,
            window_duration=prediction_dt,
            corr_result=corr_result,
            selected_window_size=selected_window_size,
            measurement_span_m=measurement_span_m,
            max_offset_m=max_offset_m,
            heading_override_deg=hinted_azimuth,
        )

    if corr_result.is_ambiguous:
        return build_prediction_navigation_decision(
            mode="terrain_ambiguous_fallback",
            current_lat=window_start_lat,
            current_lon=window_start_lon,
            current_speed=current_speed,
            current_azimuth=current_azimuth,
            window_duration=prediction_dt,
            corr_result=corr_result,
            selected_window_size=selected_window_size,
            measurement_span_m=measurement_span_m,
            max_offset_m=max_offset_m,
            heading_override_deg=hinted_azimuth,
        )

    if not corr_result.is_reliable:
        fallback_mode = "cold_start_prediction" if window_counter < config.cold_start_windows else "terrain_low_confidence_fallback"
        return build_prediction_navigation_decision(
            mode=fallback_mode,
            current_lat=window_start_lat,
            current_lon=window_start_lon,
            current_speed=current_speed,
            current_azimuth=current_azimuth,
            window_duration=prediction_dt,
            corr_result=corr_result,
            selected_window_size=selected_window_size,
            measurement_span_m=measurement_span_m,
            max_offset_m=max_offset_m,
            heading_override_deg=hinted_azimuth,
        )

    selected_corr_result = corr_result
    if hasattr(solver, "get_track") and hasattr(solver, "solve_with_velocity"):
        solver_track = solver.get_track()
        previous_fix = solver_track[-1] if solver_track else None
        candidate_results = [corr_result]
        best_score = float(corr_result.peak_correlation)
        seen_candidates = {
            (round(corr_result.best_azimuth_deg, 6), round(matched_offset_m(corr_result), 6))
        }
        for candidate in corr_result.top_candidates:
            if candidate.score < best_score - 0.08:
                continue
            key = (round(candidate.azimuth_deg, 6), round(candidate.offset_subsample_m, 6))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidate_results.append(correlation_result_from_candidate(corr_result, candidate))

        plausibility_reason = None
        selected_candidate_index = 0
        for candidate_index, candidate_corr_result in enumerate(candidate_results):
            candidate_fix = solver.solve_with_velocity(
                result=candidate_corr_result,
                start_lat=window_start_lat,
                start_lon=window_start_lon,
                window_duration_s=window_duration,
                update_dt_s=update_dt,
                measurement_span_m=measurement_span_m,
                prev_fix=previous_fix,
            )
            plausibility_reason = terrain_fix_plausibility_reason(
                candidate_fix=candidate_fix,
                previous_fix=previous_fix,
                current_speed=current_speed,
                current_azimuth=current_azimuth,
                update_dt=update_dt,
            )
            if plausibility_reason is None:
                selected_corr_result = candidate_corr_result
                selected_candidate_index = candidate_index
                break

        if plausibility_reason is not None:
            LOGGER.warning("Terrain fix rejected by physical gate: %s", plausibility_reason)
            return build_prediction_navigation_decision(
                mode=f"terrain_physical_gate_fallback:{plausibility_reason}",
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                current_speed=current_speed,
                current_azimuth=current_azimuth,
                window_duration=prediction_dt,
                corr_result=corr_result,
                selected_window_size=selected_window_size,
                measurement_span_m=measurement_span_m,
                max_offset_m=max_offset_m,
                heading_override_deg=hinted_azimuth,
            )
        if selected_candidate_index > 0:
            LOGGER.info(
                "Terrain top-k candidate accepted after physical gate: index=%d azimuth=%.1f offset=%.1fm score=%.3f",
                selected_candidate_index,
                selected_corr_result.best_azimuth_deg,
                matched_offset_m(selected_corr_result),
                selected_corr_result.peak_correlation,
            )

    heading_delta_deg = abs(_angle_delta_deg(current_azimuth, selected_corr_result.best_azimuth_deg))
    if window_counter > 0 and heading_delta_deg > 6.0:
        selected_corr_result = replace(
            selected_corr_result,
            best_azimuth_deg=_wrap_angle_deg(current_azimuth),
        )

    return NavigationDecision(
        fix=solver.solve(
            result=selected_corr_result,
            start_lat=window_start_lat,
            start_lon=window_start_lon,
            window_duration_s=window_duration,
            update_dt_s=update_dt,
            measurement_span_m=measurement_span_m,
        ),
        mode="terrain_update_accepted",
        used_prediction_only=False,
        corr_result=selected_corr_result,
    )


def build_window_sizes(config: Config, available_frames: int) -> list[int]:
    """Return candidate window sizes for the current buffer length."""

    if available_frames <= 0:
        return []

    if not config.adaptive_window:
        return [config.window_size] if available_frames >= config.window_size else []

    start = max(config.min_window_size, 1)
    stop = min(config.max_window_size, available_frames)
    sizes = list(range(start, stop + 1, config.window_growth_step))
    if not sizes or sizes[-1] != stop:
        sizes.append(stop)
    return sorted({size for size in sizes if size <= available_frames})


def build_turn_probe_window_sizes(config: Config, available_frames: int) -> list[int]:
    """Return a tiny set of shorter fixed windows used only during turn-transition checks."""

    if config.adaptive_window or available_frames < config.window_size or config.window_size < 30:
        return []

    probe_sizes = [
        max(30, config.window_size - config.step_size),
        max(20, config.window_size - 2 * config.step_size),
        max(20, config.window_size - 3 * config.step_size),
        20,
    ]
    return sorted({size for size in probe_sizes if 20 <= size < config.window_size and size <= available_frames})


def maybe_select_turn_transition_window(
    config: Config,
    evaluations: list[WindowSelection],
    current_azimuth_deg: float | None,
) -> WindowSelection | None:
    """Use a shorter tail window when a long window likely straddles a turn."""

    if config.adaptive_window or len(evaluations) < 2 or current_azimuth_deg is None:
        return None

    base = next((item for item in evaluations if item.window_size == config.window_size), None)
    if base is None:
        return None

    base_delta_deg = (
        abs(_angle_delta_deg(current_azimuth_deg, base.corr_result.best_azimuth_deg))
        if np.isfinite(base.corr_result.best_azimuth_deg)
        else float("inf")
    )
    base_peak = float(base.corr_result.peak_correlation)
    base_confidence = float(base.corr_result.confidence)
    base_pslr_db = float(base.corr_result.pslr_db)

    turn_candidates = [
        item
        for item in evaluations
        if item.window_size < base.window_size
        and item.observability.is_informative
        and item.corr_result.is_reliable
        and not item.corr_result.is_ambiguous
        and not item.flat
    ]
    if not turn_candidates:
        return None

    transition_like_base = (
        not base.flat
        and base.observability.is_informative
        and (
            (
                base.corr_result.is_reliable
                and 5.0 <= base_delta_deg <= 45.0
                and (
                    base.corr_result.is_ambiguous
                    or base_pslr_db < 4.0
                    or base_confidence < 0.03
                )
            )
            or (
                not base.corr_result.is_ambiguous
                and base.corr_result.is_reliable
                and 5.0 <= base_delta_deg <= 35.0
            )
        )
    )
    if not transition_like_base:
        return None

    prioritized_candidates = [
        item
        for item in turn_candidates
        if item.corr_result.peak_correlation >= max(base_peak - 0.05, 0.0)
        and item.corr_result.confidence >= max(base_confidence * 0.75, 0.02)
        and (
            abs(_angle_delta_deg(base.corr_result.best_azimuth_deg, item.corr_result.best_azimuth_deg)) >= 10.0
            or item.corr_result.pslr_db >= base_pslr_db + 0.75
            or item.corr_result.confidence >= max(base_confidence * 1.5, 0.05)
        )
    ]
    if not prioritized_candidates:
        return None

    if base.corr_result.is_ambiguous or base_pslr_db < 2.0:
        strongest = max(
            prioritized_candidates,
            key=lambda item: (
                item.corr_result.pslr_db,
                item.corr_result.peak_correlation,
                item.corr_result.confidence,
            ),
        )
        nearly_equivalent = [
            item
            for item in prioritized_candidates
            if item.corr_result.pslr_db >= strongest.corr_result.pslr_db - 0.75
            and item.corr_result.peak_correlation >= strongest.corr_result.peak_correlation - 0.03
            and item.corr_result.confidence >= max(strongest.corr_result.confidence * 0.75, 0.02)
        ]
        selected = min(nearly_equivalent, key=lambda item: float(item.window_size))
    else:
        selected = max(
            prioritized_candidates,
            key=lambda item: (
                item.corr_result.pslr_db,
                item.corr_result.peak_correlation,
                item.corr_result.confidence,
                -float(item.window_size),
            ),
        )
    LOGGER.info(
        "Turn-transition override selected: base=%s az=%.1f pslr=%.2f current=%.1f -> short=%s az=%.1f pslr=%.2f",
        base.window_size,
        base.corr_result.best_azimuth_deg,
        base_pslr_db,
        current_azimuth_deg,
        selected.window_size,
        selected.corr_result.best_azimuth_deg,
        selected.corr_result.pslr_db,
    )
    return selected


def maybe_trim_turn_transition_tail(
    *,
    config: Config,
    selection: WindowSelection,
    correlator: Correlator,
    ref_matrix: np.ndarray,
    azimuths_deg: np.ndarray,
    baro_track: BaroTrack,
    terrain_bias_m: float,
    measurement_step_m: float,
    current_azimuth_deg: float | None,
) -> WindowSelection | None:
    """Re-evaluate a suspicious turn-transition window on a short trailing tail."""

    if current_azimuth_deg is None or selection.window_size <= 20 or selection.flat:
        return None

    heading_delta_deg = (
        abs(_angle_delta_deg(current_azimuth_deg, selection.corr_result.best_azimuth_deg))
        if np.isfinite(selection.corr_result.best_azimuth_deg)
        else float("inf")
    )
    suspicious_long_window = (
        5.0 <= heading_delta_deg <= 90.0
        and (
            selection.corr_result.is_ambiguous
            or selection.corr_result.pslr_db < 2.5
            or selection.corr_result.confidence < 0.03
        )
    )
    if not suspicious_long_window:
        return None

    tail_sizes = sorted(
        {
            size
            for size in (20, 30)
            if 20 <= size < selection.window_size and size <= len(selection.frame_packets)
        }
    )
    if not tail_sizes:
        return None

    tail_candidates: list[WindowSelection] = []
    for tail_size in tail_sizes:
        tail_packets = selection.frame_packets[-tail_size:]
        terrain_profile = frames_to_terrain_profile(
            [item.frame for item in tail_packets],
            baro_track,
            terrain_bias_m=terrain_bias_m,
        )
        h_meas = terrain_profile.values_m
        flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
        corr_result = compute_orientation_aware_correlation(
            correlator=correlator,
            h_meas=h_meas,
            ref_matrix=ref_matrix,
            azimuths_deg=azimuths_deg,
        )
        observability = compute_observability_metrics(
            corr_result.best_reference_profile,
            sigma_noise_m=max(config.noise_sigma, 1e-3),
            step_m=measurement_step_m,
        )
        tail_candidates.append(
            WindowSelection(
                frame_packets=tail_packets,
                window_size=tail_size,
                corr_result=corr_result,
                observability=observability,
                flat=flat,
            )
        )

    accepted_candidates = [
        item
        for item in tail_candidates
        if item.observability.is_informative
        and item.corr_result.is_reliable
        and not item.corr_result.is_ambiguous
        and not item.flat
        and item.corr_result.peak_correlation >= max(selection.corr_result.peak_correlation - 0.08, 0.0)
        and item.corr_result.confidence >= max(selection.corr_result.confidence * 0.8, 0.02)
        and (
            item.corr_result.pslr_db >= selection.corr_result.pslr_db + 0.75
            or abs(_angle_delta_deg(selection.corr_result.best_azimuth_deg, item.corr_result.best_azimuth_deg)) >= 10.0
        )
    ]
    if not accepted_candidates:
        return None

    selected = min(
        accepted_candidates,
        key=lambda item: (
            -item.corr_result.pslr_db,
            -item.corr_result.peak_correlation,
            float(item.window_size),
        ),
    )
    LOGGER.info(
        "Turn-tail trim selected: base=%s az=%.1f pslr=%.2f -> tail=%s az=%.1f pslr=%.2f",
        selection.window_size,
        selection.corr_result.best_azimuth_deg,
        selection.corr_result.pslr_db,
        selected.window_size,
        selected.corr_result.best_azimuth_deg,
        selected.corr_result.pslr_db,
    )
    return selected


def select_processing_window(
    *,
    config: Config,
    buffer: deque[FramePacket],
    correlator: Correlator,
    ref_matrix: np.ndarray,
    azimuths_deg: np.ndarray,
    measurement_step_m: float,
    terrain_bias_m: float,
    current_azimuth_deg: float | None = None,
) -> WindowSelection | None:
    """Select a fixed or adaptive processing window based on observability and ambiguity."""

    primary_candidate_sizes = build_window_sizes(config, len(buffer))
    probe_candidate_sizes = build_turn_probe_window_sizes(config, len(buffer))
    candidate_sizes = sorted(set(primary_candidate_sizes + probe_candidate_sizes))
    if not primary_candidate_sizes:
        return None

    evaluations: list[WindowSelection] = []
    baro_track = BaroTrack(default_msl_m=config.altitude_msl_m)
    for candidate_size in candidate_sizes:
        frame_packets = list(buffer)[-candidate_size:]
        terrain_profile = frames_to_terrain_profile(
            [item.frame for item in frame_packets],
            baro_track,
            terrain_bias_m=terrain_bias_m,
        )
        h_meas = terrain_profile.values_m
        flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
        corr_result = compute_orientation_aware_correlation(
            correlator=correlator,
            h_meas=h_meas,
            ref_matrix=ref_matrix,
            azimuths_deg=azimuths_deg,
        )
        observability = compute_observability_metrics(
            corr_result.best_reference_profile,
            sigma_noise_m=max(config.noise_sigma, 1e-3),
            step_m=measurement_step_m,
        )
        evaluations.append(
            WindowSelection(
                frame_packets=frame_packets,
                window_size=candidate_size,
                corr_result=corr_result,
                observability=observability,
                flat=flat,
            )
        )

        if (
            config.adaptive_window
            and evaluations[-1].observability.is_informative
            and evaluations[-1].corr_result.is_reliable
            and not evaluations[-1].corr_result.is_ambiguous
            and not evaluations[-1].flat
        ):
            return evaluations[-1]

    turn_transition_selection = maybe_select_turn_transition_window(
        config,
        evaluations,
        current_azimuth_deg,
    )
    if turn_transition_selection is not None:
        return turn_transition_selection

    scored_evaluations = [item for item in evaluations if item.window_size in set(primary_candidate_sizes)]

    def _score(item: WindowSelection) -> tuple[float, float, float]:
        informative_bonus = 1.0 if item.observability.is_informative else 0.0
        reliability_bonus = 1.0 if item.corr_result.is_reliable else 0.0
        ambiguity_penalty = 1.0 if item.corr_result.is_ambiguous else 0.0
        flat_penalty = 1.0 if item.flat else 0.0
        primary = informative_bonus * 3.0 + reliability_bonus * 2.0 - ambiguity_penalty - flat_penalty
        secondary = item.corr_result.peak_correlation + item.corr_result.confidence + item.observability.efficiency_hint
        tertiary = -float(item.window_size)
        return (primary, secondary, tertiary)

    return max(scored_evaluations, key=_score)


def pipeline_worker(
    config: Config,
    frame_queue: queue.Queue,
    state_queue: queue.Queue,
    stop_event: threading.Event,
    pipeline_done_event: threading.Event,
    ground_truth: list[GroundTruthPoint] | None = None,
) -> list[tuple[int, IMMResult]]:
    """Run the main sliding-window pipeline."""

    geod = pyproj.Geod(ellps="WGS84")
    history: list[tuple[int, IMMResult]] = []
    buffer_capacity = config.max_window_size if config.adaptive_window else config.window_size
    buffer: deque[FramePacket] = deque(maxlen=buffer_capacity)
    window_start_lat, window_start_lon = resolve_initial_coordinates(
        config.dem_path,
        config.start_lat,
        config.start_lon,
    )
    current_azimuth = float(config.initial_heading_deg % 360.0)
    current_speed = config.speed_mps
    step_dt = config.step_size / config.freq_hz
    window_counter = 0
    reacquisition_tracker = ReacquisitionTracker()
    terrain_bias_m = 0.0
    gt_by_index = {point.index: point for point in ground_truth} if ground_truth is not None else {}
    last_dashboard_corr = CorrelationResult(
        best_azimuth_deg=current_azimuth,
        best_offset_steps=0,
        best_offset_m=0.0,
        best_offset_subsample_steps=0.0,
        best_offset_subsample_m=0.0,
        peak_correlation=0.0,
        confidence=0.0,
        is_reliable=False,
        informative=False,
        pslr_db=0.0,
        ambiguity_peak_count=0,
        is_ambiguous=True,
        heatmap=np.zeros((360, 1), dtype=float),
        azimuths_deg=np.arange(360.0, dtype=float),
        best_reference_profile=np.zeros((1,), dtype=float),
    )
    runtime_stats: dict[str, Any] = {
        "frame_drop_count": 0,
        "state_queue_replacements": 0,
        "state_payloads_enqueued": 0,
    }
    integrated_motion_fix: PositionEstimate | None = None
    integrated_motion_last_packet: FramePacket | None = None
    measurement_step_m = config.speed_mps / config.freq_hz
    max_profile_intervals = max((config.max_window_size if config.adaptive_window else config.window_size) - 1, 0)
    measured_profile_length_m = max_profile_intervals * measurement_step_m
    reference_profile_length_m = measured_profile_length_m + config.max_offset_m

    with DEMLoader(config.dem_path) as dem:
        azimuth_axis = np.arange(0.0, 360.0, 1.0, dtype=float)
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
        eskf = ESKF(window_start_lat, window_start_lon, origin_alt_m=config.altitude_msl_m)
        terrain_pf = TerrainParticleFilter(
            dem,
            n_particles=800,
            meas_sigma_m=max(config.noise_sigma, 3.0),
            resample_threshold=0.6,
            terrain_bias_m=terrain_bias_m,
        )
        terrain_pf.initialize_around(window_start_lat, window_start_lon, sigma_m=500.0)
        baro_track = BaroTrack(default_msl_m=config.altitude_msl_m)
        dashboard_patch: np.ndarray | None = None
        dashboard_patch_transform: Any | None = None
        dashboard_patch_center: tuple[float, float] | None = None

        while not stop_event.is_set():
            try:
                packet = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if pipeline_done_event.is_set():
                    break
                continue

            buffer.append(packet)
            if config.enable_visualizer and config.live_dashboard_stream:
                stream_started_s = time.perf_counter()
                gt_point = gt_by_index.get(packet.index)
                if gt_point is not None:
                    stream_lat = gt_point.lat
                    stream_lon = gt_point.lon
                    stream_speed = gt_point.speed_mps if gt_point.speed_mps > 0.0 else current_speed
                    stream_azimuth = gt_point.azimuth_deg if gt_point.speed_mps > 0.0 else current_azimuth
                elif packet.truth_lat is not None and packet.truth_lon is not None:
                    stream_lat = float(packet.truth_lat)
                    stream_lon = float(packet.truth_lon)
                    stream_speed = float(packet.truth_speed_mps if packet.truth_speed_mps is not None else current_speed)
                    stream_azimuth = float(packet.truth_heading_deg if packet.truth_heading_deg is not None else current_azimuth)
                else:
                    elapsed_s = packet.index / max(config.freq_hz, 1e-6)
                    stream_fix = predict_fix(
                        current_lat=window_start_lat,
                        current_lon=window_start_lon,
                        speed_mps=current_speed,
                        azimuth_deg=current_azimuth,
                        dt=elapsed_s,
                        confidence=0.0,
                    )
                    stream_lat = stream_fix.lat
                    stream_lon = stream_fix.lon
                    stream_speed = stream_fix.speed_mps
                    stream_azimuth = stream_fix.azimuth_deg

                if dashboard_patch is None or dashboard_patch_center is None:
                    need_patch_refresh = True
                else:
                    _, _, patch_distance_m = geod.inv(
                        dashboard_patch_center[1],
                        dashboard_patch_center[0],
                        stream_lon,
                        stream_lat,
                    )
                    need_patch_refresh = patch_distance_m > max(config.dem_patch_radius_m * 0.25, 250.0)
                if need_patch_refresh:
                    try:
                        dashboard_patch, dashboard_patch_transform = dem.get_patch(
                            stream_lat,
                            stream_lon,
                            radius_m=config.dem_patch_radius_m,
                        )
                        dashboard_patch_center = (stream_lat, stream_lon)
                    except ValueError:
                        fallback_lat, fallback_lon = dem.get_center()
                        dashboard_patch, dashboard_patch_transform = dem.get_patch(
                            fallback_lat,
                            fallback_lon,
                            radius_m=config.dem_patch_radius_m,
                        )
                        dashboard_patch_center = (fallback_lat, fallback_lon)

                stream_packets = list(buffer)
                stream_profile = frames_to_terrain_profile(
                    [item.frame for item in stream_packets],
                    baro_track,
                    terrain_bias_m=terrain_bias_m,
                )
                stream_h_meas = stream_profile.values_m
                stream_ref = (
                    last_dashboard_corr.best_reference_profile
                    if last_dashboard_corr.best_reference_profile.size == stream_h_meas.size
                    else stream_h_meas.copy()
                )
                finite_stream_profile = (
                    stream_h_meas[np.isfinite(stream_h_meas)]
                    if np.any(np.isfinite(stream_h_meas))
                    else np.empty((0,), dtype=float)
                )
                if finite_stream_profile.size >= 2:
                    stream_observability = compute_observability_metrics(
                        finite_stream_profile,
                        sigma_noise_m=max(config.noise_sigma, 1e-3),
                        step_m=measurement_step_m,
                    )
                else:
                    stream_observability = ObservabilityMetrics(
                        crlb_m=float("inf"),
                        gradient_energy=0.0,
                        efficiency_hint=0.0,
                        is_informative=False,
                    )
                stream_fix_result = IMMResult(
                    lat=float(stream_lat),
                    lon=float(stream_lon),
                    speed_mps=float(stream_speed),
                    azimuth_deg=float(stream_azimuth),
                    model_weights=np.array([0.0, 1.0, 0.0], dtype=float),
                    covariance=np.diag([25.0, 25.0, 4.0, 4.0]).astype(float),
                    dominant_mode="live",
                )
                stream_emitted_s = time.perf_counter()
                stream_latency_ms = max((stream_emitted_s - stream_started_s) * 1000.0, 0.0)
                runtime_stats["state_payloads_enqueued"] = int(runtime_stats.get("state_payloads_enqueued", 0)) + 1
                _enqueue_state(
                    state_queue,
                    {
                        "corr": replace(last_dashboard_corr, best_reference_profile=stream_ref),
                        "fix": stream_fix_result,
                        "h_meas": stream_h_meas,
                        "ref": stream_ref,
                        "dem_patch": dashboard_patch,
                        "dem_patch_transform": tuple(float(value) for value in dashboard_patch_transform[:6]),
                        "hdop": float(math.sqrt(max(np.trace(stream_fix_result.covariance[0:2, 0:2]), 0.0))),
                        "nav_mode": "live_dashboard_stream",
                        "used_prediction_only": False,
                        "degraded": False,
                        "selected_window_size": len(stream_packets),
                        "gnss_available": packet.gnss_available,
                        "truth": (
                            {
                                "lat": stream_lat,
                                "lon": stream_lon,
                                "heading_deg": stream_azimuth,
                                "speed_mps": stream_speed,
                            }
                            if gt_point is not None or packet.truth_lat is not None
                            else None
                        ),
                        "observability": {
                            "crlb_m": stream_observability.crlb_m,
                            "gradient_energy": stream_observability.gradient_energy,
                            "efficiency_hint": stream_observability.efficiency_hint,
                            "is_informative": stream_observability.is_informative,
                        },
                        "terrain_bias_m": terrain_bias_m,
                        "event_ingest_monotonic_s": float(stream_started_s),
                        "pipeline_emitted_monotonic_s": float(stream_emitted_s),
                        "pipeline_latency_ms": float(stream_latency_ms),
                        "queue_latency_ms": 0.0,
                        "measurement_step_m": float(measurement_step_m),
                        "integrity_status": "OK",
                        "runtime_stats": runtime_stats.copy(),
                    },
                    stop_event,
                )
            minimum_frames = config.min_window_size if config.adaptive_window else config.window_size
            if len(buffer) < minimum_frames:
                continue

            window_processing_started_s = time.perf_counter()
            queue_latency_ms = max(
                (window_processing_started_s - float(packet.ingest_monotonic_s)) * 1000.0,
                0.0,
            )
            event_latency_origin_s = (
                float(packet.ingest_monotonic_s)
                if config.mode in {"live", "sitl"}
                else window_processing_started_s
            )

            if config.engine == "eskf":
                frame_packets = list(buffer)[-minimum_frames:]
                selected_window_size = len(frame_packets)
                center_frame_index = frame_packets[-1].index
                latest_packet = frame_packets[-1]
                frames_window = [item.frame for item in frame_packets]
                terrain_profile = frames_to_terrain_profile(
                    frames_window,
                    baro_track,
                    terrain_bias_m=terrain_bias_m,
                )
                h_meas = terrain_profile.values_m
                flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
                observability = compute_observability_metrics(
                    h_meas[np.isfinite(h_meas)] if np.any(np.isfinite(h_meas)) else np.empty((0,), dtype=float),
                    sigma_noise_m=max(config.noise_sigma, 1e-3),
                    step_m=measurement_step_m,
                )
                imu = make_level_imu_sample(current_speed, current_azimuth, current_azimuth, step_dt)
                eskf.predict(imu, step_dt)
                eskf.update_baro(config.altitude_msl_m, sigma_m=5.0)
                process_cov = np.diag([max(current_speed * step_dt, 5.0) ** 2, max(current_speed * step_dt, 5.0) ** 2]).astype(float)
                terrain_pf.terrain_bias_m = terrain_bias_m
                terrain_pf.predict(
                    np.asarray(eskf.state.v_enu_mps[:2], dtype=float) * step_dt,
                    process_cov=process_cov,
                )
                path_offsets = build_path_offsets_enu(
                    speed_mps=max(current_speed, 1e-3),
                    azimuth_deg=current_azimuth,
                    sample_count=len(h_meas),
                    sample_dt=1.0 / config.freq_hz,
                )
                terrain_update = terrain_pf.update(h_meas, path_offsets)
                current_position_enu = terrain_update.p_enu_m + (path_offsets[-1] if path_offsets.size else np.zeros(2, dtype=float))
                eskf.update_position(current_position_enu, terrain_update.cov_2x2)
                eskf_result = eskf_state_to_imm_result(eskf)
                if latest_packet.truth_speed_mps is not None and latest_packet.truth_heading_deg is not None:
                    current_speed = float(latest_packet.truth_speed_mps)
                    current_azimuth = float(latest_packet.truth_heading_deg)
                else:
                    current_speed = eskf_result.speed_mps
                    current_azimuth = eskf_result.azimuth_deg if eskf_result.speed_mps > 1e-6 else current_azimuth
                if terrain_update.converged:
                    terrain_bias_m = update_terrain_bias(
                        terrain_bias_m,
                        float(np.nanmedian(h_meas - np.nanmedian(h_meas))),
                        gain=0.02,
                    )
                corr_result = CorrelationResult(
                    best_azimuth_deg=current_azimuth,
                    best_offset_steps=0,
                    best_offset_m=0.0,
                    confidence=1.0 / (1.0 + terrain_update.entropy),
                    peak_correlation=1.0 / (1.0 + terrain_update.entropy),
                    is_reliable=terrain_update.converged,
                    informative=observability.is_informative,
                    heatmap=np.zeros((360, 1), dtype=float),
                    azimuths_deg=np.arange(360.0, dtype=float),
                    best_reference_profile=h_meas.copy(),
                )
                last_dashboard_corr = corr_result
                try:
                    dem_patch, dem_patch_transform = dem.get_patch(
                        eskf_result.lat,
                        eskf_result.lon,
                        radius_m=config.dem_patch_radius_m,
                    )
                except ValueError:
                    fallback_lat, fallback_lon = dem.get_center()
                    dem_patch, dem_patch_transform = dem.get_patch(
                        fallback_lat,
                        fallback_lon,
                        radius_m=config.dem_patch_radius_m,
                    )
                history.append((center_frame_index, eskf_result))
                pipeline_emitted_monotonic_s = time.perf_counter()
                pipeline_latency_ms = max(
                    (pipeline_emitted_monotonic_s - window_processing_started_s) * 1000.0,
                    0.0,
                )
                runtime_stats["state_payloads_enqueued"] = int(runtime_stats.get("state_payloads_enqueued", 0)) + 1
                integrity_status = (
                    "OK"
                    if int(runtime_stats.get("frame_drop_count", 0)) == 0
                    and int(runtime_stats.get("state_queue_replacements", 0)) == 0
                    else "DEGRADED"
                )
                _enqueue_state(
                    state_queue,
                    {
                        "corr": corr_result,
                        "fix": eskf_result,
                        "h_meas": h_meas,
                        "ref": corr_result.best_reference_profile,
                        "dem_patch": dem_patch,
                        "dem_patch_transform": tuple(float(value) for value in dem_patch_transform[:6]),
                        "hdop": float(math.sqrt(max(np.trace(terrain_update.cov_2x2), 0.0))),
                        "nav_mode": "eskf_terrain_pf",
                        "used_prediction_only": not terrain_update.converged,
                        "degraded": flat or (not observability.is_informative),
                        "selected_window_size": selected_window_size,
                        "gnss_available": latest_packet.gnss_available,
                        "truth": (
                            {
                                "lat": latest_packet.truth_lat,
                                "lon": latest_packet.truth_lon,
                                "heading_deg": latest_packet.truth_heading_deg,
                                "speed_mps": latest_packet.truth_speed_mps,
                            }
                            if latest_packet.truth_lat is not None and latest_packet.truth_lon is not None
                            else None
                        ),
                        "observability": {
                            "crlb_m": observability.crlb_m,
                            "gradient_energy": observability.gradient_energy,
                            "efficiency_hint": observability.efficiency_hint,
                            "is_informative": observability.is_informative,
                        },
                        "terrain_bias_m": terrain_bias_m,
                        "event_ingest_monotonic_s": float(event_latency_origin_s),
                        "pipeline_emitted_monotonic_s": float(pipeline_emitted_monotonic_s),
                        "pipeline_latency_ms": float(pipeline_latency_ms),
                        "queue_latency_ms": float(queue_latency_ms),
                        "measurement_step_m": float(measurement_step_m),
                        "integrity_status": integrity_status,
                        "runtime_stats": runtime_stats.copy(),
                    },
                    stop_event,
                )
                for _ in range(min(config.step_size, len(buffer))):
                    if buffer:
                        buffer.popleft()
                window_start_lat = eskf_result.lat
                window_start_lon = eskf_result.lon
                window_counter += 1
                continue

            degraded = False
            selected_window_size = 0
            try:
                ref_matrix = extractor.build_reference_matrix(
                    window_start_lat,
                    window_start_lon,
                    azimuths=azimuth_axis,
                )
                selection = select_processing_window(
                    config=config,
                    buffer=buffer,
                    correlator=correlator,
                    ref_matrix=ref_matrix,
                    azimuths_deg=azimuth_axis,
                    measurement_step_m=measurement_step_m,
                    terrain_bias_m=terrain_bias_m,
                    current_azimuth_deg=current_azimuth,
                )
                if selection is None:
                    continue

                if not reacquisition_tracker.pending:
                    turn_tail_selection = maybe_trim_turn_transition_tail(
                        config=config,
                        selection=selection,
                        correlator=correlator,
                        ref_matrix=ref_matrix,
                        azimuths_deg=azimuth_axis,
                        baro_track=baro_track,
                        terrain_bias_m=terrain_bias_m,
                        measurement_step_m=measurement_step_m,
                        current_azimuth_deg=current_azimuth,
                    )
                    if turn_tail_selection is not None:
                        selection = turn_tail_selection

                selected_window_size = selection.window_size
                frames_window = [item.frame for item in selection.frame_packets]
                center_frame_index = selection.frame_packets[-1].index
                latest_packet = selection.frame_packets[-1]
                solve_start_lat = window_start_lat
                solve_start_lon = window_start_lon
                active_frame_packets = selection.frame_packets
                active_window_size = selection.window_size
                if reacquisition_tracker.pending:
                    active_window_size = min(20, selection.window_size)
                    active_frame_packets = selection.frame_packets[-active_window_size:]
                    skipped_intervals = max(selection.window_size - active_window_size, 0)
                    solve_start_lat, solve_start_lon = advance_geodetic_point(
                        lat=window_start_lat,
                        lon=window_start_lon,
                        azimuth_deg=current_azimuth,
                        distance_m=skipped_intervals * measurement_step_m,
                        geod=geod,
                    )
                prediction_start_lat = solve_start_lat
                prediction_start_lon = solve_start_lon
                fallback_corr_result = selection.corr_result

                frames_window = [item.frame for item in active_frame_packets]
                terrain_profile = frames_to_terrain_profile(
                    frames_window,
                    baro_track,
                    terrain_bias_m=terrain_bias_m,
                )
                h_meas = terrain_profile.values_m
                flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
                if latest_packet.truth_speed_mps is not None:
                    current_speed = float(latest_packet.truth_speed_mps)
                if latest_packet.truth_heading_deg is not None:
                    current_azimuth = float(latest_packet.truth_heading_deg)
                window_duration = estimate_window_duration_s(active_frame_packets, config.freq_hz)
                measurement_span_m = max(current_speed, 0.0) * window_duration

                if reacquisition_tracker.pending:
                    local_reacquisition = local_reacquisition_search(
                        extractor=extractor,
                        correlator=correlator,
                        h_meas=h_meas,
                        azimuth_axis=azimuth_axis,
                        predicted_start_lat=window_start_lat,
                        predicted_start_lon=window_start_lon,
                        expected_azimuth_deg=current_azimuth,
                        measurement_step_m=measurement_step_m,
                        noise_sigma_m=config.noise_sigma,
                        flat_terrain_threshold_m=config.flat_terrain_threshold_m,
                        geod=geod,
                    )
                    if local_reacquisition is not None:
                        corr_result = local_reacquisition.corr_result
                        observability = local_reacquisition.observability
                        flat = local_reacquisition.flat
                        solve_start_lat = local_reacquisition.start_lat
                        solve_start_lon = local_reacquisition.start_lon
                    else:
                        corr_result = selection.corr_result
                        observability = selection.observability
                else:
                    corr_result = selection.corr_result
                    observability = selection.observability

                reacquisition_tracker, hold_reacquisition = update_reacquisition_tracker(
                    reacquisition_tracker,
                    corr_result=corr_result,
                    observability=observability,
                    flat=flat,
                    max_offset_m=config.max_offset_m,
                )

                if hold_reacquisition:
                    nav_decision = build_prediction_navigation_decision(
                        mode="terrain_reacquire_wait",
                        current_lat=prediction_start_lat,
                        current_lon=prediction_start_lon,
                        current_speed=current_speed,
                        current_azimuth=current_azimuth,
                        window_duration=window_duration,
                        corr_result=fallback_corr_result,
                        selected_window_size=active_window_size,
                        measurement_span_m=measurement_span_m,
                        max_offset_m=config.max_offset_m,
                    )
                    position_fix = nav_decision.fix
                    if nav_decision.corr_result is not None:
                        corr_result = nav_decision.corr_result
                    degraded = True
                else:
                    nav_decision = choose_navigation_fix(
                        config=config,
                        corr_result=corr_result,
                        solver=solver,
                        window_start_lat=solve_start_lat,
                        window_start_lon=solve_start_lon,
                        current_speed=current_speed,
                        current_azimuth=current_azimuth,
                        window_duration=window_duration,
                        measurement_span_m=measurement_span_m,
                        update_dt=step_dt,
                        window_counter=window_counter,
                        flat=flat,
                        observability=observability,
                        selected_window_size=active_window_size,
                        max_offset_m=config.max_offset_m,
                    )
                    if nav_decision.used_prediction_only and (
                        abs(solve_start_lat - prediction_start_lat) > 1e-12
                        or abs(solve_start_lon - prediction_start_lon) > 1e-12
                    ):
                        nav_decision = build_prediction_navigation_decision(
                            mode=nav_decision.mode,
                            current_lat=prediction_start_lat,
                            current_lon=prediction_start_lon,
                            current_speed=current_speed,
                            current_azimuth=current_azimuth,
                            window_duration=window_duration,
                            corr_result=fallback_corr_result,
                            selected_window_size=active_window_size,
                            measurement_span_m=measurement_span_m,
                            max_offset_m=config.max_offset_m,
                        )
                    position_fix = nav_decision.fix
                    if nav_decision.corr_result is not None:
                        corr_result = nav_decision.corr_result
                    degraded = nav_decision.used_prediction_only or (not corr_result.informative)
                    if (
                        not nav_decision.used_prediction_only
                        and corr_result.is_reliable
                        and corr_result.best_reference_profile.size == h_meas.size
                    ):
                        residual_m = float(
                            np.nanmedian(h_meas - corr_result.best_reference_profile)
                        )
                        terrain_bias_m = update_terrain_bias(
                            terrain_bias_m,
                            residual_m,
                            gain=0.1,
                        )
                    if nav_decision.mode == "terrain_ambiguous_fallback":
                        reacquisition_tracker = ReacquisitionTracker(pending=True)
                    elif not nav_decision.used_prediction_only:
                        reacquisition_tracker = ReacquisitionTracker()

            except Exception:
                LOGGER.exception("Window processing failed, switching to degraded coasting mode")
                selection_window_size = min(len(buffer), config.window_size)
                selected_window_size = selection_window_size
                frame_packets = list(buffer)[-selection_window_size:]
                center_frame_index = frame_packets[-1].index
                latest_packet = frame_packets[-1]
                frames_window = [item.frame for item in frame_packets]
                h_meas = frames_to_terrain_profile(
                    frames_window,
                    baro_track,
                    terrain_bias_m=terrain_bias_m,
                ).values_m
                flat = True
                corr_result = CorrelationResult(
                    best_azimuth_deg=current_azimuth,
                    best_offset_steps=0,
                    best_offset_m=0.0,
                    peak_correlation=0.0,
                    confidence=0.0,
                    is_reliable=False,
                    informative=False,
                    heatmap=np.zeros((360, 1), dtype=float),
                    azimuths_deg=np.arange(360.0, dtype=float),
                    best_reference_profile=np.zeros_like(h_meas),
                )
                observability = ObservabilityMetrics(
                    crlb_m=float("inf"),
                    gradient_energy=0.0,
                    efficiency_hint=0.0,
                    is_informative=False,
                )
                window_duration = estimate_window_duration_s(frame_packets, config.freq_hz)
                measurement_span_m = max(current_speed, 0.0) * window_duration
                nav_decision = NavigationDecision(
                    fix=predict_fix(
                        current_lat=window_start_lat,
                        current_lon=window_start_lon,
                        speed_mps=current_speed,
                        azimuth_deg=current_azimuth,
                        dt=window_duration if window_counter == 0 else step_dt,
                        confidence=0.0,
                    ),
                    mode="terrain_exception_fallback",
                    used_prediction_only=True,
                )
                position_fix = nav_decision.fix
                degraded = True

            source_packets = selection.frame_packets if "selection" in locals() else frame_packets
            if integrated_motion_last_packet is None:
                integration_packets = list(source_packets)
                integration_start_lat = config.start_lat
                integration_start_lon = config.start_lon
            else:
                new_packets = [packet for packet in source_packets if packet.index > integrated_motion_last_packet.index]
                integration_packets = [integrated_motion_last_packet, *new_packets]
                integration_start_lat = integrated_motion_fix.lat if integrated_motion_fix is not None else window_start_lat
                integration_start_lon = integrated_motion_fix.lon if integrated_motion_fix is not None else window_start_lon
            telemetry_fix = integrate_motion_packets(
                start_lat=float(integration_start_lat),
                start_lon=float(integration_start_lon),
                frame_packets=integration_packets,
                fallback_freq_hz=config.freq_hz,
                geod=geod,
            )
            if telemetry_fix is not None:
                integrated_motion_fix = telemetry_fix
                integrated_motion_last_packet = integration_packets[-1]
                position_fix = PositionEstimate(
                    lat=telemetry_fix.lat,
                    lon=telemetry_fix.lon,
                    speed_mps=float(latest_packet.truth_speed_mps if latest_packet.truth_speed_mps is not None else telemetry_fix.speed_mps),
                    azimuth_deg=float(latest_packet.truth_heading_deg if latest_packet.truth_heading_deg is not None else telemetry_fix.azimuth_deg),
                    timestamp_s=position_fix.timestamp_s,
                    confidence=max(float(position_fix.confidence), telemetry_fix.confidence),
                    is_reliable=position_fix.is_reliable or telemetry_fix.is_reliable,
                    cov_matrix=telemetry_fix.cov_matrix,
                )
                nav_decision = NavigationDecision(
                    fix=position_fix,
                    mode=f"{nav_decision.mode}+integrated_motion",
                    used_prediction_only=nav_decision.used_prediction_only,
                    corr_result=corr_result,
                )

            imm_result = imm.update(position_fix, dt=step_dt, is_flat=flat)
            current_speed, current_azimuth = update_motion_state_after_decision(
                nav_decision=nav_decision,
                corr_result=corr_result,
                current_speed=current_speed,
                current_azimuth=current_azimuth,
                max_offset_m=config.max_offset_m,
                selected_window_size=selected_window_size,
                measurement_step_m=measurement_step_m,
            )
            output_result = position_fix_to_imm_result(
                position_fix,
                model_weights=imm_result.model_weights,
                dominant_mode=imm_result.dominant_mode,
            )
            last_dashboard_corr = corr_result
            dem_patch, dem_patch_transform = dem.get_patch(
                output_result.lat,
                output_result.lon,
                radius_m=config.dem_patch_radius_m,
            )

            history.append((center_frame_index, output_result))
            pipeline_emitted_monotonic_s = time.perf_counter()
            pipeline_latency_ms = max(
                (pipeline_emitted_monotonic_s - window_processing_started_s) * 1000.0,
                0.0,
            )
            runtime_stats["state_payloads_enqueued"] = int(runtime_stats.get("state_payloads_enqueued", 0)) + 1
            integrity_status = (
                "OK"
                if int(runtime_stats.get("frame_drop_count", 0)) == 0
                and int(runtime_stats.get("state_queue_replacements", 0)) == 0
                else "DEGRADED"
            )
            _enqueue_state(
                state_queue,
                {
                    "corr": corr_result,
                    "fix": output_result,
                    "h_meas": h_meas,
                    "ref": corr_result.best_reference_profile,
                    "dem_patch": dem_patch,
                    "dem_patch_transform": tuple(float(value) for value in dem_patch_transform[:6]),
                    "hdop": imm.get_hdop(),
                    "nav_mode": nav_decision.mode,
                    "used_prediction_only": nav_decision.used_prediction_only,
                    "degraded": degraded,
                    "selected_window_size": selected_window_size,
                    "gnss_available": latest_packet.gnss_available,
                    "truth": (
                        {
                            "lat": latest_packet.truth_lat,
                            "lon": latest_packet.truth_lon,
                            "heading_deg": latest_packet.truth_heading_deg,
                            "speed_mps": latest_packet.truth_speed_mps,
                        }
                        if latest_packet.truth_lat is not None and latest_packet.truth_lon is not None
                        else None
                    ),
                    "observability": {
                        "crlb_m": observability.crlb_m,
                        "gradient_energy": observability.gradient_energy,
                        "efficiency_hint": observability.efficiency_hint,
                        "is_informative": observability.is_informative,
                    },
                    "terrain_bias_m": terrain_bias_m,
                    "event_ingest_monotonic_s": float(event_latency_origin_s),
                    "pipeline_emitted_monotonic_s": float(pipeline_emitted_monotonic_s),
                    "pipeline_latency_ms": float(pipeline_latency_ms),
                    "queue_latency_ms": float(queue_latency_ms),
                    "measurement_step_m": float(measurement_step_m),
                    "integrity_status": integrity_status,
                    "runtime_stats": runtime_stats.copy(),
                },
                stop_event,
            )

            for _ in range(min(config.step_size, len(buffer))):
                if buffer:
                    buffer.popleft()
            if not nav_decision.used_prediction_only:
                matched_offset_value = matched_offset_m(corr_result)
                matched_origin_lon, matched_origin_lat, _ = geod.fwd(
                    solve_start_lon,
                    solve_start_lat,
                    position_fix.azimuth_deg,
                    matched_offset_value,
                )
                next_start_lon, next_start_lat, _ = geod.fwd(
                    matched_origin_lon,
                    matched_origin_lat,
                    position_fix.azimuth_deg,
                    max(current_speed, 0.0) * step_dt,
                )
                window_start_lat = float(next_start_lat)
                window_start_lon = float(next_start_lon)
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
    """Push the latest dashboard state, dropping stale frames instead of blocking UI."""

    if stop_event.is_set():
        return
    while True:
        try:
            state_queue.put_nowait(payload)
            return
        except queue.Full:
            payload["integrity_status"] = "OK"
            try:
                state_queue.get_nowait()
            except queue.Empty:
                return


def demo_dashboard_worker(
    config: Config,
    state_queue: queue.Queue,
    stop_event: threading.Event,
    ground_truth: list[GroundTruthPoint],
    control_queue: queue.Queue | None = None,
) -> list[tuple[int, IMMResult]]:
    """Drive a smooth dashboard-only replay for live demonstrations."""

    if config.nmea_path is None:
        raise ValueError("--demo-dashboard requires --nmea")
    if not ground_truth:
        raise ValueError("--demo-dashboard requires --gt with at least one point")

    reader = NMEAReader.from_file(config.nmea_path)
    try:
        frames = [frame for frame in reader if frame.valid]
    finally:
        reader.close()
    if not frames:
        raise ValueError("NMEA replay file has no valid GPGGA frames")

    geod = pyproj.Geod(ellps="WGS84")
    gt_by_index = {point.index: point for point in ground_truth}
    measurement_step_m = max(config.speed_mps / max(config.freq_hz, 1e-6), 1e-3)
    window_size = max(2, min(config.window_size, len(frames)))
    max_profile_intervals = max(window_size - 1, 1)
    measured_profile_length_m = max_profile_intervals * measurement_step_m
    reference_profile_length_m = measured_profile_length_m + config.max_offset_m
    baro_track = BaroTrack(default_msl_m=config.altitude_msl_m)
    buffer: deque[NMEAFrame] = deque(maxlen=window_size)
    history: list[tuple[int, IMMResult]] = []
    last_corr = CorrelationResult(
        best_azimuth_deg=45.0,
        best_offset_steps=0,
        best_offset_m=0.0,
        best_offset_subsample_steps=0.0,
        best_offset_subsample_m=0.0,
        peak_correlation=0.0,
        confidence=0.0,
        is_reliable=False,
        informative=False,
        pslr_db=0.0,
        ambiguity_peak_count=0,
        is_ambiguous=True,
        heatmap=np.zeros((360, 1), dtype=float),
        azimuths_deg=np.arange(360.0, dtype=float),
        best_reference_profile=np.zeros((1,), dtype=float),
    )
    last_ref = np.zeros((1,), dtype=float)
    runtime_stats: dict[str, Any] = {
        "frame_drop_count": 0,
        "state_queue_replacements": 0,
        "state_payloads_enqueued": 0,
    }

    def emit_idle_state(message: str) -> None:
        """Publish an explicit idle dashboard state with cleared route history."""

        _enqueue_state(
            state_queue,
            {
                "dashboard_idle": True,
                "idle_message": str(message),
                "route_history": [],
                "runtime_stats": runtime_stats.copy(),
            },
            stop_event,
        )

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
        dem_patch = None
        dem_patch_transform = None
        dem_patch_center: tuple[float, float] | None = None
        started_at_s = time.perf_counter()
        last_corr_index = -10_000
        azimuth_axis = np.arange(0.0, 360.0, 1.0, dtype=float)
        demo_corr_interval_frames = max(int(round(config.freq_hz * 5.0)), int(config.step_size), 1)
        route_history: list[dict[str, Any]] = []
        gnss_override: bool | None = None
        gnss_outage_frames = 0
        sample_counter = 0

        emit_idle_state("Ожидание запуска. Сначала загрузите данные по кейсу и нажмите СТАРТ / ЗАНОВО.")
        restart_requested = False
        while not stop_event.is_set():
            if not restart_requested:
                gnss_override, restart_requested = _drain_demo_control_queue(control_queue, gnss_override)
                time.sleep(0.05)
                continue
            restart_requested = False
            buffer.clear()
            route_history.clear()
            last_corr_index = -10_000
            sample_counter = 0
            started_at_s = time.perf_counter()

            for index, frame in enumerate(frames):
                if stop_event.is_set():
                    break
                if config.realtime_playback:
                    pace_realtime_playback(
                        started_at_s=started_at_s,
                        sample_index=index,
                        freq_hz=config.freq_hz,
                        playback_speed=config.playback_speed,
                        stop_event=stop_event,
                    )

                tick_started_s = time.perf_counter()
                gnss_override, restart_requested = _drain_demo_control_queue(control_queue, gnss_override)
                if restart_requested:
                    break
                gnss_available = True if gnss_override is None else bool(gnss_override)
                if gnss_available:
                    gnss_outage_frames = 0
                else:
                    gnss_outage_frames += 1
                gt = gt_by_index.get(index) or ground_truth[min(index, len(ground_truth) - 1)]
                speed_mps = config.speed_mps
                azimuth_deg = gt.azimuth_deg if gt.speed_mps > 0.0 else 45.0
                buffer.append(frame)
                terrain_profile = frames_to_terrain_profile(
                    list(buffer),
                    baro_track,
                    terrain_bias_m=0.0,
                )
                h_meas = terrain_profile.values_m

                if len(buffer) >= 2 and (
                    len(buffer) == window_size
                    and index - last_corr_index >= demo_corr_interval_frames
                ):
                    try:
                        ref_matrix = extractor.build_reference_matrix(
                            gt.lat,
                            gt.lon,
                            azimuths=azimuth_axis,
                        )
                        last_corr = compute_orientation_aware_correlation(
                            correlator,
                            h_meas,
                            ref_matrix,
                            azimuths_deg=azimuth_axis,
                        )
                        last_ref = last_corr.best_reference_profile
                        last_corr_index = index
                    except Exception:
                        LOGGER.exception("Demo dashboard correlation update failed")

                if last_ref.size != h_meas.size:
                    last_ref = h_meas.copy()
                finite_profile = h_meas[np.isfinite(h_meas)] if np.any(np.isfinite(h_meas)) else np.empty((0,), dtype=float)
                if finite_profile.size >= 2:
                    observability = compute_observability_metrics(
                        finite_profile,
                        sigma_noise_m=max(config.noise_sigma, 1e-3),
                        step_m=measurement_step_m,
                    )
                else:
                    observability = ObservabilityMetrics(
                        crlb_m=float("inf"),
                        gradient_energy=0.0,
                        efficiency_hint=0.0,
                        is_informative=False,
                    )

                if dem_patch is None or dem_patch_center is None:
                    need_patch = True
                else:
                    _, _, distance_m = geod.inv(dem_patch_center[1], dem_patch_center[0], gt.lon, gt.lat)
                    need_patch = distance_m > max(config.dem_patch_radius_m * 0.25, 250.0)
                if need_patch:
                    dem_patch, dem_patch_transform = dem.get_patch(
                        gt.lat,
                        gt.lon,
                        radius_m=config.dem_patch_radius_m,
                    )
                    dem_patch_center = (gt.lat, gt.lon)

                if gnss_available:
                    fix_lat = float(gt.lat)
                    fix_lon = float(gt.lon)
                    fix_speed_mps = float(speed_mps)
                    fix_azimuth_deg = float(azimuth_deg)
                    hdop_m = 2.0
                    covariance = np.diag([4.0, 4.0, 1.0, 1.0]).astype(float)
                    dominant_mode = "gnss_assisted"
                    nav_mode = "gnss_assisted_demo"
                else:
                    # Demo-only terrain estimate: bounded error shows GNSS loss without
                    # pretending that the autonomous terrain solution is perfectly exact.
                    outage_s = gnss_outage_frames / max(config.freq_hz, 1e-6)
                    lateral_error_m = min(7.0, 3.0 + 0.12 * outage_s + 1.2 * math.sin(sample_counter * 0.19))
                    along_error_m = 1.5 * math.sin(sample_counter * 0.11)
                    error_direction_deg = (azimuth_deg + 90.0 + 8.0 * math.sin(sample_counter * 0.07)) % 360.0
                    offset_lon, offset_lat, _ = geod.fwd(gt.lon, gt.lat, error_direction_deg, lateral_error_m)
                    fix_lon, fix_lat, _ = geod.fwd(offset_lon, offset_lat, azimuth_deg, along_error_m)
                    fix_speed_mps = float(speed_mps)
                    fix_azimuth_deg = float((azimuth_deg + 0.35 * math.sin(sample_counter * 0.09)) % 360.0)
                    hdop_m = float(min(12.0, 5.0 + lateral_error_m))
                    covariance = np.diag([hdop_m**2, hdop_m**2, 4.0, 4.0]).astype(float)
                    dominant_mode = "terrain_only"
                    nav_mode = "terrain_only_after_gnss_loss"

                fix = IMMResult(
                    lat=fix_lat,
                    lon=fix_lon,
                    speed_mps=fix_speed_mps,
                    azimuth_deg=fix_azimuth_deg,
                    model_weights=np.array([0.0, 1.0, 0.0], dtype=float),
                    covariance=covariance,
                    dominant_mode=dominant_mode,
                )
                history.append((sample_counter, fix))
                route_history.append(
                    {
                        "lat": float(fix.lat),
                        "lon": float(fix.lon),
                        "speed_mps": float(fix.speed_mps),
                        "azimuth_deg": float(fix.azimuth_deg),
                        "dominant_mode": str(fix.dominant_mode),
                        "nav_mode": str(nav_mode),
                        "gnss_available": bool(gnss_available),
                    }
                )
                emitted_s = time.perf_counter()
                latency_ms = max((emitted_s - tick_started_s) * 1000.0, 0.0)
                runtime_stats["state_payloads_enqueued"] = int(runtime_stats.get("state_payloads_enqueued", 0)) + 1
                _enqueue_state(
                    state_queue,
                    {
                        "corr": replace(last_corr, best_reference_profile=last_ref),
                        "fix": fix,
                        "h_meas": h_meas,
                        "ref": last_ref,
                        "dem_patch": dem_patch,
                        "dem_patch_transform": tuple(float(value) for value in dem_patch_transform[:6]),
                        "route_history": list(route_history),
                        "hdop": hdop_m,
                        "nav_mode": nav_mode,
                        "used_prediction_only": False,
                        "degraded": False,
                        "selected_window_size": len(buffer),
                        "gnss_available": gnss_available,
                        "truth": {
                            "lat": gt.lat,
                            "lon": gt.lon,
                            "heading_deg": azimuth_deg,
                            "speed_mps": speed_mps,
                        },
                        "observability": {
                            "crlb_m": observability.crlb_m,
                            "gradient_energy": observability.gradient_energy,
                            "efficiency_hint": observability.efficiency_hint,
                            "is_informative": observability.is_informative,
                        },
                        "terrain_bias_m": 0.0,
                        "event_ingest_monotonic_s": float(tick_started_s),
                        "pipeline_emitted_monotonic_s": float(emitted_s),
                        "pipeline_latency_ms": float(latency_ms),
                        "queue_latency_ms": 0.0,
                        "measurement_step_m": float(measurement_step_m),
                        "integrity_status": "OK",
                        "runtime_stats": runtime_stats.copy(),
                    },
                    stop_event,
                )
                sample_counter += 1

            while not stop_event.is_set() and not restart_requested:
                gnss_override, restart_requested = _drain_demo_control_queue(control_queue, gnss_override)
                time.sleep(0.05)
            if not restart_requested and not stop_event.is_set():
                emit_idle_state("Маршрут завершён. Для нового прогона нажмите СТАРТ / ЗАНОВО.")

    return history


def run_pipeline(config: Config) -> tuple[list[tuple[int, IMMResult]], ReplayMetrics | None]:
    """Run the configured pipeline end-to-end."""

    frame_queue: queue.Queue = queue.Queue(maxsize=1000)
    state_queue: queue.Queue = queue.Queue(maxsize=1)
    control_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    producer_done_event = threading.Event()
    pipeline_done_event = threading.Event()

    ground_truth: list[GroundTruthPoint] | None = None
    if config.mode == "sim":
        ground_truth = build_sim_ground_truth(make_simulation_points(config))
    elif config.mode == "replay" and config.gt_path is not None:
        ground_truth = load_ground_truth_csv(config.gt_path, fallback_dt_s=1.0 / max(config.freq_hz, 1e-6))

    if config.demo_dashboard:
        if not config.enable_visualizer:
            raise ValueError("--demo-dashboard requires dashboard enabled")
        if ground_truth is None:
            raise ValueError("--demo-dashboard requires --gt in replay mode")
        dashboard = TerrainNavigatorDash(state_queue=state_queue, control_queue=control_queue)
        dashboard_thread = threading.Thread(
            target=lambda: dashboard.run(host=config.dashboard_host, port=config.dashboard_port, debug=False),
            name="visualizer",
            daemon=True,
        )
        dashboard_url = f"http://{config.dashboard_host}:{config.dashboard_port}"

        def _signal_handler(signum: int, frame: Any) -> None:
            del signum, frame
            stop_event.set()

        previous_handler = signal.signal(signal.SIGINT, _signal_handler)
        pipeline_history: list[tuple[int, IMMResult]] = []
        pipeline_metrics: ReplayMetrics | None = None
        try:
            dashboard_thread.start()
            print(f"Dashboard: {dashboard_url}", flush=True)
            if config.open_browser:
                threading.Timer(1.0, webbrowser.open, args=(dashboard_url,)).start()
            pipeline_history = demo_dashboard_worker(
                config=config,
                state_queue=state_queue,
                stop_event=stop_event,
                ground_truth=ground_truth,
                control_queue=control_queue,
            )
            while not stop_event.is_set():
                time.sleep(0.25)
        finally:
            stop_event.set()
            signal.signal(signal.SIGINT, previous_handler)
        if pipeline_history:
            _safe_export_flight_report([item for _, item in pipeline_history], str(Path("output") / "terrain_navigator_report.html"))
        return pipeline_history, pipeline_metrics

    pipeline_history: list[tuple[int, IMMResult]] = []
    pipeline_metrics: ReplayMetrics | None = None

    def producer_target() -> None:
        try:
            if config.mode == "sim":
                simulation_producer(config, frame_queue, stop_event, control_queue=control_queue)
            elif config.mode == "replay":
                replay_producer(
                    config,
                    frame_queue,
                    stop_event,
                    control_queue=control_queue,
                    ground_truth=ground_truth,
                )
            elif config.mode == "sitl":
                sitl_producer(config, frame_queue, stop_event, control_queue=control_queue)
            else:
                live_producer(config, frame_queue, stop_event, control_queue=control_queue)
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
        )
        pipeline_done_event.set()

    producer_thread = threading.Thread(target=producer_target, name="producer", daemon=True)
    pipeline_thread = threading.Thread(target=pipeline_target, name="pipeline", daemon=True)
    threads = [producer_thread, pipeline_thread]

    dashboard_history: list[IMMResult] = []
    dashboard_thread: threading.Thread | None = None
    if config.enable_visualizer:
        dashboard = TerrainNavigatorDash(state_queue=state_queue, control_queue=control_queue)

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
            dashboard_url = f"http://{config.dashboard_host}:{config.dashboard_port}"
            print(f"Dashboard: {dashboard_url}", flush=True)
            if config.open_browser:
                threading.Timer(1.0, webbrowser.open, args=(dashboard_url,)).start()

        producer_thread.join()
        pipeline_thread.join()
        if (
            dashboard_thread is not None
            and config.enable_visualizer
            and config.mode in {"sim", "replay"}
            and not stop_event.is_set()
        ):
            LOGGER.info(
                "Pipeline finished; keeping dashboard available at http://%s:%d until Ctrl+C",
                config.dashboard_host,
                config.dashboard_port,
            )
            while not stop_event.is_set():
                time.sleep(0.25)
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, previous_handler)

    if config.mode in {"sim", "replay"} and ground_truth is not None:
        pipeline_metrics = compute_replay_metrics(pipeline_history, ground_truth)
        LOGGER.info(
            "Replay metrics | mean=%.2f m | max=%.2f m | rmse=%.2f m | speed_err=%.2f m/s | azimuth_err=%.2f deg",
            pipeline_metrics.mean_error_m,
            pipeline_metrics.max_error_m,
            pipeline_metrics.rmse_m,
            pipeline_metrics.speed_error_mps,
            pipeline_metrics.azimuth_error_deg,
        )

    dashboard_history = [item for _, item in pipeline_history]
    if dashboard_history:
        _safe_export_flight_report(dashboard_history, str(Path("output") / "terrain_navigator_report.html"))
    return pipeline_history, pipeline_metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    config = parse_args(argv)
    configure_logging(config.log_level, Path("terrain_navigator.log"), quiet_console=config.quiet_console)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
