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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pyproj

from correlator import Correlator, CorrelationResult, ObservabilityMetrics, compute_observability_metrics
from constants import FIXED_BARO_ALTITUDE_M
from dem_loader import DEMLoader
from imm_filter import IMMFilter, IMMResult
from nmea_parser import NMEAFrame, NMEAReader, parse_line
from position_solver import PositionEstimate, PositionSolver
from profile_extractor import ProfileExtractor, is_flat_terrain
from sim_generator import SimulationConfig, TrajectoryPoint, format_gpgga, generate_points
from sitl_bridge import SITLBridge
from visualizer import TerrainNavigatorDash, export_flight_report


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the full TERRAIN NAVIGATOR pipeline."""

    mode: str
    dem_path: Path
    start_lat: float
    start_lon: float
    trajectory: int
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
    log_level: str = "INFO"


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


@dataclass(frozen=True)
class WindowSelection:
    """Selected processing window and its diagnostics."""

    frame_packets: list[FramePacket]
    window_size: int
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
    parser.add_argument("--trajectory", type=int, default=1, choices=(1, 2, 3), help="Simulation trajectory id")
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
    parser.add_argument("--log-level", default="INFO", help="Logging level")
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
        altitude_msl_m=FIXED_BARO_ALTITUDE_M,
        noise_sigma=args.noise,
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
    )


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
        for row in reader:
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
            speed_mps = 0.0
            azimuth_deg = 0.0
        else:
            prev = rows[idx - 1]
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


def simulation_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> list[GroundTruthPoint]:
    """Produce NMEA frames from the simulation generator."""

    points = make_simulation_points(config)
    for point in points:
        if stop_event.is_set():
            break
        frame = parse_line(format_gpgga(point.timestamp_s, point.radar_alt_measured))
        if frame is None:
            continue
        enqueue_frame(
            frame_queue,
            FramePacket(
                index=point.index,
                frame=frame,
                gnss_available=True,
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
) -> None:
    """Produce frames from a recorded NMEA log."""

    assert config.nmea_path is not None
    reader = NMEAReader.from_file(config.nmea_path)
    try:
        valid_index = 0
        for frame in reader:
            if stop_event.is_set():
                break
            enqueue_frame(frame_queue, FramePacket(index=valid_index, frame=frame, gnss_available=True), stop_event)
            if frame.valid:
                valid_index += 1
    finally:
        reader.close()


def live_producer(
    config: Config,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Produce frames from a live UDP stream."""

    reader = NMEAReader.from_udp(config.udp_host, config.udp_port)
    valid_index = 0
    try:
        while not stop_event.is_set():
            got_frame = False
            for frame in reader:
                got_frame = True
                enqueue_frame(frame_queue, FramePacket(index=valid_index, frame=frame, gnss_available=True), stop_event)
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
    window_counter: int,
    flat: bool,
    observability: ObservabilityMetrics,
) -> NavigationDecision:
    """Select between accepted terrain fix and predictive fallback."""

    if window_counter < config.cold_start_windows:
        return NavigationDecision(
            fix=predict_fix(
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                speed_mps=current_speed,
                azimuth_deg=current_azimuth,
                dt=window_duration,
                confidence=corr_result.confidence,
            ),
            mode="cold_start_prediction",
            used_prediction_only=True,
        )

    if flat:
        return NavigationDecision(
            fix=predict_fix(
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                speed_mps=current_speed,
                azimuth_deg=current_azimuth,
                dt=window_duration,
                confidence=corr_result.confidence,
            ),
            mode="terrain_flat_fallback",
            used_prediction_only=True,
        )

    if not observability.is_informative:
        return NavigationDecision(
            fix=predict_fix(
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                speed_mps=current_speed,
                azimuth_deg=current_azimuth,
                dt=window_duration,
                confidence=corr_result.confidence,
            ),
            mode="terrain_uninformative_fallback",
            used_prediction_only=True,
        )

    if corr_result.is_ambiguous:
        return NavigationDecision(
            fix=predict_fix(
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                speed_mps=current_speed,
                azimuth_deg=current_azimuth,
                dt=window_duration,
                confidence=corr_result.confidence,
            ),
            mode="terrain_ambiguous_fallback",
            used_prediction_only=True,
        )

    if not corr_result.is_reliable:
        return NavigationDecision(
            fix=predict_fix(
                current_lat=window_start_lat,
                current_lon=window_start_lon,
                speed_mps=current_speed,
                azimuth_deg=current_azimuth,
                dt=window_duration,
                confidence=corr_result.confidence,
            ),
            mode="terrain_low_confidence_fallback",
            used_prediction_only=True,
        )

    return NavigationDecision(
        fix=solver.solve(
            result=corr_result,
            start_lat=window_start_lat,
            start_lon=window_start_lon,
            window_duration_s=window_duration,
        ),
        mode="terrain_update_accepted",
        used_prediction_only=False,
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


def select_processing_window(
    *,
    config: Config,
    buffer: deque[FramePacket],
    correlator: Correlator,
    ref_matrix: np.ndarray,
    measurement_step_m: float,
) -> WindowSelection | None:
    """Select a fixed or adaptive processing window based on observability and ambiguity."""

    candidate_sizes = build_window_sizes(config, len(buffer))
    if not candidate_sizes:
        return None

    evaluations: list[WindowSelection] = []
    azimuth_axis = np.arange(0.0, ref_matrix.shape[0], 1.0)
    for candidate_size in candidate_sizes:
        frame_packets = list(buffer)[-candidate_size:]
        h_meas = np.array(
            [config.altitude_msl_m - item.frame.radar_alt_m for item in frame_packets],
            dtype=float,
        )
        flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
        corr_result = correlator.compute(
            h_meas=h_meas,
            ref_matrix=ref_matrix,
            azimuths_deg=azimuth_axis,
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
            and observability.is_informative
            and corr_result.is_reliable
            and not corr_result.is_ambiguous
            and not flat
        ):
            return evaluations[-1]

    def _score(item: WindowSelection) -> tuple[float, float, float]:
        informative_bonus = 1.0 if item.observability.is_informative else 0.0
        reliability_bonus = 1.0 if item.corr_result.is_reliable else 0.0
        ambiguity_penalty = 1.0 if item.corr_result.is_ambiguous else 0.0
        flat_penalty = 1.0 if item.flat else 0.0
        primary = informative_bonus * 3.0 + reliability_bonus * 2.0 - ambiguity_penalty - flat_penalty
        secondary = item.corr_result.peak_correlation + item.corr_result.confidence + item.observability.efficiency_hint
        tertiary = -float(item.window_size)
        return (primary, secondary, tertiary)

    return max(evaluations, key=_score)


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
    current_azimuth = 45.0
    current_speed = config.speed_mps
    step_dt = config.step_size / config.freq_hz
    default_window_duration = config.window_size / config.freq_hz
    window_counter = 0
    measurement_step_m = config.speed_mps / config.freq_hz
    measured_profile_length_m = config.max_window_size * measurement_step_m if config.adaptive_window else config.window_size * measurement_step_m
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
            minimum_frames = config.min_window_size if config.adaptive_window else config.window_size
            if len(buffer) < minimum_frames:
                continue

            ref_matrix = extractor.build_reference_matrix(window_start_lat, window_start_lon)
            selection = select_processing_window(
                config=config,
                buffer=buffer,
                correlator=correlator,
                ref_matrix=ref_matrix,
                measurement_step_m=measurement_step_m,
            )
            if selection is None:
                continue

            frames_window = [item.frame for item in selection.frame_packets]
            center_frame_index = selection.frame_packets[-1].index
            latest_packet = selection.frame_packets[-1]
            h_meas = np.array(
                [config.altitude_msl_m - frame.radar_alt_m for frame in frames_window],
                dtype=float,
            )
            flat = selection.flat
            corr_result = selection.corr_result
            observability = selection.observability
            window_duration = selection.window_size / config.freq_hz

            nav_decision = choose_navigation_fix(
                config=config,
                corr_result=corr_result,
                solver=solver,
                window_start_lat=window_start_lat,
                window_start_lon=window_start_lon,
                current_speed=current_speed,
                current_azimuth=current_azimuth,
                window_duration=window_duration,
                window_counter=window_counter,
                flat=flat,
                observability=observability,
            )
            position_fix = nav_decision.fix

            imm_result = imm.update(position_fix, dt=step_dt, is_flat=flat)
            current_azimuth = imm_result.azimuth_deg
            current_speed = imm_result.speed_mps
            dem_patch, _ = dem.get_patch(imm_result.lat, imm_result.lon, radius_m=config.dem_patch_radius_m)

            history.append((center_frame_index, imm_result))
            _enqueue_state(
                state_queue,
                {
                    "corr": corr_result,
                    "fix": imm_result,
                    "h_meas": h_meas,
                    "ref": corr_result.best_reference_profile,
                    "dem_patch": dem_patch,
                    "hdop": imm.get_hdop(),
                    "nav_mode": nav_decision.mode,
                    "used_prediction_only": nav_decision.used_prediction_only,
                    "selected_window_size": selection.window_size,
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
                },
                stop_event,
            )

            for _ in range(min(config.step_size, len(buffer))):
                if buffer:
                    buffer.popleft()
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
    control_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    producer_done_event = threading.Event()
    pipeline_done_event = threading.Event()

    ground_truth: list[GroundTruthPoint] | None = None
    if config.mode == "sim":
        ground_truth = build_sim_ground_truth(make_simulation_points(config))
    elif config.mode == "replay" and config.gt_path is not None:
        ground_truth = load_ground_truth_csv(config.gt_path)

    pipeline_history: list[tuple[int, IMMResult]] = []
    pipeline_metrics: ReplayMetrics | None = None

    def producer_target() -> None:
        try:
            if config.mode == "sim":
                simulation_producer(config, frame_queue, stop_event)
            elif config.mode == "replay":
                replay_producer(config, frame_queue, stop_event)
            elif config.mode == "sitl":
                sitl_producer(config, frame_queue, stop_event, control_queue=control_queue)
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

        producer_thread.join()
        pipeline_thread.join()
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, previous_handler)

    if config.mode in {"sim", "replay"} and ground_truth is not None:
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
    if dashboard_history:
        export_flight_report(dashboard_history, str(Path("output") / "terrain_navigator_report.html"))
    return pipeline_history, pipeline_metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    config = parse_args(argv)
    configure_logging(config.log_level, Path("terrain_navigator.log"))
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
