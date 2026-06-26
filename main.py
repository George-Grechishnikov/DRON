"""Main orchestrator for TERRAIN NAVIGATOR."""

from __future__ import annotations

import argparse
import csv
import logging
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

from correlator import Correlator, CorrelationResult
from dem_loader import DEMLoader
from imm_filter import IMMFilter, IMMResult
from nmea_parser import NMEAFrame, NMEAReader, parse_line
from position_solver import PositionEstimate, PositionSolver
from profile_extractor import ProfileExtractor, is_flat_terrain
from sim_generator import SimulationConfig, TrajectoryPoint, format_gpgga, generate_points
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
    log_level: str = "INFO"


@dataclass(frozen=True)
class FramePacket:
    """Frame packet moving through producer/pipeline queues."""

    index: int
    frame: NMEAFrame


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
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sim", action="store_true", help="Simulation mode")
    mode_group.add_argument("--live", action="store_true", help="Live UDP mode")
    mode_group.add_argument("--replay", action="store_true", help="Replay NMEA log mode")

    parser.add_argument("--dem", required=True, type=Path, help="Path to DEM GeoTIFF")
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
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into a Config object."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.replay and args.nmea is None:
        parser.error("--nmea is required in replay mode")

    mode = "sim" if args.sim else "live" if args.live else "replay"
    return Config(
        mode=mode,
        dem_path=args.dem,
        start_lat=args.lat,
        start_lon=args.lon,
        trajectory=args.trajectory,
        nmea_path=args.nmea,
        gt_path=args.gt,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        enable_visualizer=not args.no_visualizer,
        seed=args.seed,
        speed_mps=args.speed,
        altitude_msl_m=args.altitude_msl,
        noise_sigma=args.noise,
        window_size=args.window_size,
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
        enqueue_frame(frame_queue, FramePacket(index=point.index, frame=frame), stop_event)
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
            enqueue_frame(frame_queue, FramePacket(index=valid_index, frame=frame), stop_event)
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
                enqueue_frame(frame_queue, FramePacket(index=valid_index, frame=frame), stop_event)
                if frame.valid:
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


def pipeline_worker(
    config: Config,
    frame_queue: queue.Queue,
    state_queue: queue.Queue,
    stop_event: threading.Event,
    pipeline_done_event: threading.Event,
    ground_truth: list[GroundTruthPoint] | None = None,
) -> list[tuple[int, IMMResult]]:
    """Run the main sliding-window pipeline."""

    history: list[tuple[int, IMMResult]] = []
    buffer: deque[FramePacket] = deque(maxlen=config.window_size)
    current_lat = config.start_lat
    current_lon = config.start_lon
    current_azimuth = 45.0
    current_speed = config.speed_mps
    step_dt = config.step_size / config.freq_hz
    window_duration = config.window_size / config.freq_hz
    window_counter = 0

    with DEMLoader(config.dem_path) as dem:
        extractor = ProfileExtractor(dem, profile_length_m=config.window_size * config.speed_mps / config.freq_hz)
        correlator = Correlator(
            profile_length_m=config.window_size * config.speed_mps / config.freq_hz,
            step_m=30.0,
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

            frames_window = [item.frame for item in buffer]
            center_frame_index = buffer[-1].index
            h_meas = np.array([frame.radar_alt_m for frame in frames_window], dtype=float)
            flat = is_flat_terrain(h_meas, threshold_m=config.flat_terrain_threshold_m)
            ref_matrix = extractor.build_reference_matrix(current_lat, current_lon)
            corr_result = correlator.sliding_window_compute(
                frames_buffer=frames_window,
                ref_matrix=ref_matrix,
                speed_mps=config.speed_mps,
                freq_hz=config.freq_hz,
                azimuths_deg=np.arange(0.0, ref_matrix.shape[0], 1.0),
            )

            if window_counter < config.cold_start_windows:
                position_fix = predict_fix(
                    current_lat=current_lat,
                    current_lon=current_lon,
                    speed_mps=current_speed,
                    azimuth_deg=current_azimuth,
                    dt=step_dt,
                    confidence=corr_result.confidence,
                )
            else:
                position_fix = solver.solve(
                    result=corr_result,
                    start_lat=current_lat,
                    start_lon=current_lon,
                    window_duration_s=window_duration,
                )

            imm_result = imm.update(position_fix, dt=step_dt, is_flat=flat)
            current_lat = imm_result.lat
            current_lon = imm_result.lon
            current_azimuth = imm_result.azimuth_deg
            current_speed = imm_result.speed_mps
            dem_patch, _ = dem.get_patch(current_lat, current_lon, radius_m=config.dem_patch_radius_m)

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
                },
                stop_event,
            )

            for _ in range(min(config.step_size, len(buffer))):
                if buffer:
                    buffer.popleft()
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
