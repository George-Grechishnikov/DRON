"""Synthetic NMEA generator for TERRAIN NAVIGATOR.

Example:
    python sim_generator.py \
        --dem data/dem.tif \
        --lat 60.5 \
        --lon 90.3 \
        --trajectory 1 \
        --noise 2.0 \
        --freq 5 \
        --output file \
        --out-nmea output/traj1.nmea \
        --out-csv output/traj1_ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import rasterio
from constants import FIXED_BARO_ALTITUDE_M
from pyproj import Geod
from rasterio.transform import Affine


LOGGER = logging.getLogger(__name__)
WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class TrajectoryPoint:
    """One simulated sample along the route."""

    index: int
    timestamp_s: float
    lat: float
    lon: float
    alt_msl: float
    terrain_h: float
    radar_alt_measured: float


@dataclass(frozen=True)
class SyntheticImuPoint:
    """One synthetic IMU row aligned with the generated trajectory."""

    timestamp_s: float
    accel_x_mps2: float
    accel_y_mps2: float
    accel_z_mps2: float
    gyro_x_rps: float
    gyro_y_rps: float
    gyro_z_rps: float


@dataclass(frozen=True)
class Segment:
    """Route segment definition."""

    azimuth_deg: float
    distance_m: float
    speed_mps: float
    altitude_start_m: float
    altitude_end_m: float


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation parameters."""

    dem_path: Path
    start_lat: float
    start_lon: float
    frequency_hz: float
    noise_sigma_m: float
    output_mode: str
    out_nmea: Path | None
    out_csv: Path | None
    out_imu: Path | None
    udp_host: str
    udp_port: int
    random_seed: int
    altitude_msl_m: float
    speed_mps: float
    azimuth_deg: float
    duration_s: float | None
    length_km: float | None
    trajectory_id: int | None
    beam_width_deg: float = 0.0
    beam_aggregation: str = "mean"
    spike_prob: float = 0.0
    spike_sigma_m: float = 50.0
    dropout_prob: float = 0.0
    water_multipath_extra_sigma_m: float = 0.0
    realistic: bool = False
    surface_bias_mode: str = "none"
    surface_bias_m: float = 0.0
    surface_mask: Path | None = None


def nmea_checksum(sentence: str) -> str:
    """Return the XOR checksum for an NMEA payload without '$' and '*'."""

    checksum = 0
    for char in sentence:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def format_gpgga(timestamp_s: float, radar_alt_m: float) -> str:
    """Build a minimal valid GPGGA sentence with the measured radar altitude."""

    whole_seconds = int(timestamp_s)
    fractional = timestamp_s - whole_seconds
    hours = (whole_seconds // 3600) % 24
    minutes = (whole_seconds % 3600) // 60
    seconds = (whole_seconds % 60) + fractional
    utc_token = f"{hours:02d}{minutes:02d}{seconds:06.3f}"
    fields = [
        "GPGGA",
        utc_token,
        "",
        "",
        "",
        "",
        "1",
        "08",
        "1.0",
        "" if not np.isfinite(radar_alt_m) else f"{radar_alt_m:.1f}",
        "M",
        "0.0",
        "M",
        "",
        "",
    ]
    payload = ",".join(fields)
    return f"${payload}*{nmea_checksum(payload)}\r\n"


def _fractional_index(transform: Affine, lon: float, lat: float) -> tuple[float, float]:
    """Convert geographic coordinates to fractional raster row/column indices."""

    col, row = (~transform) * (lon, lat)
    return float(row), float(col)


def sample_dem_bilinear(
    dataset: rasterio.io.DatasetReader, data: np.ndarray, lat: float, lon: float
) -> float:
    """Sample a DEM with bilinear interpolation."""

    if not (
        dataset.bounds.left <= lon <= dataset.bounds.right
        and dataset.bounds.bottom <= lat <= dataset.bounds.top
    ):
        raise ValueError(
            f"Coordinate ({lat:.6f}, {lon:.6f}) is outside DEM bounds "
            f"{dataset.bounds!s}"
        )

    row, col = _fractional_index(dataset.transform, lon, lat)
    if row < 0 or col < 0 or row >= dataset.height - 1 or col >= dataset.width - 1:
        raise ValueError(
            f"Coordinate ({lat:.6f}, {lon:.6f}) is too close to the DEM edge "
            "for bilinear interpolation"
        )

    row0 = int(math.floor(row))
    col0 = int(math.floor(col))
    row1 = row0 + 1
    col1 = col0 + 1
    dy = row - row0
    dx = col - col0

    q11 = float(data[row0, col0])
    q21 = float(data[row0, col1])
    q12 = float(data[row1, col0])
    q22 = float(data[row1, col1])

    if dataset.nodata is not None and any(
        math.isclose(value, float(dataset.nodata)) for value in (q11, q21, q12, q22)
    ):
        raise ValueError(
            f"Coordinate ({lat:.6f}, {lon:.6f}) intersects a nodata region in the DEM"
        )

    return (
        q11 * (1.0 - dx) * (1.0 - dy)
        + q21 * dx * (1.0 - dy)
        + q12 * (1.0 - dx) * dy
        + q22 * dx * dy
    )


def build_segments(config: SimulationConfig) -> list[Segment]:
    """Create route segments for either a preset trajectory or a custom route."""

    if config.trajectory_id is not None:
        if config.trajectory_id == 1:
            return [
                Segment(
                    azimuth_deg=45.0,
                    distance_m=10_000.0,
                    speed_mps=50.0,
                    altitude_start_m=config.altitude_msl_m,
                    altitude_end_m=config.altitude_msl_m,
                )
            ]
        if config.trajectory_id == 2:
            return [
                Segment(
                    azimuth_deg=0.0,
                    distance_m=5_000.0,
                    speed_mps=50.0,
                    altitude_start_m=config.altitude_msl_m,
                    altitude_end_m=config.altitude_msl_m,
                ),
                Segment(
                    azimuth_deg=45.0,
                    distance_m=5_000.0,
                    speed_mps=50.0,
                    altitude_start_m=config.altitude_msl_m,
                    altitude_end_m=config.altitude_msl_m,
                ),
            ]
        if config.trajectory_id == 3:
            return [
                Segment(
                    azimuth_deg=45.0,
                    distance_m=5_000.0,
                    speed_mps=50.0,
                    altitude_start_m=config.altitude_msl_m,
                    altitude_end_m=config.altitude_msl_m + 500.0,
                )
            ]
        raise ValueError(f"Unsupported trajectory id: {config.trajectory_id}")

    if config.duration_s is None and config.length_km is None:
        raise ValueError("Either --trajectory or one of --duration-s/--length-km is required")

    distance_m = (
        config.length_km * 1000.0
        if config.length_km is not None
        else config.speed_mps * float(config.duration_s)
    )
    return [
        Segment(
            azimuth_deg=config.azimuth_deg,
            distance_m=distance_m,
            speed_mps=config.speed_mps,
            altitude_start_m=config.altitude_msl_m,
            altitude_end_m=config.altitude_msl_m,
        )
    ]


def _sample_disc_terrain(
    dataset: rasterio.io.DatasetReader,
    data: np.ndarray,
    lat: float,
    lon: float,
    radius_m: float,
    aggregation: str,
) -> tuple[float, np.ndarray]:
    """Sample terrain over a small disc footprint and aggregate it."""

    if radius_m <= 1e-6:
        terrain = sample_dem_bilinear(dataset, data, lat, lon)
        return terrain, np.array([terrain], dtype=float)

    bearings = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=float)
    lons, lats, _ = WGS84_GEOD.fwd(
        np.full(bearings.shape, lon, dtype=float),
        np.full(bearings.shape, lat, dtype=float),
        bearings,
        np.full(bearings.shape, radius_m, dtype=float),
    )
    samples = [sample_dem_bilinear(dataset, data, lat, lon)]
    for point_lat, point_lon in zip(lats, lons):
        samples.append(sample_dem_bilinear(dataset, data, float(point_lat), float(point_lon)))
    values = np.asarray(samples, dtype=float)
    if aggregation == "min":
        return float(np.nanmin(values)), values
    return float(np.nanmean(values)), values


def _segment_bank_angle_deg(segments: Sequence[Segment], segment_index: int) -> float:
    """Estimate a simple bank angle on turning segments from heading change."""

    previous_azimuth = segments[segment_index - 1].azimuth_deg if segment_index > 0 else segments[segment_index].azimuth_deg
    current_azimuth = segments[segment_index].azimuth_deg
    delta = ((current_azimuth - previous_azimuth + 180.0) % 360.0) - 180.0
    return float(min(abs(delta) * 0.5, 25.0))


def _is_water_like(terrain_samples: np.ndarray) -> bool:
    """Heuristic water detector for synthetic DEM experiments."""

    values = np.asarray(terrain_samples, dtype=float)
    if values.size == 0:
        return False
    return bool(np.nanstd(values) < 0.05 and abs(float(np.nanmean(values))) < 1.0)


def _surface_bias_at(
    *,
    config: SimulationConfig,
    dataset: rasterio.io.DatasetReader,
    data: np.ndarray,
    lat: float,
    lon: float,
    terrain_samples: np.ndarray,
    mask_dataset: rasterio.io.DatasetReader | None,
    mask_data: np.ndarray | None,
) -> float:
    """Return a surface-related bias for canopy/snow-like conditions."""

    if config.surface_bias_mode == "none" or abs(config.surface_bias_m) <= 1e-9:
        return 0.0
    if config.surface_bias_mode in {"forest", "snow"}:
        return float(config.surface_bias_m)
    if config.surface_bias_mode == "mask":
        if mask_dataset is None or mask_data is None:
            return 0.0
        mask_value = sample_dem_bilinear(mask_dataset, mask_data, lat, lon)
        return float(config.surface_bias_m if mask_value > 0.5 else 0.0)
    raise ValueError(f"Unsupported surface_bias_mode: {config.surface_bias_mode}")


def apply_sensor_model(
    *,
    dataset: rasterio.io.DatasetReader,
    data: np.ndarray,
    lat: float,
    lon: float,
    alt_msl: float,
    terrain_h: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    bank_angle_deg: float,
    mask_dataset: rasterio.io.DatasetReader | None = None,
    mask_data: np.ndarray | None = None,
) -> float:
    """Apply optional realistic radar-altimeter effects on top of the true terrain."""

    true_agl = alt_msl - terrain_h
    if not config.realistic:
        return float(true_agl + rng.normal(0.0, config.noise_sigma_m))

    beam_radius_m = max(true_agl, 0.0) * math.tan(math.radians(config.beam_width_deg) * 0.5)
    beam_terrain_h, terrain_samples = _sample_disc_terrain(
        dataset=dataset,
        data=data,
        lat=lat,
        lon=lon,
        radius_m=beam_radius_m,
        aggregation=config.beam_aggregation,
    )
    surface_bias_m = _surface_bias_at(
        config=config,
        dataset=dataset,
        data=data,
        lat=lat,
        lon=lon,
        terrain_samples=terrain_samples,
        mask_dataset=mask_dataset,
        mask_data=mask_data,
    )
    measured_agl = alt_msl - (beam_terrain_h + surface_bias_m)
    cos_term = math.cos(math.radians(bank_angle_deg))
    if abs(cos_term) > 1e-6:
        measured_agl = measured_agl / cos_term

    noise_sigma = config.noise_sigma_m
    if _is_water_like(terrain_samples):
        noise_sigma += config.water_multipath_extra_sigma_m
    measured_agl += float(rng.normal(0.0, noise_sigma))

    if config.spike_prob > 0.0 and float(rng.random()) < config.spike_prob:
        measured_agl += float(rng.normal(0.0, config.spike_sigma_m))

    if config.dropout_prob > 0.0 and float(rng.random()) < config.dropout_prob:
        return float("nan")

    return float(measured_agl)


def iter_points(
    dataset: rasterio.io.DatasetReader,
    data: np.ndarray,
    segments: Sequence[Segment],
    start_lat: float,
    start_lon: float,
    frequency_hz: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    mask_dataset: rasterio.io.DatasetReader | None = None,
    mask_data: np.ndarray | None = None,
) -> Iterator[TrajectoryPoint]:
    """Yield simulated route samples for the given segments."""

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    step_time_s = 1.0 / frequency_hz
    current_lat = start_lat
    current_lon = start_lon
    current_time_s = 0.0
    index = 0

    for segment_index, segment in enumerate(segments):
        nominal_step_distance_m = segment.speed_mps * step_time_s
        num_steps = max(1, int(round(segment.distance_m / nominal_step_distance_m)))
        step_distance_m = segment.distance_m / num_steps
        bank_angle_deg = _segment_bank_angle_deg(segments, segment_index)

        for step_idx in range(num_steps):
            if index == 0 and step_idx == 0:
                lat = current_lat
                lon = current_lon
            else:
                lon, lat, _ = WGS84_GEOD.fwd(current_lon, current_lat, segment.azimuth_deg, step_distance_m)
                current_lat = lat
                current_lon = lon

            progress = step_idx / max(num_steps - 1, 1)
            alt_msl = (
                segment.altitude_start_m
                + (segment.altitude_end_m - segment.altitude_start_m) * progress
            )
            terrain_h = sample_dem_bilinear(dataset, data, lat, lon)
            radar_alt = apply_sensor_model(
                dataset=dataset,
                data=data,
                lat=lat,
                lon=lon,
                alt_msl=alt_msl,
                terrain_h=terrain_h,
                config=config,
                rng=rng,
                bank_angle_deg=bank_angle_deg,
                mask_dataset=mask_dataset,
                mask_data=mask_data,
            )

            yield TrajectoryPoint(
                index=index,
                timestamp_s=current_time_s,
                lat=lat,
                lon=lon,
                alt_msl=alt_msl,
                terrain_h=terrain_h,
                radar_alt_measured=radar_alt,
            )

            current_time_s += step_time_s
            index += 1


def generate_points(config: SimulationConfig) -> list[TrajectoryPoint]:
    """Generate all simulated points for a run."""

    LOGGER.info("Opening DEM: %s", config.dem_path)
    rng = np.random.default_rng(config.random_seed)
    with rasterio.open(config.dem_path) as dataset:
        data = dataset.read(1, out_dtype="float64").copy()
        if dataset.nodata is not None:
            data[np.isclose(data, float(dataset.nodata))] = np.nan
        segments = build_segments(config)
        LOGGER.info("Generating %d segment(s) at %.2f Hz", len(segments), config.frequency_hz)
        mask_dataset = None
        mask_data = None
        if config.surface_mask is not None:
            mask_dataset = rasterio.open(config.surface_mask)
            mask_data = mask_dataset.read(1, out_dtype="float64").copy()
            if mask_dataset.nodata is not None:
                mask_data[np.isclose(mask_data, float(mask_dataset.nodata))] = 0.0
        try:
            return list(
                iter_points(
                    dataset=dataset,
                    data=data,
                    segments=segments,
                    start_lat=config.start_lat,
                    start_lon=config.start_lon,
                    frequency_hz=config.frequency_hz,
                    config=config,
                    rng=rng,
                    mask_dataset=mask_dataset,
                    mask_data=mask_data,
                )
            )
        finally:
            if mask_dataset is not None:
                mask_dataset.close()


def write_nmea_file(points: Iterable[TrajectoryPoint], output_path: Path) -> None:
    """Write generated NMEA sentences to a file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        for point in points:
            handle.write(format_gpgga(point.timestamp_s, point.radar_alt_measured))
    LOGGER.info("Wrote NMEA output to %s", output_path)


def write_csv(points: Iterable[TrajectoryPoint], output_path: Path) -> None:
    """Write ground-truth CSV for validation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "timestamp_s",
                "lat",
                "lon",
                "alt_msl",
                "terrain_h",
                "radar_alt_measured",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.index,
                    f"{point.timestamp_s:.3f}",
                    f"{point.lat:.8f}",
                    f"{point.lon:.8f}",
                    f"{point.alt_msl:.3f}",
                    f"{point.terrain_h:.3f}",
                    f"{point.radar_alt_measured:.3f}",
                ]
            )
    LOGGER.info("Wrote ground-truth CSV to %s", output_path)


def generate_imu_points(points: Sequence[TrajectoryPoint]) -> list[SyntheticImuPoint]:
    """Generate a simple level-flight IMU stream from trajectory samples."""

    if not points:
        return []

    imu_points: list[SyntheticImuPoint] = []
    velocities = []
    headings = []
    for index, point in enumerate(points):
        if index == 0:
            velocities.append(np.zeros(2, dtype=float))
            headings.append(0.0)
            continue
        prev = points[index - 1]
        dt = max(point.timestamp_s - prev.timestamp_s, 1e-6)
        azimuth_deg, _, distance_m = WGS84_GEOD.inv(prev.lon, prev.lat, point.lon, point.lat)
        speed_mps = distance_m / dt
        azimuth_rad = math.radians(float(azimuth_deg) % 360.0)
        velocities.append(np.array([speed_mps * math.sin(azimuth_rad), speed_mps * math.cos(azimuth_rad)], dtype=float))
        headings.append(float(azimuth_rad))
    velocities[0] = velocities[1].copy() if len(velocities) > 1 else np.zeros(2, dtype=float)
    headings[0] = headings[1] if len(headings) > 1 else 0.0

    for index, point in enumerate(points):
        if index == 0:
            dt = max(points[1].timestamp_s - point.timestamp_s, 1e-6) if len(points) > 1 else 1.0
            velocity_prev = velocities[0]
            heading_prev = headings[0]
        else:
            dt = max(point.timestamp_s - points[index - 1].timestamp_s, 1e-6)
            velocity_prev = velocities[index - 1]
            heading_prev = headings[index - 1]
        velocity = velocities[index]
        accel_enu = np.array([(velocity[0] - velocity_prev[0]) / dt, (velocity[1] - velocity_prev[1]) / dt, 0.0], dtype=float)
        heading = headings[index]
        rotation = np.array(
            [
                [math.sin(heading), math.cos(heading), 0.0],
                [-math.cos(heading), math.sin(heading), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        specific_force_body = rotation @ (accel_enu - np.array([0.0, 0.0, -9.80665], dtype=float))
        yaw_rate = ((heading - heading_prev + math.pi) % (2.0 * math.pi)) - math.pi
        yaw_rate /= max(dt, 1e-6)
        imu_points.append(
            SyntheticImuPoint(
                timestamp_s=point.timestamp_s,
                accel_x_mps2=float(specific_force_body[0]),
                accel_y_mps2=float(specific_force_body[1]),
                accel_z_mps2=float(specific_force_body[2]),
                gyro_x_rps=0.0,
                gyro_y_rps=0.0,
                gyro_z_rps=float(yaw_rate),
            )
        )
    return imu_points


def write_imu_csv(points: Sequence[TrajectoryPoint], output_path: Path) -> None:
    """Write a synthetic IMU CSV synchronized to the generated trajectory."""

    imu_points = generate_imu_points(points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp_s",
                "accel_x_mps2",
                "accel_y_mps2",
                "accel_z_mps2",
                "gyro_x_rps",
                "gyro_y_rps",
                "gyro_z_rps",
            ]
        )
        for point in imu_points:
            writer.writerow(
                [
                    f"{point.timestamp_s:.3f}",
                    f"{point.accel_x_mps2:.6f}",
                    f"{point.accel_y_mps2:.6f}",
                    f"{point.accel_z_mps2:.6f}",
                    f"{point.gyro_x_rps:.6f}",
                    f"{point.gyro_y_rps:.6f}",
                    f"{point.gyro_z_rps:.6f}",
                ]
            )
    LOGGER.info("Wrote IMU CSV to %s", output_path)


def stream_udp(points: Iterable[TrajectoryPoint], host: str, port: int) -> None:
    """Send generated NMEA sentences to a UDP socket."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for point in points:
            sock.sendto(format_gpgga(point.timestamp_s, point.radar_alt_measured).encode("ascii"), (host, port))
    LOGGER.info("Sent NMEA stream to udp://%s:%d", host, port)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Generate synthetic NMEA radar-altimeter data")
    parser.add_argument("--dem", required=True, type=Path, help="Path to DEM GeoTIFF")
    parser.add_argument("--lat", required=True, type=float, help="Start latitude in decimal degrees")
    parser.add_argument("--lon", required=True, type=float, help="Start longitude in decimal degrees")
    parser.add_argument("--trajectory", type=int, choices=(1, 2, 3), help="Preset trajectory id")
    parser.add_argument("--azimuth", type=float, default=45.0, help="Custom route azimuth in degrees")
    parser.add_argument("--speed", type=float, default=50.0, help="Vehicle speed in m/s")
    parser.add_argument("--duration-s", type=float, help="Custom route duration in seconds")
    parser.add_argument("--length-km", type=float, help="Custom route length in kilometers")
    parser.add_argument("--freq", type=float, default=5.0, help="NMEA output frequency in Hz")
    parser.add_argument("--noise", type=float, default=2.0, help="Radar altimeter Gaussian noise sigma in meters")
    parser.add_argument("--realistic", action="store_true", help="Enable realistic radar-altimeter error model")
    parser.add_argument("--beam-width-deg", type=float, default=0.0, help="Beam width in degrees for footprint smearing")
    parser.add_argument("--beam-aggregation", choices=("mean", "min"), default="mean", help="Footprint aggregation mode")
    parser.add_argument("--spike-prob", type=float, default=0.0, help="Probability of gross outlier per sample")
    parser.add_argument("--spike-sigma", type=float, default=50.0, help="Sigma of gross outlier noise in meters")
    parser.add_argument("--dropout-prob", type=float, default=0.0, help="Probability of missing radar-altimeter sample")
    parser.add_argument("--water-multipath-extra-sigma", type=float, default=0.0, help="Extra sigma over water-like surfaces")
    parser.add_argument("--surface-bias-mode", choices=("none", "forest", "snow", "mask"), default="none", help="Surface bias mode for canopy/snow effects")
    parser.add_argument("--surface-bias-m", type=float, default=0.0, help="Surface bias in meters added above bare-earth DEM")
    parser.add_argument("--surface-mask", type=Path, help="Optional raster mask for --surface-bias-mode mask")
    parser.add_argument("--output", choices=("file", "udp"), default="file", help="Output mode")
    parser.add_argument("--out-nmea", type=Path, help="Path to the .nmea output file")
    parser.add_argument("--out-csv", type=Path, help="Path to the ground-truth CSV output")
    parser.add_argument("--out-imu", type=Path, help="Path to the synthetic IMU CSV output")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP host for --output udp")
    parser.add_argument("--udp-port", type=int, default=10110, help="UDP port for --output udp")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise reproducibility")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> SimulationConfig:
    """Parse CLI arguments into a typed configuration object."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.output == "file" and args.out_nmea is None:
        parser.error("--out-nmea is required when --output file")
    if args.out_csv is None:
        parser.error("--out-csv is required")
    if args.trajectory is None and args.duration_s is None and args.length_km is None:
        parser.error("Provide --trajectory or one of --duration-s/--length-km")

    return SimulationConfig(
        dem_path=args.dem,
        start_lat=args.lat,
        start_lon=args.lon,
        frequency_hz=args.freq,
        noise_sigma_m=args.noise,
        output_mode=args.output,
        out_nmea=args.out_nmea,
        out_csv=args.out_csv,
        out_imu=args.out_imu,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        random_seed=args.seed,
        altitude_msl_m=FIXED_BARO_ALTITUDE_M,
        speed_mps=args.speed,
        azimuth_deg=args.azimuth,
        duration_s=args.duration_s,
        length_km=args.length_km,
        trajectory_id=args.trajectory,
        beam_width_deg=args.beam_width_deg,
        beam_aggregation=args.beam_aggregation,
        spike_prob=args.spike_prob,
        spike_sigma_m=args.spike_sigma,
        dropout_prob=args.dropout_prob,
        water_multipath_extra_sigma_m=args.water_multipath_extra_sigma,
        realistic=bool(args.realistic),
        surface_bias_mode=args.surface_bias_mode,
        surface_bias_m=args.surface_bias_m,
        surface_mask=args.surface_mask,
    )


def configure_logging(level: str) -> None:
    """Configure module logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    config = parse_args(argv)
    configure_logging("INFO")
    points = generate_points(config)
    if config.out_csv is not None:
        write_csv(points, config.out_csv)
    if config.out_imu is not None:
        write_imu_csv(points, config.out_imu)
    if config.output_mode == "file":
        if config.out_nmea is None:
            raise ValueError("out_nmea must be provided in file mode")
        write_nmea_file(points, config.out_nmea)
    else:
        stream_udp(points, config.udp_host, config.udp_port)
    LOGGER.info("Generated %d points", len(points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
