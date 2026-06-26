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
    udp_host: str
    udp_port: int
    random_seed: int
    altitude_msl_m: float
    speed_mps: float
    azimuth_deg: float
    duration_s: float | None
    length_km: float | None
    trajectory_id: int | None


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
        f"{radar_alt_m:.1f}",
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


def iter_points(
    dataset: rasterio.io.DatasetReader,
    data: np.ndarray,
    segments: Sequence[Segment],
    start_lat: float,
    start_lon: float,
    frequency_hz: float,
    noise_sigma_m: float,
    rng: np.random.Generator,
) -> Iterator[TrajectoryPoint]:
    """Yield simulated route samples for the given segments."""

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    step_time_s = 1.0 / frequency_hz
    current_lat = start_lat
    current_lon = start_lon
    current_time_s = 0.0
    index = 0

    for segment in segments:
        nominal_step_distance_m = segment.speed_mps * step_time_s
        num_steps = max(1, int(round(segment.distance_m / nominal_step_distance_m)))
        step_distance_m = segment.distance_m / num_steps

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
            radar_alt = alt_msl - terrain_h + float(rng.normal(0.0, noise_sigma_m))

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
        return list(
            iter_points(
                dataset=dataset,
                data=data,
                segments=segments,
                start_lat=config.start_lat,
                start_lon=config.start_lon,
                frequency_hz=config.frequency_hz,
                noise_sigma_m=config.noise_sigma_m,
                rng=rng,
            )
        )


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
    parser.add_argument("--altitude-msl", type=float, default=1500.0, help="Absolute altitude MSL in meters")
    parser.add_argument("--duration-s", type=float, help="Custom route duration in seconds")
    parser.add_argument("--length-km", type=float, help="Custom route length in kilometers")
    parser.add_argument("--freq", type=float, default=5.0, help="NMEA output frequency in Hz")
    parser.add_argument("--noise", type=float, default=2.0, help="Radar altimeter Gaussian noise sigma in meters")
    parser.add_argument("--output", choices=("file", "udp"), default="file", help="Output mode")
    parser.add_argument("--out-nmea", type=Path, help="Path to the .nmea output file")
    parser.add_argument("--out-csv", type=Path, help="Path to the ground-truth CSV output")
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
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        random_seed=args.seed,
        altitude_msl_m=args.altitude_msl,
        speed_mps=args.speed,
        azimuth_deg=args.azimuth,
        duration_s=args.duration_s,
        length_km=args.length_km,
        trajectory_id=args.trajectory,
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
