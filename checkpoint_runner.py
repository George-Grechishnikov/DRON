"""Case input adapter for the Адриадна terrain-navigation pipeline.

The hackathon case input is a DEM GeoTIFF plus radar-altimeter messages in
NMEA-0183 GPGGA/GNGGA format. The adapter validates that stream, runs the replay
pipeline from the DEM center by default, and writes both local ENU-like
trajectory coordinates and global WGS84 coordinates.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import pyproj
import rasterio
from rasterio.enums import Resampling

from constants import FIXED_BARO_ALTITUDE_M
from main import Config, configure_logging, resolve_initial_coordinates, run_pipeline
from measurement_layer import parse_nmea_timestamp_to_seconds
from nmea_parser import NMEAReader
from sim_generator import format_gpgga


WGS84_GEOD = pyproj.Geod(ellps="WGS84")


def read_heights(path: Path) -> np.ndarray:
    """Read one numeric height per non-empty line."""

    values: list[float] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip().replace(",", ".")
            if not stripped:
                continue
            try:
                values.append(float(stripped))
            except ValueError as exc:
                raise ValueError(f"Invalid height at {path}:{line_no}: {line.rstrip()!r}") from exc
    if not values:
        raise ValueError(f"Height file is empty: {path}")
    return np.asarray(values, dtype=float)


def validate_nmea_file(path: Path) -> tuple[int, float | None]:
    """Validate a case NMEA file and return (valid_frame_count, inferred_freq_hz)."""

    reader = NMEAReader.from_file(path)
    try:
        frames = list(reader)
    finally:
        reader.close()
    valid_frames = [frame for frame in frames if frame.valid and np.isfinite(frame.radar_alt_m)]
    if not valid_frames:
        raise ValueError(f"NMEA file has no valid GPGGA/GNGGA radar-altimeter frames: {path}")

    timestamps = np.asarray(
        [parse_nmea_timestamp_to_seconds(frame.timestamp_utc) for frame in valid_frames],
        dtype=float,
    )
    timestamps = timestamps[np.isfinite(timestamps)]
    if timestamps.size < 2:
        return len(valid_frames), None

    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-6]
    if deltas.size == 0:
        return len(valid_frames), None
    inferred_freq_hz = 1.0 / float(np.median(deltas))
    freq_tolerance_hz = 1e-6
    if inferred_freq_hz < (1.0 - freq_tolerance_hz) or inferred_freq_hz > (10.0 + freq_tolerance_hz):
        raise ValueError(
            "NMEA message frequency must be within 1-10 Hz by case requirements; "
            f"inferred {inferred_freq_hz:.2f} Hz from timestamps"
        )
    return len(valid_frames), float(inferred_freq_hz)
def resolve_runtime_window_params(
    sample_count: int,
    requested_window_size: int,
    requested_step_size: int,
) -> tuple[int, int]:
    """Clamp replay window parameters to the actual uploaded track length."""

    if sample_count <= 1:
        raise ValueError("Need at least 2 height samples to build a terrain window")

    effective_window_size = max(2, min(int(requested_window_size), int(sample_count)))
    effective_step_size = max(1, min(int(requested_step_size), effective_window_size))
    return effective_window_size, effective_step_size


def write_nmea_from_heights(
    *,
    heights_m: np.ndarray,
    output_path: Path,
    freq_hz: float,
    input_kind: str,
    baro_alt_m: float,
) -> None:
    """Write a temporary GPGGA stream consumed by the existing replay mode."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="") as handle:
        for index, height_m in enumerate(heights_m):
            if input_kind == "radar":
                radar_alt_m = float(height_m)
            elif input_kind == "terrain":
                radar_alt_m = float(baro_alt_m - height_m)
            else:
                raise ValueError(f"Unsupported input_kind: {input_kind}")
            timestamp_s = index / max(freq_hz, 1e-9)
            handle.write(format_gpgga(timestamp_s, radar_alt_m))


def _to_wgs84(dataset: rasterio.io.DatasetReader, x: float, y: float) -> tuple[float, float]:
    if dataset.crs is None:
        raise ValueError("DEM has no CRS; cannot convert start point to WGS84")
    if dataset.crs.to_epsg() == 4326:
        return float(y), float(x)
    transformer = pyproj.Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(x), float(y))
    return float(lat), float(lon)


def _pixel_to_wgs84(dataset: rasterio.io.DatasetReader, col: float, row: float) -> tuple[float, float]:
    x, y = dataset.transform * (float(col) + 0.5, float(row) + 0.5)
    return _to_wgs84(dataset, x, y)


def _local_m_to_wgs84(dataset: rasterio.io.DatasetReader, east_m: float, north_m: float) -> tuple[float, float]:
    if dataset.crs is not None and dataset.crs.to_epsg() != 4326:
        x = float(dataset.bounds.left) + float(east_m)
        y = float(dataset.bounds.bottom) + float(north_m)
        return _to_wgs84(dataset, x, y)

    # Geographic DEM: treat local meters from south-west corner using geodesics.
    lon0 = float(dataset.bounds.left)
    lat0 = float(dataset.bounds.bottom)
    lon_east, lat_east, _ = WGS84_GEOD.fwd(lon0, lat0, 90.0, float(east_m))
    lon_final, lat_final, _ = WGS84_GEOD.fwd(lon_east, lat_east, 0.0, float(north_m))
    return float(lat_final), float(lon_final)


def resolve_start_latlon(dem_path: Path, start_x: float, start_y: float, xy_mode: str) -> tuple[float, float, str]:
    """Resolve checkpoint (x, y) to WGS84 latitude/longitude."""

    with rasterio.open(dem_path) as dataset:
        if xy_mode == "auto":
            if 0.0 <= start_x < dataset.width and 0.0 <= start_y < dataset.height:
                lat, lon = _pixel_to_wgs84(dataset, start_x, start_y)
                return lat, lon, "pixel"
            if (
                dataset.bounds.left <= start_x <= dataset.bounds.right
                and dataset.bounds.bottom <= start_y <= dataset.bounds.top
            ):
                lat, lon = _to_wgs84(dataset, start_x, start_y)
                return lat, lon, "crs"
            lat, lon = _local_m_to_wgs84(dataset, start_x, start_y)
            return lat, lon, "local-m"
        if xy_mode == "pixel":
            lat, lon = _pixel_to_wgs84(dataset, start_x, start_y)
            return lat, lon, "pixel"
        if xy_mode == "crs":
            lat, lon = _to_wgs84(dataset, start_x, start_y)
            return lat, lon, "crs"
        if xy_mode == "local-m":
            lat, lon = _local_m_to_wgs84(dataset, start_x, start_y)
            return lat, lon, "local-m"
        raise ValueError(f"Unsupported xy_mode: {xy_mode}")


def _local_offsets_from_start(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    azimuth_deg, _, distance_m = WGS84_GEOD.inv(lon0, lat0, lon, lat)
    azimuth_rad = math.radians(float(azimuth_deg))
    return float(distance_m * math.sin(azimuth_rad)), float(distance_m * math.cos(azimuth_rad))


def write_trajectory_csv(
    *,
    output_path: Path,
    history: list[tuple[int, object]],
    start_lat: float,
    start_lon: float,
    freq_hz: float,
) -> None:
    """Write estimated trajectory in local and global coordinates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "timestamp_s",
                "local_x_m",
                "local_y_m",
                "lat",
                "lon",
                "speed_mps",
                "azimuth_deg",
            ]
        )
        for frame_index, result in history:
            lat = float(getattr(result, "lat"))
            lon = float(getattr(result, "lon"))
            local_x_m, local_y_m = _local_offsets_from_start(start_lat, start_lon, lat, lon)
            writer.writerow(
                [
                    int(frame_index),
                    f"{int(frame_index) / max(freq_hz, 1e-9):.3f}",
                    f"{local_x_m:.3f}",
                    f"{local_y_m:.3f}",
                    f"{lat:.8f}",
                    f"{lon:.8f}",
                    f"{float(getattr(result, 'speed_mps')):.3f}",
                    f"{float(getattr(result, 'azimuth_deg')) % 360.0:.3f}",
                ]
            )


def write_trajectory_html(
    *,
    dem_path: Path,
    history: list[tuple[int, object]],
    output_path: Path,
    start_lat: float,
    start_lon: float,
) -> None:
    """Write a static HTML visualization with DEM and estimated trajectory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as dataset:
        scale = max(dataset.width / 900.0, dataset.height / 650.0, 1.0)
        out_width = max(int(dataset.width / scale), 2)
        out_height = max(int(dataset.height / scale), 2)
        dem = dataset.read(out_shape=(1, out_height, out_width), resampling=Resampling.bilinear)[0].astype(float)
        if dataset.nodata is not None:
            dem[np.isclose(dem, float(dataset.nodata))] = np.nan
        left, bottom, right, top = dataset.bounds
        if dataset.crs is not None and dataset.crs.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            lon_left, lat_bottom = transformer.transform(left, bottom)
            lon_right, lat_top = transformer.transform(right, top)
        else:
            lon_left, lat_bottom, lon_right, lat_top = left, bottom, right, top

    lats = [float(getattr(result, "lat")) for _, result in history]
    lons = [float(getattr(result, "lon")) for _, result in history]
    speeds = [float(getattr(result, "speed_mps")) for _, result in history]
    headings = [float(getattr(result, "azimuth_deg")) % 360.0 for _, result in history]
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=dem,
            x=np.linspace(float(lon_left), float(lon_right), dem.shape[1]),
            y=np.linspace(float(lat_top), float(lat_bottom), dem.shape[0]),
            colorscale="Earth",
            name="DEM",
            showscale=True,
            colorbar={"title": "Высота, м"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+lines",
            marker={"color": "#2d9cdb", "size": 5},
            line={"color": "#7fd3ff", "width": 2},
            name="Оцененная траектория",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[start_lon],
            y=[start_lat],
            mode="markers+text",
            marker={"color": "#45ff8a", "size": 13, "symbol": "circle-open", "line": {"width": 3}},
            text=["СТАРТ"],
            textposition="bottom right",
            name="Старт",
        )
    )
    if history:
        last = history[-1][1]
        figure.add_trace(
            go.Scatter(
                x=[float(getattr(last, "lon"))],
                y=[float(getattr(last, "lat"))],
                mode="markers+text",
                marker={"color": "#ffe66d", "size": 16, "line": {"color": "#101820", "width": 2}},
                text=["ИСКАТЬ ЗДЕСЬ"],
                textposition="top right",
                name="Где искать БПЛА",
            )
        )
    title_suffix = ""
    if speeds:
        title_suffix = f" | скорость {speeds[-1]:.1f} м/с | курс {headings[-1]:.1f}°"
    figure.update_layout(
        template="plotly_dark",
        title=f"Адриадна: траектория по проверочному набору{title_suffix}",
        xaxis_title="Долгота",
        yaxis_title="Широта",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Адриадна on the hackathon DEM + NMEA case format")
    parser.add_argument("--dem", required=True, type=Path, help="DEM GeoTIFF from experts")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--nmea", type=Path, help="NMEA-0183 GPGGA/GNGGA radar-altimeter log")
    input_group.add_argument("--heights", type=Path, help="Legacy text file: one height in meters per line")
    parser.add_argument("--start-x", type=float, help="Optional initial x coordinate; defaults to DEM center")
    parser.add_argument("--start-y", type=float, help="Optional initial y coordinate; defaults to DEM center")
    parser.add_argument("--xy-mode", choices=("auto", "pixel", "crs", "local-m"), default="auto", help="How to interpret start x/y")
    parser.add_argument("--heading", type=float, default=0.0, help="Initial heading/course hint, degrees clockwise from north")
    parser.add_argument("--speed", required=True, type=float, help="Initial aircraft speed, m/s")
    parser.add_argument("--freq", type=float, help="Radar-altimeter frequency, Hz; inferred from NMEA timestamps when omitted")
    parser.add_argument("--input-kind", choices=("radar", "terrain"), default="radar", help="heights file contains radar AGL heights or already reconstructed terrain heights")
    parser.add_argument("--baro-alt", type=float, default=FIXED_BARO_ALTITUDE_M, help="Constant barometric altitude MSL, meters")
    parser.add_argument("--window-size", type=int, default=64, help="Correlation window size in samples")
    parser.add_argument("--step-size", type=int, default=1, help="Trajectory output/update step in samples")
    parser.add_argument("--max-offset", type=float, default=0.0, help="Correlation offset search radius in meters")
    parser.add_argument("--dem-patch-radius", type=float, default=5000.0, help="Dashboard/DEM patch radius in meters")
    parser.add_argument("--out-dir", type=Path, default=Path("output") / "checkpoint", help="Output directory")
    parser.add_argument("--dashboard", action="store_true", help="Also run live Dash visualization")
    parser.add_argument("--open-browser", action="store_true", help="Open Dash in browser when --dashboard is used")
    parser.add_argument("--quiet-console", action="store_true", help="Reduce console logs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.freq is not None and args.freq <= 0:
        parser.error("--freq must be positive")
    if args.window_size <= 1:
        parser.error("--window-size must be greater than 1")
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    if (args.start_x is None) != (args.start_y is None):
        parser.error("--start-x and --start-y must be provided together")

    configure_logging("WARNING", Path("terrain_navigator.log"), quiet_console=bool(args.quiet_console))
    if args.start_x is None and args.start_y is None:
        start_lat, start_lon = resolve_initial_coordinates(args.dem, float("nan"), float("nan"))
        resolved_xy_mode = "dem-center"
    else:
        start_lat, start_lon, resolved_xy_mode = resolve_start_latlon(args.dem, float(args.start_x), float(args.start_y), args.xy_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nmea_path = args.out_dir / "case_input.nmea"
    csv_path = args.out_dir / "trajectory_estimated.csv"
    html_path = args.out_dir / "trajectory_visualization.html"
    if args.nmea is not None:
        valid_count, inferred_freq_hz = validate_nmea_file(args.nmea)
        nmea_path = args.nmea
        freq_hz = float(args.freq if args.freq is not None else (inferred_freq_hz if inferred_freq_hz is not None else 5.0))
        if not (1.0 <= freq_hz <= 10.0):
            parser.error("--freq must be within 1-10 Hz by case requirements")
        sample_count = valid_count
    else:
        if args.freq is None:
            parser.error("--freq is required when --heights is used")
        if not (1.0 <= float(args.freq) <= 10.0):
            parser.error("--freq must be within 1-10 Hz by case requirements")
        heights_m = read_heights(args.heights)
        sample_count = len(heights_m)
        write_nmea_from_heights(
            heights_m=heights_m,
            output_path=nmea_path,
            freq_hz=float(args.freq),
            input_kind=str(args.input_kind),
            baro_alt_m=float(args.baro_alt),
        )
        valid_count, _ = validate_nmea_file(nmea_path)
        freq_hz = float(args.freq)

    effective_window_size, effective_step_size = resolve_runtime_window_params(
        sample_count,
        int(args.window_size),
        int(args.step_size),
    )

    config = Config(
        mode="replay",
        dem_path=args.dem,
        start_lat=start_lat,
        start_lon=start_lon,
        trajectory=1,
        nmea_path=nmea_path,
        gt_path=None,
        sitl_connection="udp:127.0.0.1:14550",
        sitl_gnss_drop_after_s=None,
        sitl_gnss_recover_after_s=None,
        udp_host="127.0.0.1",
        udp_port=10110,
        dashboard_host="127.0.0.1",
        dashboard_port=8050,
        enable_visualizer=bool(args.dashboard),
        seed=42,
        speed_mps=float(args.speed),
        altitude_msl_m=float(args.baro_alt),
        noise_sigma=2.0,
        initial_heading_deg=float(args.heading % 360.0),
        window_size=effective_window_size,
        adaptive_window=False,
        min_window_size=effective_window_size,
        max_window_size=effective_window_size,
        window_growth_step=effective_step_size,
        step_size=effective_step_size,
        freq_hz=freq_hz,
        dem_patch_radius_m=float(args.dem_patch_radius),
        max_offset_m=float(args.max_offset),
        flat_terrain_threshold_m=15.0,
        cold_start_windows=1,
        log_level="WARNING",
        quiet_console=bool(args.quiet_console),
        open_browser=bool(args.open_browser),
        realtime_playback=bool(args.dashboard),
        playback_speed=1.0,
        live_dashboard_stream=True,
        demo_dashboard=False,
        engine="legacy",
    )
    history, _ = run_pipeline(config)
    write_trajectory_csv(
        output_path=csv_path,
        history=history,
        start_lat=start_lat,
        start_lon=start_lon,
        freq_hz=freq_hz,
    )
    write_trajectory_html(
        dem_path=args.dem,
        history=history,
        output_path=html_path,
        start_lat=start_lat,
        start_lon=start_lon,
    )
    print(f"Resolved start: lat={start_lat:.8f}, lon={start_lon:.8f}, xy_mode={resolved_xy_mode}")
    print(f"NMEA input: {nmea_path}")
    print(f"Valid NMEA frames: {valid_count}, freq_hz={freq_hz:.3f}")
    print(
        "Effective window params: "
        f"window_size={effective_window_size}, step_size={effective_step_size}, samples={sample_count}"
    )
    print(f"Trajectory CSV: {csv_path}")
    print(f"Visualization HTML: {html_path}")
    print(f"Estimated points: {len(history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
