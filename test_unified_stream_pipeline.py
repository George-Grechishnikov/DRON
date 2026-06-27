from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin

from main import Config, run_pipeline


def _write_demo_dem(path: Path) -> np.ndarray:
    rows, cols = 1400, 1400
    y_grid, x_grid = np.indices((rows, cols), dtype=float)
    terrain = (
        250.0
        + 35.0 * np.sin(x_grid / 19.0)
        + 24.0 * np.cos(y_grid / 21.0)
        + 10.0 * np.sin((x_grid + y_grid) / 29.0)
        + 0.03 * x_grid
        + 0.04 * y_grid
    ).astype("float32")
    transform = from_origin(89.8, 61.0, 0.0005, 0.0005)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(terrain, 1)
    return terrain


def _write_samples_jsonl(dem_path: Path, terrain: np.ndarray, samples_path: Path) -> None:
    geod = Geod(ellps="WGS84")
    lat = 60.5
    lon = 90.3
    with rasterio.open(dem_path) as dataset, samples_path.open("w", encoding="utf-8") as handle:
        for index in range(70):
            if index > 0:
                lon, lat, _ = geod.fwd(lon, lat, 45.0, 10.0)
            row, col = dataset.index(lon, lat)
            terrain_h = float(terrain[row, col])
            timestamp_s = index / 5.0
            sample = {
                "timestamp_s": timestamp_s,
                "lat": lat,
                "lon": lon,
                "alt_msl": 1500.0,
                "radar_alt_m": 1500.0 - terrain_h,
                "terrain_h": terrain_h,
                "heading_deg": 45.0,
                "speed_mps": 50.0,
                "gnss_available": timestamp_s < 6.0,
                "nav_mode": "GNSS" if timestamp_s < 6.0 else "TERRAIN_NAV",
                "truth_lat": lat,
                "truth_lon": lon,
            }
            handle.write(json.dumps(sample) + "\n")


def test_unified_stream_pipeline_creates_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dem_path = tmp_path / "unified_stream_dem.tif"
    samples_path = tmp_path / "samples.jsonl"
    report_path = tmp_path / "output" / "unified_stream_report.html"

    terrain = _write_demo_dem(dem_path)
    _write_samples_jsonl(dem_path, terrain, samples_path)

    config = Config(
        mode="samples",
        dem_path=dem_path,
        start_lat=60.5,
        start_lon=90.3,
        trajectory=1,
        config_path=None,
        nmea_path=None,
        samples_path=samples_path,
        gt_path=None,
        barometer_path=None,
        udp_host="127.0.0.1",
        udp_port=10110,
        dashboard_host="127.0.0.1",
        dashboard_port=8050,
        enable_visualizer=False,
        seed=42,
        speed_mps=50.0,
        altitude_msl_m=1500.0,
        noise_sigma=0.0,
        window_size=20,
        step_size=10,
        freq_hz=5.0,
        dem_patch_radius_m=1000.0,
        max_offset_m=300.0,
        flat_terrain_threshold_m=15.0,
        cold_start_windows=2,
        gnss_drop_after_s=None,
        report_path=report_path,
        log_level="WARNING",
    )

    history, metrics = run_pipeline(config)

    assert len(history) >= 3
    assert metrics is None
    assert report_path.exists()
    assert "Demo Report" in report_path.read_text(encoding="utf-8")
