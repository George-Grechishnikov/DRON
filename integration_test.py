from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.transform import from_origin
import numpy as np

from main import Config, run_pipeline


def _write_integration_dem(path: Path) -> None:
    rows, cols = 2200, 2200
    y, x = np.indices((rows, cols), dtype=float)
    terrain = (
        250.0
        + 40.0 * np.sin(x / 17.0)
        + 28.0 * np.cos(y / 23.0)
        + 12.0 * np.sin((x + y) / 31.0)
        + 0.03 * x
        + 0.05 * y
    ).astype("float32")
    transform = from_origin(89.8, 61.2, 0.0005, 0.0005)
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


def test_full_sim_pipeline_produces_reasonable_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dem_path = tmp_path / "integration_dem.tif"
    _write_integration_dem(dem_path)

    config = Config(
        mode="sim",
        dem_path=dem_path,
        start_lat=60.5,
        start_lon=90.3,
        trajectory=1,
        nmea_path=None,
        gt_path=None,
        sitl_connection="udp:127.0.0.1:14550",
        sitl_gnss_drop_after_s=None,
        sitl_gnss_recover_after_s=None,
        udp_host="127.0.0.1",
        udp_port=10110,
        dashboard_host="127.0.0.1",
        dashboard_port=8050,
        enable_visualizer=False,
        seed=42,
        speed_mps=50.0,
        altitude_msl_m=1500.0,
        noise_sigma=0.8,
        window_size=50,
        adaptive_window=False,
        min_window_size=50,
        max_window_size=50,
        window_growth_step=10,
        step_size=10,
        freq_hz=5.0,
        dem_patch_radius_m=5000.0,
        max_offset_m=2000.0,
        flat_terrain_threshold_m=15.0,
        cold_start_windows=3,
        log_level="INFO",
    )

    history, metrics = run_pipeline(config)

    assert len(history) >= 10
    assert metrics is not None
    assert np.isfinite(metrics.mean_error_m)
    assert np.isfinite(metrics.max_error_m)
    assert np.isfinite(metrics.rmse_m)
    assert np.isfinite(metrics.speed_error_mps)
    assert np.isfinite(metrics.azimuth_error_deg)
    assert (tmp_path / "output" / "terrain_navigator_report.html").exists()
