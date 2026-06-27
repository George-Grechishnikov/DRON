from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nmea_parser import parse_line
from sim_generator import SimulationConfig, generate_imu_points, generate_points, nmea_checksum, write_csv, write_nmea_file


def _write_test_dem(path: Path) -> None:
    data = np.fromfunction(lambda r, c: 100.0 + r * 0.5 + c * 0.25, (800, 800), dtype=float)
    transform = from_origin(89.0, 61.0, 0.002, 0.002)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data.astype("float32"), 1)


def _write_water_dem(path: Path) -> None:
    data = np.zeros((800, 800), dtype="float32")
    data[:, 400:] = np.fromfunction(lambda r, c: 100.0 + r * 0.25 + c * 0.1, (800, 400), dtype=float).astype("float32")
    transform = from_origin(89.0, 61.0, 0.002, 0.002)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)


def _base_config(dem_path: Path, *, start_lon: float = 90.3) -> SimulationConfig:
    return SimulationConfig(
        dem_path=dem_path,
        start_lat=60.5,
        start_lon=start_lon,
        frequency_hz=5.0,
        noise_sigma_m=0.0,
        output_mode="file",
        out_nmea=None,
        out_csv=None,
        out_imu=None,
        udp_host="127.0.0.1",
        udp_port=10110,
        random_seed=7,
        altitude_msl_m=1500.0,
        speed_mps=50.0,
        azimuth_deg=45.0,
        duration_s=10.0,
        length_km=None,
        trajectory_id=None,
    )


def test_trajectory_1_generates_valid_nmea_and_csv(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    nmea_path = tmp_path / "traj1.nmea"
    csv_path = tmp_path / "traj1.csv"
    _write_test_dem(dem_path)

    config = SimulationConfig(**{**_base_config(dem_path).__dict__, "out_nmea": nmea_path, "out_csv": csv_path})

    points = generate_points(config)
    write_nmea_file(points, nmea_path)
    write_csv(points, csv_path)

    assert len(points) == 50

    nmea_lines = nmea_path.read_text(encoding="ascii").splitlines()
    assert len(nmea_lines) == 50
    for line in nmea_lines:
        payload, checksum = line[1:].split("*")
        assert nmea_checksum(payload) == checksum

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 50
    assert rows[0]["index"] == "0"


def test_realistic_false_is_reproducible_and_matches_baseline(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_test_dem(dem_path)
    config = SimulationConfig(**{**_base_config(dem_path).__dict__, "noise_sigma_m": 1.5, "random_seed": 11})

    points_a = generate_points(config)
    points_b = generate_points(config)

    assert len(points_a) == len(points_b)
    assert np.allclose(
        [point.radar_alt_measured for point in points_a],
        [point.radar_alt_measured for point in points_b],
        atol=1e-9,
    )


def test_dropout_writes_empty_nmea_altitude_field(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    nmea_path = tmp_path / "dropout.nmea"
    _write_test_dem(dem_path)
    config = SimulationConfig(
        **{
            **_base_config(dem_path).__dict__,
            "realistic": True,
            "dropout_prob": 1.0,
            "out_nmea": nmea_path,
        }
    )

    points = generate_points(config)
    write_nmea_file(points, nmea_path)
    first_line = nmea_path.read_text(encoding="ascii").splitlines()[0]
    frame = parse_line(first_line)

    assert ",,M," in first_line
    assert frame is not None
    assert np.isnan(frame.radar_alt_m)


def test_spike_model_produces_large_outliers_when_enabled(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_test_dem(dem_path)
    clean_config = SimulationConfig(**{**_base_config(dem_path).__dict__, "noise_sigma_m": 0.0, "random_seed": 21})
    spike_config = SimulationConfig(
        **{
            **_base_config(dem_path).__dict__,
            "noise_sigma_m": 0.0,
            "random_seed": 21,
            "realistic": True,
            "spike_prob": 1.0,
            "spike_sigma_m": 30.0,
        }
    )

    clean_points = generate_points(clean_config)
    spike_points = generate_points(spike_config)
    delta = np.abs(
        np.asarray([point.radar_alt_measured for point in spike_points], dtype=float)
        - np.asarray([point.radar_alt_measured for point in clean_points], dtype=float)
    )

    assert float(np.max(delta)) > 10.0


def test_water_multipath_increases_measurement_variance_over_water(tmp_path: Path) -> None:
    dem_path = tmp_path / "water_dem.tif"
    _write_water_dem(dem_path)
    base_water = SimulationConfig(
        **{
            **_base_config(dem_path, start_lon=89.1).__dict__,
            "noise_sigma_m": 0.5,
            "random_seed": 5,
            "realistic": True,
        }
    )
    noisy_water = SimulationConfig(
        **{
            **_base_config(dem_path, start_lon=89.1).__dict__,
            "noise_sigma_m": 0.5,
            "random_seed": 5,
            "realistic": True,
            "water_multipath_extra_sigma_m": 8.0,
        }
    )

    base_points = generate_points(base_water)
    noisy_points = generate_points(noisy_water)
    base_residual = np.asarray([point.alt_msl - point.terrain_h - point.radar_alt_measured for point in base_points], dtype=float)
    noisy_residual = np.asarray([point.alt_msl - point.terrain_h - point.radar_alt_measured for point in noisy_points], dtype=float)

    assert float(np.nanstd(noisy_residual)) > float(np.nanstd(base_residual))


def test_surface_bias_shifts_measured_radar_altitude_without_touching_ground_truth(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_test_dem(dem_path)
    base_config = SimulationConfig(
        **{
            **_base_config(dem_path).__dict__,
            "noise_sigma_m": 0.0,
            "random_seed": 9,
            "realistic": True,
        }
    )
    forest_config = SimulationConfig(
        **{
            **_base_config(dem_path).__dict__,
            "noise_sigma_m": 0.0,
            "random_seed": 9,
            "realistic": True,
            "surface_bias_mode": "forest",
            "surface_bias_m": 20.0,
        }
    )

    base_points = generate_points(base_config)
    forest_points = generate_points(forest_config)

    assert np.isclose(base_points[0].terrain_h, forest_points[0].terrain_h, atol=1e-9)
    assert np.isclose(
        base_points[0].radar_alt_measured - forest_points[0].radar_alt_measured,
        20.0,
        atol=1e-6,
    )


def test_generate_imu_points_matches_trajectory_length(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    _write_test_dem(dem_path)
    points = generate_points(_base_config(dem_path))
    imu_points = generate_imu_points(points)

    assert len(imu_points) == len(points)
    assert np.isfinite([row.accel_z_mps2 for row in imu_points]).all()
