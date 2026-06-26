from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sim_generator import SimulationConfig, generate_points, nmea_checksum, write_csv, write_nmea_file


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


def test_trajectory_1_generates_valid_nmea_and_csv(tmp_path: Path) -> None:
    dem_path = tmp_path / "dem.tif"
    nmea_path = tmp_path / "traj1.nmea"
    csv_path = tmp_path / "traj1.csv"
    _write_test_dem(dem_path)

    config = SimulationConfig(
        dem_path=dem_path,
        start_lat=60.5,
        start_lon=90.3,
        frequency_hz=5.0,
        noise_sigma_m=0.0,
        output_mode="file",
        out_nmea=nmea_path,
        out_csv=csv_path,
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
