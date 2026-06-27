"""Production preflight checks for TERRAIN NAVIGATOR."""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PYTHON_FILES = [
    "case_runner.py",
    "case_reader.py",
    "correlation_fallback.py",
    "sample_ingest.py",
    "sample_validator.py",
    "sim_generator.py",
    "nmea_parser.py",
    "dem_loader.py",
    "profile_extractor.py",
    "correlator.py",
    "position_solver.py",
    "imm_filter.py",
    "visualizer.py",
    "main.py",
]


def compile_sources() -> None:
    """Compile project modules before running heavier checks."""

    for relative_path in PYTHON_FILES:
        py_compile.compile(str(PROJECT_ROOT / relative_path), doraise=True)


def run_tests() -> None:
    """Run the full pytest suite."""

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def write_demo_dem(path: Path) -> np.ndarray:
    """Create a deterministic DEM that covers the default demo route."""

    rows, cols = 1200, 1200
    y_grid, x_grid = np.indices((rows, cols), dtype=float)
    terrain = (
        250.0
        + 40.0 * np.sin(x_grid / 17.0)
        + 28.0 * np.cos(y_grid / 23.0)
        + 12.0 * np.sin((x_grid + y_grid) / 31.0)
        + 0.03 * x_grid
        + 0.05 * y_grid
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


def write_demo_samples(dem_path: Path, samples_path: Path, terrain: np.ndarray) -> None:
    """Write a short unified sample stream with a GNSS loss transition."""

    geod = Geod(ellps="WGS84")
    lat = 60.5
    lon = 90.3
    with rasterio.open(dem_path) as dataset, samples_path.open("w", encoding="utf-8") as handle:
        for index in range(60):
            if index > 0:
                lon, lat, _ = geod.fwd(lon, lat, 45.0, 10.0)
            row, col = dataset.index(lon, lat)
            terrain_h = float(terrain[row, col])
            timestamp = index / 5.0
            sample = {
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "alt_msl": 1500.0,
                "heading_deg": 45.0,
                "ground_speed_mps": 50.0,
                "radar_alt_m": 1500.0 - terrain_h,
                "gnss_available": timestamp < 6.0,
            }
            handle.write(json.dumps(sample) + "\n")


def run_jsonl_smoke() -> None:
    """Run the main pipeline against a generated unified sample JSONL stream."""

    from main import main

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        dem_path = temp_root / "preflight_dem.tif"
        samples_path = temp_root / "samples.jsonl"
        terrain = write_demo_dem(dem_path)
        write_demo_samples(dem_path, samples_path, terrain)

        previous_cwd = Path.cwd()
        try:
            os.chdir(temp_root)
            exit_code = main(
                [
                    "--samples-jsonl",
                    str(samples_path),
                    "--dem",
                    str(dem_path),
                    "--lat",
                    "60.5",
                    "--lon",
                    "90.3",
                    "--no-visualizer",
                    "--window-size",
                    "20",
                    "--step-size",
                    "10",
                    "--dem-patch-radius",
                    "1000",
                    "--max-offset",
                    "300",
                    "--log-level",
                    "WARNING",
                ]
            )
            report_path = temp_root / "output" / "terrain_navigator_report.html"
            if exit_code != 0 or not report_path.exists():
                raise RuntimeError("JSONL smoke run did not produce the expected report")
        finally:
            os.chdir(previous_cwd)


def main(argv: list[str] | None = None) -> int:
    """Run the complete preflight suite."""

    parser = argparse.ArgumentParser(description="Run TERRAIN NAVIGATOR production preflight")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest and run only compile + smoke checks")
    args = parser.parse_args(argv)

    compile_sources()
    if not args.skip_tests:
        run_tests()
    run_jsonl_smoke()
    print("Preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
