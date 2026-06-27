"""Generate and verify a large unified-sample dataset for stress testing."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Geod


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import Config, load_unified_samples_jsonl, run_pipeline
from sample_validator import validate_unified_sample


WGS84 = Geod(ellps="WGS84")


def generate_unified_samples(
    output_path: Path,
    dem_path: Path,
    count: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Write a large unified JSONL stream aligned to the DEM."""

    if count <= 0:
        raise ValueError("count must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / sample_rate_hz
    center_lat = 60.555
    center_lon = 90.378
    alt_msl_nominal = 1500.0

    started_at = time.perf_counter()
    with rasterio.open(dem_path) as dataset, output_path.open("w", encoding="utf-8") as handle:
        terrain = dataset.read(1)
        bounds = dataset.bounds
        prev_lat: float | None = None
        prev_lon: float | None = None
        last_sample: dict[str, Any] | None = None

        def terrain_height(lat: float, lon: float) -> float:
            row, col = dataset.index(lon, lat)
            row = min(max(row, 0), dataset.height - 1)
            col = min(max(col, 0), dataset.width - 1)
            return float(terrain[row, col])

        for index in range(count):
            t = index * dt
            east_m = (
                2200.0 * np.sin(2.0 * np.pi * t / 7200.0)
                + 900.0 * np.sin(2.0 * np.pi * t / 1800.0)
                + 250.0 * np.sin(2.0 * np.pi * t / 300.0)
            )
            north_m = (
                1800.0 * np.sin(2.0 * np.pi * t / 5400.0 + 0.5)
                + 700.0 * np.sin(2.0 * np.pi * t / 2400.0)
            )
            bearing = float(np.degrees(np.arctan2(east_m, north_m)) % 360.0)
            distance_m = float(np.hypot(east_m, north_m))
            lon, lat, _ = WGS84.fwd(center_lon, center_lat, bearing, distance_m)
            if not (bounds.left < lon < bounds.right and bounds.bottom < lat < bounds.top):
                raise RuntimeError(f"Generated sample {index} left DEM bounds")

            terrain_h = terrain_height(lat, lon)
            alt_msl = alt_msl_nominal + 0.6 * np.sin(2.0 * np.pi * t / 8000.0) + 0.25 * np.cos(
                2.0 * np.pi * t / 1700.0
            )
            radar_alt_m = max(float(alt_msl - terrain_h), 20.0)
            if prev_lat is None or prev_lon is None:
                heading_deg = 0.0
                speed_mps = 0.0
            else:
                fwd_azimuth, _, step_distance_m = WGS84.inv(prev_lon, prev_lat, lon, lat)
                heading_deg = float(fwd_azimuth % 360.0)
                speed_mps = float(step_distance_m / dt)

            sample = {
                "timestamp_s": round(t, 3),
                "lat": round(float(lat), 8),
                "lon": round(float(lon), 8),
                "alt_msl": round(float(alt_msl), 3),
                "radar_alt_m": round(radar_alt_m, 3),
                "terrain_h": round(terrain_h, 3),
                "heading_deg": round(heading_deg, 3),
                "speed_mps": round(speed_mps, 3),
                "gnss_available": t < 6.0,
                "nav_mode": "GNSS" if t < 6.0 else "TERRAIN_NAV",
                "truth_lat": round(float(lat), 8),
                "truth_lon": round(float(lon), 8),
            }
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            prev_lat = float(lat)
            prev_lon = float(lon)
            last_sample = sample

    duration_s = time.perf_counter() - started_at
    return {
        "count": count,
        "sample_rate_hz": sample_rate_hz,
        "generation_seconds": duration_s,
        "output_bytes": output_path.stat().st_size,
        "last_sample": last_sample,
    }


def stream_validate_samples(path: Path) -> dict[str, Any]:
    """Validate a JSONL sample file without loading it fully into memory."""

    started_at = time.perf_counter()
    validated = 0
    invalid = 0
    first_error: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            errors = validate_unified_sample(payload)
            if errors:
                invalid += 1
                if first_error is None:
                    first_error = f"line {line_number}: {errors[0]}"
            validated += 1
    return {
        "validated": validated,
        "invalid": invalid,
        "first_error": first_error,
        "validation_seconds": time.perf_counter() - started_at,
    }


def stream_count_samples(path: Path) -> dict[str, Any]:
    """Count samples through the main JSONL loader interface."""

    started_at = time.perf_counter()
    count = 0
    last_timestamp = None
    for payload in load_unified_samples_jsonl(path):
        count += 1
        last_timestamp = payload.get("timestamp_s")
    return {
        "counted": count,
        "last_timestamp_s": last_timestamp,
        "load_seconds": time.perf_counter() - started_at,
    }


def run_pipeline_subset(
    samples_path: Path,
    dem_path: Path,
    pipeline_samples: int,
    sample_rate_hz: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the terrain pipeline on an initial subset of the large dataset."""

    subset_path = output_dir / f"samples_subset_{pipeline_samples}.jsonl"
    report_path = output_dir / "stress_pipeline_report.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    with samples_path.open("r", encoding="utf-8") as source, subset_path.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source):
            if index >= pipeline_samples:
                break
            target.write(line)

    config = Config(
        mode="samples",
        dem_path=dem_path,
        start_lat=60.56274504,
        start_lon=90.37800000,
        trajectory=1,
        config_path=None,
        nmea_path=None,
        samples_path=subset_path,
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
        window_size=50,
        step_size=10,
        freq_hz=sample_rate_hz,
        dem_patch_radius_m=1000.0,
        max_offset_m=300.0,
        flat_terrain_threshold_m=15.0,
        cold_start_windows=2,
        gnss_drop_after_s=None,
        report_path=report_path,
        log_level="WARNING",
    )
    history, metrics = run_pipeline(config)
    duration_s = time.perf_counter() - started_at
    return {
        "subset_samples": pipeline_samples,
        "pipeline_seconds": duration_s,
        "history_points": len(history),
        "report_path": str(report_path),
        "metrics_present": metrics is not None,
        "report_exists": report_path.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Stress test TERRAIN NAVIGATOR with a million unified samples")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of unified samples to generate")
    parser.add_argument(
        "--pipeline-samples",
        type=int,
        default=5000,
        help="Number of initial samples to run through the full pipeline",
    )
    parser.add_argument("--sample-rate", type=float, default=5.0, help="Unified sample rate in Hz")
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path("input") / "incoming" / "dem" / "terrain.tif",
        help="DEM path for sample generation and subset pipeline run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "stress_1m",
        help="Directory for large sample files and benchmark outputs",
    )
    args = parser.parse_args(argv)

    samples_path = args.output_dir / f"unified_samples_{args.count}.jsonl"
    summary_path = args.output_dir / "stress_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generation = generate_unified_samples(samples_path, args.dem, args.count, args.sample_rate)
    validation = stream_validate_samples(samples_path)
    loading = stream_count_samples(samples_path)
    pipeline = run_pipeline_subset(samples_path, args.dem, args.pipeline_samples, args.sample_rate, args.output_dir)
    summary = {
        "generation": generation,
        "validation": validation,
        "loading": loading,
        "pipeline_subset": pipeline,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
