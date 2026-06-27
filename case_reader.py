"""Load case-style terrain-navigation inputs from config-driven files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from nmea_parser import NMEAReader


@dataclass(frozen=True)
class TruthSample:
    """One ground-truth row from truth.csv."""

    timestamp_s: float
    lat: float
    lon: float
    alt_msl: float
    heading_deg: float
    speed_mps: float


@dataclass(frozen=True)
class BarometerSample:
    """One barometer row from barometer.csv."""

    timestamp_s: float
    baro_alt_m: float


@dataclass(frozen=True)
class CaseInputConfig:
    """Resolved input file paths and timing settings for case playback."""

    dem_path: Path
    radar_data_path: Path
    truth_path: Path
    barometer_path: Path
    sample_rate_hz: float
    gnss_drop_after_s: float | None = None


def _require_columns(path: Path, header: list[str], required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{path} missing required column(s): {', '.join(missing)}")


def load_truth_samples(path: Path) -> list[TruthSample]:
    """Load case truth.csv rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        _require_columns(path, header, ("timestamp", "lat", "lon", "alt_msl", "heading_deg", "speed_mps"))
        return [
            TruthSample(
                timestamp_s=float(row["timestamp"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                alt_msl=float(row["alt_msl"]),
                heading_deg=float(row["heading_deg"]),
                speed_mps=float(row["speed_mps"]),
            )
            for row in reader
        ]


def load_barometer_samples(path: Path) -> list[BarometerSample]:
    """Load case barometer.csv rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        _require_columns(path, header, ("timestamp", "baro_alt_m"))
        return [
            BarometerSample(
                timestamp_s=float(row["timestamp"]),
                baro_alt_m=float(row["baro_alt_m"]),
            )
            for row in reader
        ]


def _validate_alignment(
    truth_samples: list[TruthSample],
    barometer_samples: list[BarometerSample],
    sample_rate_hz: float,
) -> None:
    if len(truth_samples) != len(barometer_samples):
        raise ValueError(
            "truth.csv and barometer.csv must contain the same number of rows "
            f"({len(truth_samples)} != {len(barometer_samples)})"
        )
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    tolerance_s = max(1.0 / sample_rate_hz, 0.25)
    for index, (truth, baro) in enumerate(zip(truth_samples, barometer_samples), start=1):
        if abs(truth.timestamp_s - baro.timestamp_s) > tolerance_s:
            raise ValueError(
                f"truth.csv and barometer.csv timestamps diverge at row {index}: "
                f"{truth.timestamp_s} vs {baro.timestamp_s}"
            )


def iter_case_unified_samples(config: CaseInputConfig) -> Iterator[dict[str, Any]]:
    """Yield unified samples from radar_data.nmea + truth.csv + barometer.csv."""

    truth_samples = load_truth_samples(config.truth_path)
    barometer_samples = load_barometer_samples(config.barometer_path)
    _validate_alignment(truth_samples, barometer_samples, config.sample_rate_hz)

    reader = NMEAReader.from_file(config.radar_data_path)
    try:
        radar_frames = [frame for frame in reader if frame.valid]
    finally:
        reader.close()

    if len(radar_frames) != len(truth_samples):
        raise ValueError(
            "radar_data.nmea and truth.csv must contain the same number of valid samples "
            f"({len(radar_frames)} != {len(truth_samples)})"
        )

    for index, (frame, truth, baro) in enumerate(zip(radar_frames, truth_samples, barometer_samples)):
        gnss_available = config.gnss_drop_after_s is None or truth.timestamp_s < config.gnss_drop_after_s
        yield {
            "timestamp_s": truth.timestamp_s,
            "lat": truth.lat if gnss_available else None,
            "lon": truth.lon if gnss_available else None,
            "alt_msl": baro.baro_alt_m,
            "radar_alt_m": frame.radar_alt_m,
            "terrain_h": baro.baro_alt_m - frame.radar_alt_m,
            "heading_deg": truth.heading_deg,
            "speed_mps": truth.speed_mps,
            "gnss_available": gnss_available,
            "nav_mode": "GNSS" if gnss_available else "TERRAIN_NAV",
            "truth_lat": truth.lat,
            "truth_lon": truth.lon,
        }
