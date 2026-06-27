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
class RadarSample:
    """One validated radar sample from radar_data.nmea."""

    line_number: int
    timestamp_s: float
    radar_alt_m: float


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


def _parse_nmea_time_to_seconds(token: str) -> float:
    """Parse an NMEA HHMMSS(.sss) token into seconds-of-day."""

    value = token.strip()
    if not value:
        raise ValueError("empty NMEA timestamp")
    if "." in value:
        head, tail = value.split(".", 1)
        fractional = float(f"0.{tail}")
    else:
        head = value
        fractional = 0.0
    if len(head) < 6 or not head.isdigit():
        raise ValueError(f"invalid NMEA timestamp '{token}'")
    hours = int(head[:-4])
    minutes = int(head[-4:-2])
    seconds = int(head[-2:])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid NMEA timestamp '{token}'")
    return hours * 3600.0 + minutes * 60.0 + seconds + fractional


def load_radar_samples(path: Path) -> list[RadarSample]:
    """Load radar_data.nmea rows with explicit validation and timestamps."""

    reader = NMEAReader.from_file(path)
    radar_samples: list[RadarSample] = []
    invalid_lines: list[int] = []
    try:
        for line_number, frame in enumerate(reader, start=1):
            if not frame.valid:
                invalid_lines.append(line_number)
                continue
            try:
                timestamp_s = _parse_nmea_time_to_seconds(frame.timestamp_utc)
            except ValueError as exc:
                raise ValueError(
                    f"radar_data.nmea contains invalid timestamp on line {line_number}: {exc}"
                ) from exc
            radar_samples.append(
                RadarSample(
                    line_number=line_number,
                    timestamp_s=timestamp_s,
                    radar_alt_m=float(frame.radar_alt_m),
                )
            )
    finally:
        reader.close()

    if invalid_lines:
        preview = ", ".join(str(number) for number in invalid_lines[:10])
        suffix = "..." if len(invalid_lines) > 10 else ""
        raise ValueError(
            "radar_data.nmea contains invalid or checksum-failed frames at line(s): "
            f"{preview}{suffix}"
        )
    return radar_samples


def _normalize_radar_relative_timestamps(radar_samples: list[RadarSample]) -> list[float]:
    """Return monotonic relative radar timestamps with midnight rollover handling."""

    if not radar_samples:
        return []
    normalized: list[float] = []
    day_offset_s = 0.0
    previous_timestamp_s = radar_samples[0].timestamp_s
    origin_timestamp_s = previous_timestamp_s
    for sample in radar_samples:
        timestamp_s = sample.timestamp_s
        if timestamp_s + day_offset_s < previous_timestamp_s - 43200.0:
            day_offset_s += 86400.0
        adjusted_timestamp_s = timestamp_s + day_offset_s
        normalized.append(adjusted_timestamp_s - origin_timestamp_s)
        previous_timestamp_s = adjusted_timestamp_s
    return normalized


def _validate_alignment(
    truth_samples: list[TruthSample],
    barometer_samples: list[BarometerSample],
    radar_samples: list[RadarSample],
    sample_rate_hz: float,
) -> None:
    if len(truth_samples) != len(barometer_samples):
        raise ValueError(
            "truth.csv and barometer.csv must contain the same number of rows "
            f"({len(truth_samples)} != {len(barometer_samples)})"
        )
    if len(truth_samples) != len(radar_samples):
        raise ValueError(
            "radar_data.nmea and truth.csv must contain the same number of samples "
            f"({len(radar_samples)} != {len(truth_samples)})"
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
    truth_origin = truth_samples[0].timestamp_s if truth_samples else 0.0
    radar_relative_timestamps = _normalize_radar_relative_timestamps(radar_samples)
    for index, (truth, radar_relative_s) in enumerate(zip(truth_samples, radar_relative_timestamps), start=1):
        truth_relative_s = truth.timestamp_s - truth_origin
        if abs(truth_relative_s - radar_relative_s) > tolerance_s:
            raise ValueError(
                "radar_data.nmea timestamps diverge from truth.csv at row "
                f"{index}: {radar_relative_s:.3f}s vs {truth_relative_s:.3f}s"
            )


def iter_case_unified_samples(config: CaseInputConfig) -> Iterator[dict[str, Any]]:
    """Yield unified samples from radar_data.nmea + truth.csv + barometer.csv."""

    truth_samples = load_truth_samples(config.truth_path)
    barometer_samples = load_barometer_samples(config.barometer_path)
    radar_samples = load_radar_samples(config.radar_data_path)
    _validate_alignment(truth_samples, barometer_samples, radar_samples, config.sample_rate_hz)

    for index, (radar, truth, baro) in enumerate(zip(radar_samples, truth_samples, barometer_samples)):
        gnss_available = config.gnss_drop_after_s is None or truth.timestamp_s < config.gnss_drop_after_s
        yield {
            "timestamp_s": truth.timestamp_s,
            "lat": truth.lat if gnss_available else None,
            "lon": truth.lon if gnss_available else None,
            "alt_msl": baro.baro_alt_m,
            "radar_alt_m": radar.radar_alt_m,
            "terrain_h": baro.baro_alt_m - radar.radar_alt_m,
            "heading_deg": truth.heading_deg,
            "speed_mps": truth.speed_mps,
            "gnss_available": gnss_available,
            "nav_mode": "GNSS" if gnss_available else "INIT",
            "truth_lat": truth.lat,
            "truth_lon": truth.lon,
        }
