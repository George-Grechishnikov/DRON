"""Normalize ad hoc sample files into TERRAIN NAVIGATOR unified JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sample_validator import validate_unified_sample


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp_s": ("timestamp_s", "timestamp", "time", "ts"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "longitude"),
    "alt_msl": ("alt_msl", "altitude_msl", "altitude", "alt"),
    "radar_alt_m": ("radar_alt_m", "radar_alt", "agl", "relative_alt"),
    "terrain_h": ("terrain_h", "terrain_height", "ground_height"),
    "heading_deg": ("heading_deg", "heading", "course_deg", "yaw_deg"),
    "speed_mps": ("speed_mps", "ground_speed_mps", "speed", "ground_speed"),
    "gnss_available": ("gnss_available", "gnss", "gps_available", "gps_ok"),
    "nav_mode": ("nav_mode", "mode", "navigation_mode"),
    "truth_lat": ("truth_lat", "gt_lat", "ground_truth_lat"),
    "truth_lon": ("truth_lon", "gt_lon", "ground_truth_lon"),
    "estimated_lat": ("estimated_lat", "est_lat"),
    "estimated_lon": ("estimated_lon", "est_lon"),
    "correlation_score": ("correlation_score", "score", "corr_score"),
    "correlation_heatmap": ("correlation_heatmap", "heatmap"),
    "best_azimuth_deg": ("best_azimuth_deg", "best_azimuth"),
    "best_offset_m": ("best_offset_m", "best_offset"),
}

BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}


def _find_alias(row: dict[str, Any], canonical_field: str) -> Any:
    for alias in FIELD_ALIASES[canonical_field]:
        if alias in row:
            return row[alias]
    return None


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in BOOL_TRUE:
        return True
    if normalized in BOOL_FALSE:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one loose sample dict into unified sample format."""

    sample = {
        "timestamp_s": float(_find_alias(row, "timestamp_s")),
        "lat": _parse_optional_float(_find_alias(row, "lat")),
        "lon": _parse_optional_float(_find_alias(row, "lon")),
        "alt_msl": float(_find_alias(row, "alt_msl")),
        "radar_alt_m": float(_find_alias(row, "radar_alt_m")),
        "terrain_h": _parse_optional_float(_find_alias(row, "terrain_h")),
        "heading_deg": _parse_optional_float(_find_alias(row, "heading_deg")),
        "speed_mps": _parse_optional_float(_find_alias(row, "speed_mps")),
        "gnss_available": _parse_bool(_find_alias(row, "gnss_available")),
        "nav_mode": str(_find_alias(row, "nav_mode") or ("GNSS" if _parse_bool(_find_alias(row, "gnss_available")) else "TERRAIN_NAV")).upper(),
        "truth_lat": _parse_optional_float(_find_alias(row, "truth_lat")),
        "truth_lon": _parse_optional_float(_find_alias(row, "truth_lon")),
        "estimated_lat": _parse_optional_float(_find_alias(row, "estimated_lat")),
        "estimated_lon": _parse_optional_float(_find_alias(row, "estimated_lon")),
        "correlation_score": _parse_optional_float(_find_alias(row, "correlation_score")),
        "correlation_heatmap": _find_alias(row, "correlation_heatmap"),
        "best_azimuth_deg": _parse_optional_float(_find_alias(row, "best_azimuth_deg")),
        "best_offset_m": _parse_optional_float(_find_alias(row, "best_offset_m")),
    }
    errors = validate_unified_sample(sample)
    if errors:
        raise ValueError("; ".join(errors))
    return sample


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            return [dict(item) for item in payload["samples"]]
        return [payload]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError("JSON input must be an object, a list of objects, or {'samples': [...]}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load arbitrary sample data from csv/json/jsonl."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use .csv, .json, or .jsonl")


def write_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    """Write normalized unified samples to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Normalize random sample files into unified sample JSONL")
    parser.add_argument("input_path", type=Path, help="Input .csv, .json, or .jsonl file")
    parser.add_argument("output_path", type=Path, help="Output unified JSONL path")
    args = parser.parse_args(argv)

    if not args.input_path.exists():
        raise SystemExit(f"Input file not found: {args.input_path}")

    rows = load_samples(args.input_path)
    normalized = [normalize_sample(row) for row in rows]
    write_jsonl(normalized, args.output_path)
    print(f"Normalized {len(normalized)} sample(s) to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
