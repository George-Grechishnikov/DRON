"""Validate unified sample streams before running the terrain pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "timestamp_s",
    "alt_msl",
    "radar_alt_m",
    "gnss_available",
)
OPTIONAL_MODE_VALUES = {"GNSS", "TERRAIN_NAV", "INIT", "LOST", "TERRAIN"}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_unified_sample(sample: dict[str, Any]) -> list[str]:
    """Return validation errors for one unified sample dictionary."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in sample and not (field == "timestamp_s" and "timestamp" in sample):
            errors.append(f"missing required field: {field}")

    timestamp = sample.get("timestamp_s", sample.get("timestamp"))
    if timestamp is not None and not _is_finite_number(timestamp):
        errors.append("timestamp_s must be a finite number")

    for numeric_field in ("alt_msl", "radar_alt_m", "terrain_h", "heading_deg", "speed_mps", "ground_speed_mps"):
        if numeric_field in sample and sample[numeric_field] is not None and not _is_finite_number(sample[numeric_field]):
            errors.append(f"{numeric_field} must be a finite number or null")

    for coord_field in ("lat", "lon", "truth_lat", "truth_lon", "estimated_lat", "estimated_lon"):
        if coord_field in sample and sample[coord_field] is not None and not _is_finite_number(sample[coord_field]):
            errors.append(f"{coord_field} must be a finite number or null")

    if "lat" in sample and sample["lat"] is not None and not (-90.0 <= float(sample["lat"]) <= 90.0):
        errors.append("lat must be within [-90, 90]")
    if "truth_lat" in sample and sample["truth_lat"] is not None and not (-90.0 <= float(sample["truth_lat"]) <= 90.0):
        errors.append("truth_lat must be within [-90, 90]")
    for lon_field in ("lon", "truth_lon", "estimated_lon"):
        if lon_field in sample and sample[lon_field] is not None and not (-180.0 <= float(sample[lon_field]) <= 180.0):
            errors.append(f"{lon_field} must be within [-180, 180]")

    if "gnss_available" in sample and not isinstance(sample["gnss_available"], bool):
        errors.append("gnss_available must be boolean")

    if "nav_mode" in sample and sample["nav_mode"] is not None:
        mode = str(sample["nav_mode"]).upper()
        if mode not in OPTIONAL_MODE_VALUES:
            errors.append(f"nav_mode must be one of {sorted(OPTIONAL_MODE_VALUES)}")

    heatmap = sample.get("correlation_heatmap")
    if heatmap is not None and not isinstance(heatmap, (list, tuple)):
        errors.append("correlation_heatmap must be a nested list/tuple or null")

    return errors


def load_jsonl_samples(path: Path) -> list[dict[str, Any]]:
    """Load JSONL samples from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Unified sample file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Unified sample path is not a file: {path}")

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            samples.append(payload)
    return samples


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Validate TERRAIN NAVIGATOR unified sample JSONL")
    parser.add_argument("samples_jsonl", type=Path, help="Path to unified sample JSONL file")
    args = parser.parse_args(argv)

    try:
        samples = load_jsonl_samples(args.samples_jsonl)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Pass an existing JSONL file, for example: output/samples.jsonl", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    failures = 0
    for index, sample in enumerate(samples, start=1):
        errors = validate_unified_sample(sample)
        if errors:
            failures += 1
            print(f"sample {index} invalid:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)

    if failures:
        print(f"Validation failed: {failures} invalid sample(s)", file=sys.stderr)
        return 1

    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
