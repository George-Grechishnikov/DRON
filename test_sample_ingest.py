from __future__ import annotations

import json
from pathlib import Path

from sample_ingest import load_samples, normalize_sample, write_jsonl


def test_normalize_sample_accepts_alias_fields() -> None:
    sample = normalize_sample(
        {
            "timestamp": "1.5",
            "latitude": "60.5",
            "longitude": "90.3",
            "altitude": "1500",
            "agl": "1200",
            "heading": "45",
            "ground_speed": "50",
            "gps_ok": "true",
            "mode": "gnss",
            "gt_lat": "60.5",
            "gt_lon": "90.3",
        }
    )

    assert sample["timestamp_s"] == 1.5
    assert sample["lat"] == 60.5
    assert sample["lon"] == 90.3
    assert sample["nav_mode"] == "GNSS"


def test_load_samples_supports_json_array(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "timestamp_s": 1.0,
                    "lat": 60.5,
                    "lon": 90.3,
                    "alt_msl": 1500.0,
                    "radar_alt_m": 1200.0,
                    "gnss_available": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = load_samples(input_path)

    assert len(rows) == 1
    assert rows[0]["lat"] == 60.5


def test_write_jsonl_writes_normalized_samples(tmp_path: Path) -> None:
    output_path = tmp_path / "normalized.jsonl"
    write_jsonl(
        [
            {
                "timestamp_s": 1.0,
                "lat": 60.5,
                "lon": 90.3,
                "alt_msl": 1500.0,
                "radar_alt_m": 1200.0,
                "terrain_h": None,
                "heading_deg": 45.0,
                "speed_mps": 50.0,
                "gnss_available": True,
                "nav_mode": "GNSS",
                "truth_lat": None,
                "truth_lon": None,
                "estimated_lat": None,
                "estimated_lon": None,
                "correlation_score": None,
                "correlation_heatmap": None,
                "best_azimuth_deg": None,
                "best_offset_m": None,
            }
        ],
        output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"timestamp_s": 1.0' in lines[0]
