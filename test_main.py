from __future__ import annotations

from pathlib import Path
import queue
import threading

import numpy as np

from imm_filter import IMMResult
from main import (
    GroundTruthPoint,
    UnifiedSample,
    coerce_unified_sample,
    compute_replay_metrics,
    load_unified_samples_jsonl,
    parse_args,
    unified_sample_producer,
)


def test_parse_args_replay_mode() -> None:
    config = parse_args(
        [
            "--replay",
            "--dem",
            "data/dem.tif",
            "--nmea",
            "logs/flight.nmea",
            "--gt",
            "logs/gt.csv",
        ]
    )

    assert config.mode == "replay"
    assert config.nmea_path == Path("logs/flight.nmea")
    assert config.gt_path == Path("logs/gt.csv")


def test_parse_args_accepts_gnss_drop_after() -> None:
    config = parse_args(
        [
            "--sim",
            "--dem",
            "data/dem.tif",
            "--gnss-drop-after",
            "30",
        ]
    )

    assert config.gnss_drop_after_s == 30.0


def test_parse_args_samples_jsonl_mode() -> None:
    config = parse_args(
        [
            "--samples-jsonl",
            "logs/samples.jsonl",
            "--dem",
            "data/dem.tif",
        ]
    )

    assert config.mode == "samples"
    assert config.samples_path == Path("logs/samples.jsonl")


def test_parse_args_unified_stream_alias() -> None:
    config = parse_args(
        [
            "--unified-stream",
            "logs/samples.jsonl",
            "--dem",
            "data/dem.tif",
        ]
    )

    assert config.mode == "samples"
    assert config.samples_path == Path("logs/samples.jsonl")


def test_parse_args_accepts_report_path() -> None:
    config = parse_args(
        [
            "--sim",
            "--dem",
            "data/dem.tif",
            "--report-path",
            "reports/demo.html",
        ]
    )

    assert config.report_path == Path("reports/demo.html")


def test_coerce_unified_sample_from_bridge_dict() -> None:
    sample = coerce_unified_sample(
        {
            "timestamp": 1.5,
            "lat": 60.5,
            "lon": 90.3,
            "alt_msl": 1500.0,
            "heading_deg": 45.0,
            "ground_speed_mps": 50.0,
            "radar_alt_m": 1200.0,
            "gnss_available": False,
        }
    )

    assert isinstance(sample, UnifiedSample)
    assert sample.timestamp == 1.5
    assert sample.gnss_available is False


def test_unified_sample_compatibility() -> None:
    sample = coerce_unified_sample(
        {
            "timestamp_s": 2.0,
            "lat": None,
            "lon": None,
            "alt_msl": 1500.0,
            "radar_alt_m": 1200.0,
            "terrain_h": 300.0,
            "heading_deg": None,
            "speed_mps": None,
            "gnss_available": False,
            "nav_mode": "INIT",
            "truth_lat": 60.5,
            "truth_lon": 90.3,
            "estimated_lat": 60.5001,
            "estimated_lon": 90.3001,
            "correlation_score": 0.88,
            "correlation_heatmap": [[0.1, 0.2], [0.3, 0.9]],
        }
    )

    assert sample.timestamp_s == 2.0
    assert sample.truth_lat == 60.5
    assert sample.estimated_lon == 90.3001
    assert sample.correlation_score == 0.88


def test_unified_sample_producer_enqueues_bridge_dicts() -> None:
    sample_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    unified_sample_producer(
        [
            {
                "timestamp": 0.0,
                "lat": 60.5,
                "lon": 90.3,
                "alt_msl": 1500.0,
                "heading_deg": 45.0,
                "ground_speed_mps": 50.0,
                "radar_alt_m": 1200.0,
                "gnss_available": True,
            }
        ],
        sample_queue,
        stop_event,
    )

    packet = sample_queue.get_nowait()
    assert packet.index == 0
    assert packet.sample.lat == 60.5
    assert packet.sample.gnss_available is True


def test_load_unified_samples_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        "\n"
        '{"timestamp": 0, "lat": 60.5, "lon": 90.3, "alt_msl": 1500, '
        '"heading_deg": 45, "ground_speed_mps": 50, "radar_alt_m": 1200, '
        '"gnss_available": true}\n',
        encoding="utf-8",
    )

    samples = list(load_unified_samples_jsonl(samples_path))

    assert len(samples) == 1
    assert samples[0]["lat"] == 60.5


def test_compute_replay_metrics_returns_non_negative_values() -> None:
    history = [
        (
            1,
            IMMResult(
                lat=60.5,
                lon=90.3,
                speed_mps=50.0,
                azimuth_deg=45.0,
                model_weights=np.array([0.2, 0.7, 0.1], dtype=float),
                covariance=np.eye(4),
                dominant_mode="cruise",
            ),
        )
    ]
    ground_truth = [
        GroundTruthPoint(
            index=1,
            timestamp_s=1.0,
            lat=60.5001,
            lon=90.3001,
            speed_mps=49.0,
            azimuth_deg=44.0,
        )
    ]

    metrics = compute_replay_metrics(history, ground_truth)

    assert metrics.mean_error_m >= 0.0
    assert metrics.max_error_m >= 0.0
    assert metrics.rmse_m >= 0.0
    assert metrics.speed_error_mps >= 0.0
    assert metrics.azimuth_error_deg >= 0.0
