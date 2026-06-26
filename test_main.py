from __future__ import annotations

from pathlib import Path

import numpy as np

from imm_filter import IMMResult
from main import GroundTruthPoint, compute_replay_metrics, parse_args


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
