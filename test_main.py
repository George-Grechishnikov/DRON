from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from constants import FIXED_BARO_ALTITUDE_M
from imm_filter import IMMResult
from main import Config, GroundTruthPoint, build_window_sizes, choose_navigation_fix, compute_replay_metrics, parse_args
from position_solver import PositionEstimate
from correlator import CorrelationResult, ObservabilityMetrics


class StubSolver:
    def __init__(self, fix: PositionEstimate) -> None:
        self.fix = fix
        self.called = False

    def solve(
        self,
        result: CorrelationResult,
        start_lat: float,
        start_lon: float,
        window_duration_s: float,
    ) -> PositionEstimate:
        del result, start_lat, start_lon, window_duration_s
        self.called = True
        return self.fix


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


def test_parse_args_adaptive_window_mode() -> None:
    config = parse_args(
        [
            "--sim",
            "--dem",
            "data/dem.tif",
            "--adaptive-window",
            "--min-window-size",
            "20",
            "--max-window-size",
            "50",
            "--window-growth-step",
            "10",
        ]
    )

    assert config.adaptive_window is True
    assert config.min_window_size == 20
    assert config.max_window_size == 50
    assert config.window_growth_step == 10


def test_parse_args_sitl_mode() -> None:
    config = parse_args(
        [
            "--sitl",
            "--dem",
            "data/dem.tif",
            "--sitl-connect",
            "udp:127.0.0.1:14550",
            "--sitl-gnss-drop-after",
            "15",
        ]
    )

    assert config.mode == "sitl"
    assert config.sitl_connection == "udp:127.0.0.1:14550"
    assert config.sitl_gnss_drop_after_s == 15.0
    assert config.altitude_msl_m == FIXED_BARO_ALTITUDE_M


def test_parse_args_uses_dem_center_when_lat_lon_are_omitted() -> None:
    config = parse_args(
        [
            "--live",
            "--dem",
            "data/dem.tif",
        ]
    )

    assert math.isnan(config.start_lat)
    assert math.isnan(config.start_lon)


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


def test_build_window_sizes_returns_single_size_when_adaptive_is_disabled() -> None:
    sizes = build_window_sizes(_config(), available_frames=80)

    assert sizes == [50]


def test_build_window_sizes_expands_candidates_when_adaptive_is_enabled() -> None:
    config = Config(
        **{
            **_config().__dict__,
            "adaptive_window": True,
            "min_window_size": 20,
            "max_window_size": 50,
            "window_growth_step": 10,
        }
    )

    sizes = build_window_sizes(config, available_frames=50)

    assert sizes == [20, 30, 40, 50]


def _config() -> Config:
    return Config(
        mode="sim",
        dem_path=Path("data/dem.tif"),
        start_lat=60.5,
        start_lon=90.3,
        trajectory=1,
        nmea_path=None,
        gt_path=None,
        sitl_connection="udp:127.0.0.1:14550",
        sitl_gnss_drop_after_s=None,
        sitl_gnss_recover_after_s=None,
        udp_host="127.0.0.1",
        udp_port=10110,
        dashboard_host="127.0.0.1",
        dashboard_port=8050,
        enable_visualizer=False,
        seed=42,
        speed_mps=50.0,
        altitude_msl_m=FIXED_BARO_ALTITUDE_M,
        noise_sigma=2.0,
        adaptive_window=False,
        min_window_size=50,
        max_window_size=50,
        window_growth_step=10,
    )


def _corr(*, is_reliable: bool = True, is_ambiguous: bool = False, confidence: float = 0.8) -> CorrelationResult:
    return CorrelationResult(
        best_azimuth_deg=45.0,
        best_offset_steps=10,
        best_offset_m=300.0,
        best_offset_subsample_steps=10.2,
        best_offset_subsample_m=306.0,
        peak_correlation=0.9,
        confidence=confidence,
        is_reliable=is_reliable,
        pslr_db=7.0,
        ambiguity_peak_count=1 if not is_ambiguous else 3,
        peak_isolation_m=300.0,
        is_ambiguous=is_ambiguous,
        heatmap=np.array([[0.2, 0.9]], dtype=float),
        azimuths_deg=np.array([45.0], dtype=float),
        best_reference_profile=np.ones((8,), dtype=float),
    )


def _solver_fix() -> PositionEstimate:
    return PositionEstimate(
        lat=60.6,
        lon=90.4,
        speed_mps=40.0,
        azimuth_deg=50.0,
        timestamp_s=1.0,
        confidence=0.9,
        is_reliable=True,
        cov_matrix=np.eye(2),
    )


def test_choose_navigation_fix_uses_solver_when_terrain_update_is_accepted() -> None:
    solver = StubSolver(_solver_fix())

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=_corr(),
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True),
    )

    assert solver.called is True
    assert decision.mode == "terrain_update_accepted"
    assert decision.used_prediction_only is False
    assert decision.fix == solver.fix


def test_choose_navigation_fix_falls_back_when_terrain_is_ambiguous() -> None:
    solver = StubSolver(_solver_fix())

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=_corr(is_ambiguous=True),
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True),
    )

    assert solver.called is False
    assert decision.mode == "terrain_ambiguous_fallback"
    assert decision.used_prediction_only is True
    assert decision.fix.is_reliable is False


def test_choose_navigation_fix_falls_back_when_terrain_is_uninformative() -> None:
    solver = StubSolver(_solver_fix())

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=_corr(),
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(
            crlb_m=float("inf"),
            gradient_energy=0.0,
            efficiency_hint=0.0,
            is_informative=False,
        ),
    )

    assert solver.called is False
    assert decision.mode == "terrain_uninformative_fallback"
    assert decision.used_prediction_only is True
