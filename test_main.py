from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from constants import FIXED_BARO_ALTITUDE_M
from imm_filter import IMMResult
from main import (
    Config,
    FramePacket,
    GroundTruthPoint,
    NavigationDecision,
    ReacquisitionTracker,
    WindowSelection,
    build_turn_probe_window_sizes,
    build_window_sizes,
    choose_navigation_fix,
    compute_replay_metrics,
    estimate_window_duration_s,
    maybe_select_turn_transition_window,
    maybe_trim_turn_transition_tail,
    maybe_update_heading_from_correlation,
    parse_args,
    predict_fix,
    update_reacquisition_tracker,
    update_motion_state_after_decision,
)
from position_solver import PositionEstimate, PositionSolver
from correlator import CorrelationCandidate, CorrelationResult, ObservabilityMetrics
from measurement_layer import BaroTrack
from nmea_parser import NMEAFrame


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
        update_dt_s: float | None = None,
        measurement_span_m: float = 0.0,
    ) -> PositionEstimate:
        del result, start_lat, start_lon, window_duration_s, update_dt_s, measurement_span_m
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


def test_parse_args_accepts_eskf_engine() -> None:
    config = parse_args(
        [
            "--sim",
            "--dem",
            "data/dem.tif",
            "--engine",
            "eskf",
        ]
    )

    assert config.engine == "eskf"


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


def test_estimate_window_duration_uses_nmea_timestamps() -> None:
    packets = [
        FramePacket(
            index=0,
            frame=NMEAFrame(timestamp_utc="123519.000", radar_alt_m=100.0, raw="", valid=True),
        ),
        FramePacket(
            index=1,
            frame=NMEAFrame(timestamp_utc="123521.500", radar_alt_m=100.0, raw="", valid=True),
        ),
    ]

    duration_s = estimate_window_duration_s(packets, fallback_freq_hz=5.0)

    assert duration_s == 2.5


def test_estimate_window_duration_falls_back_for_invalid_timestamps() -> None:
    packets = [
        FramePacket(
            index=0,
            frame=NMEAFrame(timestamp_utc="", radar_alt_m=100.0, raw="", valid=True),
        ),
        FramePacket(
            index=1,
            frame=NMEAFrame(timestamp_utc="", radar_alt_m=100.0, raw="", valid=True),
        ),
        FramePacket(
            index=2,
            frame=NMEAFrame(timestamp_utc="", radar_alt_m=100.0, raw="", valid=True),
        ),
    ]

    duration_s = estimate_window_duration_s(packets, fallback_freq_hz=4.0)

    assert duration_s == 0.5


def test_build_turn_probe_window_sizes_includes_short_tail() -> None:
    sizes = build_turn_probe_window_sizes(_config(), available_frames=60)

    assert sizes == [20, 30, 40]


def test_maybe_select_turn_transition_window_prefers_short_tail_for_ambiguous_base() -> None:
    config = _config()
    base = WindowSelection(
        frame_packets=[],
        window_size=50,
        corr_result=_corr(
            is_ambiguous=True,
            confidence=0.02,
            pslr_db=0.4,
            azimuth_deg=18.0,
            peak_correlation=0.90,
        ),
        observability=_obs(),
        flat=False,
    )
    short_tail = WindowSelection(
        frame_packets=[],
        window_size=20,
        corr_result=_corr(
            is_ambiguous=False,
            confidence=0.09,
            pslr_db=3.8,
            azimuth_deg=46.0,
            peak_correlation=0.89,
        ),
        observability=_obs(),
        flat=False,
    )

    selected = maybe_select_turn_transition_window(config, [base, short_tail], current_azimuth_deg=0.0)

    assert selected == short_tail


def test_maybe_select_turn_transition_window_keeps_base_when_short_tail_is_not_better() -> None:
    config = _config()
    base = WindowSelection(
        frame_packets=[],
        window_size=50,
        corr_result=_corr(
            is_ambiguous=True,
            confidence=0.02,
            pslr_db=0.4,
            azimuth_deg=18.0,
            peak_correlation=0.90,
        ),
        observability=_obs(),
        flat=False,
    )
    weak_short_tail = WindowSelection(
        frame_packets=[],
        window_size=20,
        corr_result=_corr(
            is_ambiguous=False,
            confidence=0.02,
            pslr_db=0.7,
            azimuth_deg=21.0,
            peak_correlation=0.82,
        ),
        observability=_obs(),
        flat=False,
    )

    selected = maybe_select_turn_transition_window(config, [base, weak_short_tail], current_azimuth_deg=0.0)

    assert selected is None


def test_maybe_trim_turn_transition_tail_returns_none_when_window_is_not_suspicious() -> None:
    selection = WindowSelection(
        frame_packets=[],
        window_size=50,
        corr_result=_corr(
            is_ambiguous=False,
            confidence=0.12,
            pslr_db=5.0,
            azimuth_deg=3.0,
            peak_correlation=0.92,
        ),
        observability=_obs(),
        flat=False,
    )

    selected = maybe_trim_turn_transition_tail(
        config=_config(),
        selection=selection,
        correlator=None,  # type: ignore[arg-type]
        ref_matrix=np.empty((0, 0), dtype=float),
        azimuths_deg=np.empty((0,), dtype=float),
        baro_track=BaroTrack(default_msl_m=FIXED_BARO_ALTITUDE_M),
        terrain_bias_m=0.0,
        measurement_step_m=10.0,
        current_azimuth_deg=0.0,
    )

    assert selected is None


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


def _corr(
    *,
    is_reliable: bool = True,
    is_ambiguous: bool = False,
    confidence: float = 0.8,
    pslr_db: float = 7.0,
    azimuth_deg: float = 45.0,
    peak_correlation: float = 0.9,
    offset_m: float = 300.0,
) -> CorrelationResult:
    offset_steps = int(round(offset_m / 30.0))
    return CorrelationResult(
        best_azimuth_deg=azimuth_deg,
        best_offset_steps=offset_steps,
        best_offset_m=offset_m,
        best_offset_subsample_steps=float(offset_steps),
        best_offset_subsample_m=offset_m,
        peak_correlation=peak_correlation,
        confidence=confidence,
        is_reliable=is_reliable,
        pslr_db=pslr_db,
        ambiguity_peak_count=1 if not is_ambiguous else 3,
        peak_isolation_m=300.0,
        is_ambiguous=is_ambiguous,
        heatmap=np.array([[0.2, 0.9]], dtype=float),
        azimuths_deg=np.array([azimuth_deg], dtype=float),
        best_reference_profile=np.ones((8,), dtype=float),
    )


def _obs(*, is_informative: bool = True, efficiency_hint: float = 0.7) -> ObservabilityMetrics:
    return ObservabilityMetrics(
        crlb_m=4.0,
        efficiency_hint=efficiency_hint,
        gradient_energy=0.2,
        is_informative=is_informative,
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
        measurement_span_m=490.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True),
    )

    assert solver.called is True
    assert decision.mode == "terrain_update_accepted"
    assert decision.used_prediction_only is False
    assert decision.fix == solver.fix


def test_choose_navigation_fix_rejects_physically_implausible_jump() -> None:
    solver = PositionSolver()
    solver.solve(
        result=_corr(offset_m=0.0, azimuth_deg=45.0),
        start_lat=60.5,
        start_lon=90.3,
        window_duration_s=10.0,
        update_dt_s=2.0,
        measurement_span_m=0.0,
    )

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=_corr(offset_m=2000.0, azimuth_deg=45.0),
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        measurement_span_m=100.0,
        update_dt=2.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True),
    )

    assert decision.used_prediction_only is True
    assert decision.mode.startswith("terrain_physical_gate_fallback")


def test_choose_navigation_fix_accepts_plausible_top_k_alternative() -> None:
    solver = PositionSolver()
    solver.solve(
        result=_corr(offset_m=0.0, azimuth_deg=45.0),
        start_lat=60.5,
        start_lon=90.3,
        window_duration_s=10.0,
        update_dt_s=2.0,
        measurement_span_m=0.0,
    )
    primary = _corr(offset_m=2000.0, azimuth_deg=45.0, peak_correlation=0.9)
    alternative = CorrelationCandidate(
        azimuth_deg=45.0,
        offset_steps=0,
        offset_m=0.0,
        offset_subsample_steps=0.0,
        offset_subsample_m=0.0,
        score=0.88,
        ncc_score=0.88,
        msd_score=0.88,
    )
    corr = CorrelationResult(
        **{
            **primary.__dict__,
            "top_candidates": (alternative,),
        }
    )

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=corr,
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        measurement_span_m=100.0,
        update_dt=2.0,
        window_counter=5,
        flat=False,
        observability=ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True),
    )

    assert decision.used_prediction_only is False
    assert decision.corr_result is not None
    assert decision.corr_result.best_offset_m == 0.0


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
        measurement_span_m=490.0,
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
        measurement_span_m=490.0,
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


def test_navigation_decision_can_mark_degraded_fallback_mode() -> None:
    solver = StubSolver(_solver_fix())

    decision = choose_navigation_fix(
        config=_config(),
        corr_result=_corr(is_reliable=False, confidence=0.0),
        solver=solver,
        window_start_lat=60.5,
        window_start_lon=90.3,
        current_speed=50.0,
        current_azimuth=45.0,
        window_duration=10.0,
        measurement_span_m=490.0,
        window_counter=5,
        flat=True,
        observability=ObservabilityMetrics(
            crlb_m=float("inf"),
            gradient_energy=0.0,
            efficiency_hint=0.0,
            is_informative=False,
        ),
    )

    assert decision.used_prediction_only is True
    assert decision.fix.is_reliable is False


def test_ambiguous_turn_transition_keeps_prediction_heading_when_pslr_is_low() -> None:
    corr = _corr(is_ambiguous=True, confidence=0.05)
    corr = CorrelationResult(
        **{
            **corr.__dict__,
            "best_azimuth_deg": 92.0,
            "best_offset_m": 120.0,
            "best_offset_subsample_m": 125.0,
            "pslr_db": 0.3,
            "ambiguity_peak_count": 3,
        }
    )

    hinted = maybe_update_heading_from_correlation(
        current_azimuth_deg=45.0,
        corr_result=corr,
        max_offset_m=2000.0,
        selected_window_size=30,
        measurement_step_m=10.0,
        used_prediction_only=True,
    )

    assert hinted == 45.0


def test_ambiguous_top_k_low_offset_candidate_can_softly_update_heading() -> None:
    corr = _corr(is_ambiguous=True, confidence=0.05, peak_correlation=0.94)
    candidate = CorrelationCandidate(
        azimuth_deg=45.0,
        offset_steps=3,
        offset_m=30.0,
        offset_subsample_steps=3.0,
        offset_subsample_m=30.0,
        score=0.93,
        ncc_score=0.88,
        msd_score=0.88,
    )
    corr = CorrelationResult(
        **{
            **corr.__dict__,
            "best_azimuth_deg": 20.0,
            "best_offset_m": 0.0,
            "best_offset_subsample_m": 0.0,
            "top_candidates": (candidate,),
            "pslr_db": 0.3,
            "ambiguity_peak_count": 3,
        }
    )

    hinted = maybe_update_heading_from_correlation(
        current_azimuth_deg=1.5,
        corr_result=corr,
        max_offset_m=2000.0,
        selected_window_size=50,
        measurement_step_m=10.0,
        used_prediction_only=True,
    )

    assert hinted == 9.5


def test_prediction_heading_cue_updates_next_motion_state() -> None:
    corr = _corr(
        is_reliable=False,
        is_ambiguous=False,
        confidence=0.1,
        pslr_db=4.0,
        azimuth_deg=1.5,
        peak_correlation=0.98,
        offset_m=0.0,
    )
    decision = NavigationDecision(
        fix=predict_fix(
            current_lat=60.5,
            current_lon=90.3,
            speed_mps=50.0,
            azimuth_deg=1.5,
            dt=10.0,
            confidence=0.1,
        ),
        mode="cold_start_prediction",
        used_prediction_only=True,
    )

    speed, azimuth = update_motion_state_after_decision(
        nav_decision=decision,
        corr_result=corr,
        current_speed=50.0,
        current_azimuth=45.0,
        max_offset_m=2000.0,
        selected_window_size=50,
        measurement_step_m=10.0,
    )

    assert speed == 50.0
    assert azimuth == 1.5


def test_reacquisition_waits_for_two_stable_clean_windows_after_ambiguity() -> None:
    tracker = ReacquisitionTracker(pending=True)
    obs = ObservabilityMetrics(crlb_m=50.0, gradient_energy=2.0, efficiency_hint=0.4, is_informative=True)
    corr_first = CorrelationResult(
        **{
            **_corr().__dict__,
            "best_azimuth_deg": 82.0,
            "best_offset_m": 240.0,
            "best_offset_subsample_m": 245.0,
            "pslr_db": 4.5,
            "confidence": 0.2,
        }
    )
    tracker, hold_first = update_reacquisition_tracker(
        tracker,
        corr_result=corr_first,
        observability=obs,
        flat=False,
        max_offset_m=2000.0,
    )
    assert hold_first is True
    assert tracker.pending is True
    assert tracker.stable_windows == 1

    corr_second = CorrelationResult(
        **{
            **corr_first.__dict__,
            "best_azimuth_deg": 86.0,
            "best_offset_subsample_m": 255.0,
        }
    )
    tracker, hold_second = update_reacquisition_tracker(
        tracker,
        corr_result=corr_second,
        observability=obs,
        flat=False,
        max_offset_m=2000.0,
    )
    assert hold_second is False
    assert tracker.pending is False
    assert tracker.stable_windows == 0
