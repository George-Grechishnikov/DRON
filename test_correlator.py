from __future__ import annotations

import time

import numpy as np

from correlator import Correlator, build_heatmap, compute_crlb_position, compute_observability_metrics
from nmea_parser import NMEAFrame


def _make_reference_matrix(
    azimuth_count: int,
    length: int,
    offset_steps: int,
) -> np.ndarray:
    total_length = length + offset_steps
    rng = np.random.default_rng(2026)
    base = rng.normal(0.0, 1.0, size=(azimuth_count, total_length))
    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    smoothed = np.array(
        [np.convolve(row, kernel, mode="same") for row in base],
        dtype=float,
    )
    trend = np.linspace(-0.5, 0.5, total_length)
    for azimuth in range(azimuth_count):
        smoothed[azimuth] += trend * (azimuth / max(azimuth_count - 1, 1))
    return smoothed


def test_compute_recovers_best_azimuth_and_offset() -> None:
    azimuths = np.arange(360, dtype=float)
    length = 80
    offset_steps = 20
    ref_matrix = _make_reference_matrix(360, length, offset_steps)
    rng = np.random.default_rng(123)
    h_meas = ref_matrix[45, 10 : 10 + length] + rng.normal(0.0, 0.03, size=length)

    correlator = Correlator(profile_length_m=2400.0, step_m=30.0, max_offset_m=600.0)
    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.best_azimuth_deg == 45.0
    assert result.best_offset_steps == 10
    assert result.best_offset_m == 300.0
    assert abs(result.best_offset_subsample_steps - 10.0) < 1.0
    assert result.peak_correlation > 0.95
    assert result.best_reference_profile.shape == h_meas.shape


def test_build_heatmap_normalizes_output() -> None:
    azimuths = np.arange(3, dtype=float)
    ref_matrix = _make_reference_matrix(3, 20, 5)
    h_meas = ref_matrix[1, 2:22]
    correlator = Correlator(profile_length_m=600.0, step_m=30.0, max_offset_m=150.0)
    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    heatmap = build_heatmap(result)
    assert heatmap.shape == result.heatmap.shape
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) <= 1.0


def test_sliding_window_compute_uses_nmea_frames() -> None:
    azimuths = np.arange(16, dtype=float)
    ref_matrix = _make_reference_matrix(16, 30, 6)
    profile = ref_matrix[7, 3:33]
    frames = [
        NMEAFrame(timestamp_utc=str(idx), radar_alt_m=float(value), raw="", valid=True)
        for idx, value in enumerate(profile)
    ]

    correlator = Correlator(profile_length_m=900.0, step_m=30.0, max_offset_m=180.0)
    result = correlator.sliding_window_compute(
        frames_buffer=frames,
        ref_matrix=ref_matrix,
        speed_mps=50.0,
        freq_hz=5.0,
        azimuths_deg=azimuths,
    )

    assert result.best_azimuth_deg == 7.0
    assert result.best_offset_steps == 3


def test_compute_performance_under_half_second() -> None:
    azimuths = np.arange(360, dtype=float)
    length = 128
    offset_steps = 67
    ref_matrix = _make_reference_matrix(360, length, offset_steps)
    h_meas = ref_matrix[120, 25 : 25 + length]
    correlator = Correlator(profile_length_m=3840.0, step_m=30.0, max_offset_m=2010.0)

    started_at = time.perf_counter()
    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)
    elapsed = time.perf_counter() - started_at

    assert result.best_azimuth_deg == 120.0
    assert elapsed < 0.5


def test_compute_returns_ambiguity_metrics() -> None:
    azimuths = np.arange(8, dtype=float)
    ref_matrix = _make_reference_matrix(8, 40, 8)
    h_meas = ref_matrix[3, 4:44]
    correlator = Correlator(profile_length_m=1200.0, step_m=30.0, max_offset_m=240.0)

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert np.isfinite(result.pslr_db)
    assert result.ambiguity_peak_count >= 1
    assert result.peak_isolation_m >= 0.0
    assert isinstance(result.is_ambiguous, bool)


def test_compute_supports_feature_mode() -> None:
    azimuths = np.arange(24, dtype=float)
    ref_matrix = _make_reference_matrix(24, 48, 10)
    h_meas = ref_matrix[9, 5:53]
    correlator = Correlator(
        profile_length_m=1440.0,
        step_m=30.0,
        max_offset_m=300.0,
        use_terrain_features=True,
    )

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.best_azimuth_deg == 9.0
    assert result.best_offset_steps == 5


def test_compute_supports_top_k_feature_refinement() -> None:
    azimuths = np.arange(32, dtype=float)
    ref_matrix = _make_reference_matrix(32, 64, 12)
    h_meas = ref_matrix[11, 6:70]
    correlator = Correlator(
        profile_length_m=1920.0,
        step_m=30.0,
        max_offset_m=360.0,
        feature_refine_top_k=5,
    )

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.best_azimuth_deg == 11.0
    assert result.best_offset_steps == 6


def test_compute_crlb_position_is_finite_for_informative_profile() -> None:
    profile = np.array([100.0, 120.0, 90.0, 140.0, 80.0, 130.0], dtype=float)

    crlb_m = compute_crlb_position(profile, sigma_noise_m=3.0, step_m=30.0)

    assert np.isfinite(crlb_m)
    assert crlb_m > 0.0


def test_compute_observability_metrics_detects_flat_profile() -> None:
    profile = np.full((10,), 100.0, dtype=float)

    metrics = compute_observability_metrics(profile, sigma_noise_m=3.0, step_m=30.0)

    assert metrics.is_informative is False
    assert np.isinf(metrics.crlb_m)
