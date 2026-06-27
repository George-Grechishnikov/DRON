from __future__ import annotations

import time

import numpy as np

from constants import FIXED_BARO_ALTITUDE_M
from correlator import PSR_THRESHOLD, Correlator, build_heatmap, compute_crlb_position, compute_observability_metrics, parabolic_vertex
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

    assert np.isclose(result.best_azimuth_deg, 45.0, atol=0.5)
    assert result.best_offset_steps == 10
    assert result.best_offset_m == 300.0
    assert abs(result.best_offset_subsample_steps - 10.0) < 1.0
    assert result.sigma_offset_m >= 0.0
    assert result.sigma_azimuth_m >= 0.0
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
        NMEAFrame(
            timestamp_utc=str(idx),
            radar_alt_m=float(FIXED_BARO_ALTITUDE_M - value),
            raw="",
            valid=True,
        )
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

    assert np.isclose(result.best_azimuth_deg, 7.0, atol=0.5)
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

    assert np.isclose(result.best_azimuth_deg, 120.0, atol=0.5)
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


def test_peak_quality_marks_single_sharp_peak_as_reliable() -> None:
    heatmap = np.zeros((360, 24), dtype=float)
    heatmap[120, 10] = 1.0
    heatmap[120, 17] = 0.2

    confidence, peak_to_sidelobe, peak_to_mean = Correlator._peak_quality(heatmap, 120, 10)

    assert confidence > 0.0
    assert peak_to_sidelobe > PSR_THRESHOLD
    assert peak_to_mean > 1.0


def test_peak_quality_marks_two_similar_peaks_as_unreliable() -> None:
    heatmap = np.zeros((360, 24), dtype=float)
    heatmap[90, 8] = 1.0
    heatmap[200, 16] = 0.99

    confidence, _, _ = Correlator._peak_quality(heatmap, 90, 8)

    assert confidence < 0.05


def test_peak_quality_uses_cyclic_azimuth_mask() -> None:
    heatmap = np.zeros((360, 24), dtype=float)
    heatmap[0, 10] = 1.0
    heatmap[359, 10] = 0.95
    heatmap[40, 10] = 0.4

    confidence, _, _ = Correlator._peak_quality(heatmap, 0, 10)

    assert np.isclose(confidence, 0.6, atol=1e-6)


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

    assert np.isclose(result.best_azimuth_deg, 9.0, atol=0.5)
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

    assert np.isclose(result.best_azimuth_deg, 11.0, atol=0.5)
    assert result.best_offset_steps == 6


def test_hybrid_metric_preserves_absolute_level_better_than_ncc() -> None:
    base = np.array([120.0, 135.0, 128.0, 145.0, 132.0, 150.0, 141.0, 155.0], dtype=float)
    ref_matrix = np.array([np.concatenate([base + 100.0, base])], dtype=float)
    h_meas = base.copy()
    azimuths = np.array([0.0], dtype=float)

    ncc_correlator = Correlator(
        profile_length_m=240.0,
        step_m=30.0,
        max_offset_m=240.0,
        metric="ncc",
    )
    hybrid_correlator = Correlator(
        profile_length_m=240.0,
        step_m=30.0,
        max_offset_m=240.0,
        metric="hybrid",
        alpha=0.5,
        beta=0.5,
        msd_scale_m2=25.0,
    )

    ncc_result = ncc_correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)
    hybrid_result = hybrid_correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert ncc_result.best_offset_steps == 0
    assert hybrid_result.best_offset_steps == len(base)
    assert hybrid_result.msd_peak > ncc_result.msd_peak


def test_hybrid_metric_reduces_false_confidence_on_flat_terrain() -> None:
    azimuths = np.array([0.0], dtype=float)
    noise = np.array([0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.1, -0.1], dtype=float)
    h_meas = 100.0 + noise
    ref_matrix = (130.0 + noise).reshape(1, -1)

    ncc_correlator = Correlator(
        profile_length_m=240.0,
        step_m=30.0,
        max_offset_m=0.0,
        metric="ncc",
    )
    hybrid_correlator = Correlator(
        profile_length_m=240.0,
        step_m=30.0,
        max_offset_m=0.0,
        metric="hybrid",
        alpha=0.5,
        beta=0.5,
        msd_scale_m2=4.0,
    )

    ncc_result = ncc_correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)
    hybrid_result = hybrid_correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert hybrid_result.peak_correlation < ncc_result.peak_correlation
    assert hybrid_result.confidence < ncc_result.confidence


def test_masked_correlation_handles_nodata_without_losing_valid_peak() -> None:
    azimuths = np.array([0.0], dtype=float)
    h_meas = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    ref_matrix = np.array([[5.0, np.nan, 10.0, 20.0, 30.0, 40.0, np.nan]], dtype=float)
    correlator = Correlator(
        profile_length_m=120.0,
        step_m=30.0,
        max_offset_m=90.0,
        metric="hybrid",
        min_valid_fraction=0.5,
    )

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.informative is True
    assert result.best_offset_steps == 2
    assert result.peak_correlation > 0.5


def test_masked_correlation_marks_window_uninformative_when_too_many_nans() -> None:
    azimuths = np.array([0.0], dtype=float)
    h_meas = np.array([10.0, np.nan, np.nan, 40.0], dtype=float)
    ref_matrix = np.array([[np.nan, 20.0, np.nan, 40.0, np.nan, 60.0]], dtype=float)
    correlator = Correlator(
        profile_length_m=120.0,
        step_m=30.0,
        max_offset_m=60.0,
        metric="hybrid",
        min_valid_fraction=0.5,
    )

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.informative is False
    assert result.is_reliable is False
    assert result.peak_correlation == 0.0


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


def test_compute_returns_uninformative_result_for_flat_measurement() -> None:
    azimuths = np.arange(16, dtype=float)
    ref_matrix = np.ones((16, 40), dtype=float) * 100.0
    h_meas = np.ones((20,), dtype=float) * 100.0
    correlator = Correlator(profile_length_m=600.0, step_m=30.0, max_offset_m=180.0)

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert result.informative is False
    assert result.is_reliable is False
    assert result.peak_correlation == 0.0


def test_parabolic_vertex_returns_subsample_delta_and_curvature() -> None:
    delta, curvature = parabolic_vertex(0.8, 1.0, 0.9)

    assert -0.5 <= delta <= 0.5
    assert delta > 0.0
    assert curvature < 0.0


def test_compute_supports_cyclic_azimuth_subsample_interpolation() -> None:
    azimuths = np.arange(360, dtype=float)
    ref_matrix = _make_reference_matrix(360, 40, 8)
    h_meas = ref_matrix[0, 4:44]
    ref_matrix[359, 4:44] = h_meas - 0.001
    correlator = Correlator(profile_length_m=1200.0, step_m=30.0, max_offset_m=240.0)

    result = correlator.compute(h_meas, ref_matrix, azimuths_deg=azimuths)

    assert 0.0 <= result.best_azimuth_deg < 360.0


def test_compute_rejects_mismatched_azimuth_grid_metadata() -> None:
    azimuths = np.array([10.0, 20.0], dtype=float)
    ref_matrix = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=float).view(np.ndarray)

    class ReferenceArray(np.ndarray):
        pass

    reference = np.asarray(ref_matrix, dtype=float).view(ReferenceArray)
    reference.azimuths_deg = np.array([11.0, 21.0], dtype=float)
    correlator = Correlator(profile_length_m=60.0, step_m=30.0, max_offset_m=0.0)

    with np.testing.assert_raises(ValueError):
        correlator.compute(np.array([1.0, 2.0, 3.0], dtype=float), reference, azimuths_deg=azimuths)


def test_sharp_peak_has_smaller_sigma_than_flat_peak() -> None:
    sharp_heatmap = np.array([[0.2, 0.6, 1.0, 0.6, 0.2]], dtype=float)
    flat_heatmap = np.array([[0.8, 0.9, 1.0, 0.9, 0.8]], dtype=float)

    _, sharp_curvature = Correlator._subsample_peak_1d(sharp_heatmap[0], 2, cyclic=False)
    _, flat_curvature = Correlator._subsample_peak_1d(flat_heatmap[0], 2, cyclic=False)
    sharp_sigma = Correlator._curvature_to_sigma(sharp_curvature)
    flat_sigma = Correlator._curvature_to_sigma(flat_curvature)

    assert sharp_sigma < flat_sigma
