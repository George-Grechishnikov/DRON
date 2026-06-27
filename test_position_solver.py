from __future__ import annotations

import numpy as np
import pyproj

from correlator import CorrelationResult
from position_solver import PositionEstimate, PositionSolver, meters_to_degrees


def _make_result(
    azimuth_deg: float,
    offset_m: float,
    *,
    sigma_offset_m: float = 20.0,
    sigma_azimuth_m: float = 40.0,
    informative: bool = True,
) -> CorrelationResult:
    return CorrelationResult(
        best_azimuth_deg=azimuth_deg,
        best_offset_steps=int(offset_m // 30.0),
        best_offset_m=offset_m,
        sigma_offset_m=sigma_offset_m,
        sigma_azimuth_m=sigma_azimuth_m,
        peak_correlation=0.9,
        confidence=0.6,
        is_reliable=True,
        informative=informative,
        heatmap=np.zeros((1, 1), dtype=float),
        azimuths_deg=np.array([azimuth_deg], dtype=float),
        best_reference_profile=np.zeros((10,), dtype=float),
    )


def test_solve_matches_geodetic_forward_solution() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    solver = PositionSolver(geod=geod)
    result = _make_result(azimuth_deg=45.0, offset_m=1500.0)

    fix = solver.solve(
        result=result,
        start_lat=60.5,
        start_lon=90.3,
        window_duration_s=10.0,
        measurement_span_m=0.0,
    )
    expected_lon, expected_lat, _ = geod.fwd(90.3, 60.5, 45.0, 1500.0)

    assert np.isclose(fix.lat, expected_lat, atol=1e-7)
    assert np.isclose(fix.lon, expected_lon, atol=1e-7)
    assert np.isclose(fix.speed_mps, 0.0)
    assert fix.azimuth_deg == 45.0
    assert fix.cov_matrix.shape == (2, 2)


def test_estimate_velocity_from_two_fixes() -> None:
    solver = PositionSolver()
    fix1 = PositionEstimate(
        lat=60.5,
        lon=90.3,
        speed_mps=0.0,
        azimuth_deg=0.0,
        timestamp_s=100.0,
        confidence=1.0,
        is_reliable=True,
        cov_matrix=np.eye(2),
    )
    fix2 = PositionEstimate(
        lat=60.5005,
        lon=90.301,
        speed_mps=0.0,
        azimuth_deg=0.0,
        timestamp_s=110.0,
        confidence=1.0,
        is_reliable=True,
        cov_matrix=np.eye(2),
    )

    speed_mps, azimuth_deg = solver.estimate_velocity_from_two_fixes(fix1, fix2)

    assert speed_mps > 0.0
    assert 0.0 <= azimuth_deg < 360.0


def test_meters_to_degrees_returns_small_positive_deltas() -> None:
    delta_lat_deg, delta_lon_deg = meters_to_degrees(
        lat=60.5,
        delta_m_lat=100.0,
        delta_m_lon=100.0,
    )

    assert delta_lat_deg > 0.0
    assert delta_lon_deg > 0.0


def test_history_keeps_last_ten_fixes() -> None:
    solver = PositionSolver()
    result = _make_result(azimuth_deg=30.0, offset_m=300.0)

    for _ in range(12):
        solver.solve(result, start_lat=60.5, start_lon=90.3, window_duration_s=10.0, measurement_span_m=0.0)

    track = solver.get_track()
    assert len(track) == 10


def test_second_fix_derives_motion_from_previous_fix() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    solver = PositionSolver(geod=geod)

    first = solver.solve(
        _make_result(azimuth_deg=45.0, offset_m=500.0),
        start_lat=60.5,
        start_lon=90.3,
        window_duration_s=10.0,
        measurement_span_m=500.0,
    )
    second = solver.solve(
        _make_result(azimuth_deg=45.0, offset_m=500.0),
        start_lat=first.lat,
        start_lon=first.lon,
        window_duration_s=10.0,
        measurement_span_m=500.0,
    )

    assert np.isclose(second.speed_mps, 100.0, atol=1e-6)
    assert np.isclose(second.azimuth_deg, 45.0, atol=1e-2)
    assert np.isclose(second.timestamp_s - first.timestamp_s, 10.0, atol=1e-9)


def test_solve_can_return_end_of_window_fix_from_matched_origin() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    solver = PositionSolver(geod=geod)
    result = _make_result(azimuth_deg=45.0, offset_m=100.0)

    fix = solver.solve(
        result,
        start_lat=60.5,
        start_lon=90.3,
        window_duration_s=10.0,
        measurement_span_m=400.0,
    )
    matched_lon, matched_lat, _ = geod.fwd(90.3, 60.5, 45.0, 100.0)
    expected_lon, expected_lat, _ = geod.fwd(matched_lon, matched_lat, 45.0, 400.0)

    assert np.isclose(fix.lat, expected_lat, atol=1e-7)
    assert np.isclose(fix.lon, expected_lon, atol=1e-7)
    assert np.isclose(fix.speed_mps, 40.0, atol=1e-9)


def test_position_covariance_is_anisotropic_and_rotated_by_azimuth() -> None:
    solver = PositionSolver()
    result = _make_result(
        azimuth_deg=0.0,
        offset_m=300.0,
        sigma_offset_m=10.0,
        sigma_azimuth_m=50.0,
    )

    cov = solver._position_covariance(result, azimuth_deg=0.0)

    assert cov.shape == (2, 2)
    assert np.isclose(cov[0, 0], 50.0**2, atol=1e-6)
    assert np.isclose(cov[1, 1], 10.0**2, atol=1e-6)
    assert np.allclose(cov, cov.T, atol=1e-9)


def test_uninformative_result_inflates_position_covariance() -> None:
    solver = PositionSolver()
    informative = _make_result(
        azimuth_deg=45.0,
        offset_m=300.0,
        sigma_offset_m=10.0,
        sigma_azimuth_m=20.0,
        informative=True,
    )
    uninformative = _make_result(
        azimuth_deg=45.0,
        offset_m=300.0,
        sigma_offset_m=10.0,
        sigma_azimuth_m=20.0,
        informative=False,
    )

    informative_cov = solver._position_covariance(informative, azimuth_deg=45.0)
    uninformative_cov = solver._position_covariance(uninformative, azimuth_deg=45.0)

    assert np.trace(uninformative_cov) > np.trace(informative_cov)
