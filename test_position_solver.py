from __future__ import annotations

import numpy as np
import pyproj

from correlator import CorrelationResult
from position_solver import PositionEstimate, PositionSolver, meters_to_degrees


def _make_result(azimuth_deg: float, offset_m: float) -> CorrelationResult:
    return CorrelationResult(
        best_azimuth_deg=azimuth_deg,
        best_offset_steps=int(offset_m // 30.0),
        best_offset_m=offset_m,
        peak_correlation=0.9,
        confidence=0.6,
        is_reliable=True,
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
    )
    expected_lon, expected_lat, _ = geod.fwd(90.3, 60.5, 45.0, 1500.0)

    assert np.isclose(fix.lat, expected_lat, atol=1e-7)
    assert np.isclose(fix.lon, expected_lon, atol=1e-7)
    assert np.isclose(fix.speed_mps, 150.0)
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
        solver.solve(result, start_lat=60.5, start_lon=90.3, window_duration_s=10.0)

    track = solver.get_track()
    assert len(track) == 10
