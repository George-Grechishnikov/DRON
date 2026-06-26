from __future__ import annotations

import numpy as np
import pyproj

from imm_filter import IMMFilter
from position_solver import PositionEstimate


def _make_fix(
    *,
    geod: pyproj.Geod,
    start_lat: float,
    start_lon: float,
    azimuth_deg: float,
    distance_m: float,
    speed_mps: float,
    timestamp_s: float,
) -> PositionEstimate:
    lon, lat, _ = geod.fwd(start_lon, start_lat, azimuth_deg, distance_m)
    return PositionEstimate(
        lat=float(lat),
        lon=float(lon),
        speed_mps=float(speed_mps),
        azimuth_deg=float(azimuth_deg),
        timestamp_s=float(timestamp_s),
        confidence=0.8,
        is_reliable=True,
        cov_matrix=np.diag([25.0, 25.0]).astype(float),
    )


def test_straight_line_dominates_cruise_mode() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    imm = IMMFilter()
    start_lat = 60.5
    start_lon = 90.3

    result = None
    for step in range(1, 7):
        fix = _make_fix(
            geod=geod,
            start_lat=start_lat,
            start_lon=start_lon,
            azimuth_deg=45.0,
            distance_m=step * 100.0,
            speed_mps=50.0,
            timestamp_s=float(step * 2),
        )
        result = imm.update(fix, dt=2.0, is_flat=False)

    assert result is not None
    assert result.dominant_mode == "cruise"
    assert result.model_weights[1] > 0.7


def test_sharp_turn_dominates_turn_mode() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    imm = IMMFilter()
    start_lat = 60.5
    start_lon = 90.3
    azimuths = [0.0, 0.0, 0.0, 60.0, 90.0, 90.0]

    results = []
    for step, azimuth in enumerate(azimuths, start=1):
        fix = _make_fix(
            geod=geod,
            start_lat=start_lat,
            start_lon=start_lon,
            azimuth_deg=azimuth,
            distance_m=step * 120.0,
            speed_mps=55.0,
            timestamp_s=float(step * 2),
        )
        results.append(imm.update(fix, dt=2.0, is_flat=False))

    turn_window = results[3:5]
    assert all(result.dominant_mode == "turn" for result in turn_window)
    assert all(result.model_weights[2] > 0.45 for result in turn_window)


def test_flat_terrain_increases_hdop() -> None:
    geod = pyproj.Geod(ellps="WGS84")
    imm = IMMFilter()
    fix = _make_fix(
        geod=geod,
        start_lat=60.5,
        start_lon=90.3,
        azimuth_deg=30.0,
        distance_m=100.0,
        speed_mps=30.0,
        timestamp_s=2.0,
    )

    imm.update(fix, dt=2.0, is_flat=True)
    hdop = imm.get_hdop()

    assert hdop > 0.0
