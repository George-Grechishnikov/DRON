"""Position and velocity solver for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyproj

from correlator import CorrelationResult


@dataclass(frozen=True)
class PositionEstimate:
    """Geodetic navigation fix derived from correlation output."""

    lat: float
    lon: float
    speed_mps: float
    azimuth_deg: float
    timestamp_s: float
    confidence: float
    is_reliable: bool
    cov_matrix: np.ndarray


def normalize_azimuth_deg(azimuth_deg: float) -> float:
    """Normalize azimuth into the [0, 360) interval."""

    return float(azimuth_deg % 360.0)


def meters_to_degrees(
    lat: float,
    delta_m_lat: float,
    delta_m_lon: float,
) -> Tuple[float, float]:
    """Convert local meter offsets to approximate latitude/longitude deltas."""

    lat_rad = math.radians(lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    if abs(meters_per_deg_lon) < 1e-9:
        raise ValueError("Longitude degree size is too small near the poles")
    return (delta_m_lat / meters_per_deg_lat, delta_m_lon / meters_per_deg_lon)


class PositionSolver:
    """Convert correlation maxima into geodetic fixes and track history."""

    def __init__(self, geod: pyproj.Geod | None = None) -> None:
        self.geod = geod or pyproj.Geod(ellps="WGS84")
        self._history: deque[PositionEstimate] = deque(maxlen=10)

    def solve(
        self,
        result: CorrelationResult,
        start_lat: float,
        start_lon: float,
        window_duration_s: float,
    ) -> PositionEstimate:
        """Convert the best azimuth and offset into a geodetic position fix."""

        if window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")

        azimuth_deg = normalize_azimuth_deg(result.best_azimuth_deg)
        end_lon, end_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            azimuth_deg,
            result.best_offset_m,
        )
        speed_mps = result.best_offset_m / window_duration_s
        sigma_pos = max(1.0, 150.0 * (1.0 - result.peak_correlation))
        cov_matrix = np.diag([sigma_pos**2, sigma_pos**2]).astype(float)

        fix = PositionEstimate(
            lat=float(end_lat),
            lon=float(end_lon),
            speed_mps=float(speed_mps),
            azimuth_deg=azimuth_deg,
            timestamp_s=float(time.time()),
            confidence=float(result.confidence),
            is_reliable=bool(result.is_reliable),
            cov_matrix=cov_matrix,
        )
        self._history.append(fix)
        return fix

    def estimate_velocity_from_two_fixes(
        self,
        fix1: PositionEstimate,
        fix2: PositionEstimate,
    ) -> Tuple[float, float]:
        """Estimate speed and azimuth between two sequential fixes."""

        dt = fix2.timestamp_s - fix1.timestamp_s
        if dt <= 0:
            raise ValueError("fix2.timestamp_s must be greater than fix1.timestamp_s")

        azimuth_forward, _, distance_m = self.geod.inv(
            fix1.lon,
            fix1.lat,
            fix2.lon,
            fix2.lat,
        )
        speed_mps = distance_m / dt
        return float(speed_mps), normalize_azimuth_deg(azimuth_forward)

    def get_track(self) -> List[PositionEstimate]:
        """Return the recent position-fix history."""

        return list(self._history)
