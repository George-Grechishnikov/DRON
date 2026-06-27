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
        update_dt_s: float | None = None,
        measurement_span_m: float = 0.0,
    ) -> PositionEstimate:
        """Convert the best azimuth and offset into a geodetic position fix."""

        if window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")

        previous_fix = self._history[-1] if self._history else None
        fix = self.solve_with_velocity(
            result=result,
            start_lat=start_lat,
            start_lon=start_lon,
            window_duration_s=window_duration_s,
            update_dt_s=update_dt_s,
            measurement_span_m=measurement_span_m,
            prev_fix=previous_fix,
        )
        self._history.append(fix)
        return fix

    def solve_with_velocity(
        self,
        result: CorrelationResult,
        start_lat: float,
        start_lon: float,
        window_duration_s: float,
        update_dt_s: float | None,
        measurement_span_m: float,
        prev_fix: PositionEstimate | None,
    ) -> PositionEstimate:
        """Convert correlation output into a fix, deriving motion from sequential fixes when possible."""

        if window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")
        if measurement_span_m < 0:
            raise ValueError("measurement_span_m must be non-negative")

        azimuth_deg = normalize_azimuth_deg(result.best_azimuth_deg)
        offset_m = (
            result.best_offset_subsample_m
            if np.isfinite(result.best_offset_subsample_m)
            and (result.best_offset_m == 0.0 or result.best_offset_subsample_m > 0.0)
            else result.best_offset_m
        )
        origin_lon, origin_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            azimuth_deg,
            offset_m,
        )
        end_lon, end_lat, _ = self.geod.fwd(
            origin_lon,
            origin_lat,
            azimuth_deg,
            measurement_span_m,
        )
        speed_mps = measurement_span_m / window_duration_s
        timestamp_s = (
            prev_fix.timestamp_s + (float(update_dt_s) if update_dt_s is not None else window_duration_s)
            if prev_fix is not None
            else float(time.time())
        )
        cov_matrix = self._position_covariance(result, azimuth_deg)

        fix = PositionEstimate(
            lat=float(end_lat),
            lon=float(end_lon),
            speed_mps=float(speed_mps),
            azimuth_deg=azimuth_deg,
            timestamp_s=timestamp_s,
            confidence=float(result.confidence),
            is_reliable=bool(result.is_reliable),
            cov_matrix=cov_matrix,
        )

        if prev_fix is not None:
            measured_speed_mps, measured_azimuth_deg = self.estimate_velocity_from_two_fixes(prev_fix, fix)
            fix = PositionEstimate(
                lat=fix.lat,
                lon=fix.lon,
                speed_mps=float(measured_speed_mps),
                azimuth_deg=float(measured_azimuth_deg),
                timestamp_s=fix.timestamp_s,
                confidence=fix.confidence,
                is_reliable=fix.is_reliable,
                cov_matrix=fix.cov_matrix,
            )
        return fix

    def _position_covariance(
        self,
        result: CorrelationResult,
        azimuth_deg: float,
    ) -> np.ndarray:
        """Build a 2x2 EN covariance from along-track and cross-track uncertainty."""

        sigma_along = self._sanitize_sigma(result.sigma_offset_m, fallback_m=150.0 * (1.0 - result.peak_correlation))
        sigma_cross = self._sanitize_sigma(result.sigma_azimuth_m, fallback_m=max(sigma_along, 25.0))

        if not result.informative:
            sigma_along *= 5.0
            sigma_cross *= 5.0

        azimuth_rad = math.radians(normalize_azimuth_deg(azimuth_deg))
        basis = np.array(
            [
                [math.sin(azimuth_rad), math.cos(azimuth_rad)],
                [math.cos(azimuth_rad), -math.sin(azimuth_rad)],
            ],
            dtype=float,
        )
        cov_track = np.diag([sigma_along**2, sigma_cross**2]).astype(float)
        cov_matrix = basis @ cov_track @ basis.T
        return cov_matrix.astype(float)

    @staticmethod
    def _sanitize_sigma(sigma_m: float, *, fallback_m: float) -> float:
        """Return a stable positive sigma in meters."""

        if np.isfinite(sigma_m) and sigma_m > 1e-6:
            return float(sigma_m)
        return float(max(fallback_m, 1.0))

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
