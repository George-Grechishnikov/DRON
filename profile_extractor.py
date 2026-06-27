"""Reference terrain profile extraction for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pyproj import Geod
from scipy.signal import savgol_filter

from dem_loader import DEMLoader


LOGGER = logging.getLogger(__name__)
WGS84_GEOD = Geod(ellps="WGS84")


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-std normalized profile."""

    values = np.asarray(profile, dtype=float)
    if values.size == 0:
        return values.copy()

    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if np.isnan(std) or std == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def is_flat_terrain(profile: np.ndarray, threshold_m: float = 15.0) -> bool:
    """Return True when the terrain variation is too small for reliable matching."""

    values = np.asarray(profile, dtype=float)
    if values.size == 0:
        return True
    return float(np.nanstd(values)) < threshold_m


def _smooth_profile(values: np.ndarray, window_length: int = 5) -> np.ndarray:
    """Smooth a profile before derivative estimation."""

    profile = np.asarray(values, dtype=float)
    if profile.size < 5:
        return profile.copy()

    candidate = min(window_length, profile.size if profile.size % 2 == 1 else profile.size - 1)
    if candidate < 5:
        return profile.copy()
    return savgol_filter(profile, window_length=candidate, polyorder=2, mode="interp")


def extract_terrain_features(profile: np.ndarray, step_m: float = 30.0) -> np.ndarray:
    """Build a normalized terrain feature vector from height, slope, and curvature."""

    values = np.asarray(profile, dtype=float)
    if values.ndim != 1:
        raise ValueError("profile must be a 1D array")
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    if values.size == 0:
        return np.empty((0,), dtype=float)

    smoothed = _smooth_profile(values)
    slope = np.gradient(smoothed, step_m)
    curvature = np.gradient(slope, step_m)

    height_norm = normalize_profile(smoothed) * 0.5
    slope_norm = normalize_profile(slope) * 0.3
    curvature_norm = normalize_profile(curvature) * 0.2
    return np.concatenate([height_norm, slope_norm, curvature_norm]).astype(float, copy=False)


@dataclass(frozen=True)
class CachedReferenceMatrix:
    """Cached matrix metadata."""

    center_lat: float
    center_lon: float
    azimuths: tuple[float, ...]
    matrix: np.ndarray


class ReferenceMatrix(np.ndarray):
    """ndarray with attached azimuth grid metadata."""

    azimuths_deg: np.ndarray

    def __new__(cls, data: np.ndarray, azimuths_deg: np.ndarray) -> "ReferenceMatrix":
        obj = np.asarray(data, dtype=float).view(cls)
        obj.azimuths_deg = np.asarray(azimuths_deg, dtype=float).copy()
        return obj

    def __array_finalize__(self, obj: object | None) -> None:
        if obj is None:
            return
        self.azimuths_deg = getattr(obj, "azimuths_deg", np.empty((0,), dtype=float))


class ProfileExtractor:
    """Build reference terrain profiles for all azimuths around a center point."""

    def __init__(
        self,
        dem: DEMLoader,
        profile_length_m: float,
        step_m: float = 30.0,
        cache_reuse_m: float | None = None,
    ) -> None:
        if profile_length_m <= 0:
            raise ValueError("profile_length_m must be positive")
        if step_m <= 0:
            raise ValueError("step_m must be positive")

        self.dem = dem
        self.profile_length_m = float(profile_length_m)
        self.step_m = float(step_m)
        self.cache_reuse_m = float(cache_reuse_m if cache_reuse_m is not None else (0.25 * self.step_m))
        self.center_lat: float | None = None
        self.center_lon: float | None = None
        self._cached: CachedReferenceMatrix | None = None

    def build_reference_matrix(
        self,
        center_lat: float,
        center_lon: float,
        azimuths: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a reference profile matrix for all requested azimuths."""

        if azimuths is None:
            azimuths = np.arange(0.0, 360.0, 1.0)
        azimuth_array = np.asarray(azimuths, dtype=float)
        azimuth_key = tuple(float(value) for value in azimuth_array.tolist())
        if self._can_reuse_cache(center_lat, center_lon, azimuth_key):
            LOGGER.info(
                "Reusing cached reference matrix for center=(%.6f, %.6f)",
                center_lat,
                center_lon,
            )
            assert self._cached is not None
            self.update_center(center_lat, center_lon)
            return ReferenceMatrix(self._cached.matrix.copy(), np.asarray(self._cached.azimuths, dtype=float))

        started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            profiles = list(
                executor.map(
                    lambda azimuth: self.dem.get_profile_along_azimuth(
                        lat=center_lat,
                        lon=center_lon,
                        azimuth_deg=float(azimuth),
                        distance_m=self.profile_length_m,
                        step_m=self.step_m,
                    ),
                    azimuth_array,
                )
            )

        matrix = np.vstack(profiles).astype(float, copy=False)
        self._cached = CachedReferenceMatrix(
            center_lat=center_lat,
            center_lon=center_lon,
            azimuths=azimuth_key,
            matrix=matrix.copy(),
        )
        self.update_center(center_lat, center_lon)
        elapsed = time.perf_counter() - started_at
        LOGGER.info(
            "Built reference matrix shape=%s for center=(%.6f, %.6f) in %.3fs",
            matrix.shape,
            center_lat,
            center_lon,
            elapsed,
        )
        return ReferenceMatrix(matrix, azimuth_array)

    def update_center(self, lat: float, lon: float) -> None:
        """Update the currently tracked center location."""

        self.center_lat = float(lat)
        self.center_lon = float(lon)

    def _can_reuse_cache(
        self, center_lat: float, center_lon: float, azimuth_key: tuple[float, ...]
    ) -> bool:
        if self._cached is None:
            return False
        if self._cached.azimuths != azimuth_key:
            return False
        _, _, distance_m = WGS84_GEOD.inv(
            self._cached.center_lon,
            self._cached.center_lat,
            center_lon,
            center_lat,
        )
        return distance_m < self.cache_reuse_m
