"""Reference terrain profile extraction for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pyproj import Geod

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


@dataclass(frozen=True)
class CachedReferenceMatrix:
    """Cached matrix metadata."""

    center_lat: float
    center_lon: float
    azimuths: tuple[float, ...]
    matrix: np.ndarray


class ProfileExtractor:
    """Build reference terrain profiles for all azimuths around a center point."""

    def __init__(self, dem: DEMLoader, profile_length_m: float, step_m: float = 30.0) -> None:
        if profile_length_m <= 0:
            raise ValueError("profile_length_m must be positive")
        if step_m <= 0:
            raise ValueError("step_m must be positive")

        self.dem = dem
        self.profile_length_m = float(profile_length_m)
        self.step_m = float(step_m)
        self.center_lat: float | None = None
        self.center_lon: float | None = None
        self._cached: CachedReferenceMatrix | None = None

    def build_reference_matrix(
        self,
        center_lat: float,
        center_lon: float,
        azimuths: np.ndarray = np.arange(0, 360, 1.0),
    ) -> np.ndarray:
        """Build a reference profile matrix for all requested azimuths."""

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
            return self._cached.matrix.copy()

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
        return matrix

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
        return distance_m < 200.0
