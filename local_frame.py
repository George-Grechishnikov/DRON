"""Local ENU frame helpers for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math

import numpy as np
import pyproj


def _normalize_enu_input(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size not in {2, 3}:
        raise ValueError("ENU input must be a 1D vector of length 2 or 3")
    return vector


class LocalFrame:
    """Local east-north-up frame around a geodetic origin."""

    def __init__(self, lat0: float, lon0: float, alt0_m: float = 0.0) -> None:
        self._geod = pyproj.Geod(ellps="WGS84")
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.alt0_m = float(alt0_m)

    def to_enu(
        self,
        lat: float,
        lon: float,
        alt_m: float | None = None,
    ) -> np.ndarray:
        """Convert geodetic coordinates into local ENU meters."""

        azimuth_deg, _, distance_m = self._geod.inv(self.lon0, self.lat0, float(lon), float(lat))
        azimuth_rad = math.radians(float(azimuth_deg) % 360.0)
        east_m = distance_m * math.sin(azimuth_rad)
        north_m = distance_m * math.cos(azimuth_rad)
        if alt_m is None:
            return np.array([east_m, north_m], dtype=float)
        up_m = float(alt_m) - self.alt0_m
        return np.array([east_m, north_m, up_m], dtype=float)

    def to_geodetic(self, enu_m: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, float] | tuple[float, float, float]:
        """Convert local ENU coordinates back to geodetic coordinates."""

        vector = _normalize_enu_input(enu_m)
        east_m = float(vector[0])
        north_m = float(vector[1])
        distance_m = math.hypot(east_m, north_m)
        azimuth_deg = (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0
        lon, lat, _ = self._geod.fwd(self.lon0, self.lat0, azimuth_deg, distance_m)
        if vector.size == 2:
            return (float(lat), float(lon))
        return (float(lat), float(lon), self.alt0_m + float(vector[2]))

    def rebase(
        self,
        lat0: float,
        lon0: float,
        alt0_m: float | None = None,
        points_enu: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Move the origin and optionally remap ENU points into the new frame."""

        remapped: np.ndarray | None = None
        if points_enu is not None:
            points = np.asarray(points_enu, dtype=float)
            original_shape = points.shape
            if points.ndim == 1:
                points = points.reshape(1, -1)
            remapped_points = []
            for point in points:
                geodetic = self.to_geodetic(point)
                if len(geodetic) == 2:
                    remapped_points.append([float(v) for v in geodetic])
                else:
                    remapped_points.append([float(v) for v in geodetic])
            old_lat0 = self.lat0
            old_lon0 = self.lon0
            old_alt0 = self.alt0_m
            self.lat0 = float(lat0)
            self.lon0 = float(lon0)
            if alt0_m is not None:
                self.alt0_m = float(alt0_m)
            remapped_vectors = []
            for point_geodetic in remapped_points:
                if len(point_geodetic) == 2:
                    remapped_vectors.append(self.to_enu(point_geodetic[0], point_geodetic[1]))
                else:
                    remapped_vectors.append(self.to_enu(point_geodetic[0], point_geodetic[1], point_geodetic[2]))
            remapped = np.asarray(remapped_vectors, dtype=float).reshape(original_shape)
            self.lat0 = old_lat0
            self.lon0 = old_lon0
            self.alt0_m = old_alt0

        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        if alt0_m is not None:
            self.alt0_m = float(alt0_m)
        return remapped
