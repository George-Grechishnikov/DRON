"""DEM loading utilities for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from pyproj import Geod
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from scipy.ndimage import map_coordinates


LOGGER = logging.getLogger(__name__)
WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class DEMMetadata:
    """Metadata summary for an opened DEM."""

    path: str
    width: int
    height: int
    crs: str
    bounds: tuple[float, float, float, float]
    nodata: float | None
    dtype: str
    transform: Affine


class DEMLoader:
    """Load a DEM and expose helpers for elevation and profile sampling."""

    def __init__(self, path: str | Path, max_cache_size: int = 20) -> None:
        self.path = str(path)
        self.max_cache_size = max_cache_size
        self._dataset = rasterio.open(self.path)
        self._owns_vrt = False
        self._raster = self._dataset
        if self._dataset.crs is None:
            raise ValueError("DEM has no CRS defined")
        if self._dataset.crs.to_epsg() != 4326:
            self._raster = WarpedVRT(
                self._dataset,
                crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            self._owns_vrt = True

        self.metadata = DEMMetadata(
            path=str(self.path),
            width=self._raster.width,
            height=self._raster.height,
            crs=str(self._raster.crs),
            bounds=(
                float(self._raster.bounds.left),
                float(self._raster.bounds.bottom),
                float(self._raster.bounds.right),
                float(self._raster.bounds.top),
            ),
            nodata=None if self._raster.nodata is None else float(self._raster.nodata),
            dtype=str(self._raster.dtypes[0]),
            transform=self._raster.transform,
        )
        self._patch_cache: OrderedDict[tuple[float, float, float], tuple[np.ndarray, Affine]] = OrderedDict()
        local_path = Path(self.path)
        file_size = local_path.stat().st_size if local_path.exists() else 0
        LOGGER.info(
            "Loaded DEM %s (%d bytes), bbox=%s, crs=%s",
            self.path,
            file_size,
            self.metadata.bounds,
            self.metadata.crs,
        )

    def close(self) -> None:
        """Close underlying raster resources."""

        if self._owns_vrt:
            self._raster.close()
            self._owns_vrt = False
        self._dataset.close()

    def __enter__(self) -> "DEMLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_elevation(self, lat: float, lon: float) -> float:
        """Return bilinearly interpolated elevation at lat/lon."""

        self._validate_bounds(lat, lon)
        row, col = self._fractional_index(lat, lon)
        data = self._read_window_array(row, col)
        sampled = map_coordinates(data, [[row], [col]], order=1, mode="nearest")[0]
        if np.isnan(sampled):
            return float("nan")
        return float(sampled)

    def get_patch(self, lat: float, lon: float, radius_m: float = 5000) -> tuple[np.ndarray, Affine]:
        """Return a DEM patch centered around lat/lon with the given metric radius."""

        key = self._patch_key(lat, lon, radius_m)
        cached = self._patch_cache.get(key)
        if cached is not None:
            self._patch_cache.move_to_end(key)
            return cached[0].copy(), cached[1]

        self._validate_bounds(lat, lon)
        north_lon, north_lat, _ = WGS84_GEOD.fwd(lon, lat, 0.0, radius_m)
        east_lon, east_lat, _ = WGS84_GEOD.fwd(lon, lat, 90.0, radius_m)
        south_lon, south_lat, _ = WGS84_GEOD.fwd(lon, lat, 180.0, radius_m)
        west_lon, west_lat, _ = WGS84_GEOD.fwd(lon, lat, 270.0, radius_m)

        min_lon = min(west_lon, east_lon)
        max_lon = max(west_lon, east_lon)
        min_lat = min(south_lat, north_lat)
        max_lat = max(south_lat, north_lat)
        self._validate_bounds(min_lat, min_lon)
        self._validate_bounds(max_lat, max_lon)

        window = windows.from_bounds(
            left=min_lon,
            bottom=min_lat,
            right=max_lon,
            top=max_lat,
            transform=self._raster.transform,
        )
        window = window.round_offsets().round_lengths()
        patch = self._raster.read(1, window=window, masked=True).filled(np.nan).astype(np.float64)
        if self.metadata.nodata is not None:
            patch[np.isclose(patch, self.metadata.nodata)] = np.nan
        transform = self._raster.window_transform(window)

        self._patch_cache[key] = (patch, transform)
        self._patch_cache.move_to_end(key)
        while len(self._patch_cache) > self.max_cache_size:
            self._patch_cache.popitem(last=False)
        return patch.copy(), transform

    def get_profile_along_azimuth(
        self,
        lat: float,
        lon: float,
        azimuth_deg: float,
        distance_m: float,
        step_m: float = 30.0,
    ) -> np.ndarray:
        """Return a sampled terrain profile along a geodesic ray."""

        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if step_m <= 0:
            raise ValueError("step_m must be positive")

        steps = int(math.floor(distance_m / step_m)) + 1
        distances = np.linspace(0.0, distance_m, steps)
        lons, lats, _ = WGS84_GEOD.fwd(
            np.full(steps, lon, dtype=float),
            np.full(steps, lat, dtype=float),
            np.full(steps, azimuth_deg, dtype=float),
            distances,
        )
        values = [self.get_elevation(point_lat, point_lon) for point_lon, point_lat in zip(lons, lats)]
        return np.asarray(values, dtype=float)

    def preload_route(
        self, waypoints: list[tuple[float, float]], radius_m: float = 5000
    ) -> None:
        """Preload DEM patches around route waypoints."""

        if not waypoints:
            return
        with ThreadPoolExecutor(max_workers=min(4, len(waypoints))) as executor:
            futures = [
                executor.submit(self.get_patch, lat, lon, radius_m)
                for lat, lon in waypoints
            ]
            for future in futures:
                future.result()

    def _fractional_index(self, lat: float, lon: float) -> tuple[float, float]:
        col, row = (~self._raster.transform) * (lon, lat)
        return float(row), float(col)

    def _read_window_array(self, row: float, col: float) -> np.ndarray:
        data = self._raster.read(1, masked=True).filled(np.nan).astype(np.float64)
        if self.metadata.nodata is not None:
            data[np.isclose(data, self.metadata.nodata)] = np.nan
        return data

    def _validate_bounds(self, lat: float, lon: float) -> None:
        left, bottom, right, top = self.metadata.bounds
        if not (left <= lon <= right and bottom <= lat <= top):
            raise ValueError(
                f"Coordinate ({lat:.6f}, {lon:.6f}) is outside DEM bounds "
                f"({bottom:.6f}, {left:.6f}) - ({top:.6f}, {right:.6f})"
            )

    @staticmethod
    def _patch_key(lat: float, lon: float, radius_m: float) -> tuple[float, float, float]:
        return (round(lat, 3), round(lon, 3), round(radius_m, 1))
