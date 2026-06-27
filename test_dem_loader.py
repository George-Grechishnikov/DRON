from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import dem_loader
from dem_loader import DEMLoader


def _build_memory_dem() -> MemoryFile:
    data = np.fromfunction(lambda r, c: 100.0 + r * 2.0 + c * 3.0, (100, 100), dtype=float)
    transform = from_origin(10.0, 50.0, 0.01, 0.01)
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data.astype("float32"), 1)
    return memfile


def test_get_elevation_returns_interpolated_value() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            value = dem.get_elevation(49.505, 10.505)
    finally:
        memfile.close()

    assert 348.0 < value < 352.0


def test_get_profile_along_azimuth_returns_1d_profile() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            profile = dem.get_profile_along_azimuth(
                lat=49.5,
                lon=10.5,
                azimuth_deg=90.0,
                distance_m=1000.0,
                step_m=250.0,
            )
    finally:
        memfile.close()

    assert profile.ndim == 1
    assert profile.shape == (5,)
    assert np.all(np.diff(profile) > 0)
    assert np.isfinite(profile[0])


def test_sample_points_matches_scalar_sampling_for_multiple_points() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            lats = np.array([49.505, 49.495, 49.485], dtype=float)
            lons = np.array([10.505, 10.515, 10.525], dtype=float)

            batch_values = dem._sample_points(lats, lons)
            scalar_values = np.array(
                [dem.get_elevation(lat, lon) for lat, lon in zip(lats, lons)],
                dtype=float,
            )
    finally:
        memfile.close()

    assert batch_values.shape == (3,)
    assert np.allclose(batch_values, scalar_values, atol=1e-6, equal_nan=True)


def test_get_profile_along_azimuth_matches_batch_sampling() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            distance_m = 1000.0
            step_m = 250.0
            steps = int(distance_m / step_m) + 1
            distances = np.linspace(0.0, distance_m, steps)
            lons, lats, _ = dem_loader.WGS84_GEOD.fwd(
                np.full(steps, 10.5, dtype=float),
                np.full(steps, 49.5, dtype=float),
                np.full(steps, 90.0, dtype=float),
                distances,
            )

            profile = dem.get_profile_along_azimuth(
                lat=49.5,
                lon=10.5,
                azimuth_deg=90.0,
                distance_m=distance_m,
                step_m=step_m,
            )
            batch_values = dem._sample_points(
                np.asarray(lats, dtype=float),
                np.asarray(lons, dtype=float),
            )
    finally:
        memfile.close()

    assert np.allclose(profile, batch_values, atol=1e-6, equal_nan=True)


def test_get_patch_returns_array_and_transform() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            patch, transform = dem.get_patch(49.5, 10.5, radius_m=500.0)
    finally:
        memfile.close()

    assert patch.ndim == 2
    assert patch.size > 0
    assert hasattr(transform, "a")


def test_get_profile_along_azimuth_starts_with_local_elevation() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            center_lat = 49.5
            center_lon = 10.5
            elevation = dem.get_elevation(center_lat, center_lon)
            profile = dem.get_profile_along_azimuth(
                lat=center_lat,
                lon=center_lon,
                azimuth_deg=45.0,
                distance_m=300.0,
                step_m=30.0,
            )
    finally:
        memfile.close()

    assert profile.size >= 2
    assert np.isfinite(profile[0])
    assert abs(float(profile[0]) - float(elevation)) < 1e-6
