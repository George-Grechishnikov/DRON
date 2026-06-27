from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

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


def test_get_elevation_tolerates_boundary_rounding() -> None:
    memfile = _build_memory_dem()
    try:
        with DEMLoader(memfile.name) as dem:
            value = dem.get_elevation(49.500001, 11.000001)
    finally:
        memfile.close()

    assert np.isfinite(value)
