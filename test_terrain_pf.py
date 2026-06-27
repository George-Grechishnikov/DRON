from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from dem_loader import DEMLoader
from terrain_pf import TerrainParticleFilter


def _write_pf_dem(path: Path, *, flat: bool = False) -> None:
    rows, cols = 300, 300
    y, x = np.indices((rows, cols), dtype=float)
    if flat:
        terrain = np.full((rows, cols), 100.0, dtype="float32")
    else:
        terrain = (
            200.0
            + 30.0 * np.sin(x / 12.0)
            + 25.0 * np.cos(y / 17.0)
            + 0.3 * x
            - 0.2 * y
        ).astype("float32")
    transform = from_origin(89.8, 60.8, 0.001, 0.001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(terrain, 1)


def test_particle_filter_converges_from_kilometer_scale_error(tmp_path: Path) -> None:
    dem_path = tmp_path / "pf_dem.tif"
    _write_pf_dem(dem_path, flat=False)

    with DEMLoader(dem_path) as dem:
        pf = TerrainParticleFilter(dem, n_particles=2500, meas_sigma_m=6.0, resample_threshold=0.6)
        center_lat, center_lon = dem.get_center()
        pf.initialize_global(center_lat, center_lon, radius_m=2500.0)

        true_enu = np.array([800.0, -400.0], dtype=float)
        offsets = np.stack(
            [
                np.linspace(0.0, 900.0, 40),
                100.0 * np.sin(np.linspace(0.0, np.pi / 2.0, 40)),
            ],
            axis=1,
        )
        path_points = true_enu[np.newaxis, :] + offsets
        geodetic = np.array([pf.frame.to_geodetic(point) for point in path_points], dtype=float)
        measured = dem._sample_points(geodetic[:, 0], geodetic[:, 1])

        for _ in range(5):
            result = pf.update(measured, offsets)

        error_m = float(np.linalg.norm(result.p_enu_m - true_enu))
        assert error_m < 500.0


def test_particle_filter_expands_uncertainty_on_flat_terrain(tmp_path: Path) -> None:
    dem_path = tmp_path / "flat_dem.tif"
    _write_pf_dem(dem_path, flat=True)

    with DEMLoader(dem_path) as dem:
        pf = TerrainParticleFilter(dem, n_particles=400, meas_sigma_m=8.0, resample_threshold=0.6)
        center_lat, center_lon = dem.get_center()
        pf.initialize_global(center_lat, center_lon, radius_m=2000.0)
        offsets = np.stack([np.linspace(0.0, 300.0, 20), np.zeros(20)], axis=1)
        measured = np.full((20,), 100.0, dtype=float)

        result = pf.update(measured, offsets)

        assert result.converged is False
        assert np.trace(result.cov_2x2) > 10_000.0


def test_particle_filter_resamples_and_normalizes_weights(tmp_path: Path) -> None:
    dem_path = tmp_path / "pf_dem.tif"
    _write_pf_dem(dem_path, flat=False)

    with DEMLoader(dem_path) as dem:
        pf = TerrainParticleFilter(dem, n_particles=100, meas_sigma_m=8.0, resample_threshold=0.9)
        center_lat, center_lon = dem.get_center()
        pf.initialize_global(center_lat, center_lon, radius_m=100.0)
        pf.weights = np.zeros((pf.n_particles,), dtype=float)
        pf.weights[0] = 0.99
        pf.weights[1:] = 0.01 / (pf.n_particles - 1)

        pf._resample_if_needed()

        assert np.isclose(np.sum(pf.weights), 1.0, atol=1e-9)
        assert np.allclose(pf.weights, np.full((pf.n_particles,), 1.0 / pf.n_particles), atol=1e-9)
