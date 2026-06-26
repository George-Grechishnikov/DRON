from __future__ import annotations

import numpy as np

from profile_extractor import ProfileExtractor, is_flat_terrain, normalize_profile


class SyntheticDEMLoader:
    def get_profile_along_azimuth(
        self,
        lat: float,
        lon: float,
        azimuth_deg: float,
        distance_m: float,
        step_m: float = 30.0,
    ) -> np.ndarray:
        length = int(np.floor(distance_m / step_m)) + 1
        x = np.linspace(0.0, distance_m, length)
        phase = np.deg2rad(azimuth_deg)
        return 200.0 + 40.0 * np.sin((x / 300.0) + phase) + 0.1 * azimuth_deg


def test_build_reference_matrix_shape_and_cache() -> None:
    dem = SyntheticDEMLoader()
    extractor = ProfileExtractor(dem=dem, profile_length_m=900.0, step_m=30.0)

    azimuths = np.array([0.0, 45.0, 90.0, 180.0])
    matrix1 = extractor.build_reference_matrix(60.5, 90.3, azimuths=azimuths)
    matrix2 = extractor.build_reference_matrix(60.5005, 90.3005, azimuths=azimuths)

    assert matrix1.shape == (4, 31)
    assert np.allclose(matrix1, matrix2)


def test_normalize_profile_has_zero_mean_and_unit_std() -> None:
    profile = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    normalized = normalize_profile(profile)

    assert np.isclose(np.mean(normalized), 0.0, atol=1e-9)
    assert np.isclose(np.std(normalized), 1.0, atol=1e-9)


def test_is_flat_terrain_detects_low_variance() -> None:
    flat = np.array([100.0, 101.0, 100.5, 99.5, 100.0], dtype=float)
    rough = np.array([100.0, 130.0, 90.0, 145.0, 80.0], dtype=float)

    assert is_flat_terrain(flat, threshold_m=15.0) is True
    assert is_flat_terrain(rough, threshold_m=15.0) is False
