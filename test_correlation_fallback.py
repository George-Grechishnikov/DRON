from __future__ import annotations

import numpy as np
import pytest

from correlation_fallback import cpp_backend_available, find_best_match


def test_python_correlation_fallback() -> None:
    reference_profiles = np.array(
        [
            [0.0, 0.5, -0.5, 0.25, -0.25, 0.0],
            [0.0, 1.0, 2.0, 1.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    measured_profile = np.array([1.0, 2.0, 1.0, 0.0], dtype=float)

    result = find_best_match(measured_profile, reference_profiles, max_offset_steps=2)

    assert result["best_azimuth_idx"] == 1
    assert result["best_offset_idx"] == 1
    assert result["best_score"] > 0.99
    assert result["heatmap"].shape == (2, 3)


@pytest.mark.skipif(not cpp_backend_available(), reason="C++ backend is not built")
def test_cpp_correlation_if_available() -> None:
    reference_profiles = np.array(
        [
            [0.0, 0.5, -0.5, 0.25, -0.25, 0.0],
            [0.0, 1.0, 2.0, 1.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    measured_profile = np.array([1.0, 2.0, 1.0, 0.0], dtype=float)

    result = find_best_match(measured_profile, reference_profiles, max_offset_steps=2)

    assert result["best_azimuth_idx"] == 1
    assert result["best_offset_idx"] == 1
