"""Optional C++-accelerated terrain correlation with NumPy fallback."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import correlate

try:
    import terrain_nav_core
except ImportError:
    terrain_nav_core = None


def cpp_backend_available() -> bool:
    """Return True when the optional compiled backend is importable."""

    return bool(terrain_nav_core is not None and getattr(terrain_nav_core, "AVAILABLE", False))


def normalized_correlation(measured_profile: np.ndarray, reference_profile: np.ndarray) -> float:
    """Compute normalized correlation for two same-length profiles."""

    measured = np.asarray(measured_profile, dtype=float)
    reference = np.asarray(reference_profile, dtype=float)
    if measured.shape != reference.shape:
        raise ValueError("Profiles must have the same shape")
    measured_std = float(np.nanstd(measured))
    reference_std = float(np.nanstd(reference))
    if measured_std == 0.0 or reference_std == 0.0 or np.isnan(measured_std) or np.isnan(reference_std):
        return 0.0
    centered_measured = measured - float(np.nanmean(measured))
    centered_reference = reference - float(np.nanmean(reference))
    numerator = float(np.dot(centered_measured, centered_reference))
    denominator = measured.size * measured_std * reference_std
    return 0.0 if denominator <= 0.0 else numerator / denominator


def _python_find_best_match(
    measured_profile: np.ndarray,
    reference_profiles: np.ndarray,
    max_offset_steps: int | None = None,
) -> dict[str, Any]:
    measured = np.asarray(measured_profile, dtype=float)
    references = np.asarray(reference_profiles, dtype=float)
    if measured.ndim != 1:
        raise ValueError("measured_profile must be 1D")
    if references.ndim != 2:
        raise ValueError("reference_profiles must be 2D")
    if references.shape[1] < measured.shape[0]:
        raise ValueError("reference_profiles width must be >= measured_profile length")

    total_valid_offsets = references.shape[1] - measured.shape[0] + 1
    usable_offsets = total_valid_offsets if max_offset_steps is None else min(total_valid_offsets, max_offset_steps + 1)
    if usable_offsets <= 0:
        raise ValueError("No valid offsets available")

    measured_mean = float(np.nanmean(measured))
    measured_std = float(np.nanstd(measured))
    if measured_std == 0.0 or np.isnan(measured_std):
        raise ValueError("measured_profile must have non-zero variance")

    centered_measured = measured - measured_mean
    window_length = measured.size
    heatmap = np.empty((references.shape[0], usable_offsets), dtype=float)

    for row_index, ref_profile in enumerate(references):
        reference = np.asarray(ref_profile, dtype=float)
        numerator = correlate(reference, centered_measured, mode="valid")[:usable_offsets]
        sums = np.convolve(reference, np.ones(window_length, dtype=float), mode="valid")[:usable_offsets]
        sums_sq = np.convolve(reference * reference, np.ones(window_length, dtype=float), mode="valid")[:usable_offsets]
        reference_mean = sums / window_length
        reference_var = np.maximum((sums_sq / window_length) - (reference_mean * reference_mean), 0.0)
        reference_std = np.sqrt(reference_var)
        denominator = window_length * measured_std * reference_std
        heatmap[row_index, :] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )

    best_flat_index = int(np.nanargmax(heatmap))
    best_azimuth_idx, best_offset_idx = np.unravel_index(best_flat_index, heatmap.shape)
    best_score = float(heatmap[best_azimuth_idx, best_offset_idx])
    return {
        "best_azimuth_idx": int(best_azimuth_idx),
        "best_offset_idx": int(best_offset_idx),
        "best_score": best_score,
        "heatmap": heatmap,
    }


def correlate_profiles(
    measured_profile: np.ndarray,
    reference_profiles: np.ndarray,
    max_offset_steps: int | None = None,
) -> np.ndarray:
    """Return the heatmap of correlation values across azimuths and offsets."""

    return find_best_match(measured_profile, reference_profiles, max_offset_steps=max_offset_steps)["heatmap"]


def find_best_match(
    measured_profile: np.ndarray,
    reference_profiles: np.ndarray,
    max_offset_steps: int | None = None,
) -> dict[str, Any]:
    """Dispatch to the compiled backend when present, otherwise use NumPy."""

    if cpp_backend_available():
        result = terrain_nav_core.find_best_match(
            np.asarray(measured_profile, dtype=float),
            np.asarray(reference_profiles, dtype=float),
            -1 if max_offset_steps is None else int(max_offset_steps),
        )
        return {
            "best_azimuth_idx": int(result["best_azimuth_idx"]),
            "best_offset_idx": int(result["best_offset_idx"]),
            "best_score": float(result["best_score"]),
            "heatmap": np.asarray(result["heatmap"], dtype=float),
        }
    return _python_find_best_match(measured_profile, reference_profiles, max_offset_steps=max_offset_steps)
