"""Optional C++-accelerated terrain correlation helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import terrain_nav_core
except ImportError:
    terrain_nav_core = None


def cpp_backend_available() -> bool:
    """Return True when the optional compiled backend is importable."""

    return bool(terrain_nav_core is not None and getattr(terrain_nav_core, "AVAILABLE", False))


def compute_hybrid_heatmaps(
    measured_profile: np.ndarray,
    reference_profiles: np.ndarray,
    *,
    usable_offsets: int,
    metric: str,
    alpha: float,
    beta: float,
    msd_scale_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute correlation heatmaps through the native backend when available."""

    if not cpp_backend_available():
        return None
    try:
        result: dict[str, Any] = terrain_nav_core.compute_hybrid_heatmaps(
            np.ascontiguousarray(measured_profile, dtype=np.float64),
            np.ascontiguousarray(reference_profiles, dtype=np.float64),
            int(usable_offsets),
            float(alpha),
            float(beta),
            float(msd_scale_m2),
        )
    except Exception:
        LOGGER.exception("C++ correlation backend failed; falling back to Python")
        return None

    ncc = np.asarray(result["ncc"], dtype=float)
    msd = np.asarray(result["msd"], dtype=float)
    if metric == "ncc":
        combined = ncc.copy()
    elif metric == "msd":
        combined = msd.copy()
    else:
        combined = np.asarray(result["combined"], dtype=float)
    return combined, ncc, msd
