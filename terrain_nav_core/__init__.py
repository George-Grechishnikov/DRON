"""Optional compiled terrain correlation backend."""

from __future__ import annotations

AVAILABLE = False

try:
    from ._terrain_nav_core import compute_hybrid_heatmaps

    AVAILABLE = True
except ImportError:
    pass
