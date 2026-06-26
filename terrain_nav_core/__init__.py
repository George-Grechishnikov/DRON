"""Optional compiled terrain correlation backend."""

from __future__ import annotations

AVAILABLE = False

try:
    from ._terrain_nav_core import correlate_profiles, find_best_match, normalized_correlation

    AVAILABLE = True
except ImportError:
    pass
