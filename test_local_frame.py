from __future__ import annotations

import numpy as np

from local_frame import LocalFrame


def test_local_frame_round_trip_is_consistent() -> None:
    frame = LocalFrame(60.5, 90.3, alt0_m=1500.0)

    enu = frame.to_enu(60.5005, 90.3007, 1512.0)
    lat, lon, alt_m = frame.to_geodetic(enu)

    assert np.isclose(lat, 60.5005, atol=1e-6)
    assert np.isclose(lon, 90.3007, atol=1e-6)
    assert np.isclose(alt_m, 1512.0, atol=1e-6)


def test_local_frame_rebase_preserves_geodetic_position() -> None:
    frame = LocalFrame(60.5, 90.3, alt0_m=1500.0)
    point = frame.to_enu(60.501, 90.301, 1505.0)

    remapped = frame.rebase(60.5002, 90.3002, 1499.0, points_enu=point)
    assert remapped is not None
    lat, lon, alt_m = frame.to_geodetic(remapped)

    assert np.isclose(lat, 60.501, atol=1e-6)
    assert np.isclose(lon, 90.301, atol=1e-6)
    assert np.isclose(alt_m, 1505.0, atol=1e-6)
