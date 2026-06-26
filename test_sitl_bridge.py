from __future__ import annotations

import math
from dataclasses import dataclass

from constants import FIXED_BARO_ALTITUDE_M
from sitl_bridge import SITLBridge


class StubDEM:
    def __init__(self, elevation_m: float) -> None:
        self.elevation_m = elevation_m

    def get_elevation(self, lat: float, lon: float) -> float:
        del lat, lon
        return self.elevation_m


@dataclass
class MockGlobalPositionInt:
    lat: int
    lon: int
    alt: int
    vx: int
    vy: int
    hdg: int
    time_boot_ms: int

    def get_type(self) -> str:
        return "GLOBAL_POSITION_INT"


class TimeStub:
    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def test_bridge_builds_unified_sample_from_global_position_int() -> None:
    bridge = SITLBridge(
        "mock://sitl",
        StubDEM(elevation_m=105.0),
        mock_messages=[
            MockGlobalPositionInt(
                lat=605000000,
                lon=903000000,
                alt=150000,
                vx=300,
                vy=400,
                hdg=4500,
                time_boot_ms=12000,
            )
        ],
    )

    sample = bridge.read_sample()

    assert sample is not None
    assert sample.as_dict()["gnss_available"] is True
    assert math.isclose(sample.lat, 60.5)
    assert math.isclose(sample.lon, 90.3)
    assert math.isclose(sample.alt_msl, FIXED_BARO_ALTITUDE_M)
    assert math.isclose(sample.ground_speed_mps, 5.0)
    assert math.isclose(sample.heading_deg, 45.0)
    assert math.isclose(sample.radar_alt_m, FIXED_BARO_ALTITUDE_M - 105.0)
    assert math.isclose(sample.timestamp, 12.0)


def test_bridge_drops_gnss_after_configured_interval() -> None:
    time_stub = TimeStub(100.0, 107.0)
    bridge = SITLBridge(
        "mock://sitl",
        StubDEM(elevation_m=100.0),
        gnss_drop_after_s=5.0,
        mock_messages=[
            MockGlobalPositionInt(
                lat=605000000,
                lon=903000000,
                alt=150000,
                vx=100,
                vy=0,
                hdg=9000,
                time_boot_ms=1000,
            ),
            MockGlobalPositionInt(
                lat=605000100,
                lon=903000100,
                alt=150000,
                vx=100,
                vy=0,
                hdg=9000,
                time_boot_ms=2000,
            ),
        ],
        time_fn=time_stub,
    )

    first = bridge.read_sample()
    second = bridge.read_sample()

    assert first is not None
    assert second is not None
    assert first.gnss_available is True
    assert second.gnss_available is False


def test_bridge_can_recover_gnss_after_drop_window() -> None:
    time_stub = TimeStub(10.0, 18.0, 26.0)
    bridge = SITLBridge(
        "mock://sitl",
        StubDEM(elevation_m=100.0),
        gnss_drop_after_s=5.0,
        gnss_recover_after_s=15.0,
        mock_messages=[
            MockGlobalPositionInt(605000000, 903000000, 150000, 0, 0, 0, 1000),
            MockGlobalPositionInt(605000000, 903000000, 150000, 0, 0, 0, 2000),
            MockGlobalPositionInt(605000000, 903000000, 150000, 0, 0, 0, 3000),
        ],
        time_fn=time_stub,
    )

    first = bridge.read_sample()
    second = bridge.read_sample()
    third = bridge.read_sample()

    assert first is not None and first.gnss_available is True
    assert second is not None and second.gnss_available is False
    assert third is not None and third.gnss_available is True


def test_bridge_adapts_sample_to_existing_nmea_frame() -> None:
    bridge = SITLBridge(
        "mock://sitl",
        StubDEM(elevation_m=149.2),
        mock_messages=[
            MockGlobalPositionInt(
                lat=605000000,
                lon=903000000,
                alt=150000,
                vx=0,
                vy=0,
                hdg=UINT16_SENTINEL,
                time_boot_ms=1500,
            )
        ],
    )

    sample = bridge.read_sample()
    assert sample is not None

    frame = bridge.sample_to_nmea_frame(sample)

    assert frame.valid is True
    assert math.isclose(frame.radar_alt_m, FIXED_BARO_ALTITUDE_M - 149.2, abs_tol=0.11)


def test_bridge_manual_gnss_override_takes_priority() -> None:
    bridge = SITLBridge(
        "mock://sitl",
        StubDEM(elevation_m=100.0),
        gnss_drop_after_s=0.0,
        mock_messages=[
            MockGlobalPositionInt(605000000, 903000000, 150000, 0, 0, 0, 1000),
            MockGlobalPositionInt(605000000, 903000000, 150000, 0, 0, 0, 2000),
        ],
    )

    bridge.set_gnss_enabled(True)
    first = bridge.read_sample()
    bridge.set_gnss_enabled(False)
    second = bridge.read_sample()

    assert first is not None and first.gnss_available is True
    assert second is not None and second.gnss_available is False


UINT16_SENTINEL = 65535
