"""Fallback demo-mode dataset generation for the terrain-navigation UI.

This module is intentionally isolated from the real pipeline so it can be swapped
out later without rewriting the frontend. It can be wired into backend loading
when `demo_mode: true` is enabled in config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoPoint:
    timestamp_s: float
    lat: float
    lon: float
    alt_msl: float
    radar_alt_m: float
    heading_deg: float
    speed_mps: float
    gnss_available: bool
    nav_mode: str


def generate_demo_points(count: int = 1200, sample_rate_hz: float = 5.0) -> list[DemoPoint]:
    points: list[DemoPoint] = []
    base_lat = 60.52
    base_lon = 90.37
    for index in range(count):
        timestamp_s = index / sample_rate_hz
        lat = base_lat + index * 0.00002 + math.sin(index / 40.0) * 0.00015
        lon = base_lon + index * 0.00003 + math.cos(index / 50.0) * 0.00015
        terrain_h = 1280.0 + math.sin(index / 18.0) * 90.0 + math.cos(index / 37.0) * 55.0
        alt_msl = 1500.0
        radar_alt_m = alt_msl - terrain_h
        gnss_available = timestamp_s < 180.0 or timestamp_s > 950.0
        nav_mode = "GNSS" if gnss_available else ("GNSS_LOST" if timestamp_s < 240.0 else "TERRAIN_NAV")
        heading_deg = (128.0 + math.sin(index / 35.0) * 11.0) % 360.0
        speed_mps = 21.0 + math.sin(index / 22.0) * 1.4
        points.append(
            DemoPoint(
                timestamp_s=timestamp_s,
                lat=lat,
                lon=lon,
                alt_msl=alt_msl,
                radar_alt_m=radar_alt_m,
                heading_deg=heading_deg,
                speed_mps=speed_mps,
                gnss_available=gnss_available,
                nav_mode=nav_mode,
            )
        )
    return points
