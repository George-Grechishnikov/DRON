"""Measurement conversion layer for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from constants import FIXED_BARO_ALTITUDE_M
from nmea_parser import NMEAFrame


@dataclass(frozen=True)
class BaroSample:
    """One barometric altitude sample in MSL meters."""

    timestamp_s: float
    msl_m: float


@dataclass(frozen=True)
class TerrainProfile:
    """Absolute terrain profile reconstructed from radar AGL samples."""

    values_m: np.ndarray
    timestamps_s: np.ndarray
    valid_mask: np.ndarray
    baro_bias_m: float
    terrain_bias_m: float


class BaroTrack:
    """Time-aligned barometric channel with fixed-altitude fallback."""

    def __init__(self, samples: Sequence[BaroSample] | None = None, *, default_msl_m: float = FIXED_BARO_ALTITUDE_M) -> None:
        ordered = sorted(samples or [], key=lambda item: item.timestamp_s)
        self._samples = tuple(ordered)
        self.default_msl_m = float(default_msl_m)

    def msl_at(self, timestamp_s: float) -> float:
        """Return barometric MSL altitude at the requested timestamp."""

        if not self._samples:
            return self.default_msl_m
        if timestamp_s <= self._samples[0].timestamp_s:
            return self._samples[0].msl_m
        if timestamp_s >= self._samples[-1].timestamp_s:
            return self._samples[-1].msl_m

        for left, right in zip(self._samples[:-1], self._samples[1:]):
            if left.timestamp_s <= timestamp_s <= right.timestamp_s:
                dt = right.timestamp_s - left.timestamp_s
                if dt <= 1e-12:
                    return right.msl_m
                alpha = (timestamp_s - left.timestamp_s) / dt
                return float((1.0 - alpha) * left.msl_m + alpha * right.msl_m)
        return self.default_msl_m


def parse_nmea_timestamp_to_seconds(timestamp_utc: str) -> float:
    """Convert a GPGGA timestamp token hhmmss.sss into seconds since midnight."""

    token = (timestamp_utc or "").strip()
    if not token:
        return float("nan")
    try:
        compact = token.replace(".", "")
        if len(compact) < 6:
            return float(token)
        hours = int(token[0:2])
        minutes = int(token[2:4])
        seconds = float(token[4:])
    except ValueError:
        return float("nan")
    return float(hours * 3600 + minutes * 60 + seconds)


def agl_to_terrain(
    radar_agl_m: np.ndarray,
    baro_msl_m: np.ndarray,
    baro_bias_m: float = 0.0,
    terrain_bias_m: float = 0.0,
) -> np.ndarray:
    """Convert radar AGL measurements into absolute terrain heights."""

    radar = np.asarray(radar_agl_m, dtype=float)
    baro = np.asarray(baro_msl_m, dtype=float)
    return (baro - float(baro_bias_m)) - radar - float(terrain_bias_m)


def update_terrain_bias(prev_bias_m: float, residual_m: float, gain: float) -> float:
    """Update a slow additive terrain bias estimate via exponential correction."""

    if not (0.0 <= gain <= 1.0):
        raise ValueError("gain must be within [0, 1]")
    return float((1.0 - gain) * prev_bias_m + gain * residual_m)


def frames_to_terrain_profile(
    frames: Sequence[NMEAFrame],
    baro: BaroTrack,
    baro_bias_m: float = 0.0,
    terrain_bias_m: float = 0.0,
) -> TerrainProfile:
    """Single conversion point from NMEA radar frames into an absolute terrain profile."""

    if not frames:
        return TerrainProfile(
            values_m=np.empty((0,), dtype=float),
            timestamps_s=np.empty((0,), dtype=float),
            valid_mask=np.empty((0,), dtype=bool),
            baro_bias_m=float(baro_bias_m),
            terrain_bias_m=float(terrain_bias_m),
        )

    timestamps = np.array([parse_nmea_timestamp_to_seconds(frame.timestamp_utc) for frame in frames], dtype=float)
    fallback_times = np.arange(len(frames), dtype=float)
    timestamps = np.where(np.isfinite(timestamps), timestamps, fallback_times)
    radar_agl = np.array([frame.radar_alt_m for frame in frames], dtype=float)
    valid_mask = np.isfinite(radar_agl) & np.array([frame.valid for frame in frames], dtype=bool)
    baro_msl = np.array([baro.msl_at(timestamp) for timestamp in timestamps], dtype=float)
    values_m = agl_to_terrain(
        radar_agl,
        baro_msl,
        baro_bias_m=baro_bias_m,
        terrain_bias_m=terrain_bias_m,
    )
    values_m = np.where(valid_mask, values_m, np.nan)
    return TerrainProfile(
        values_m=values_m.astype(float, copy=False),
        timestamps_s=timestamps.astype(float, copy=False),
        valid_mask=valid_mask.astype(bool, copy=False),
        baro_bias_m=float(baro_bias_m),
        terrain_bias_m=float(terrain_bias_m),
    )
