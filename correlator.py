"""Correlation engine for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import correlate

from nmea_parser import NMEAFrame, frames_to_profile
from profile_extractor import normalize_profile


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation search result."""

    best_azimuth_deg: float
    best_offset_steps: int
    best_offset_m: float
    peak_correlation: float
    confidence: float
    is_reliable: bool
    heatmap: np.ndarray
    azimuths_deg: np.ndarray
    best_reference_profile: np.ndarray


class Correlator:
    """Compute terrain-profile correlation across azimuths and offsets."""

    def __init__(
        self,
        profile_length_m: float,
        step_m: float = 30.0,
        max_offset_m: float = 2000.0,
    ) -> None:
        if profile_length_m <= 0:
            raise ValueError("profile_length_m must be positive")
        if step_m <= 0:
            raise ValueError("step_m must be positive")
        if max_offset_m < 0:
            raise ValueError("max_offset_m must be non-negative")

        self.profile_length_m = float(profile_length_m)
        self.step_m = float(step_m)
        self.max_offset_m = float(max_offset_m)

    def compute(
        self,
        h_meas: np.ndarray,
        ref_matrix: np.ndarray,
        azimuths_deg: np.ndarray | None = None,
    ) -> CorrelationResult:
        """Compute the best correlation match over azimuth and offset."""

        started_at = time.perf_counter()
        h_meas_array = np.asarray(h_meas, dtype=float)
        ref_array = np.asarray(ref_matrix, dtype=float)
        if h_meas_array.ndim != 1:
            raise ValueError("h_meas must be a 1D array")
        if ref_array.ndim != 2:
            raise ValueError("ref_matrix must be a 2D array")
        if ref_array.shape[1] < h_meas_array.shape[0]:
            raise ValueError("ref_matrix profiles must be at least as long as h_meas")

        if azimuths_deg is None:
            azimuth_axis = np.arange(ref_array.shape[0], dtype=float)
        else:
            azimuth_axis = np.asarray(azimuths_deg, dtype=float)
            if azimuth_axis.shape[0] != ref_array.shape[0]:
                raise ValueError("azimuths_deg length must match ref_matrix rows")

        max_offset_steps = int(math.floor(self.max_offset_m / self.step_m))
        total_valid_offsets = ref_array.shape[1] - h_meas_array.shape[0] + 1
        usable_offsets = min(total_valid_offsets, max_offset_steps + 1)
        if usable_offsets <= 0:
            raise ValueError("No valid offsets available for the provided input sizes")

        h_mean = float(np.nanmean(h_meas_array))
        h_std = float(np.nanstd(h_meas_array))
        if np.isnan(h_std) or h_std == 0.0:
            raise ValueError("h_meas must have non-zero variance")
        h_centered = h_meas_array - h_mean
        window_length = h_meas_array.size
        heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)

        for row_index, ref_profile in enumerate(ref_array):
            ref_values = np.asarray(ref_profile, dtype=float)
            numerator = correlate(ref_values, h_centered, mode="valid")[:usable_offsets]
            sums = np.convolve(ref_values, np.ones(window_length, dtype=float), mode="valid")[
                :usable_offsets
            ]
            sums_sq = np.convolve(ref_values * ref_values, np.ones(window_length, dtype=float), mode="valid")[
                :usable_offsets
            ]
            ref_mean = sums / window_length
            ref_var = np.maximum((sums_sq / window_length) - (ref_mean * ref_mean), 0.0)
            ref_std = np.sqrt(ref_var)
            denominator = window_length * h_std * ref_std
            corr_values = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > 0,
            )
            heatmap[row_index, :] = corr_values

        best_flat_index = int(np.nanargmax(heatmap))
        best_row, best_col = np.unravel_index(best_flat_index, heatmap.shape)
        best_azimuth_deg = float(azimuth_axis[best_row])
        best_offset_steps = int(best_col)
        best_offset_m = best_offset_steps * self.step_m
        peak = float(heatmap[best_row, best_col])
        confidence = self._compute_confidence(heatmap, peak)
        is_reliable = peak >= 0.5 and confidence >= 0.1
        best_reference_profile = ref_array[
            best_row, best_offset_steps : best_offset_steps + h_meas_array.size
        ].copy()

        elapsed = time.perf_counter() - started_at
        LOGGER.info(
            "Correlation finished in %.3fs, peak=%.4f at azimuth=%.1f offset=%.1fm",
            elapsed,
            peak,
            best_azimuth_deg,
            best_offset_m,
        )

        return CorrelationResult(
            best_azimuth_deg=best_azimuth_deg,
            best_offset_steps=best_offset_steps,
            best_offset_m=best_offset_m,
            peak_correlation=peak,
            confidence=confidence,
            is_reliable=is_reliable,
            heatmap=heatmap,
            azimuths_deg=azimuth_axis.copy(),
            best_reference_profile=best_reference_profile,
        )

    def sliding_window_compute(
        self,
        frames_buffer: list[NMEAFrame],
        ref_matrix: np.ndarray,
        speed_mps: float,
        freq_hz: float,
        azimuths_deg: np.ndarray | None = None,
    ) -> CorrelationResult:
        """Convert frames into a profile and compute the correlation result."""

        h_meas = frames_to_profile(frames_buffer, speed_mps=speed_mps, freq_hz=freq_hz)
        return self.compute(h_meas, ref_matrix, azimuths_deg=azimuths_deg)

    @staticmethod
    def _compute_confidence(heatmap: np.ndarray, peak: float) -> float:
        flat = np.sort(heatmap.ravel())
        if flat.size <= 1:
            return 1.0
        second_best = float(flat[-2])
        return max(0.0, peak - second_best)


def build_heatmap(result: CorrelationResult) -> np.ndarray:
    """Return a [0, 1]-normalized heatmap for visualization."""

    heatmap = np.asarray(result.heatmap, dtype=float)
    min_value = float(np.nanmin(heatmap))
    max_value = float(np.nanmax(heatmap))
    if max_value == min_value:
        return np.zeros_like(heatmap)
    return (heatmap - min_value) / (max_value - min_value)
