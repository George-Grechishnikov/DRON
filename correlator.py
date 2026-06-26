"""Correlation engine for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import correlate, find_peaks, windows

from nmea_parser import NMEAFrame, frames_to_profile
from profile_extractor import extract_terrain_features, normalize_profile


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation search result."""

    best_azimuth_deg: float
    best_offset_steps: int
    best_offset_m: float
    best_offset_subsample_steps: float = 0.0
    best_offset_subsample_m: float = 0.0
    peak_correlation: float = 0.0
    confidence: float = 0.0
    is_reliable: bool = False
    pslr_db: float = 0.0
    ambiguity_peak_count: int = 0
    peak_isolation_m: float = 0.0
    is_ambiguous: bool = False
    heatmap: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    azimuths_deg: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    best_reference_profile: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))


@dataclass(frozen=True)
class AmbiguityMetrics:
    """Ambiguity diagnostics for a correlation heatmap."""

    pslr_db: float
    n_peaks: int
    peak_isolation_m: float
    is_ambiguous: bool


@dataclass(frozen=True)
class ObservabilityMetrics:
    """CRLB-like terrain observability diagnostics."""

    crlb_m: float
    gradient_energy: float
    efficiency_hint: float
    is_informative: bool


class Correlator:
    """Compute terrain-profile correlation across azimuths and offsets."""

    def __init__(
        self,
        profile_length_m: float,
        step_m: float = 30.0,
        max_offset_m: float = 2000.0,
        *,
        use_terrain_features: bool = False,
        feature_refine_top_k: int = 0,
        feature_refine_weight: float = 0.3,
    ) -> None:
        if profile_length_m <= 0:
            raise ValueError("profile_length_m must be positive")
        if step_m <= 0:
            raise ValueError("step_m must be positive")
        if max_offset_m < 0:
            raise ValueError("max_offset_m must be non-negative")
        if feature_refine_top_k < 0:
            raise ValueError("feature_refine_top_k must be non-negative")
        if not (0.0 <= feature_refine_weight <= 1.0):
            raise ValueError("feature_refine_weight must be within [0, 1]")

        self.profile_length_m = float(profile_length_m)
        self.step_m = float(step_m)
        self.max_offset_m = float(max_offset_m)
        self.use_terrain_features = bool(use_terrain_features)
        self.feature_refine_top_k = int(feature_refine_top_k)
        self.feature_refine_weight = float(feature_refine_weight)

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
        window_length = h_meas_array.size
        heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)
        meas_feature = extract_terrain_features(h_meas_array, step_m=self.step_m) if self.use_terrain_features else None

        for row_index, ref_profile in enumerate(ref_array):
            ref_values = np.asarray(ref_profile, dtype=float)
            corr_values = self._compute_row_correlation(
                h_meas_array=h_meas_array,
                h_std=h_std,
                h_mean=h_mean,
                ref_values=ref_values,
                usable_offsets=usable_offsets,
                window_length=window_length,
                meas_feature=meas_feature,
            )
            heatmap[row_index, :] = corr_values

        if self.feature_refine_top_k > 0 and not self.use_terrain_features:
            self._refine_top_k_candidates(
                heatmap=heatmap,
                h_meas_array=h_meas_array,
                ref_array=ref_array,
                usable_offsets=usable_offsets,
            )

        best_flat_index = int(np.nanargmax(heatmap))
        best_row, best_col = np.unravel_index(best_flat_index, heatmap.shape)
        best_azimuth_deg = float(azimuth_axis[best_row])
        best_offset_steps = int(best_col)
        best_offset_subsample_steps = self._subsample_peak_position(heatmap[best_row], best_col)
        best_offset_m = best_offset_steps * self.step_m
        best_offset_subsample_m = best_offset_subsample_steps * self.step_m
        peak = float(heatmap[best_row, best_col])
        confidence = self._compute_confidence(heatmap, peak)
        ambiguity = self.compute_ambiguity(heatmap, self.step_m)
        is_reliable = peak >= 0.5 and confidence >= 0.1 and not ambiguity.is_ambiguous
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
            best_offset_subsample_steps=best_offset_subsample_steps,
            best_offset_subsample_m=best_offset_subsample_m,
            peak_correlation=peak,
            confidence=confidence,
            is_reliable=is_reliable,
            pslr_db=ambiguity.pslr_db,
            ambiguity_peak_count=ambiguity.n_peaks,
            peak_isolation_m=ambiguity.peak_isolation_m,
            is_ambiguous=ambiguity.is_ambiguous,
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

    def _compute_row_correlation(
        self,
        *,
        h_meas_array: np.ndarray,
        h_std: float,
        h_mean: float,
        ref_values: np.ndarray,
        usable_offsets: int,
        window_length: int,
        meas_feature: np.ndarray | None,
    ) -> np.ndarray:
        """Compute one azimuth row of the heatmap."""

        if not self.use_terrain_features:
            return self._normalized_cross_correlation(
                h_meas_array=h_meas_array,
                h_std=h_std,
                h_mean=h_mean,
                ref_values=ref_values,
                usable_offsets=usable_offsets,
                window_length=window_length,
            )

        corr_values = np.empty((usable_offsets,), dtype=float)
        for offset in range(usable_offsets):
            ref_window = ref_values[offset : offset + window_length]
            ref_feature = extract_terrain_features(ref_window, step_m=self.step_m)
            assert meas_feature is not None
            corr_values[offset] = self._feature_similarity(meas_feature, ref_feature)
        return corr_values

    def _refine_top_k_candidates(
        self,
        *,
        heatmap: np.ndarray,
        h_meas_array: np.ndarray,
        ref_array: np.ndarray,
        usable_offsets: int,
    ) -> None:
        """Refine the strongest NCC candidates using terrain feature similarity."""

        flat = heatmap.ravel()
        if flat.size == 0:
            return
        top_k = min(self.feature_refine_top_k, flat.size)
        if top_k <= 0:
            return

        meas_feature = extract_terrain_features(h_meas_array, step_m=self.step_m)
        top_indices = np.argpartition(flat, -top_k)[-top_k:]
        window_length = h_meas_array.size
        for flat_index in top_indices:
            row_index, offset = np.unravel_index(int(flat_index), heatmap.shape)
            if offset >= usable_offsets:
                continue
            ref_window = ref_array[row_index, offset : offset + window_length]
            ref_feature = extract_terrain_features(ref_window, step_m=self.step_m)
            feature_score = self._feature_similarity(meas_feature, ref_feature)
            base_score = float(heatmap[row_index, offset])
            heatmap[row_index, offset] = (
                (1.0 - self.feature_refine_weight) * base_score
                + self.feature_refine_weight * feature_score
            )

    @staticmethod
    def _normalized_cross_correlation(
        *,
        h_meas_array: np.ndarray,
        h_std: float,
        h_mean: float,
        ref_values: np.ndarray,
        usable_offsets: int,
        window_length: int,
    ) -> np.ndarray:
        """Compute normalized cross-correlation using FFT-backed convolution."""

        h_centered = h_meas_array - h_mean
        numerator = correlate(ref_values, h_centered, mode="valid", method="fft")[:usable_offsets]
        sums = np.convolve(ref_values, np.ones(window_length, dtype=float), mode="valid")[:usable_offsets]
        sums_sq = np.convolve(ref_values * ref_values, np.ones(window_length, dtype=float), mode="valid")[:usable_offsets]
        ref_mean = sums / window_length
        ref_var = np.maximum((sums_sq / window_length) - (ref_mean * ref_mean), 0.0)
        ref_std = np.sqrt(ref_var)
        denominator = window_length * h_std * ref_std
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )

    @staticmethod
    def _feature_similarity(meas_feature: np.ndarray, ref_feature: np.ndarray) -> float:
        """Compute a robust similarity score between two terrain feature vectors."""

        window = windows.hann(meas_feature.size, sym=False)
        meas = meas_feature * window
        ref = ref_feature * window
        meas_norm = np.linalg.norm(meas)
        ref_norm = np.linalg.norm(ref)
        if meas_norm <= 1e-12 or ref_norm <= 1e-12:
            return 0.0
        return float(np.dot(meas, ref) / (meas_norm * ref_norm))

    @staticmethod
    def _subsample_peak_position(r_vector: np.ndarray, peak_idx: int) -> float:
        """Parabolic interpolation around the discrete peak for sub-step precision."""

        if peak_idx <= 0 or peak_idx >= r_vector.size - 1:
            return float(peak_idx)
        alpha = float(r_vector[peak_idx - 1])
        beta = float(r_vector[peak_idx])
        gamma = float(r_vector[peak_idx + 1])
        denom = alpha - (2.0 * beta) + gamma
        if abs(denom) < 1e-12:
            return float(peak_idx)
        offset = 0.5 * (alpha - gamma) / denom
        return float(peak_idx) + float(np.clip(offset, -1.0, 1.0))

    @staticmethod
    def compute_ambiguity(heatmap: np.ndarray, step_m: float) -> AmbiguityMetrics:
        """Estimate ambiguity using radar-style peak diagnostics."""

        flat = np.asarray(heatmap, dtype=float).ravel()
        if flat.size == 0:
            return AmbiguityMetrics(pslr_db=0.0, n_peaks=0, peak_isolation_m=0.0, is_ambiguous=True)

        peak_idx = int(np.nanargmax(flat))
        peak_val = float(flat[peak_idx])
        if flat.size == 1:
            return AmbiguityMetrics(pslr_db=99.0, n_peaks=1, peak_isolation_m=float("inf"), is_ambiguous=False)

        candidate_peaks, properties = find_peaks(flat, height=max(peak_val * 0.8, peak_val - 1e-9), distance=2)
        if candidate_peaks.size == 0:
            candidate_peaks = np.array([peak_idx], dtype=int)
        if peak_idx not in candidate_peaks:
            candidate_peaks = np.append(candidate_peaks, peak_idx)
        candidate_peaks = np.unique(candidate_peaks)

        sorted_values = np.sort(flat)
        sidelobe = float(sorted_values[-2]) if sorted_values.size > 1 else 1e-10
        pslr_db = 20.0 * math.log10((peak_val + 1e-10) / (abs(sidelobe) + 1e-10))

        other_peaks = np.array([idx for idx in candidate_peaks if idx != peak_idx], dtype=int)
        if other_peaks.size == 0:
            peak_isolation_m = float("inf")
        else:
            peak_isolation_m = float(np.min(np.abs(other_peaks - peak_idx))) * step_m

        is_ambiguous = pslr_db < 3.0 or candidate_peaks.size > 2
        return AmbiguityMetrics(
            pslr_db=pslr_db,
            n_peaks=int(candidate_peaks.size),
            peak_isolation_m=peak_isolation_m,
            is_ambiguous=is_ambiguous,
        )


def build_heatmap(result: CorrelationResult) -> np.ndarray:
    """Return a [0, 1]-normalized heatmap for visualization."""

    heatmap = np.asarray(result.heatmap, dtype=float)
    min_value = float(np.nanmin(heatmap))
    max_value = float(np.nanmax(heatmap))
    if max_value == min_value:
        return np.zeros_like(heatmap)
    return (heatmap - min_value) / (max_value - min_value)


def compute_crlb_position(
    h_ref: np.ndarray,
    sigma_noise_m: float,
    step_m: float = 30.0,
) -> float:
    """Estimate a CRLB-like lower bound for 1D terrain-shift localization."""

    profile = np.asarray(h_ref, dtype=float)
    if profile.ndim != 1:
        raise ValueError("h_ref must be a 1D array")
    if profile.size == 0:
        return float("inf")
    if sigma_noise_m <= 0:
        raise ValueError("sigma_noise_m must be positive")
    if step_m <= 0:
        raise ValueError("step_m must be positive")

    gradient = np.gradient(profile, step_m)
    fisher_info = float(np.sum(gradient**2) / (sigma_noise_m**2))
    if not np.isfinite(fisher_info) or fisher_info <= 1e-12:
        return float("inf")
    return float(1.0 / math.sqrt(fisher_info))


def compute_observability_metrics(
    h_ref: np.ndarray,
    sigma_noise_m: float,
    step_m: float = 30.0,
) -> ObservabilityMetrics:
    """Build a practical observability score from gradient energy and CRLB-like bound."""

    profile = np.asarray(h_ref, dtype=float)
    if profile.ndim != 1:
        raise ValueError("h_ref must be a 1D array")
    if profile.size == 0:
        return ObservabilityMetrics(
            crlb_m=float("inf"),
            gradient_energy=0.0,
            efficiency_hint=0.0,
            is_informative=False,
        )

    gradient = np.gradient(profile, step_m)
    gradient_energy = float(np.sum(gradient**2))
    crlb_m = compute_crlb_position(profile, sigma_noise_m=sigma_noise_m, step_m=step_m)
    efficiency_hint = 0.0 if not np.isfinite(crlb_m) else float(1.0 / (1.0 + crlb_m / max(step_m, 1e-6)))
    is_informative = bool(np.isfinite(crlb_m) and crlb_m <= 250.0 and gradient_energy > 0.05)
    return ObservabilityMetrics(
        crlb_m=crlb_m,
        gradient_energy=gradient_energy,
        efficiency_hint=efficiency_hint,
        is_informative=is_informative,
    )
