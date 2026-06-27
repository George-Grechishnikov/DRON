"""Correlation engine for TERRAIN NAVIGATOR."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import correlate, windows

from measurement_layer import BaroTrack, frames_to_terrain_profile
from nmea_parser import NMEAFrame
from profile_extractor import extract_terrain_features, normalize_profile


LOGGER = logging.getLogger(__name__)
PSR_THRESHOLD = 1.3
SHAPE_FACTOR = 0.5
MIN_PROFILE_STD_M = 1e-6


@dataclass(frozen=True)
class CorrelationCandidate:
    """One local maximum from the azimuth/offset correlation heatmap."""

    azimuth_deg: float
    offset_steps: int
    offset_m: float
    offset_subsample_steps: float
    offset_subsample_m: float
    score: float
    ncc_score: float
    msd_score: float


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation search result."""

    best_azimuth_deg: float
    best_offset_steps: int
    best_offset_m: float
    best_offset_subsample_steps: float = 0.0
    best_offset_subsample_m: float = 0.0
    sigma_offset_m: float = 0.0
    sigma_azimuth_m: float = 0.0
    peak_correlation: float = 0.0
    confidence: float = 0.0
    is_reliable: bool = False
    peak_to_sidelobe: float = 0.0
    peak_to_mean: float = 0.0
    informative: bool = True
    ncc_peak: float = 0.0
    msd_peak: float = 0.0
    pslr_db: float = 0.0
    ambiguity_peak_count: int = 0
    peak_isolation_m: float = 0.0
    is_ambiguous: bool = False
    heatmap: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    azimuths_deg: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    best_reference_profile: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    top_candidates: tuple[CorrelationCandidate, ...] = field(default_factory=tuple)


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


def parabolic_vertex(y_minus: float, y0: float, y_plus: float) -> tuple[float, float]:
    """Return (delta, curvature) of the parabola vertex through 3 equally spaced points."""

    alpha = float(y_minus)
    beta = float(y0)
    gamma = float(y_plus)
    denom = alpha - (2.0 * beta) + gamma
    if abs(denom) < 1e-12:
        return (0.0, 0.0)
    delta = 0.5 * (alpha - gamma) / denom
    return (float(np.clip(delta, -0.5, 0.5)), float(denom))


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
        metric: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.5,
        msd_scale_m2: float = 100.0,
        min_valid_fraction: float = 0.5,
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
        if metric not in {"ncc", "msd", "hybrid"}:
            raise ValueError("metric must be one of: 'ncc', 'msd', 'hybrid'")
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if msd_scale_m2 <= 0.0:
            raise ValueError("msd_scale_m2 must be positive")
        if not (0.0 <= min_valid_fraction <= 1.0):
            raise ValueError("min_valid_fraction must be within [0, 1]")

        self.profile_length_m = float(profile_length_m)
        self.step_m = float(step_m)
        self.max_offset_m = float(max_offset_m)
        self.use_terrain_features = bool(use_terrain_features)
        self.feature_refine_top_k = int(feature_refine_top_k)
        self.feature_refine_weight = float(feature_refine_weight)
        self.metric = str(metric)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.msd_scale_m2 = float(msd_scale_m2)
        self.min_valid_fraction = float(min_valid_fraction)

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
            reference_azimuths = getattr(ref_matrix, "azimuths_deg", None)
            if reference_azimuths is not None and len(reference_azimuths) == ref_array.shape[0]:
                azimuth_axis = np.asarray(reference_azimuths, dtype=float)
            else:
                azimuth_axis = np.arange(ref_array.shape[0], dtype=float)
        else:
            azimuth_axis = np.asarray(azimuths_deg, dtype=float)
            if azimuth_axis.shape[0] != ref_array.shape[0]:
                raise ValueError("azimuths_deg length must match ref_matrix rows")
            reference_azimuths = getattr(ref_matrix, "azimuths_deg", None)
            if reference_azimuths is not None:
                reference_array = np.asarray(reference_azimuths, dtype=float)
                if reference_array.shape != azimuth_axis.shape or not np.allclose(reference_array, azimuth_axis, atol=1e-9):
                    raise ValueError("azimuths_deg must match the azimuth grid used to build ref_matrix")

        max_offset_steps = int(math.floor(self.max_offset_m / self.step_m))
        total_valid_offsets = ref_array.shape[1] - h_meas_array.shape[0] + 1
        usable_offsets = min(total_valid_offsets, max_offset_steps + 1)
        if usable_offsets <= 0:
            raise ValueError("No valid offsets available for the provided input sizes")

        measurement_valid_fraction = float(np.mean(np.isfinite(h_meas_array))) if h_meas_array.size else 0.0
        h_mean = float(np.nanmean(h_meas_array))
        h_std = float(np.nanstd(h_meas_array))
        if measurement_valid_fraction < self.min_valid_fraction or not self._is_informative(h_std):
            return CorrelationResult(
                best_azimuth_deg=float(azimuth_axis[0]) if azimuth_axis.size else 0.0,
                best_offset_steps=0,
                best_offset_m=0.0,
                best_offset_subsample_steps=0.0,
                best_offset_subsample_m=0.0,
                sigma_offset_m=float("inf"),
                sigma_azimuth_m=float("inf"),
                peak_correlation=0.0,
                confidence=0.0,
                is_reliable=False,
                peak_to_sidelobe=0.0,
                peak_to_mean=0.0,
                informative=False,
                ncc_peak=0.0,
                msd_peak=0.0,
                pslr_db=0.0,
                ambiguity_peak_count=0,
                peak_isolation_m=0.0,
                is_ambiguous=True,
                heatmap=np.zeros((ref_array.shape[0], usable_offsets), dtype=float),
                azimuths_deg=azimuth_axis.copy(),
                best_reference_profile=np.zeros((h_meas_array.size,), dtype=float),
            )
        window_length = h_meas_array.size
        heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)
        ncc_heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)
        msd_heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)
        valid_fraction_heatmap = np.empty((ref_array.shape[0], usable_offsets), dtype=float)
        meas_feature = extract_terrain_features(h_meas_array, step_m=self.step_m) if self.use_terrain_features else None
        h_centered = h_meas_array - h_mean

        for row_index, ref_profile in enumerate(ref_array):
            ref_values = np.asarray(ref_profile, dtype=float)
            combined_values, ncc_values, msd_values, valid_fraction_values = self._compute_row_correlation(
                h_meas_array=h_meas_array,
                h_centered=h_centered,
                h_std=h_std,
                ref_values=ref_values,
                usable_offsets=usable_offsets,
                window_length=window_length,
                meas_feature=meas_feature,
            )
            heatmap[row_index, :] = combined_values
            ncc_heatmap[row_index, :] = ncc_values
            msd_heatmap[row_index, :] = msd_values
            valid_fraction_heatmap[row_index, :] = valid_fraction_values

        invalid_mask = valid_fraction_heatmap < self.min_valid_fraction
        heatmap[invalid_mask] = -np.inf
        ncc_heatmap[invalid_mask] = 0.0
        msd_heatmap[invalid_mask] = 0.0

        finite_mask = np.isfinite(heatmap)
        if not np.any(finite_mask):
            return CorrelationResult(
                best_azimuth_deg=float(azimuth_axis[0]) if azimuth_axis.size else 0.0,
                best_offset_steps=0,
                best_offset_m=0.0,
                best_offset_subsample_steps=0.0,
                best_offset_subsample_m=0.0,
                sigma_offset_m=float("inf"),
                sigma_azimuth_m=float("inf"),
                peak_correlation=0.0,
                confidence=0.0,
                is_reliable=False,
                peak_to_sidelobe=0.0,
                peak_to_mean=0.0,
                informative=False,
                ncc_peak=0.0,
                msd_peak=0.0,
                pslr_db=0.0,
                ambiguity_peak_count=0,
                peak_isolation_m=0.0,
                is_ambiguous=True,
                heatmap=np.zeros((ref_array.shape[0], usable_offsets), dtype=float),
                azimuths_deg=azimuth_axis.copy(),
                best_reference_profile=np.zeros((h_meas_array.size,), dtype=float),
            )

        if self.feature_refine_top_k > 0 and not self.use_terrain_features:
            self._refine_top_k_candidates(
                heatmap=heatmap,
                h_meas_array=h_meas_array,
                ref_array=ref_array,
                usable_offsets=usable_offsets,
            )

        best_flat_index = int(np.nanargmax(heatmap))
        best_row, best_col = np.unravel_index(best_flat_index, heatmap.shape)
        best_offset_steps = int(best_col)
        offset_delta, offset_curvature = self._subsample_peak_1d(
            heatmap[best_row],
            best_col,
            cyclic=False,
        )
        azimuth_delta, azimuth_curvature = self._subsample_peak_2d_azimuth(heatmap, best_row, best_col)
        best_azimuth_deg = float((azimuth_axis[best_row] + azimuth_delta) % 360.0)
        best_offset_subsample_steps = float(np.clip(best_col + offset_delta, 0.0, max(usable_offsets - 1, 0)))
        best_offset_m = best_offset_steps * self.step_m
        best_offset_subsample_m = best_offset_subsample_steps * self.step_m
        sigma_offset_steps = self._curvature_to_sigma(offset_curvature)
        sigma_offset_m = sigma_offset_steps * self.step_m
        sigma_azimuth_deg = self._curvature_to_sigma(azimuth_curvature)
        sigma_azimuth_m = sigma_azimuth_deg * (self.profile_length_m * math.pi / 180.0)
        peak = float(heatmap[best_row, best_col])
        ncc_peak = float(ncc_heatmap[best_row, best_col])
        msd_peak = float(msd_heatmap[best_row, best_col])
        top_candidates = self._extract_top_candidates(
            heatmap=heatmap,
            ncc_heatmap=ncc_heatmap,
            msd_heatmap=msd_heatmap,
            azimuth_axis=azimuth_axis,
            limit=8,
        )
        confidence, peak_to_sidelobe, peak_to_mean = self._peak_quality(
            heatmap,
            best_row,
            best_col,
        )
        ambiguity = self.compute_ambiguity(heatmap, self.step_m)
        is_reliable = peak >= 0.5 and peak_to_sidelobe >= PSR_THRESHOLD and not ambiguity.is_ambiguous
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
            sigma_offset_m=sigma_offset_m,
            sigma_azimuth_m=sigma_azimuth_m,
            peak_correlation=peak,
            confidence=confidence,
            is_reliable=is_reliable,
            peak_to_sidelobe=peak_to_sidelobe,
            peak_to_mean=peak_to_mean,
            informative=True,
            ncc_peak=ncc_peak,
            msd_peak=msd_peak,
            pslr_db=ambiguity.pslr_db,
            ambiguity_peak_count=ambiguity.n_peaks,
            peak_isolation_m=ambiguity.peak_isolation_m,
            is_ambiguous=ambiguity.is_ambiguous,
            heatmap=heatmap,
            azimuths_deg=azimuth_axis.copy(),
            best_reference_profile=best_reference_profile,
            top_candidates=top_candidates,
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

        del speed_mps, freq_hz
        terrain_profile = frames_to_terrain_profile(frames_buffer, BaroTrack())
        h_meas = terrain_profile.values_m
        return self.compute(h_meas, ref_matrix, azimuths_deg=azimuths_deg)

    @staticmethod
    def _peak_quality(
        heatmap: np.ndarray,
        best_row: int,
        best_col: int,
        *,
        exclude_offset: int = 3,
        exclude_azimuth: int = 5,
    ) -> tuple[float, float, float]:
        """Return (confidence, peak_to_sidelobe, peak_to_mean) for the selected peak."""

        values = np.asarray(heatmap, dtype=float)
        if values.ndim != 2 or values.size == 0:
            return (0.0, 0.0, 0.0)

        peak = float(values[best_row, best_col])
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0 or not np.isfinite(peak):
            return (0.0, 0.0, 0.0)
        global_mean = float(np.mean(finite_values))
        peak_to_mean = peak / max(abs(global_mean), 1e-9)

        mask = np.ones(values.shape, dtype=bool)
        row_count, col_count = values.shape
        row_offsets = np.arange(-exclude_azimuth, exclude_azimuth + 1, dtype=int)
        row_indices = (best_row + row_offsets) % row_count
        col_start = max(0, best_col - exclude_offset)
        col_stop = min(col_count, best_col + exclude_offset + 1)
        mask[np.ix_(row_indices, np.arange(col_start, col_stop, dtype=int))] = False

        sidelobes = values[mask & np.isfinite(values)]
        if sidelobes.size == 0:
            return (peak, peak / 1e-9, peak_to_mean)

        max_sidelobe = float(np.nanmax(sidelobes))
        sidelobe_std = float(np.nanstd(sidelobes))
        confidence = max(0.0, peak - max_sidelobe)
        peak_to_sidelobe = peak / max(sidelobe_std, 1e-9)
        return (confidence, peak_to_sidelobe, peak_to_mean)

    @staticmethod
    def _is_informative(h_centered_std: float) -> bool:
        """Return True when the measurement profile contains enough variation."""

        return bool(np.isfinite(h_centered_std) and h_centered_std >= MIN_PROFILE_STD_M)

    def _compute_row_correlation(
        self,
        *,
        h_meas_array: np.ndarray,
        h_centered: np.ndarray,
        h_std: float,
        ref_values: np.ndarray,
        usable_offsets: int,
        window_length: int,
        meas_feature: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute one azimuth row of the heatmap and metric diagnostics."""

        if not self.use_terrain_features:
            if np.all(np.isfinite(h_meas_array)) and np.all(np.isfinite(ref_values)):
                ncc_values = self._ncc_scores(
                    ref_values=ref_values,
                    h_centered=h_centered,
                    h_std=h_std,
                    usable_offsets=usable_offsets,
                    window_length=window_length,
                )
                msd_values = self._msd_scores(
                    ref_values=ref_values,
                    h_meas=h_meas_array,
                    usable_offsets=usable_offsets,
                    window_length=window_length,
                )
                valid_fractions = np.ones((usable_offsets,), dtype=float)
                return self._combine_metric_scores(ncc_values, msd_values), ncc_values, msd_values, valid_fractions

            ncc_values = np.zeros((usable_offsets,), dtype=float)
            msd_values = np.zeros((usable_offsets,), dtype=float)
            valid_fractions = np.zeros((usable_offsets,), dtype=float)
            for offset in range(usable_offsets):
                ref_window = ref_values[offset : offset + window_length]
                ncc_value, valid_fraction = self._masked_ncc(ref_window, h_meas_array)
                msd_value = self._masked_msd(ref_window, h_meas_array)
                ncc_values[offset] = ncc_value
                msd_values[offset] = msd_value
                valid_fractions[offset] = valid_fraction
            combined = self._combine_metric_scores(ncc_values, msd_values)
            return combined, ncc_values, msd_values, valid_fractions

        corr_values = np.empty((usable_offsets,), dtype=float)
        for offset in range(usable_offsets):
            ref_window = ref_values[offset : offset + window_length]
            ref_feature = extract_terrain_features(ref_window, step_m=self.step_m)
            assert meas_feature is not None
            corr_values[offset] = self._feature_similarity(meas_feature, ref_feature)
        zeros = np.zeros((usable_offsets,), dtype=float)
        valid_fractions = np.ones((usable_offsets,), dtype=float)
        return corr_values, zeros.copy(), zeros, valid_fractions

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

    def _extract_top_candidates(
        self,
        *,
        heatmap: np.ndarray,
        ncc_heatmap: np.ndarray,
        msd_heatmap: np.ndarray,
        azimuth_axis: np.ndarray,
        limit: int,
    ) -> tuple[CorrelationCandidate, ...]:
        """Return strongest local maxima for downstream multi-hypothesis gating."""

        values = np.asarray(heatmap, dtype=float)
        finite_mask = np.isfinite(values)
        if values.ndim != 2 or values.size == 0 or not np.any(finite_mask) or limit <= 0:
            return tuple()

        sanitized = np.where(finite_mask, values, -np.inf)
        local_max = sanitized == maximum_filter(
            sanitized,
            size=(5, 5),
            mode=("wrap", "nearest"),
        )
        rows, cols = np.nonzero(local_max & finite_mask)
        if rows.size == 0:
            rows, cols = np.nonzero(finite_mask)

        scores = sanitized[rows, cols]
        order = np.argsort(scores)[::-1][:limit]
        candidates: list[CorrelationCandidate] = []
        seen: set[tuple[int, int]] = set()
        for item_index in order:
            row = int(rows[item_index])
            col = int(cols[item_index])
            key = (row, col)
            if key in seen:
                continue
            seen.add(key)
            offset_delta, _ = self._subsample_peak_1d(
                values[row],
                col,
                cyclic=False,
            )
            azimuth_delta, _ = self._subsample_peak_2d_azimuth(values, row, col)
            offset_subsample_steps = float(np.clip(col + offset_delta, 0.0, max(values.shape[1] - 1, 0)))
            candidates.append(
                CorrelationCandidate(
                    azimuth_deg=float((azimuth_axis[row] + azimuth_delta) % 360.0),
                    offset_steps=col,
                    offset_m=float(col * self.step_m),
                    offset_subsample_steps=offset_subsample_steps,
                    offset_subsample_m=float(offset_subsample_steps * self.step_m),
                    score=float(values[row, col]),
                    ncc_score=float(ncc_heatmap[row, col]),
                    msd_score=float(msd_heatmap[row, col]),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _ncc_scores(
        *,
        ref_values: np.ndarray,
        h_centered: np.ndarray,
        h_std: float,
        usable_offsets: int,
        window_length: int,
    ) -> np.ndarray:
        """Compute normalized cross-correlation using FFT-backed convolution."""

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

    def _msd_scores(
        self,
        *,
        ref_values: np.ndarray,
        h_meas: np.ndarray,
        usable_offsets: int,
        window_length: int,
    ) -> np.ndarray:
        """Compute a bounded negative-MSD score that preserves absolute terrain level."""

        windows_view = np.lib.stride_tricks.sliding_window_view(ref_values, window_length)[:usable_offsets]
        squared_error = (windows_view - h_meas[np.newaxis, :]) ** 2
        msd = np.mean(squared_error, axis=1)
        return 1.0 / (1.0 + (msd / self.msd_scale_m2))

    def _combine_metric_scores(self, ncc_values: np.ndarray, msd_values: np.ndarray) -> np.ndarray:
        """Combine metric components according to the configured scoring mode."""

        if self.metric == "ncc":
            return np.asarray(ncc_values, dtype=float)
        if self.metric == "msd":
            return np.asarray(msd_values, dtype=float)

        weight_sum = self.alpha + self.beta
        if weight_sum <= 1e-12:
            return 0.5 * (np.asarray(ncc_values, dtype=float) + np.asarray(msd_values, dtype=float))
        return (
            self.alpha * np.asarray(ncc_values, dtype=float)
            + self.beta * np.asarray(msd_values, dtype=float)
        ) / weight_sum

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
    def _masked_ncc(ref_window: np.ndarray, h_window: np.ndarray) -> tuple[float, float]:
        """Return (ncc, valid_fraction) with NaN-aware masking."""

        ref = np.asarray(ref_window, dtype=float)
        meas = np.asarray(h_window, dtype=float)
        valid = np.isfinite(ref) & np.isfinite(meas)
        valid_count = int(np.count_nonzero(valid))
        if ref.size == 0:
            return (0.0, 0.0)
        valid_fraction = valid_count / ref.size
        if valid_count < 2:
            return (0.0, float(valid_fraction))

        ref_valid = ref[valid]
        meas_valid = meas[valid]
        ref_centered = ref_valid - float(np.mean(ref_valid))
        meas_centered = meas_valid - float(np.mean(meas_valid))
        numerator = float(np.dot(ref_centered, meas_centered))
        denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(meas_centered))
        if denominator <= 1e-12:
            return (0.0, float(valid_fraction))
        return (numerator / denominator, float(valid_fraction))

    def _masked_msd(self, ref_window: np.ndarray, h_window: np.ndarray) -> float:
        """Return NaN-aware bounded negative-MSD score."""

        ref = np.asarray(ref_window, dtype=float)
        meas = np.asarray(h_window, dtype=float)
        valid = np.isfinite(ref) & np.isfinite(meas)
        if not np.any(valid):
            return 0.0
        msd = float(np.mean((ref[valid] - meas[valid]) ** 2))
        return float(1.0 / (1.0 + (msd / self.msd_scale_m2)))

    @staticmethod
    def _subsample_peak_1d(r_vector: np.ndarray, peak_idx: int, *, cyclic: bool) -> tuple[float, float]:
        """Return (delta, curvature) for a 1D peak."""

        values = np.asarray(r_vector, dtype=float)
        if values.size < 3:
            return (0.0, 0.0)
        if not cyclic and (peak_idx <= 0 or peak_idx >= values.size - 1):
            return (0.0, 0.0)

        left_idx = (peak_idx - 1) % values.size
        right_idx = (peak_idx + 1) % values.size
        return parabolic_vertex(values[left_idx], values[peak_idx], values[right_idx])

    @staticmethod
    def _subsample_peak_2d_azimuth(heatmap: np.ndarray, best_row: int, best_col: int) -> tuple[float, float]:
        """Return azimuth-axis (delta, curvature) using cyclic interpolation."""

        values = np.asarray(heatmap, dtype=float)
        if values.ndim != 2 or values.shape[0] < 3:
            return (0.0, 0.0)
        left_row = (best_row - 1) % values.shape[0]
        right_row = (best_row + 1) % values.shape[0]
        return parabolic_vertex(values[left_row, best_col], values[best_row, best_col], values[right_row, best_col])

    @staticmethod
    def _curvature_to_sigma(curvature: float) -> float:
        """Convert peak curvature into a crude sigma estimate."""

        if curvature >= -1e-12:
            return float("inf")
        return SHAPE_FACTOR / math.sqrt(max(-curvature, 1e-12))

    @staticmethod
    def compute_ambiguity(heatmap: np.ndarray, step_m: float) -> AmbiguityMetrics:
        """Estimate ambiguity using radar-style peak diagnostics."""

        values = np.asarray(heatmap, dtype=float)
        finite_mask = np.isfinite(values)
        if values.ndim != 2 or values.size == 0 or not np.any(finite_mask):
            return AmbiguityMetrics(pslr_db=0.0, n_peaks=0, peak_isolation_m=0.0, is_ambiguous=True)

        peak_row, peak_col = np.unravel_index(int(np.nanargmax(values)), values.shape)
        peak_val = float(values[peak_row, peak_col])
        finite_values = values[finite_mask]
        if finite_values.size == 1:
            return AmbiguityMetrics(pslr_db=99.0, n_peaks=1, peak_isolation_m=float("inf"), is_ambiguous=False)

        sanitized = np.where(finite_mask, values, -np.inf)
        local_max = sanitized == maximum_filter(
            sanitized,
            size=(7, 7),
            mode=("wrap", "nearest"),
        )
        peak_drop = max(abs(peak_val) * 0.2, 1e-9)
        candidate_mask = local_max & finite_mask & (sanitized >= peak_val - peak_drop)
        candidate_rows, candidate_cols = np.nonzero(candidate_mask)
        if candidate_rows.size == 0:
            candidate_rows = np.asarray([peak_row], dtype=int)
            candidate_cols = np.asarray([peak_col], dtype=int)
        if not np.any((candidate_rows == peak_row) & (candidate_cols == peak_col)):
            candidate_rows = np.append(candidate_rows, peak_row)
            candidate_cols = np.append(candidate_cols, peak_col)

        exclusion = np.zeros(values.shape, dtype=bool)
        row_offsets = np.arange(-3, 4, dtype=int)
        excluded_rows = (peak_row + row_offsets) % values.shape[0]
        col_start = max(0, peak_col - 3)
        col_stop = min(values.shape[1], peak_col + 4)
        exclusion[np.ix_(excluded_rows, np.arange(col_start, col_stop, dtype=int))] = True
        sidelobes = sanitized[finite_mask & ~exclusion]
        sidelobe = float(np.nanmax(sidelobes)) if sidelobes.size else 1e-10
        pslr_db = 20.0 * math.log10((peak_val + 1e-10) / (abs(sidelobe) + 1e-10))

        other_mask = ~((candidate_rows == peak_row) & (candidate_cols == peak_col))
        if not np.any(other_mask):
            peak_isolation_m = float("inf")
        else:
            row_delta = np.abs(candidate_rows[other_mask] - peak_row)
            cyclic_row_delta = np.minimum(row_delta, values.shape[0] - row_delta)
            col_delta_m = (candidate_cols[other_mask] - peak_col) * step_m
            azimuth_delta_m = cyclic_row_delta * (step_m * max(values.shape[1], 1) * math.pi / 180.0)
            peak_isolation_m = float(np.min(np.hypot(col_delta_m, azimuth_delta_m)))

        peak_count = int(candidate_rows.size)
        is_ambiguous = pslr_db < 3.0 or peak_count > 2
        return AmbiguityMetrics(
            pslr_db=pslr_db,
            n_peaks=peak_count,
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
