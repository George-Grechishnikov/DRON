"""Terrain particle filter for DEM-based horizontal localization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dem_loader import DEMLoader
from local_frame import LocalFrame


@dataclass(frozen=True)
class TerrainUpdateResult:
    """Posterior horizontal terrain update."""

    p_enu_m: np.ndarray
    cov_2x2: np.ndarray
    entropy: float
    converged: bool


class TerrainParticleFilter:
    """Point-mass/particle filter over horizontal DEM position."""

    def __init__(
        self,
        dem: DEMLoader,
        *,
        n_particles: int = 500,
        meas_sigma_m: float = 10.0,
        resample_threshold: float = 0.5,
        terrain_bias_m: float = 0.0,
    ) -> None:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if meas_sigma_m <= 0:
            raise ValueError("meas_sigma_m must be positive")
        if not (0.0 < resample_threshold <= 1.0):
            raise ValueError("resample_threshold must be within (0, 1]")
        self.dem = dem
        center_lat, center_lon = dem.get_center()
        self.frame = LocalFrame(center_lat, center_lon)
        self.n_particles = int(n_particles)
        self.meas_sigma_m = float(meas_sigma_m)
        self.resample_threshold = float(resample_threshold)
        self.terrain_bias_m = float(terrain_bias_m)
        self.particles_enu = np.zeros((self.n_particles, 2), dtype=float)
        self.weights = np.full((self.n_particles,), 1.0 / self.n_particles, dtype=float)
        self._rng = np.random.default_rng(2026)

    def initialize_global(self, center_lat: float, center_lon: float, radius_m: float) -> None:
        """Initialize particles over a broad disc around the requested center."""

        center_enu = self.frame.to_enu(center_lat, center_lon)
        radii = radius_m * np.sqrt(self._rng.random(self.n_particles))
        angles = self._rng.uniform(0.0, 2.0 * math.pi, size=self.n_particles)
        self.particles_enu[:, 0] = center_enu[0] + radii * np.cos(angles)
        self.particles_enu[:, 1] = center_enu[1] + radii * np.sin(angles)
        self.weights.fill(1.0 / self.n_particles)

    def initialize_around(self, lat: float, lon: float, sigma_m: float) -> None:
        """Initialize particles around a local tracking hypothesis."""

        center_enu = self.frame.to_enu(lat, lon)
        noise = self._rng.normal(0.0, sigma_m, size=(self.n_particles, 2))
        self.particles_enu = center_enu[np.newaxis, :] + noise
        self.weights.fill(1.0 / self.n_particles)

    def predict(self, dp_enu_m: np.ndarray, process_cov: np.ndarray) -> None:
        """Shift particles with process noise from the navigation core."""

        delta = np.asarray(dp_enu_m, dtype=float)
        cov = np.asarray(process_cov, dtype=float)
        if delta.shape != (2,):
            raise ValueError("dp_enu_m must have shape (2,)")
        if cov.shape != (2, 2):
            raise ValueError("process_cov must have shape (2, 2)")
        noise = self._rng.multivariate_normal(np.zeros(2, dtype=float), cov, size=self.n_particles)
        self.particles_enu += delta[np.newaxis, :] + noise

    def update(self, measured_terrain_profile: np.ndarray, path_offsets_enu: np.ndarray) -> TerrainUpdateResult:
        """Update particle weights from a terrain profile along the estimated path."""

        profile = np.asarray(measured_terrain_profile, dtype=float)
        offsets = np.asarray(path_offsets_enu, dtype=float)
        if profile.ndim != 1:
            raise ValueError("measured_terrain_profile must be 1D")
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError("path_offsets_enu must have shape (N, 2)")
        if offsets.shape[0] != profile.shape[0]:
            raise ValueError("path_offsets_enu length must match measured_terrain_profile")

        valid = np.isfinite(profile)
        if np.count_nonzero(valid) < max(3, int(math.ceil(0.5 * profile.size))) or float(np.nanstd(profile)) < 1e-6:
            self.weights.fill(1.0 / self.n_particles)
            estimate = self.estimate()
            inflated_cov = estimate.cov_2x2 + np.eye(2, dtype=float) * 10_000.0
            return TerrainUpdateResult(
                p_enu_m=estimate.p_enu_m,
                cov_2x2=inflated_cov,
                entropy=estimate.entropy,
                converged=False,
            )

        flat_positions = (self.particles_enu[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 2)
        latlon = np.array([self.frame.to_geodetic(point) for point in flat_positions], dtype=float)
        sampled = self.dem._sample_points(latlon[:, 0], latlon[:, 1]).reshape(self.n_particles, profile.size)
        sampled = sampled + self.terrain_bias_m
        residual = sampled - profile[np.newaxis, :]
        residual = residual[:, valid]
        finite_residual = np.where(np.isfinite(residual), residual, 5.0 * self.meas_sigma_m)
        mse = np.mean(finite_residual * finite_residual, axis=1)
        log_weights = np.log(np.maximum(self.weights, 1e-300)) - 0.5 * mse / (self.meas_sigma_m ** 2)
        log_weights -= float(np.max(log_weights))
        self.weights = np.exp(log_weights)
        weight_sum = float(np.sum(self.weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= weight_sum
        self._resample_if_needed()
        return self.estimate()

    def estimate(self) -> TerrainUpdateResult:
        """Return weighted mean, covariance, and entropy."""

        mean = np.average(self.particles_enu, axis=0, weights=self.weights)
        map_index = int(np.argmax(self.weights))
        estimate = self.particles_enu[map_index].copy()
        delta = self.particles_enu - mean[np.newaxis, :]
        cov = np.einsum("i,ij,ik->jk", self.weights, delta, delta)
        entropy = float(-np.sum(self.weights * np.log(np.maximum(self.weights, 1e-300))))
        converged = bool(np.trace(cov) < 50_000.0 and entropy < math.log(max(self.n_particles, 2)) * 0.75)
        return TerrainUpdateResult(
            p_enu_m=estimate.astype(float, copy=False),
            cov_2x2=(cov + np.eye(2, dtype=float) * 1e-9).astype(float, copy=False),
            entropy=entropy,
            converged=converged,
        )

    def _effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))

    def _resample_if_needed(self) -> None:
        threshold = self.resample_threshold * self.n_particles
        if self._effective_sample_size() >= threshold:
            return
        cumulative = np.cumsum(self.weights)
        step = 1.0 / self.n_particles
        start = self._rng.uniform(0.0, step)
        targets = start + step * np.arange(self.n_particles, dtype=float)
        indices = np.searchsorted(cumulative, targets, side="left")
        indices = np.clip(indices, 0, self.n_particles - 1)
        self.particles_enu = self.particles_enu[indices]
        self.weights.fill(1.0 / self.n_particles)
