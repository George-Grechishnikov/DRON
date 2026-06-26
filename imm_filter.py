"""IMM filter for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pyproj

from position_solver import PositionEstimate, normalize_azimuth_deg


def _innovation_log_score(value: np.ndarray, covariance: np.ndarray) -> float:
    """Return a numerically stable log-score for innovation compatibility."""

    dim = value.shape[0]
    cov = covariance + np.eye(dim, dtype=float) * 1e-9
    inv_cov = np.linalg.inv(cov)
    mahalanobis = float(value.T @ inv_cov @ value)
    return float(-0.5 * mahalanobis / max(dim, 1))


class KalmanModel:
    """Base linear Kalman model used inside the IMM filter."""

    name: str

    def __init__(
        self,
        name: str,
        process_noise_scale: float,
        velocity_damping: float,
    ) -> None:
        self.name = name
        self.process_noise_scale = float(process_noise_scale)
        self.velocity_damping = float(velocity_damping)
        self.state = np.zeros(4, dtype=float)
        self.covariance = np.eye(4, dtype=float) * 10.0
        self.F = np.eye(4, dtype=float)
        self.H = np.eye(4, dtype=float)
        self.Q = np.eye(4, dtype=float)
        self.R = np.eye(4, dtype=float)

    def clone(self) -> "KalmanModel":
        cloned = self.__class__()  # type: ignore[call-arg]
        cloned.state = self.state.copy()
        cloned.covariance = self.covariance.copy()
        cloned.F = self.F.copy()
        cloned.H = self.H.copy()
        cloned.Q = self.Q.copy()
        cloned.R = self.R.copy()
        return cloned

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=float).copy()
        self.covariance = np.asarray(covariance, dtype=float).copy()

    def predict(self, dt: float) -> None:
        """Run one linear prediction step."""

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, self.velocity_damping, 0.0],
                [0.0, 0.0, 0.0, self.velocity_damping],
            ],
            dtype=float,
        )
        q_pos = self.process_noise_scale * max(dt * dt, 1e-6)
        q_vel = self.process_noise_scale * max(dt, 1e-6)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray) -> float:
        """Run one linear update step and return a log-likelihood-like score."""

        self.R = measurement_cov.copy()
        innovation = measurement - (self.H @ self.state)
        innovation_cov = self.H @ self.covariance @ self.H.T + self.R
        kalman_gain = self.covariance @ self.H.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(self.covariance.shape[0], dtype=float)
        self.covariance = (identity - kalman_gain @ self.H) @ self.covariance
        return _innovation_log_score(innovation, innovation_cov)


class HoverModel(KalmanModel):
    def __init__(self) -> None:
        super().__init__(name="hover", process_noise_scale=6.0, velocity_damping=0.15)


class CruiseModel(KalmanModel):
    def __init__(self) -> None:
        super().__init__(name="cruise", process_noise_scale=2.0, velocity_damping=0.98)


class TurnModel(KalmanModel):
    def __init__(self) -> None:
        super().__init__(name="turn", process_noise_scale=4.5, velocity_damping=0.92)


@dataclass(frozen=True)
class IMMResult:
    """Fused IMM estimate."""

    lat: float
    lon: float
    speed_mps: float
    azimuth_deg: float
    model_weights: np.ndarray
    covariance: np.ndarray
    dominant_mode: str


class IMMFilter:
    """Interacting Multiple Model filter with hover, cruise, and turn modes."""

    def __init__(self, transition_matrix: np.ndarray | None = None) -> None:
        self.transition_matrix = (
            np.asarray(
                transition_matrix,
                dtype=float,
            )
            if transition_matrix is not None
            else np.array(
                [
                    [0.90, 0.08, 0.02],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.10, 0.85],
                ],
                dtype=float,
            )
        )
        self.models: List[KalmanModel] = [HoverModel(), CruiseModel(), TurnModel()]
        self.weights = np.array([0.2, 0.6, 0.2], dtype=float)
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._geod = pyproj.Geod(ellps="WGS84")
        self._last_fix: PositionEstimate | None = None
        self._last_result: IMMResult | None = None

    def update(
        self,
        position_fix: PositionEstimate,
        dt: float,
        is_flat: bool = False,
    ) -> IMMResult:
        """Run one full IMM step: mixing, predict, update, mode update, fusion."""

        if dt <= 0:
            raise ValueError("dt must be positive")

        measurement = self._fix_to_measurement(position_fix)
        measurement_cov = self._measurement_covariance(position_fix, is_flat=is_flat)

        mixed_states, mixed_covariances, normalizers = self._mix_states()
        for model, mixed_state, mixed_cov in zip(self.models, mixed_states, mixed_covariances):
            model.set_state(mixed_state, mixed_cov)
            model.predict(dt)

        compat = self._model_compatibility(position_fix, dt)
        log_posterior = np.zeros(len(self.models), dtype=float)
        for index, model in enumerate(self.models):
            base_log_likelihood = model.update(measurement, measurement_cov)
            log_posterior[index] = (
                base_log_likelihood
                + 3.0 * math.log(max(compat[index], 1e-12))
                + math.log(max(normalizers[index], 1e-12))
            )

        max_log = float(np.max(log_posterior))
        posterior = np.exp(log_posterior - max_log)
        posterior_sum = float(np.sum(posterior))
        if not np.isfinite(posterior_sum) or posterior_sum <= 0.0:
            self.weights = np.full(len(self.models), 1.0 / len(self.models), dtype=float)
        else:
            self.weights = posterior / posterior_sum

        fused_state, fused_cov = self._fuse_estimate()
        lat, lon = self._xy_to_latlon(fused_state[0], fused_state[1])
        speed_mps = float(np.hypot(fused_state[2], fused_state[3]))
        azimuth_deg = normalize_azimuth_deg(math.degrees(math.atan2(fused_state[2], fused_state[3])))
        dominant_mode = self.models[int(np.argmax(self.weights))].name

        result = IMMResult(
            lat=float(lat),
            lon=float(lon),
            speed_mps=speed_mps,
            azimuth_deg=azimuth_deg,
            model_weights=self.weights.copy(),
            covariance=fused_cov.copy(),
            dominant_mode=dominant_mode,
        )
        self._last_fix = position_fix
        self._last_result = result
        return result

    def get_hdop(self) -> float:
        """Return a GDOP-like horizontal uncertainty estimate in meters."""

        if self._last_result is None:
            return 0.0
        covariance = self._last_result.covariance
        return float(math.sqrt(max(np.trace(covariance[0:2, 0:2]), 0.0)))

    def _mix_states(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Compute mixed initial conditions for all models."""

        num_models = len(self.models)
        predicted_mode_prob = self.transition_matrix.T @ self.weights
        mixed_states: list[np.ndarray] = []
        mixed_covariances: list[np.ndarray] = []

        for target_idx in range(num_models):
            denom = max(predicted_mode_prob[target_idx], 1e-12)
            mixing_probs = (
                self.transition_matrix[:, target_idx] * self.weights
            ) / denom
            state_stack = np.vstack([model.state for model in self.models])
            mixed_state = np.sum(mixing_probs[:, None] * state_stack, axis=0)

            mixed_cov = np.zeros((4, 4), dtype=float)
            for source_idx, model in enumerate(self.models):
                delta = model.state - mixed_state
                mixed_cov += mixing_probs[source_idx] * (
                    model.covariance + np.outer(delta, delta)
                )
            mixed_states.append(mixed_state)
            mixed_covariances.append(mixed_cov)

        return mixed_states, mixed_covariances, predicted_mode_prob

    def _fuse_estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Fuse model-conditioned states into one IMM estimate."""

        state_stack = np.vstack([model.state for model in self.models])
        fused_state = np.sum(self.weights[:, None] * state_stack, axis=0)
        fused_cov = np.zeros((4, 4), dtype=float)
        for idx, model in enumerate(self.models):
            delta = model.state - fused_state
            fused_cov += self.weights[idx] * (
                model.covariance + np.outer(delta, delta)
            )
        return fused_state, fused_cov

    def _measurement_covariance(
        self,
        position_fix: PositionEstimate,
        *,
        is_flat: bool,
    ) -> np.ndarray:
        """Build measurement covariance in the local meter frame."""

        sigma_pos = float(math.sqrt(max(np.mean(np.diag(position_fix.cov_matrix)), 1.0)))
        if is_flat:
            sigma_pos *= 10.0
        sigma_vel = max(position_fix.speed_mps * 0.15, 2.0)
        return np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2]).astype(float)

    def _model_compatibility(self, position_fix: PositionEstimate, dt: float) -> np.ndarray:
        """Estimate how well the measurement matches each motion mode."""

        speed = max(position_fix.speed_mps, 0.0)
        turn_rate = 0.0
        if self._last_fix is not None:
            delta = normalize_azimuth_deg(position_fix.azimuth_deg - self._last_fix.azimuth_deg)
            signed_delta = delta if delta <= 180.0 else delta - 360.0
            turn_rate = abs(signed_delta) / dt

        hover_score = math.exp(-((speed / 3.0) ** 2))
        cruise_speed_score = 1.0 - math.exp(-((speed / 8.0) ** 2))
        cruise_turn_score = math.exp(-((turn_rate / 2.5) ** 2))
        cruise_score = max(cruise_speed_score * cruise_turn_score, 1e-6)

        turn_speed_score = 1.0 - math.exp(-((speed / 6.0) ** 2))
        turn_rate_score = 1.0 - math.exp(-((turn_rate / 3.0) ** 2))
        turn_persistence = 1.0 + min(turn_rate / 10.0, 3.0)
        turn_score = max(turn_speed_score * turn_rate_score * turn_persistence, 1e-6)

        compat = np.array(
            [
                max(hover_score, 1e-6),
                cruise_score,
                turn_score,
            ],
            dtype=float,
        )
        return compat

    def _fix_to_measurement(self, position_fix: PositionEstimate) -> np.ndarray:
        """Convert geodetic fix into local-meter position and velocity."""

        if self._origin_lat is None or self._origin_lon is None:
            self._origin_lat = position_fix.lat
            self._origin_lon = position_fix.lon

        azimuth_fwd, _, distance_m = self._geod.inv(
            self._origin_lon,
            self._origin_lat,
            position_fix.lon,
            position_fix.lat,
        )
        azimuth_rad = math.radians(normalize_azimuth_deg(azimuth_fwd))
        x_m = distance_m * math.sin(azimuth_rad)
        y_m = distance_m * math.cos(azimuth_rad)

        motion_rad = math.radians(normalize_azimuth_deg(position_fix.azimuth_deg))
        vx = position_fix.speed_mps * math.sin(motion_rad)
        vy = position_fix.speed_mps * math.cos(motion_rad)
        return np.array([x_m, y_m, vx, vy], dtype=float)

    def _xy_to_latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Convert local-meter coordinates back to latitude/longitude."""

        if self._origin_lat is None or self._origin_lon is None:
            raise RuntimeError("IMMFilter has not been initialized with a measurement")

        distance_m = math.hypot(x_m, y_m)
        azimuth_deg = normalize_azimuth_deg(math.degrees(math.atan2(x_m, y_m)))
        lon, lat, _ = self._geod.fwd(
            self._origin_lon,
            self._origin_lat,
            azimuth_deg,
            distance_m,
        )
        return float(lat), float(lon)
