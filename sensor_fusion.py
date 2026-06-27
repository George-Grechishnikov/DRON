"""Sensor-fusion helpers for adaptive measurement updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eskf import ESKF, NavState, _normalize_quaternion


@dataclass(frozen=True)
class SensorSample:
    """One sensor observation to be fused."""

    timestamp_s: float
    kind: str
    value: np.ndarray
    quality: float


def adaptive_R(kind: str, quality: float, roll_deg: float, surface_class: str) -> np.ndarray:
    """Return a dynamic measurement covariance based on sensor context."""

    q = float(np.clip(quality, 1e-3, 1.0))
    roll_scale = 1.0 + abs(float(roll_deg)) / 30.0
    if kind == "lidar":
        sigma = 1.0
        if surface_class in {"water", "clouds", "rain"}:
            sigma *= 8.0
        sigma *= roll_scale / q
        return np.array([[sigma * sigma]], dtype=float)
    if kind == "radar_alt":
        sigma = 3.0 * roll_scale / q
        return np.array([[sigma * sigma]], dtype=float)
    if kind == "radar":
        sigma = 5.0 * roll_scale / q
        return np.array([[sigma * sigma]], dtype=float)
    if kind == "baro":
        sigma = 8.0 / q
        return np.array([[sigma * sigma]], dtype=float)
    if kind == "doppler":
        sigma_xy = 1.5 * roll_scale / q
        return np.diag([sigma_xy * sigma_xy, sigma_xy * sigma_xy, 4.0]).astype(float)
    raise ValueError(f"Unsupported sensor kind: {kind}")


class AdaptiveGate:
    """Chi-squared style innovation gate."""

    def chi2_accept(
        self,
        innovation: np.ndarray,
        innovation_cov: np.ndarray,
        dof: int,
        threshold: float,
    ) -> bool:
        del dof
        innov = np.asarray(innovation, dtype=float)
        cov = np.asarray(innovation_cov, dtype=float) + np.eye(innovation_cov.shape[0], dtype=float) * 1e-9
        metric = float(innov.T @ np.linalg.inv(cov) @ innov)
        return bool(metric <= threshold)


class FederatedFusion:
    """No-reset federated fusion over several local ESKF filters."""

    def __init__(self, local_filters: dict[str, ESKF]) -> None:
        if not local_filters:
            raise ValueError("local_filters must not be empty")
        self.local_filters = dict(local_filters)

    def apply_sample(
        self,
        filter_name: str,
        sample: SensorSample,
        *,
        roll_deg: float = 0.0,
        surface_class: str = "land",
        gate: AdaptiveGate | None = None,
        threshold: float = 9.21,
    ) -> bool:
        """Apply one measurement to a local filter if it passes gating."""

        eskf = self.local_filters[filter_name]
        gate = gate or AdaptiveGate()
        measurement_cov = adaptive_R(sample.kind, sample.quality, roll_deg, surface_class)
        value = np.asarray(sample.value, dtype=float)

        if sample.kind in {"baro", "lidar", "radar_alt", "radar"}:
            innovation = np.array([value.reshape(-1)[0] - eskf.to_geodetic()[2]], dtype=float)
            innovation_cov = np.array([[eskf.covariance[2, 2] + measurement_cov[0, 0]]], dtype=float)
            if not gate.chi2_accept(innovation, innovation_cov, dof=1, threshold=threshold):
                return False
            eskf.update_baro(float(value.reshape(-1)[0]), sigma_m=math.sqrt(measurement_cov[0, 0]))
            return True

        if sample.kind == "doppler":
            velocity = value
            if velocity.shape == (2,):
                velocity = np.array([velocity[0], velocity[1], 0.0], dtype=float)
            innovation = velocity - eskf.state.v_enu_mps
            innovation_cov = eskf.covariance[3:6, 3:6] + measurement_cov
            if not gate.chi2_accept(innovation, innovation_cov, dof=3, threshold=threshold):
                return False
            eskf.update_velocity(velocity, measurement_cov)
            return True

        raise ValueError(f"Unsupported sensor kind: {sample.kind}")

    def fuse(self) -> NavState:
        """Fuse local navigation states into one federated state."""

        states = [eskf.state for eskf in self.local_filters.values()]
        covariances = [eskf.covariance for eskf in self.local_filters.values()]
        weights = []
        for covariance in covariances:
            pos_trace = float(np.trace(covariance[0:3, 0:3]))
            weights.append(1.0 / max(pos_trace, 1e-6))
        weight_array = np.asarray(weights, dtype=float)
        weight_array /= np.sum(weight_array)

        p_enu = np.sum([w * state.p_enu_m for w, state in zip(weight_array, states)], axis=0)
        v_enu = np.sum([w * state.v_enu_mps for w, state in zip(weight_array, states)], axis=0)
        accel_bias = np.sum([w * state.accel_bias for w, state in zip(weight_array, states)], axis=0)
        gyro_bias = np.sum([w * state.gyro_bias for w, state in zip(weight_array, states)], axis=0)
        quaternion = _normalize_quaternion(np.sum([w * state.q_body_to_enu for w, state in zip(weight_array, states)], axis=0))
        return NavState(
            p_enu_m=np.asarray(p_enu, dtype=float),
            v_enu_mps=np.asarray(v_enu, dtype=float),
            q_body_to_enu=quaternion.astype(float, copy=False),
            accel_bias=np.asarray(accel_bias, dtype=float),
            gyro_bias=np.asarray(gyro_bias, dtype=float),
        )
