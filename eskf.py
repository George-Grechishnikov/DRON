"""Simplified error-state Kalman filter for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from local_frame import LocalFrame


GRAVITY_ENU = np.array([0.0, 0.0, -9.80665], dtype=float)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quaternion_from_omega(omega_rps: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(omega_rps, dtype=float)
    angle = float(np.linalg.norm(omega) * dt)
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = omega / max(np.linalg.norm(omega), 1e-12)
    half = 0.5 * angle
    return np.array(
        [math.cos(half), *(math.sin(half) * axis)],
        dtype=float,
    )


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = _normalize_quaternion(quaternion)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


@dataclass(frozen=True)
class ImuSample:
    """One IMU sample in the body frame."""

    timestamp_s: float
    accel_mps2: np.ndarray
    gyro_rps: np.ndarray


@dataclass(frozen=True)
class NavState:
    """Nominal navigation state."""

    p_enu_m: np.ndarray
    v_enu_mps: np.ndarray
    q_body_to_enu: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray


class ESKF:
    """Practical ESKF-like navigation core with 15-state covariance."""

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        *,
        origin_alt_m: float = 0.0,
        accel_noise: float = 0.2,
        gyro_noise: float = math.radians(0.5),
        accel_bias_rw: float = 0.01,
        gyro_bias_rw: float = math.radians(0.05),
    ) -> None:
        self.frame = LocalFrame(origin_lat, origin_lon, alt0_m=origin_alt_m)
        self._state = NavState(
            p_enu_m=np.zeros(3, dtype=float),
            v_enu_mps=np.zeros(3, dtype=float),
            q_body_to_enu=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            accel_bias=np.zeros(3, dtype=float),
            gyro_bias=np.zeros(3, dtype=float),
        )
        self._covariance = np.eye(15, dtype=float) * 1.0
        self.accel_noise = float(accel_noise)
        self.gyro_noise = float(gyro_noise)
        self.accel_bias_rw = float(accel_bias_rw)
        self.gyro_bias_rw = float(gyro_bias_rw)

    @property
    def state(self) -> NavState:
        return NavState(
            p_enu_m=self._state.p_enu_m.copy(),
            v_enu_mps=self._state.v_enu_mps.copy(),
            q_body_to_enu=self._state.q_body_to_enu.copy(),
            accel_bias=self._state.accel_bias.copy(),
            gyro_bias=self._state.gyro_bias.copy(),
        )

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def predict(self, imu: ImuSample, dt: float) -> None:
        """Propagate the nominal state and covariance forward."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        corrected_gyro = np.asarray(imu.gyro_rps, dtype=float) - self._state.gyro_bias
        corrected_accel = np.asarray(imu.accel_mps2, dtype=float) - self._state.accel_bias

        dq = _quaternion_from_omega(corrected_gyro, dt)
        q_next = _normalize_quaternion(_quaternion_multiply(self._state.q_body_to_enu, dq))
        rotation = _rotation_matrix_from_quaternion(q_next)
        accel_enu = rotation @ corrected_accel + GRAVITY_ENU
        v_next = self._state.v_enu_mps + accel_enu * dt
        p_next = self._state.p_enu_m + self._state.v_enu_mps * dt + 0.5 * accel_enu * dt * dt

        self._state = NavState(
            p_enu_m=p_next,
            v_enu_mps=v_next,
            q_body_to_enu=q_next,
            accel_bias=self._state.accel_bias.copy(),
            gyro_bias=self._state.gyro_bias.copy(),
        )
        self._propagate_covariance(dt)

    def update_baro(self, msl_m: float, sigma_m: float) -> None:
        """Update the vertical position from a barometric altitude observation."""

        z_meas = float(msl_m) - self.frame.alt0_m
        H = np.zeros((1, 15), dtype=float)
        H[0, 2] = 1.0
        self._kalman_update(
            innovation=np.array([z_meas - self._state.p_enu_m[2]], dtype=float),
            H=H,
            R=np.array([[max(float(sigma_m), 1e-3) ** 2]], dtype=float),
        )

    def update_heading(self, heading_deg: float, sigma_deg: float) -> None:
        """Weak heading correction, e.g. magnetometer/course alignment."""

        measured_yaw = math.radians(float(heading_deg) % 360.0)
        current_yaw = _yaw_from_quaternion(self._state.q_body_to_enu)
        innovation = ((measured_yaw - current_yaw + math.pi) % (2.0 * math.pi)) - math.pi
        H = np.zeros((1, 15), dtype=float)
        H[0, 8] = 1.0
        self._kalman_update(
            innovation=np.array([innovation], dtype=float),
            H=H,
            R=np.array([[max(math.radians(float(sigma_deg)), 1e-4) ** 2]], dtype=float),
        )

    def update_position(self, p_enu_m: np.ndarray, cov_2x2: np.ndarray) -> None:
        """Correct horizontal position from terrain or GNSS-like observations."""

        position = np.asarray(p_enu_m, dtype=float)
        if position.shape != (2,):
            raise ValueError("p_enu_m must have shape (2,)")
        cov = np.asarray(cov_2x2, dtype=float)
        if cov.shape != (2, 2):
            raise ValueError("cov_2x2 must have shape (2, 2)")
        innovation = position - self._state.p_enu_m[:2]
        H = np.zeros((2, 15), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        self._kalman_update(innovation=innovation, H=H, R=cov)

    def update_velocity(self, v_enu_mps: np.ndarray, cov: np.ndarray) -> None:
        """Correct velocity from Doppler or equivalent sources."""

        velocity = np.asarray(v_enu_mps, dtype=float)
        if velocity.shape not in {(2,), (3,)}:
            raise ValueError("v_enu_mps must have shape (2,) or (3,)")
        if velocity.shape == (2,):
            velocity = np.array([velocity[0], velocity[1], self._state.v_enu_mps[2]], dtype=float)
            R = np.eye(3, dtype=float)
            R[:2, :2] = np.asarray(cov, dtype=float)
            R[2, 2] = 100.0
        else:
            R = np.asarray(cov, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("Velocity covariance must have shape (3, 3) after normalization")
        innovation = velocity - self._state.v_enu_mps
        H = np.zeros((3, 15), dtype=float)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0
        self._kalman_update(innovation=innovation, H=H, R=R)

    def to_geodetic(self) -> tuple[float, float, float]:
        """Return current latitude, longitude, and MSL altitude."""

        lat, lon, alt_m = self.frame.to_geodetic(self._state.p_enu_m)
        return (float(lat), float(lon), float(alt_m))

    def _propagate_covariance(self, dt: float) -> None:
        F = np.eye(15, dtype=float)
        F[0:3, 3:6] = np.eye(3, dtype=float) * dt
        F[3:6, 9:12] = -np.eye(3, dtype=float) * dt
        F[6:9, 12:15] = -np.eye(3, dtype=float) * dt
        q_acc = (self.accel_noise ** 2) * max(dt, 1e-6)
        q_gyro = (self.gyro_noise ** 2) * max(dt, 1e-6)
        q_ab = (self.accel_bias_rw ** 2) * max(dt, 1e-6)
        q_gb = (self.gyro_bias_rw ** 2) * max(dt, 1e-6)
        Q = np.zeros((15, 15), dtype=float)
        Q[0:3, 0:3] = np.eye(3, dtype=float) * q_acc * dt * dt
        Q[3:6, 3:6] = np.eye(3, dtype=float) * q_acc
        Q[6:9, 6:9] = np.eye(3, dtype=float) * q_gyro
        Q[9:12, 9:12] = np.eye(3, dtype=float) * q_ab
        Q[12:15, 12:15] = np.eye(3, dtype=float) * q_gb
        self._covariance = F @ self._covariance @ F.T + Q
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._covariance += np.eye(15, dtype=float) * 1e-9

    def _kalman_update(self, *, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self._covariance @ H.T + R
        K = self._covariance @ H.T @ np.linalg.inv(S)
        delta = K @ innovation
        self._inject_error_state(delta)
        identity = np.eye(self._covariance.shape[0], dtype=float)
        self._covariance = (identity - K @ H) @ self._covariance @ (identity - K @ H).T + K @ R @ K.T
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._covariance += np.eye(15, dtype=float) * 1e-9

    def _inject_error_state(self, delta: np.ndarray) -> None:
        position = self._state.p_enu_m + delta[0:3]
        velocity = self._state.v_enu_mps + delta[3:6]
        dtheta = delta[6:9]
        dq = np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]], dtype=float)
        quaternion = _normalize_quaternion(_quaternion_multiply(self._state.q_body_to_enu, dq))
        accel_bias = self._state.accel_bias + delta[9:12]
        gyro_bias = self._state.gyro_bias + delta[12:15]
        self._state = NavState(
            p_enu_m=position,
            v_enu_mps=velocity,
            q_body_to_enu=quaternion,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )
