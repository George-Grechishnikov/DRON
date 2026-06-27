from __future__ import annotations

import math

import numpy as np

from eskf import ESKF, ImuSample


def _level_imu(*, yaw_rate_rps: float = 0.0) -> ImuSample:
    return ImuSample(
        timestamp_s=0.0,
        accel_mps2=np.array([0.0, 0.0, 9.80665], dtype=float),
        gyro_rps=np.array([0.0, 0.0, yaw_rate_rps], dtype=float),
    )


def test_pure_inertial_prediction_has_limited_short_term_drift() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)

    for _ in range(50):
        eskf.predict(_level_imu(), dt=0.1)

    state = eskf.state
    assert np.linalg.norm(state.p_enu_m) < 1.0
    assert np.linalg.norm(state.v_enu_mps) < 1.0


def test_update_baro_stabilizes_vertical_channel() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    eskf.predict(
        ImuSample(
            timestamp_s=0.0,
            accel_mps2=np.array([0.0, 0.0, 10.30665], dtype=float),
            gyro_rps=np.zeros(3, dtype=float),
        ),
        dt=1.0,
    )
    pre_update_alt = eskf.to_geodetic()[2]

    eskf.update_baro(1500.0, sigma_m=0.5)
    post_update_alt = eskf.to_geodetic()[2]

    assert abs(post_update_alt - 1500.0) < abs(pre_update_alt - 1500.0)


def test_update_heading_pulls_yaw_toward_measurement() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    for _ in range(10):
        eskf.predict(_level_imu(yaw_rate_rps=math.radians(9.0)), dt=0.1)
    yaw_before = math.degrees(math.atan2(eskf.state.q_body_to_enu[3], eskf.state.q_body_to_enu[0])) * 2.0

    eskf.update_heading(0.0, sigma_deg=2.0)
    yaw_after = math.degrees(math.atan2(eskf.state.q_body_to_enu[3], eskf.state.q_body_to_enu[0])) * 2.0

    assert abs(yaw_after) < abs(yaw_before)


def test_quaternion_stays_normalized_and_covariance_is_spd() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    for _ in range(20):
        eskf.predict(_level_imu(yaw_rate_rps=math.radians(3.0)), dt=0.2)
    covariance = eskf.covariance
    eigvals = np.linalg.eigvalsh(covariance)

    assert np.isclose(np.linalg.norm(eskf.state.q_body_to_enu), 1.0, atol=1e-6)
    assert np.allclose(covariance, covariance.T, atol=1e-9)
    assert np.all(eigvals > 0.0)
