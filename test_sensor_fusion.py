from __future__ import annotations

import numpy as np

from eskf import ESKF
from sensor_fusion import AdaptiveGate, FederatedFusion, SensorSample, adaptive_R


def test_adaptive_r_degrades_lidar_in_clouds() -> None:
    clear = adaptive_R("lidar", quality=1.0, roll_deg=0.0, surface_class="land")
    clouds = adaptive_R("lidar", quality=1.0, roll_deg=0.0, surface_class="clouds")

    assert clouds[0, 0] > clear[0, 0]


def test_doppler_update_improves_velocity_estimate() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    eskf.update_velocity(np.array([20.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.5)
    baseline_error = abs(np.linalg.norm(eskf.state.v_enu_mps[:2]) - 20.0)

    fusion = FederatedFusion({"doppler": eskf})
    applied = fusion.apply_sample(
        "doppler",
        SensorSample(timestamp_s=0.0, kind="doppler", value=np.array([35.0, 0.0, 0.0], dtype=float), quality=1.0),
        surface_class="land",
        threshold=1_000.0,
    )
    improved_error = abs(np.linalg.norm(eskf.state.v_enu_mps[:2]) - 35.0)

    assert applied is True
    assert improved_error < baseline_error + 15.0


def test_gate_rejects_gross_baro_outlier() -> None:
    eskf = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    fusion = FederatedFusion({"baro": eskf})
    before_alt = eskf.to_geodetic()[2]
    applied = fusion.apply_sample(
        "baro",
        SensorSample(timestamp_s=0.0, kind="baro", value=np.array([1700.0], dtype=float), quality=1.0),
        gate=AdaptiveGate(),
        threshold=9.21,
    )
    after_alt = eskf.to_geodetic()[2]

    assert applied is False
    assert np.isclose(after_alt, before_alt, atol=1e-6)


def test_federated_fusion_combines_local_states() -> None:
    left = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    right = ESKF(60.5, 90.3, origin_alt_m=1500.0)
    left.update_velocity(np.array([10.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.5)
    right.update_velocity(np.array([20.0, 0.0, 0.0], dtype=float), np.eye(3) * 0.5)

    fused = FederatedFusion({"left": left, "right": right}).fuse()
    left_vx = left.state.v_enu_mps[0]
    right_vx = right.state.v_enu_mps[0]

    assert min(left_vx, right_vx) < fused.v_enu_mps[0] < max(left_vx, right_vx)
