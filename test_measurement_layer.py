from __future__ import annotations

import numpy as np

from constants import FIXED_BARO_ALTITUDE_M
from correlator import Correlator
from measurement_layer import BaroSample, BaroTrack, agl_to_terrain, frames_to_terrain_profile, update_terrain_bias
from nmea_parser import NMEAFrame


def test_agl_to_terrain_reconstructs_absolute_relief() -> None:
    radar_agl = np.array([500.0, 520.0, 540.0], dtype=float)
    baro_msl = np.full((3,), FIXED_BARO_ALTITUDE_M, dtype=float)

    terrain = agl_to_terrain(radar_agl, baro_msl)

    assert np.allclose(terrain, np.array([1000.0, 980.0, 960.0], dtype=float))


def test_baro_track_interpolates_between_samples() -> None:
    track = BaroTrack(
        [
            BaroSample(timestamp_s=0.0, msl_m=1500.0),
            BaroSample(timestamp_s=10.0, msl_m=1510.0),
        ]
    )

    assert np.isclose(track.msl_at(5.0), 1505.0)


def test_frames_to_terrain_profile_uses_fixed_baro_and_valid_mask() -> None:
    frames = [
        NMEAFrame(timestamp_utc="123519.000", radar_alt_m=545.4, raw="", valid=True),
        NMEAFrame(timestamp_utc="123520.000", radar_alt_m=float("nan"), raw="", valid=False),
        NMEAFrame(timestamp_utc="123521.000", radar_alt_m=500.0, raw="", valid=True),
    ]

    profile = frames_to_terrain_profile(frames, BaroTrack(default_msl_m=FIXED_BARO_ALTITUDE_M))

    assert profile.values_m.shape == (3,)
    assert np.isclose(profile.values_m[0], FIXED_BARO_ALTITUDE_M - 545.4)
    assert np.isnan(profile.values_m[1])
    assert np.isclose(profile.values_m[2], 1000.0)
    assert np.array_equal(profile.valid_mask, np.array([True, False, True], dtype=bool))


def test_frames_to_terrain_profile_applies_terrain_bias_correction() -> None:
    frames = [
        NMEAFrame(timestamp_utc="123519.000", radar_alt_m=480.0, raw="", valid=True),
        NMEAFrame(timestamp_utc="123520.000", radar_alt_m=500.0, raw="", valid=True),
    ]

    profile = frames_to_terrain_profile(
        frames,
        BaroTrack(default_msl_m=FIXED_BARO_ALTITUDE_M),
        terrain_bias_m=20.0,
    )

    assert np.allclose(profile.values_m, np.array([1000.0, 980.0], dtype=float))
    assert np.isclose(profile.terrain_bias_m, 20.0)


def test_update_terrain_bias_moves_estimate_toward_residual() -> None:
    updated = update_terrain_bias(prev_bias_m=10.0, residual_m=20.0, gain=0.25)

    assert np.isclose(updated, 12.5)


def test_terrain_bias_correction_improves_hybrid_matching() -> None:
    true_profile = np.array([1000.0, 980.0, 960.0, 940.0], dtype=float)
    biased_measurement = true_profile + 20.0
    ref_matrix = np.array(
        [[1020.0, 1000.0, 980.0, 960.0], true_profile],
        dtype=float,
    )
    azimuths = np.array([0.0, 1.0], dtype=float)
    correlator = Correlator(
        profile_length_m=120.0,
        step_m=30.0,
        max_offset_m=0.0,
        metric="hybrid",
        alpha=0.5,
        beta=0.5,
        msd_scale_m2=25.0,
    )

    no_bias_result = correlator.compute(biased_measurement, ref_matrix, azimuths_deg=azimuths)
    corrected_result = correlator.compute(biased_measurement - 20.0, ref_matrix, azimuths_deg=azimuths)

    assert no_bias_result.best_azimuth_deg == 0.0
    assert corrected_result.best_azimuth_deg == 1.0
