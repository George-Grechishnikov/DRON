from __future__ import annotations

from pathlib import Path

from case_reader import CaseInputConfig, iter_case_unified_samples
from sim_generator import nmea_checksum


def _gpgga_line(timestamp: str, radar_alt_m: float) -> str:
    payload = ",".join(
        [
            "GPGGA",
            timestamp,
            "5545.1234",
            "N",
            "03736.5678",
            "E",
            "1",
            "08",
            "0.9",
            f"{radar_alt_m:.1f}",
            "M",
            "46.9",
            "M",
            "",
            "",
        ]
    )
    return f"${payload}*{nmea_checksum(payload)}\n"


def test_iter_case_unified_samples_merges_case_inputs(tmp_path: Path) -> None:
    radar_path = tmp_path / "radar_data.nmea"
    truth_path = tmp_path / "truth.csv"
    baro_path = tmp_path / "barometer.csv"
    dem_path = tmp_path / "terrain.tif"
    dem_path.write_text("stub", encoding="utf-8")
    radar_path.write_text(
        _gpgga_line("123519.000", 1200.0) + _gpgga_line("123519.200", 1198.5),
        encoding="ascii",
    )
    truth_path.write_text(
        "\n".join(
            [
                "timestamp,lat,lon,alt_msl,heading_deg,speed_mps",
                "0.0,60.5,90.3,1500.0,45.0,50.0",
                "0.2,60.5001,90.3001,1500.0,46.0,50.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    baro_path.write_text(
        "\n".join(
            [
                "timestamp,baro_alt_m",
                "0.0,1500.0",
                "0.2,1499.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = list(
        iter_case_unified_samples(
            CaseInputConfig(
                dem_path=dem_path,
                radar_data_path=radar_path,
                truth_path=truth_path,
                barometer_path=baro_path,
                sample_rate_hz=5.0,
                gnss_drop_after_s=0.1,
            )
        )
    )

    assert len(samples) == 2
    assert samples[0]["gnss_available"] is True
    assert samples[1]["gnss_available"] is False
    assert samples[0]["alt_msl"] == 1500.0
    assert samples[1]["truth_lat"] == 60.5001
    assert samples[1]["nav_mode"] == "TERRAIN_NAV"
