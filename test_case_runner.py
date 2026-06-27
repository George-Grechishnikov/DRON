from __future__ import annotations

from pathlib import Path

import pytest

from case_runner import main as case_runner_main
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


def test_case_runner_validate_only(tmp_path: Path, capsys) -> None:
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    (dem_dir / "terrain.tif").write_text("stub", encoding="utf-8")
    (tmp_path / "radar_data.nmea").write_text(
        _gpgga_line("123519.000", 1200.0) + _gpgga_line("123519.200", 1198.5),
        encoding="ascii",
    )
    (tmp_path / "truth.csv").write_text(
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
    (tmp_path / "barometer.csv").write_text(
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dem_path: {dem_dir / 'terrain.tif'}",
                f"radar_data_path: {tmp_path / 'radar_data.nmea'}",
                f"truth_path: {tmp_path / 'truth.csv'}",
                f"barometer_path: {tmp_path / 'barometer.csv'}",
                "sample_rate_hz: 5.0",
                "visualization:",
                "  export_report_path: output/terrain_navigator_report.html",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = case_runner_main(["--config", str(config_path), "--validate-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Case inputs validated" in captured.out
    assert "parsed_samples: 2" in captured.out


def test_case_runner_validate_only_fails_on_parse_errors(tmp_path: Path) -> None:
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    (dem_dir / "terrain.tif").write_text("stub", encoding="utf-8")
    (tmp_path / "radar_data.nmea").write_text(
        _gpgga_line("123519.000", 1200.0) + _gpgga_line("123521.000", 1198.5),
        encoding="ascii",
    )
    (tmp_path / "truth.csv").write_text(
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
    (tmp_path / "barometer.csv").write_text(
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dem_path: {dem_dir / 'terrain.tif'}",
                f"radar_data_path: {tmp_path / 'radar_data.nmea'}",
                f"truth_path: {tmp_path / 'truth.csv'}",
                f"barometer_path: {tmp_path / 'barometer.csv'}",
                "sample_rate_hz: 5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="radar_data.nmea timestamps diverge from truth.csv"):
        case_runner_main(["--config", str(config_path), "--validate-only"])
