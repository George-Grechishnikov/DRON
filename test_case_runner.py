from __future__ import annotations

from pathlib import Path

from case_runner import main as case_runner_main


def test_case_runner_validate_only(tmp_path: Path, capsys) -> None:
    dem_dir = tmp_path / "dem"
    dem_dir.mkdir()
    (dem_dir / "terrain.tif").write_text("stub", encoding="utf-8")
    (tmp_path / "radar_data.nmea").write_text("", encoding="utf-8")
    (tmp_path / "truth.csv").write_text("timestamp,lat,lon,alt_msl,heading_deg,speed_mps\n", encoding="utf-8")
    (tmp_path / "barometer.csv").write_text("timestamp,baro_alt_m\n", encoding="utf-8")
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
