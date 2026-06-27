from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from main import PipelineArtifacts, ReplayMetrics
from nmea_parser import nmea_checksum
from web_backend import app, controller


def _auth_headers(origin: str | None = None) -> dict[str, str]:
    headers = {"x-dron-session-token": controller.session_token}
    if origin is not None:
        headers["origin"] = origin
    return headers


def _write_dem(path: Path) -> None:
    data = np.fromfunction(lambda r, c: 100.0 + r + c, (20, 20), dtype=float).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(90.0, 61.0, 0.01, 0.01),
    ) as dataset:
        dataset.write(data, 1)


def _gpgga_line(timestamp: str, altitude_m: float) -> str:
    payload = f"GPGGA,{timestamp},,,,,,,,{altitude_m:.1f},M,46.9,M,,"
    return f"${payload}*{nmea_checksum(payload)}\n"


def _build_case(tmp_path: Path) -> Path:
    dem_path = tmp_path / "terrain.tif"
    _write_dem(dem_path)

    radar_path = tmp_path / "radar_data.nmea"
    radar_path.write_text(
        "".join(
            [
                _gpgga_line("120000.000", 300.0),
                _gpgga_line("120000.200", 301.0),
                _gpgga_line("120000.400", 302.0),
            ]
        ),
        encoding="ascii",
    )

    truth_path = tmp_path / "truth.csv"
    truth_path.write_text(
        "\n".join(
            [
                "timestamp,lat,lon,alt_msl,heading_deg,speed_mps",
                "0.0,60.5000,90.3000,1500.0,90.0,25.0",
                "0.2,60.5001,90.3001,1500.0,90.0,25.0",
                "0.4,60.5002,90.3002,1500.0,90.0,25.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    barometer_path = tmp_path / "barometer.csv"
    barometer_path.write_text(
        "\n".join(
            [
                "timestamp,baro_alt_m",
                "0.0,1500.0",
                "0.2,1500.0",
                "0.4,1500.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dem_path: {dem_path}",
                f"radar_data_path: {radar_path}",
                f"truth_path: {truth_path}",
                f"barometer_path: {barometer_path}",
                "sample_rate_hz: 5.0",
                "gnss_drop_after_s: 0.3",
                "correlation:",
                "  max_offset_m: 2000.0",
                "  flat_terrain_threshold_m: 15.0",
                "  cold_start_windows: 1",
                "visualization:",
                f"  export_report_path: {tmp_path / 'report.html'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _reset_controller() -> None:
    controller.loaded_dataset = None
    controller.pipeline_artifacts = None
    controller.report_summary = None
    controller.processing_thread = None
    controller.processing_error = None
    controller.processing = False
    controller.playing = False
    controller.playback_speed = 1.0
    controller.playback_index = 0
    controller.playback_time_s = 0.0
    controller._last_tick_monotonic = None
    controller.force_gnss = None
    controller.settings_config_path = Path("input") / "incoming" / "config.yaml"
    controller.logs.clear()


def test_health_and_dataset_load_and_validate(tmp_path: Path) -> None:
    _reset_controller()
    config_path = _build_case(tmp_path)
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/api/dataset/load", json={"config_path": str(config_path)}, headers=_auth_headers())
    assert response.status_code == 200
    controller.settings_config_path = config_path
    payload = response.json()
    assert payload["loaded"] is True
    assert payload["radar_samples"] == 3
    assert payload["truth_samples"] == 3
    assert payload["barometer_samples"] == 3

    validate = client.get("/api/dataset/validate")
    assert validate.status_code == 200
    assert validate.json()["valid"] is True

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["total_samples"] == 3


def test_metrics_trajectory_heatmap_and_settings(tmp_path: Path) -> None:
    _reset_controller()
    config_path = _build_case(tmp_path)
    client = TestClient(app)
    load_response = client.post("/api/dataset/load", json={"config_path": str(config_path)}, headers=_auth_headers())
    assert load_response.status_code == 200
    controller.settings_config_path = config_path

    controller.pipeline_artifacts = PipelineArtifacts(
        history=[],
        metrics=ReplayMetrics(10.0, 20.0, 12.0, 1.0, 2.0),
        report_records=[
            {
                "index": 0,
                "timestamp": 0.0,
                "estimated_lat": 60.5000,
                "estimated_lon": 90.3000,
                "truth_lat": 60.5000,
                "truth_lon": 90.3000,
                "truth_alt_msl": 1500.0,
                "truth_heading_deg": 90.0,
                "truth_speed_mps": 25.0,
                "estimated_speed_mps": 24.8,
                "estimated_heading_deg": 89.0,
                "radar_alt_m": 300.0,
                "baro_alt_m": 1500.0,
                "terrain_h": 1200.0,
                "gnss_available": True,
                "mode": "GNSS",
                "terrain_active": False,
                "truth_error_m": 5.0,
                "best_azimuth_deg": 90.0,
                "best_offset_m": 0.0,
                "correlation_peak": 0.81,
            },
            {
                "index": 1,
                "timestamp": 0.2,
                "estimated_lat": 60.5001,
                "estimated_lon": 90.3001,
                "truth_lat": 60.5001,
                "truth_lon": 90.3001,
                "truth_alt_msl": 1500.0,
                "truth_heading_deg": 90.0,
                "truth_speed_mps": 25.0,
                "estimated_speed_mps": 24.7,
                "estimated_heading_deg": 88.0,
                "radar_alt_m": 301.0,
                "baro_alt_m": 1500.0,
                "terrain_h": 1199.0,
                "gnss_available": False,
                "mode": "TERRAIN_NAV",
                "terrain_active": True,
                "truth_error_m": 8.0,
                "best_azimuth_deg": 95.0,
                "best_offset_m": 20.0,
                "correlation_peak": 0.78,
            },
        ],
        report_context={"correlation_heatmap": [[0.1, 0.2], [0.3, 0.9]], "correlation_step_m": 20.0},
    )

    trajectory = client.get("/api/trajectory")
    assert trajectory.status_code == 200
    assert len(trajectory.json()["events"]) == 2

    heatmap = client.get("/api/correlation/heatmap")
    assert heatmap.status_code == 200
    assert heatmap.json()["best_azimuth"] == 90.0
    assert heatmap.json()["offsets"] == [0.0, 20.0]

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["mean_position_error_m"] == 6.5
    assert metrics.json()["time_in_terrain_nav_s"] == 0.2

    profiles = client.get("/api/profiles")
    assert profiles.status_code == 200
    assert len(profiles.json()["time"]) >= 1

    timeline = client.get("/api/timeline")
    assert timeline.status_code == 200
    assert len(timeline.json()["segments"]) >= 1

    set_speed = client.post("/api/replay/set_speed", json={"speed": 2.0}, headers=_auth_headers())
    assert set_speed.status_code == 200
    assert set_speed.json()["speed"] == 2.0

    force_off = client.post("/api/gnss/force_off", json={}, headers=_auth_headers())
    assert force_off.status_code == 200
    state_after_off = client.get("/api/state")
    assert state_after_off.status_code == 200
    assert state_after_off.json()["gnss_available"] is False

    force_on = client.post("/api/gnss/force_on", json={}, headers=_auth_headers())
    assert force_on.status_code == 200
    state_after_on = client.get("/api/state")
    assert state_after_on.status_code == 200
    assert state_after_on.json()["gnss_available"] is True

    restart = client.post("/api/replay/restart", json={}, headers=_auth_headers())
    assert restart.status_code == 200
    assert restart.json()["restarted"] is True

    settings_before = client.get("/api/settings")
    assert settings_before.status_code == 200
    updated = client.post("/api/settings", json={"gnss_drop_after_s": 1.5}, headers=_auth_headers())
    assert updated.status_code == 200
    config_payload = json.loads(json.dumps(client.get("/api/settings").json()))
    assert config_payload["gnss_drop_after_s"] == 1.5


def test_browser_origin_protection_blocks_untrusted_origin(tmp_path: Path) -> None:
    _reset_controller()
    config_path = _build_case(tmp_path)
    controller.settings_config_path = config_path
    client = TestClient(app)

    response = client.post(
        "/api/dataset/load",
        json={"config_path": str(config_path)},
        headers={"origin": "http://evil.example", "x-dron-session-token": controller.session_token},
    )

    assert response.status_code == 403


def test_mutating_endpoint_requires_session_token(tmp_path: Path) -> None:
    _reset_controller()
    config_path = _build_case(tmp_path)
    client = TestClient(app)

    response = client.post("/api/dataset/load", json={"config_path": str(config_path)})

    assert response.status_code == 403
