from __future__ import annotations

import queue
from pathlib import Path

import numpy as np

from correlator import CorrelationResult
from imm_filter import IMMResult
from visualizer import TerrainNavigatorDash, create_arrow_shape, export_flight_report


def _fake_corr() -> CorrelationResult:
    return CorrelationResult(
        best_azimuth_deg=45.0,
        best_offset_steps=3,
        best_offset_m=90.0,
        best_offset_subsample_steps=3.2,
        best_offset_subsample_m=96.0,
        peak_correlation=0.87,
        confidence=0.42,
        is_reliable=True,
        pslr_db=5.5,
        ambiguity_peak_count=1,
        peak_isolation_m=120.0,
        is_ambiguous=False,
        heatmap=np.array([[0.1, 0.2], [0.4, 0.9]], dtype=float),
        azimuths_deg=np.array([0.0, 45.0], dtype=float),
        best_reference_profile=np.array([100.0, 105.0, 103.0], dtype=float),
    )


def _fake_fix() -> IMMResult:
    return IMMResult(
        lat=60.5001,
        lon=90.3002,
        speed_mps=52.0,
        azimuth_deg=44.0,
        model_weights=np.array([0.1, 0.75, 0.15], dtype=float),
        covariance=np.diag([36.0, 49.0, 4.0, 4.0]).astype(float),
        dominant_mode="cruise",
    )


def test_update_all_panels_returns_four_figures_and_store() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    state_queue.put(
        {
            "corr": _fake_corr(),
            "fix": _fake_fix(),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "hdop": 12.0,
            "nav_mode": "terrain_update_accepted",
            "used_prediction_only": False,
            "selected_window_size": 40,
            "gnss_available": False,
            "truth": {
                "lat": 60.5002,
                "lon": 90.3003,
                "heading_deg": 44.0,
                "speed_mps": 52.0,
            },
            "observability": {
                "crlb_m": 42.0,
                "gradient_energy": 1.3,
                "efficiency_hint": 0.42,
                "is_informative": True,
            },
        }
    )

    heatmap_fig, terrain_fig, profiles_fig, telemetry_fig, store = dash_app.update_all_panels(
        1,
        {"history": []},
    )

    assert len(heatmap_fig.data) >= 2
    assert len(terrain_fig.data) >= 4
    assert len(profiles_fig.data) >= 2
    assert len(telemetry_fig.data) >= 2
    assert len(store["history"]) == 1
    assert any("Nav mode" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("Window size" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("GNSS" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("CRLB" in annotation["text"] for annotation in telemetry_fig.layout.annotations)


def test_create_arrow_shape_returns_three_segments() -> None:
    shapes = create_arrow_shape(60.5, 90.3, 45.0)
    assert len(shapes) == 3
    assert all(shape["type"] == "line" for shape in shapes)


def test_export_flight_report_writes_html(tmp_path: Path) -> None:
    output_path = tmp_path / "report.html"
    export_flight_report([_fake_fix(), _fake_fix()], str(output_path))
    assert output_path.exists()
    assert "plotly" in output_path.read_text(encoding="utf-8").lower()


def test_send_control_command_pushes_to_control_queue() -> None:
    state_queue: queue.Queue = queue.Queue()
    control_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue, control_queue=control_queue)

    dash_app.send_control_command({"type": "set_gnss_enabled", "enabled": False})

    assert control_queue.get_nowait() == {"type": "set_gnss_enabled", "enabled": False}
