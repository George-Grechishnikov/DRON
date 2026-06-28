from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np

from correlator import CorrelationResult
from imm_filter import IMMResult
from visualizer import (
    TerrainNavigatorDash,
    build_velocity_annotation,
    create_arrow_shape,
    create_probability_ellipse_traces,
    export_flight_report,
)


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


def test_update_all_panels_returns_figures_and_store() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    state_queue.put(
        {
            "corr": _fake_corr(),
            "fix": _fake_fix(),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "dem_patch_transform": (0.001, 0.0, 90.299, 0.0, -0.001, 60.501),
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
            "event_ingest_monotonic_s": time.perf_counter() - 0.05,
            "pipeline_emitted_monotonic_s": time.perf_counter() - 0.02,
            "pipeline_latency_ms": 50.0,
            "integrity_status": "OK",
            "runtime_stats": {
                "frame_drop_count": 0,
                "state_queue_replacements": 0,
                "state_payloads_enqueued": 1,
            },
        }
    )

    heatmap_fig, terrain_fig, profiles_fig, telemetry_fig, metrics_panel, store = dash_app.update_all_panels(
        1,
        {"history": []},
    )

    assert len(heatmap_fig.data) >= 2
    assert len(terrain_fig.data) >= 6
    assert len(profiles_fig.data) >= 2
    assert len(telemetry_fig.data) >= 2
    assert len(metrics_panel) >= 8
    assert len(store["history"]) == 1
    assert "latest_state" in store
    assert any("Режим навигации" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("Размер окна" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("GNSS" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("CRLB" in annotation["text"] for annotation in telemetry_fig.layout.annotations)
    assert any("Измеренный профиль" in str(trace.name) for trace in profiles_fig.data)
    assert any("Остаток" in str(trace.name) for trace in profiles_fig.data)
    assert any("Зона вероятности" in str(trace.name) for trace in terrain_fig.data)
    assert any("Скорость:" in annotation["text"] for annotation in terrain_fig.layout.annotations)
    assert "reaction_latency_ms" in store["latest_state"]
    assert "runtime_stats" in store["latest_state"]
    assert min(terrain_fig.data[0]["x"]) > 90.0
    assert min(terrain_fig.data[0]["y"]) > 60.0


def test_metrics_panel_can_be_rebuilt_from_store_summary() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    state_queue.put(
        {
            "corr": _fake_corr(),
            "fix": _fake_fix(),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "dem_patch_transform": (0.001, 0.0, 90.299, 0.0, -0.001, 60.501),
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
            "event_ingest_monotonic_s": time.perf_counter() - 0.05,
            "pipeline_emitted_monotonic_s": time.perf_counter() - 0.02,
            "pipeline_latency_ms": 50.0,
            "integrity_status": "OK",
            "runtime_stats": {
                "frame_drop_count": 0,
                "state_queue_replacements": 0,
                "state_payloads_enqueued": 1,
            },
        }
    )

    _, _, _, _, _, store = dash_app.update_all_panels(1, {"history": []})
    metrics_panel = dash_app._build_metrics_panel(store["latest_state"])

    values = [str(card.children[0].children) for card in metrics_panel]
    assert len(metrics_panel) >= 8
    assert "Задержка реакции" in values
    assert "Целостность" in values

def test_update_all_panels_keeps_all_queued_route_points() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    for idx in range(3):
        base_fix = _fake_fix()
        fix = IMMResult(
            lat=base_fix.lat + idx * 0.0001,
            lon=base_fix.lon + idx * 0.0001,
            speed_mps=base_fix.speed_mps,
            azimuth_deg=base_fix.azimuth_deg,
            model_weights=base_fix.model_weights,
            covariance=base_fix.covariance,
            dominant_mode=base_fix.dominant_mode,
        )
        state_queue.put(
            {
                "corr": _fake_corr(),
                "fix": fix,
                "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
                "ref": np.array([100.0, 105.0, 103.0], dtype=float),
                "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                "dem_patch_transform": (0.001, 0.0, 90.299, 0.0, -0.001, 60.501),
                "observability": {},
                "runtime_stats": {},
            }
        )

    _, terrain_fig, _, _, metrics_panel, store = dash_app.update_all_panels(1, {"history": []})

    assert len(store["history"]) == 3
    assert len(metrics_panel) >= 8
    assert terrain_fig.layout.uirevision == "terrain-map-zoom"
    trajectory_trace = next(trace for trace in terrain_fig.data if "Траектория" in str(trace.name))
    assert len(trajectory_trace.x) == 3
    assert trajectory_trace.mode == "markers"


def test_create_arrow_shape_returns_three_segments() -> None:
    shapes = create_arrow_shape(60.5, 90.3, 45.0)
    assert len(shapes) == 3
    assert all(shape["type"] == "line" for shape in shapes)


def test_create_probability_ellipse_traces_returns_two_filled_contours() -> None:
    traces = create_probability_ellipse_traces(60.5, 90.3, _fake_fix().covariance)

    assert len(traces) == 2
    assert all(trace.fill == "toself" for trace in traces)


def test_build_velocity_annotation_includes_speed_and_heading() -> None:
    text = build_velocity_annotation(_fake_fix())

    assert "Скорость:" in text
    assert "Курс:" in text


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


def test_send_restart_route_command_pushes_to_control_queue() -> None:
    state_queue: queue.Queue = queue.Queue()
    control_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue, control_queue=control_queue)

    dash_app.send_control_command({"type": "restart_route"})

    assert control_queue.get_nowait() == {"type": "restart_route"}


def test_manual_gnss_override_is_reflected_in_dashboard_store() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    dash_app._manual_gnss_enabled = False
    state_queue.put(
        {
            "corr": _fake_corr(),
            "fix": _fake_fix(),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "dem_patch_transform": (0.001, 0.0, 90.299, 0.0, -0.001, 60.501),
            "observability": {},
            "runtime_stats": {},
            "gnss_available": True,
        }
    )

    _, _, _, _, _, store = dash_app.update_all_panels(1, {"history": []})

    assert store["latest_state"]["gnss_available"] is False
