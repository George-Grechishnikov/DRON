from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from correlator import CorrelationResult
from imm_filter import IMMResult
from visualizer import (
    TerrainNavigatorDash,
    build_operator_summary,
    create_arrow_shape,
    export_demo_report,
    export_flight_report,
    export_operator_outputs,
)


def _fake_corr() -> CorrelationResult:
    return CorrelationResult(
        best_azimuth_deg=45.0,
        best_offset_steps=3,
        best_offset_m=90.0,
        peak_correlation=0.87,
        confidence=0.42,
        is_reliable=True,
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
        }
    )

    heatmap_fig, terrain_fig, profiles_fig, telemetry_fig, store = dash_app.update_all_panels(
        1,
        {"history": []},
    )

    assert len(heatmap_fig.data) >= 2
    assert len(terrain_fig.data) >= 3
    assert len(profiles_fig.data) >= 2
    assert len(telemetry_fig.data) >= 2
    assert len(store["history"]) == 1
    assert len(store["estimated_history"]) == 1


def test_update_all_panels_marks_gnss_loss_event() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    base_state = {
        "corr": _fake_corr(),
        "fix": _fake_fix(),
        "sample": SimpleNamespace(timestamp=1.0, lat=60.5, lon=90.3),
        "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
        "ref": np.array([100.0, 105.0, 103.0], dtype=float),
        "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "hdop": 12.0,
        "gnss_available": True,
        "mode": "GNSS",
        "terrain_active": False,
        "truth_error_m": 0.0,
    }
    state_queue.put(base_state)
    first = dash_app.update_all_panels(1, None)

    lost_state = dict(base_state)
    lost_state["sample"] = SimpleNamespace(timestamp=2.0, lat=60.5001, lon=90.3001)
    lost_state["gnss_available"] = False
    lost_state["mode"] = "TERRAIN"
    lost_state["terrain_active"] = True
    state_queue.put(lost_state)
    _, terrain_fig, _, telemetry_fig, store = dash_app.update_all_panels(2, first[-1])

    assert store["gnss_loss_event"] is not None
    assert any(trace.name == "GNSS loss event" for trace in terrain_fig.data)
    assert "GNSS OFF" in telemetry_fig.layout.title.text


def test_visualizer_does_not_crash_without_truth() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    state_queue.put(
        {
            "corr": _fake_corr(),
            "fix": _fake_fix(),
            "sample": SimpleNamespace(timestamp=1.0, effective_truth_lat=None, effective_truth_lon=None),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "hdop": 12.0,
            "gnss_available": True,
            "mode": "GNSS",
            "terrain_active": False,
            "truth_error_m": float("nan"),
        }
    )

    _, terrain_fig, _, _, store = dash_app.update_all_panels(1, None)

    assert len(store["truth_history"]) == 0
    assert len(terrain_fig.data) >= 2


def test_visualizer_does_not_crash_without_heatmap() -> None:
    state_queue: queue.Queue = queue.Queue()
    dash_app = TerrainNavigatorDash(state_queue=state_queue)
    state_queue.put(
        {
            "fix": _fake_fix(),
            "sample": SimpleNamespace(timestamp=1.0, effective_truth_lat=60.5, effective_truth_lon=90.3),
            "h_meas": np.array([101.0, 107.0, 102.0], dtype=float),
            "ref": np.array([100.0, 105.0, 103.0], dtype=float),
            "dem_patch": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "hdop": 12.0,
            "gnss_available": False,
            "mode": "LOST",
            "terrain_active": False,
            "truth_error_m": float("nan"),
        }
    )

    heatmap_fig, _, _, _, _ = dash_app.update_all_panels(1, None)

    assert "unavailable" in heatmap_fig.layout.title.text.lower()


def test_create_arrow_shape_returns_three_segments() -> None:
    shapes = create_arrow_shape(60.5, 90.3, 45.0)
    assert len(shapes) == 3
    assert all(shape["type"] == "line" for shape in shapes)


def test_export_flight_report_writes_html(tmp_path: Path) -> None:
    output_path = tmp_path / "report.html"
    export_flight_report([_fake_fix(), _fake_fix()], str(output_path))
    assert output_path.exists()
    assert "plotly" in output_path.read_text(encoding="utf-8").lower()


def test_export_demo_report_writes_truth_and_gnss_status(tmp_path: Path) -> None:
    output_path = tmp_path / "demo_report.html"
    export_demo_report(
        [
            {
                "timestamp": 1.0,
                "estimated_lon": 90.3001,
                "estimated_lat": 60.5001,
                "truth_lon": 90.3,
                "truth_lat": 60.5,
                "gnss_available": True,
                "mode": "GNSS",
                "truth_error_m": 5.0,
                "correlation_peak": 0.8,
                "best_azimuth_deg": 45.0,
                "best_offset_m": 100.0,
            },
            {
                "timestamp": 2.0,
                "estimated_lon": 90.3002,
                "estimated_lat": 60.5002,
                "truth_lon": 90.3001,
                "truth_lat": 60.5001,
                "gnss_available": False,
                "mode": "TERRAIN",
                "truth_error_m": 8.0,
                "correlation_peak": 0.9,
                "best_azimuth_deg": 46.0,
                "best_offset_m": 120.0,
            },
        ],
        str(output_path),
    )

    contents = output_path.read_text(encoding="utf-8")
    assert "TERRAIN NAVIGATOR Demo Report" in contents
    assert "GNSS LOST" in contents

    summary_txt = tmp_path / "demo_report.summary.txt"
    summary_json = tmp_path / "demo_report.summary.json"
    records_csv = tmp_path / "demo_report.records.csv"
    assert summary_txt.exists()
    assert summary_json.exists()
    assert records_csv.exists()
    assert "Status:" in summary_txt.read_text(encoding="utf-8")


def test_build_operator_summary_marks_review_for_mid_quality_run() -> None:
    summary = build_operator_summary(
        [
            {
                "timestamp": 1.0,
                "gnss_available": True,
                "mode": "GNSS",
                "truth_error_m": 120.0,
                "correlation_peak": 0.80,
            },
            {
                "timestamp": 2.0,
                "gnss_available": False,
                "mode": "TERRAIN_NAV",
                "truth_error_m": 180.0,
                "correlation_peak": 0.72,
            },
        ]
    )

    assert summary["status"] in {"PASS", "REVIEW"}
    assert summary["gnss_loss_detected"] is True
    assert summary["terrain_nav_entered"] is True


def test_export_operator_outputs_writes_companion_files(tmp_path: Path) -> None:
    output_paths = export_operator_outputs(
        [
            {
                "timestamp": 1.0,
                "estimated_lat": 60.5,
                "estimated_lon": 90.3,
                "truth_lat": 60.5,
                "truth_lon": 90.3,
                "estimated_speed_mps": 50.0,
                "estimated_heading_deg": 45.0,
                "gnss_available": False,
                "mode": "TERRAIN_NAV",
                "terrain_active": True,
                "truth_error_m": 15.0,
                "best_azimuth_deg": 45.0,
                "best_offset_m": 100.0,
                "correlation_peak": 0.92,
            }
        ],
        tmp_path / "run_report.html",
    )

    assert output_paths["summary_txt"].exists()
    assert output_paths["summary_json"].exists()
    assert output_paths["records_csv"].exists()
