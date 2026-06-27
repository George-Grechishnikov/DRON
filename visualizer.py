"""Plotly Dash dashboard and report exports for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
import json
import queue
import csv
from pathlib import Path
from typing import Any, List

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as offline_plot
from plotly.subplots import make_subplots

from correlator import CorrelationResult, build_heatmap
from imm_filter import IMMResult


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _derive_output_paths(report_path: str | Path) -> dict[str, Path]:
    """Build companion output paths next to the HTML report."""

    base = Path(report_path)
    suffix = base.suffix or ".html"
    stem = base.name[: -len(suffix)] if base.name.endswith(suffix) else base.stem
    return {
        "html": base,
        "summary_txt": base.with_name(f"{stem}.summary.txt"),
        "summary_json": base.with_name(f"{stem}.summary.json"),
        "records_csv": base.with_name(f"{stem}.records.csv"),
    }


def build_operator_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a non-technical run summary for operators and reviewers."""

    if not records:
        return {
            "status": "NO_DATA",
            "status_label": "No data",
            "headline": "No processed navigation windows were produced.",
            "operator_message": "Run finished without usable output windows.",
            "windows_processed": 0,
            "gnss_loss_detected": False,
            "terrain_nav_entered": False,
        }

    timestamps = [_finite_float(record.get("timestamp")) for record in records]
    valid_timestamps = [value for value in timestamps if value is not None]
    errors = [_finite_float(record.get("truth_error_m")) for record in records]
    finite_errors = [value for value in errors if value is not None]
    correlations = [_finite_float(record.get("correlation_peak")) for record in records]
    finite_correlations = [value for value in correlations if value is not None]
    gnss_flags = [bool(record.get("gnss_available", True)) for record in records]
    modes = [str(record.get("mode", "INIT")).upper() for record in records]
    gnss_loss_detected = any(not flag for flag in gnss_flags)
    terrain_nav_entered = any(mode == "TERRAIN_NAV" for mode in modes)
    final = records[-1]

    avg_error_m = sum(finite_errors) / len(finite_errors) if finite_errors else None
    max_error_m = max(finite_errors) if finite_errors else None
    final_error_m = _finite_float(final.get("truth_error_m"))
    avg_correlation = sum(finite_correlations) / len(finite_correlations) if finite_correlations else None
    best_correlation = max(finite_correlations) if finite_correlations else None
    duration_s = (
        valid_timestamps[-1] - valid_timestamps[0]
        if len(valid_timestamps) >= 2
        else 0.0
    )

    if terrain_nav_entered and avg_correlation is not None and avg_correlation >= 0.85 and (
        avg_error_m is None or avg_error_m <= 250.0
    ):
        status = "PASS"
        status_label = "Pass"
        headline = "Terrain navigation stayed usable after GNSS loss."
    elif terrain_nav_entered and avg_correlation is not None and avg_correlation >= 0.65:
        status = "REVIEW"
        status_label = "Needs review"
        headline = "Terrain navigation engaged, but quality should be reviewed."
    else:
        status = "FAIL"
        status_label = "Fail"
        headline = "The run did not show a stable terrain-navigation result."

    operator_bits = [
        f"Processed {len(records)} navigation windows over {duration_s:.1f} s.",
        f"GNSS loss detected: {'yes' if gnss_loss_detected else 'no'}.",
        f"Terrain navigation entered: {'yes' if terrain_nav_entered else 'no'}.",
    ]
    if avg_error_m is not None:
        operator_bits.append(f"Average trajectory error: {avg_error_m:.1f} m.")
    if max_error_m is not None:
        operator_bits.append(f"Maximum trajectory error: {max_error_m:.1f} m.")
    if avg_correlation is not None:
        operator_bits.append(f"Average correlation peak: {avg_correlation:.3f}.")

    return {
        "status": status,
        "status_label": status_label,
        "headline": headline,
        "operator_message": " ".join(operator_bits),
        "windows_processed": len(records),
        "start_timestamp_s": valid_timestamps[0] if valid_timestamps else None,
        "end_timestamp_s": valid_timestamps[-1] if valid_timestamps else None,
        "duration_s": duration_s,
        "gnss_loss_detected": gnss_loss_detected,
        "terrain_nav_entered": terrain_nav_entered,
        "final_mode": str(final.get("mode", "INIT")),
        "final_gnss_available": bool(final.get("gnss_available", True)),
        "average_truth_error_m": avg_error_m,
        "max_truth_error_m": max_error_m,
        "final_truth_error_m": final_error_m,
        "average_correlation_peak": avg_correlation,
        "best_correlation_peak": best_correlation,
    }


def export_operator_outputs(records: list[dict[str, Any]], report_path: str | Path) -> dict[str, Path]:
    """Export operator-facing summary files next to the HTML report."""

    output_paths = _derive_output_paths(report_path)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_operator_summary(records)
    summary_text = "\n".join(
        [
            "TERRAIN NAVIGATOR Run Summary",
            f"Status: {summary['status_label']}",
            f"Headline: {summary['headline']}",
            f"Message: {summary['operator_message']}",
            f"Final mode: {summary.get('final_mode', 'INIT')}",
            f"Final GNSS: {'ON' if summary.get('final_gnss_available', True) else 'OFF'}",
        ]
    )
    output_paths["summary_txt"].write_text(summary_text + "\n", encoding="utf-8")
    output_paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if records:
        fieldnames = list(records[0].keys())
        with output_paths["records_csv"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    return output_paths


def create_arrow_shape(lat: float, lon: float, azimuth: float, length_deg: float = 0.002) -> list[dict[str, Any]]:
    """Create plotly line shapes representing the velocity vector arrow."""

    angle_rad = math.radians(azimuth)
    end_lon = lon + length_deg * math.sin(angle_rad)
    end_lat = lat + length_deg * math.cos(angle_rad)

    head_angle_left = math.radians(azimuth - 150.0)
    head_angle_right = math.radians(azimuth + 150.0)
    head_scale = length_deg * 0.25
    left_lon = end_lon + head_scale * math.sin(head_angle_left)
    left_lat = end_lat + head_scale * math.cos(head_angle_left)
    right_lon = end_lon + head_scale * math.sin(head_angle_right)
    right_lat = end_lat + head_scale * math.cos(head_angle_right)

    return [
        {
            "type": "line",
            "x0": lon,
            "y0": lat,
            "x1": end_lon,
            "y1": end_lat,
            "line": {"color": "#ff6b6b", "width": 3},
        },
        {
            "type": "line",
            "x0": end_lon,
            "y0": end_lat,
            "x1": left_lon,
            "y1": left_lat,
            "line": {"color": "#ff6b6b", "width": 3},
        },
        {
            "type": "line",
            "x0": end_lon,
            "y0": end_lat,
            "x1": right_lon,
            "y1": right_lat,
            "line": {"color": "#ff6b6b", "width": 3},
        },
    ]


def export_flight_report(history: List[IMMResult], path: str) -> None:
    """Export a lightweight HTML report with the fused trajectory."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[item.lon for item in history],
            y=[item.lat for item in history],
            mode="lines+markers",
            name="IMM track",
            line={"color": "#2d9cdb", "width": 3},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="TERRAIN NAVIGATOR Flight Report",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offline_plot(figure, filename=str(output_path), auto_open=False, include_plotlyjs="cdn")


def export_demo_report(records: list[dict[str, Any]], path: str, context: dict[str, Any] | None = None) -> None:
    """Export a jury-facing HTML report with truth, estimate, GNSS, and correlation."""

    if not records:
        return

    context = context or {}
    timestamps = [float(record["timestamp"]) for record in records]
    estimated_lons = [float(record["estimated_lon"]) for record in records]
    estimated_lats = [float(record["estimated_lat"]) for record in records]
    truth_lons = [record["truth_lon"] for record in records]
    truth_lats = [record["truth_lat"] for record in records]
    correlation_peaks = [float(record["correlation_peak"]) for record in records]
    truth_errors = [float(record["truth_error_m"]) for record in records]
    gnss_available = [bool(record["gnss_available"]) for record in records]
    final_heatmap = context.get("correlation_heatmap")
    final_dem_patch = context.get("dem_patch")
    final_dem_extent = context.get("dem_extent")

    gnss_loss_index = next((idx for idx, available in enumerate(gnss_available) if not available), None)
    gnss_loss_time = timestamps[gnss_loss_index] if gnss_loss_index is not None else None

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            "Trajectory on DEM Patch",
            "Correlation Heatmap",
            "Correlation Peak",
            "Truth Error and GNSS Status",
        ),
    )
    if final_dem_patch is not None and final_dem_extent is not None:
        dem_patch = np.asarray(final_dem_patch, dtype=float)
        extent = final_dem_extent
        x_axis = np.linspace(float(extent["left"]), float(extent["right"]), dem_patch.shape[1])
        y_axis = np.linspace(float(extent["top"]), float(extent["bottom"]), dem_patch.shape[0])
        figure.add_trace(
            go.Heatmap(
                z=dem_patch,
                x=x_axis,
                y=y_axis,
                colorscale="Earth",
                opacity=0.85,
                showscale=False,
                name="DEM patch",
            ),
            row=1,
            col=1,
        )
    if any(value is not None for value in truth_lons) and any(value is not None for value in truth_lats):
        figure.add_trace(
            go.Scatter(
                x=truth_lons,
                y=truth_lats,
                mode="lines",
                name="Truth trajectory",
                line={"color": "#f4f7fb", "width": 3},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=estimated_lons,
            y=estimated_lats,
            mode="lines+markers",
            name="Estimated trajectory",
            line={"color": "#2d9cdb", "width": 3},
            marker={"size": 5},
        ),
            row=1,
            col=1,
        )
    if gnss_loss_index is not None:
        figure.add_trace(
            go.Scatter(
                x=[truth_lons[gnss_loss_index] if truth_lons[gnss_loss_index] is not None else estimated_lons[gnss_loss_index]],
                y=[truth_lats[gnss_loss_index] if truth_lats[gnss_loss_index] is not None else estimated_lats[gnss_loss_index]],
                mode="markers+text",
                name="GNSS loss event",
                marker={"color": "#ffd166", "size": 13, "symbol": "x"},
                text=["GNSS LOST"],
                textposition="top center",
            ),
            row=1,
            col=1,
        )

    if final_heatmap is not None:
        heatmap = np.asarray(final_heatmap, dtype=float)
        figure.add_trace(
            go.Heatmap(
                z=heatmap,
                colorscale="Viridis",
                colorbar={"title": "r norm"},
                name="Correlation heatmap",
            ),
            row=2,
            col=1,
        )
        final_record = records[-1]
        figure.add_trace(
            go.Scatter(
                x=[float(final_record["best_offset_m"])],
                y=[float(final_record["best_azimuth_deg"])],
                mode="markers",
                marker={"color": "#ff5a5f", "symbol": "star", "size": 14},
                name="Final peak",
            ),
            row=2,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=correlation_peaks,
            mode="lines+markers",
            name="Correlation peak",
            line={"color": "#27ae60", "width": 3},
            marker={"size": 5},
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=truth_errors,
            mode="lines",
            name="Truth error, m",
            line={"color": "#ff6b6b", "width": 3},
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=[0 if available else 1 for available in gnss_available],
            mode="lines",
            name="GNSS lost flag",
            line={"color": "#ffd166", "width": 2, "shape": "hv"},
            yaxis="y4",
        ),
        row=4,
        col=1,
    )

    if gnss_loss_time is not None:
        for row in (3, 4):
            figure.add_vline(
                x=gnss_loss_time,
                line={"color": "#ffd166", "dash": "dash", "width": 2},
                row=row,
                col=1,
            )

    final = records[-1]
    operator_summary = build_operator_summary(records)
    summary = (
        f"Status: {operator_summary['status_label']}<br>"
        f"Final mode: {final['mode']}<br>"
        f"GNSS: {'ON' if final['gnss_available'] else 'OFF'}<br>"
        f"Final error: {float(final['truth_error_m']):.1f} m<br>"
        f"Peak correlation: {float(final['correlation_peak']):.3f}<br>"
        f"Best azimuth: {float(final['best_azimuth_deg']):.1f} deg<br>"
        f"Best offset: {float(final['best_offset_m']):.0f} m"
    )
    figure.update_layout(
        template="plotly_dark",
        title="TERRAIN NAVIGATOR Demo Report",
        height=1320,
        annotations=[
            *list(figure.layout.annotations),
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 0.99,
                "showarrow": False,
                "align": "right",
                "text": summary,
                "font": {"size": 13},
                "bgcolor": "rgba(16, 24, 32, 0.75)",
                "bordercolor": "#2d9cdb",
                "borderwidth": 1,
            },
        ],
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
    )
    figure.update_xaxes(title_text="Longitude", row=1, col=1)
    figure.update_yaxes(title_text="Latitude", row=1, col=1)
    figure.update_xaxes(title_text="Offset index", row=2, col=1)
    figure.update_yaxes(title_text="Azimuth index", row=2, col=1)
    figure.update_xaxes(title_text="Time, s", row=3, col=1)
    figure.update_yaxes(title_text="Correlation", row=3, col=1)
    figure.update_xaxes(title_text="Time, s", row=4, col=1)
    figure.update_yaxes(title_text="Error, m", row=4, col=1)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offline_plot(figure, filename=str(output_path), auto_open=False, include_plotlyjs="cdn")
    export_operator_outputs(records, output_path)


class TerrainNavigatorDash:
    """Real-time Plotly Dash dashboard for TERRAIN NAVIGATOR state updates."""

    def __init__(self, state_queue: queue.Queue) -> None:
        self.state_queue = state_queue
        self.app = Dash(__name__)
        self.app.layout = html.Div(
            style={
                "backgroundColor": "#101820",
                "color": "#f4f7fb",
                "minHeight": "100vh",
                "padding": "18px",
                "fontFamily": "Segoe UI, Arial, sans-serif",
            },
            children=[
                html.H1("TERRAIN NAVIGATOR", style={"marginBottom": "12px"}),
                dcc.Store(
                    id="history-store",
                    data={
                        "estimated_history": [],
                        "truth_history": [],
                        "gnss_loss_event": None,
                    },
                ),
                dcc.Interval(id="dashboard-interval", interval=500, n_intervals=0),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gridTemplateRows": "1fr 1fr",
                        "gap": "14px",
                    },
                    children=[
                        dcc.Graph(id="correlation-heatmap"),
                        dcc.Graph(id="terrain-map"),
                        dcc.Graph(id="profiles-graph"),
                        dcc.Graph(id="telemetry-graph"),
                    ],
                ),
            ],
        )
        self._register_callbacks()

    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
        """Start the Dash server."""

        self.app.run(host=host, port=port, debug=debug)

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output("correlation-heatmap", "figure"),
            Output("terrain-map", "figure"),
            Output("profiles-graph", "figure"),
            Output("telemetry-graph", "figure"),
            Output("history-store", "data"),
            Input("dashboard-interval", "n_intervals"),
            State("history-store", "data"),
        )
        def _callback(n_intervals: int, data_store: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
            return self.update_all_panels(n_intervals, data_store)

    def update_all_panels(
        self,
        n_intervals: int,
        data_store: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Read one state update and rebuild all dashboard panels."""

        del n_intervals
        try:
            state = self.state_queue.get_nowait()
        except queue.Empty:
            return no_update, no_update, no_update, no_update, no_update

        store = data_store or {
            "estimated_history": [],
            "truth_history": [],
            "gnss_loss_event": None,
        }
        estimated_history = list(store.get("estimated_history", store.get("history", [])))
        truth_history = list(store.get("truth_history", []))
        fix = state["fix"]
        sample = state.get("sample")
        gnss_available = bool(state.get("gnss_available", True))
        mode = str(state.get("mode", "GNSS" if gnss_available else "TERRAIN_NAV"))
        timestamp = float(getattr(sample, "timestamp", len(estimated_history)))

        estimated_history.append(
            {
                "lat": float(fix.lat),
                "lon": float(fix.lon),
                "speed_mps": float(fix.speed_mps),
                "azimuth_deg": float(fix.azimuth_deg),
                "dominant_mode": str(fix.dominant_mode),
                "timestamp": timestamp,
                "mode": mode,
                "gnss_available": gnss_available,
            }
        )
        truth_lat = (
            getattr(sample, "effective_truth_lat", getattr(sample, "truth_lat", getattr(sample, "lat", None)))
            if sample is not None
            else None
        )
        truth_lon = (
            getattr(sample, "effective_truth_lon", getattr(sample, "truth_lon", getattr(sample, "lon", None)))
            if sample is not None
            else None
        )
        if truth_lat is not None and truth_lon is not None:
            truth_history.append(
                {
                    "lat": float(truth_lat),
                    "lon": float(truth_lon),
                    "timestamp": timestamp,
                    "gnss_available": gnss_available,
                }
            )

        previous = estimated_history[-2] if len(estimated_history) > 1 else None
        if (
            store.get("gnss_loss_event") is None
            and previous is not None
            and bool(previous.get("gnss_available", True))
            and not gnss_available
        ):
            store["gnss_loss_event"] = {
                "timestamp": timestamp,
                "lat": float(truth_lat if truth_lat is not None else fix.lat),
                "lon": float(truth_lon if truth_lon is not None else fix.lon),
            }

        store["estimated_history"] = estimated_history[-500:]
        store["truth_history"] = truth_history[-500:]
        store["history"] = store["estimated_history"]

        heatmap_figure = self._build_correlation_figure(state.get("corr"), state)
        terrain_map_figure = self._build_map_figure(state, store)
        profiles_figure = self._build_profiles_figure(state)
        telemetry_figure = self._build_telemetry_figure(state, store)
        return heatmap_figure, terrain_map_figure, profiles_figure, telemetry_figure, store

    def _build_correlation_figure(self, corr: CorrelationResult | None, state: dict[str, Any]) -> go.Figure:
        incoming_heatmap = state.get("correlation_heatmap")
        if corr is None and incoming_heatmap is None:
            figure = go.Figure()
            figure.update_layout(
                template="plotly_dark",
                title="Correlation Heatmap | unavailable",
                xaxis_title="Offset",
                yaxis_title="Azimuth",
            )
            return figure

        if corr is not None:
            heatmap = build_heatmap(corr)
            best_offset_m = corr.best_offset_m
            best_azimuth_deg = corr.best_azimuth_deg
            peak_correlation = corr.peak_correlation
            azimuth_axis = corr.azimuths_deg
            x_offsets = np.arange(heatmap.shape[1], dtype=float) * (
                corr.best_offset_m / max(corr.best_offset_steps, 1)
                if corr.best_offset_steps > 0
                else 30.0
            )
        else:
            heatmap = np.asarray(incoming_heatmap, dtype=float)
            best_offset_m = float(state.get("best_offset_m", 0.0))
            best_azimuth_deg = float(state.get("best_azimuth_deg", 0.0))
            peak_correlation = float(state.get("correlation_score", 0.0))
            azimuth_axis = np.arange(heatmap.shape[0], dtype=float)
            x_offsets = np.arange(heatmap.shape[1], dtype=float)

        figure = go.Figure(
            data=[
                go.Heatmap(
                    z=heatmap,
                    x=x_offsets,
                    y=azimuth_axis,
                    colorscale="Viridis",
                    colorbar={"title": "r norm"},
                ),
                go.Scatter(
                    x=[best_offset_m],
                    y=[best_azimuth_deg],
                    mode="markers",
                    marker={"color": "#ff5a5f", "symbol": "star", "size": 14},
                    name="Peak",
                ),
            ]
        )
        figure.update_layout(
            template="plotly_dark",
            title=f"Correlation Heatmap | Peak: {peak_correlation:.3f} @ {best_azimuth_deg:.0f} deg",
            xaxis_title="Смещение, м",
            yaxis_title="Азимут, град",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        return figure

    def _build_map_figure(self, state: dict[str, Any], store: dict[str, Any]) -> go.Figure:
        patch = np.asarray(state.get("dem_patch", np.zeros((2, 2), dtype=float)), dtype=float)
        fix: IMMResult = state["fix"]
        estimated_history = list(store.get("estimated_history", []))
        truth_history = list(store.get("truth_history", []))
        extent = state.get("dem_extent")
        x_axis = None
        y_axis = None
        if extent is not None and patch.size:
            x_axis = np.linspace(float(extent["left"]), float(extent["right"]), patch.shape[1])
            y_axis = np.linspace(float(extent["top"]), float(extent["bottom"]), patch.shape[0])
        figure = go.Figure()
        if patch.size:
            figure.add_trace(
                go.Heatmap(
                    z=patch,
                    x=x_axis,
                    y=y_axis,
                    colorscale="Earth",
                    opacity=0.8,
                    showscale=False,
                    name="DEM patch",
                )
            )
        if truth_history:
            figure.add_trace(
                go.Scatter(
                    x=[entry["lon"] for entry in truth_history],
                    y=[entry["lat"] for entry in truth_history],
                    mode="lines",
                    line={"color": "#f4f7fb", "width": 3},
                    name="Truth trajectory",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[entry["lon"] for entry in estimated_history],
                y=[entry["lat"] for entry in estimated_history],
                mode="lines+markers",
                line={"color": "#2d9cdb", "width": 3},
                marker={"size": 6},
                name="Estimated trajectory",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[fix.lon],
                y=[fix.lat],
                mode="markers",
                marker={"color": "#ff5a5f", "size": 11},
                name="Current position",
            )
        )
        gnss_loss_event = store.get("gnss_loss_event")
        if gnss_loss_event is not None:
            figure.add_trace(
                go.Scatter(
                    x=[gnss_loss_event["lon"]],
                    y=[gnss_loss_event["lat"]],
                    mode="markers+text",
                    marker={"color": "#ffd166", "size": 13, "symbol": "x"},
                    text=["GNSS LOST"],
                    textposition="top center",
                    name="GNSS loss event",
                )
            )
        gnss_status = "GNSS ON" if state.get("gnss_available", True) else "GNSS OFF"
        primary_mode = f"NAV MODE: {state.get('mode', 'INIT')}"
        figure.update_layout(
            template="plotly_dark",
            title=f"DEM Map | {gnss_status} | {primary_mode}",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
            shapes=create_arrow_shape(fix.lat, fix.lon, fix.azimuth_deg),
            legend={"orientation": "h", "y": 1.02, "x": 0.0},
        )
        return figure

    def _build_profiles_figure(self, state: dict[str, Any]) -> go.Figure:
        h_meas = np.asarray(state.get("h_meas", np.array([], dtype=float)), dtype=float)
        best_ref = np.asarray(state.get("ref", np.array([], dtype=float)), dtype=float)
        corr = state.get("corr")
        x_axis = np.arange(h_meas.size, dtype=float) * 30.0
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=x_axis,
                y=h_meas,
                mode="lines",
                line={"color": "#2d9cdb", "width": 3},
                name="H_meas",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_axis[: best_ref.size],
                y=best_ref,
                mode="lines",
                line={"color": "#ff6b6b", "dash": "dot", "width": 3},
                name="H_ref",
            )
        )
        figure.update_layout(
            template="plotly_dark",
            title="Профили высот",
            xaxis_title="Расстояние, м",
            yaxis_title="Высота, м",
            annotations=[
                {
                    "x": 0.98,
                    "y": 0.95,
                    "xref": "paper",
                    "yref": "paper",
                    "text": f"r={float(state.get('correlation_score', getattr(corr, 'peak_correlation', 0.0))):.3f}",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#f4f7fb"},
                }
            ],
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        return figure

    def _build_telemetry_figure(self, state: dict[str, Any], store: dict[str, Any]) -> go.Figure:
        fix: IMMResult = state["fix"]
        hdop = float(state.get("hdop", math.sqrt(max(np.trace(fix.covariance[0:2, 0:2]), 0.0))))
        corr = state.get("corr")
        gnss_available = bool(state.get("gnss_available", True))
        terrain_active = bool(state.get("terrain_active", False))
        mode = str(state.get("mode", "GNSS" if gnss_available else "TERRAIN_NAV"))
        truth_error_m = state.get("truth_error_m")
        gnss_text = "GNSS ON" if gnss_available else "GNSS OFF"
        mode_text = f"NAV MODE: {mode}"
        banner = (
            "GNSS signal lost, switching to terrain-based correction"
            if terrain_active
            else "GNSS aiding active, terrain matching running in parallel"
        )
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=["Hover", "Cruise", "Turn"],
                y=fix.model_weights.tolist(),
                marker_color=["#4aa3df", "#27ae60", "#f39c12"],
                name="Mode weights",
            )
        )
        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(fix.speed_mps),
                domain={"x": [0.60, 0.98], "y": [0.45, 0.98]},
                title={"text": "Скорость, м/с"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff6b6b"},
                    "bgcolor": "#1f2937",
                },
            )
        )
        truth_error_value = float(truth_error_m) if truth_error_m is not None else float("nan")
        figure.update_layout(
            template="plotly_dark",
            title=f"System State | {gnss_text} | {mode_text}",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
            annotations=[
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.02,
                    "y": 0.31,
                    "showarrow": False,
                    "align": "left",
                    "text": (
                        f"{gnss_text}<br>"
                        f"{mode_text}<br>"
                        f"{banner}<br><br>"
                        f"Lat: {fix.lat:.6f}<br>"
                        f"Lon: {fix.lon:.6f}<br>"
                        f"Estimated speed: {fix.speed_mps:.1f} m/s<br>"
                        f"Estimated heading: {fix.azimuth_deg:.1f}°<br>"
                        f"Best azimuth: {float(getattr(corr, 'best_azimuth_deg', state.get('best_azimuth_deg', 0.0))):.1f}°<br>"
                        f"Best offset: {float(getattr(corr, 'best_offset_m', state.get('best_offset_m', 0.0))):.0f} m<br>"
                        f"Correlation peak: {float(state.get('correlation_score', getattr(corr, 'peak_correlation', 0.0))):.3f}<br>"
                        f"Truth error: {truth_error_value:.1f} m<br>"
                        f"IMM model: {fix.dominant_mode}<br>"
                        f"HDOP: {hdop:.1f} m"
                    ),
                    "font": {"size": 13},
                }
            ],
        )
        return figure
