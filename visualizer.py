"""Plotly Dash dashboard for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
import queue
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, dcc, html, no_update
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as offline_plot

from correlator import CorrelationResult, build_heatmap
from imm_filter import IMMResult


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


def export_flight_report(history: list[IMMResult], path: str) -> None:
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


class TerrainNavigatorDash:
    """Real-time Plotly Dash dashboard for TERRAIN NAVIGATOR state updates."""

    def __init__(self, state_queue: queue.Queue, control_queue: queue.Queue | None = None) -> None:
        self.state_queue = state_queue
        self.control_queue = control_queue
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
                html.Div(
                    style={"display": "flex", "gap": "10px", "marginBottom": "12px"},
                    children=[
                        html.Button(
                            "GNSS ON",
                            id="gnss-on-button",
                            n_clicks=0,
                            style={"padding": "10px 14px", "backgroundColor": "#27ae60", "color": "#ffffff", "border": "none"},
                        ),
                        html.Button(
                            "GNSS OFF",
                            id="gnss-off-button",
                            n_clicks=0,
                            style={"padding": "10px 14px", "backgroundColor": "#c0392b", "color": "#ffffff", "border": "none"},
                        ),
                    ],
                ),
                dcc.Store(id="history-store", data={"history": []}),
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
            Output("gnss-on-button", "title"),
            Input("gnss-on-button", "n_clicks"),
            Input("gnss-off-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def _control_callback(gnss_on_clicks: int, gnss_off_clicks: int) -> str:
            del gnss_on_clicks, gnss_off_clicks
            return self.handle_gnss_button_click()

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

    def handle_gnss_button_click(self) -> str:
        """Handle a GNSS control button event by pushing a command to the control queue."""

        if self.control_queue is None:
            return "GNSS control unavailable"

        from dash import callback_context

        triggered = callback_context.triggered
        if not triggered:
            return "GNSS control idle"
        prop_id = triggered[0]["prop_id"].split(".", 1)[0]
        command = None
        if prop_id == "gnss-on-button":
            command = {"type": "set_gnss_enabled", "enabled": True}
        elif prop_id == "gnss-off-button":
            command = {"type": "set_gnss_enabled", "enabled": False}
        if command is None:
            return "GNSS control ignored"
        self.send_control_command(command)
        return f"Queued {prop_id}"

    def send_control_command(self, command: dict[str, Any]) -> None:
        """Send a control command to the producer side."""

        try:
            self.control_queue.put_nowait(command)
        except queue.Full:
            try:
                self.control_queue.get_nowait()
            except queue.Empty:
                pass
            self.control_queue.put_nowait(command)

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

        store = data_store or {"history": []}
        history = list(store.get("history", []))
        fix = state["fix"]
        history.append(
            {
                "lat": float(fix.lat),
                "lon": float(fix.lon),
                "speed_mps": float(fix.speed_mps),
                "azimuth_deg": float(fix.azimuth_deg),
                "dominant_mode": str(fix.dominant_mode),
            }
        )
        store["history"] = history[-500:]

        heatmap_figure = self._build_correlation_figure(state["corr"])
        terrain_map_figure = self._build_map_figure(state, store["history"])
        profiles_figure = self._build_profiles_figure(state)
        telemetry_figure = self._build_telemetry_figure(state)
        return heatmap_figure, terrain_map_figure, profiles_figure, telemetry_figure, store

    def _build_correlation_figure(self, corr: CorrelationResult) -> go.Figure:
        heatmap = build_heatmap(corr)
        x_offsets = np.arange(heatmap.shape[1], dtype=float) * (
            corr.best_offset_m / max(corr.best_offset_steps, 1) if corr.best_offset_steps > 0 else 30.0
        )
        figure = go.Figure(
            data=[
                go.Heatmap(
                    z=heatmap,
                    x=x_offsets,
                    y=corr.azimuths_deg,
                    colorscale="Viridis",
                    colorbar={"title": "r norm"},
                ),
                go.Scatter(
                    x=[corr.best_offset_m],
                    y=[corr.best_azimuth_deg],
                    mode="markers",
                    marker={"color": "#ff5a5f", "symbol": "star", "size": 14},
                    name="Peak",
                ),
            ]
        )
        figure.update_layout(
            template="plotly_dark",
            title=f"Correlation Heatmap | peak={corr.peak_correlation:.3f} @ {corr.best_azimuth_deg:.0f} deg",
            xaxis_title="Offset, m",
            yaxis_title="Azimuth, deg",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        return figure

    def _build_map_figure(self, state: dict[str, Any], history: list[dict[str, Any]]) -> go.Figure:
        patch = np.asarray(state.get("dem_patch", np.zeros((2, 2), dtype=float)), dtype=float)
        fix: IMMResult = state["fix"]
        truth = state.get("truth")
        figure = go.Figure()
        figure.add_trace(
            go.Heatmap(
                z=patch,
                colorscale="Earth",
                opacity=0.8,
                showscale=False,
                name="DEM patch",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[entry["lon"] for entry in history],
                y=[entry["lat"] for entry in history],
                mode="lines+markers",
                line={"color": "#2d9cdb", "width": 3},
                marker={"size": 6},
                name="Trajectory",
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
        if truth is not None:
            figure.add_trace(
                go.Scatter(
                    x=[truth["lon"]],
                    y=[truth["lat"]],
                    mode="markers",
                    marker={"color": "#27ae60", "size": 10, "symbol": "diamond"},
                    name="Truth position",
                )
            )
        figure.update_layout(
            template="plotly_dark",
            title="Terrain Map and Trajectory",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
            shapes=create_arrow_shape(fix.lat, fix.lon, fix.azimuth_deg),
        )
        return figure

    def _build_profiles_figure(self, state: dict[str, Any]) -> go.Figure:
        h_meas = np.asarray(state.get("h_meas", np.array([], dtype=float)), dtype=float)
        best_ref = np.asarray(state.get("ref", np.array([], dtype=float)), dtype=float)
        corr: CorrelationResult = state["corr"]
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
            title="Terrain Profiles",
            xaxis_title="Distance, m",
            yaxis_title="Elevation, m",
            annotations=[
                {
                    "x": 0.98,
                    "y": 0.95,
                    "xref": "paper",
                    "yref": "paper",
                    "text": f"r={corr.peak_correlation:.3f}",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#f4f7fb"},
                }
            ],
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        return figure

    def _build_telemetry_figure(self, state: dict[str, Any]) -> go.Figure:
        fix: IMMResult = state["fix"]
        corr: CorrelationResult = state["corr"]
        hdop = float(state.get("hdop", math.sqrt(max(np.trace(fix.covariance[0:2, 0:2]), 0.0))))
        nav_mode = str(state.get("nav_mode", "unknown"))
        used_prediction_only = bool(state.get("used_prediction_only", False))
        selected_window_size = int(state.get("selected_window_size", 0))
        gnss_available = bool(state.get("gnss_available", True))
        observability = dict(state.get("observability", {}))
        crlb_m = float(observability.get("crlb_m", float("inf")))
        terrain_informative = bool(observability.get("is_informative", False))
        terrain_status = "FALLBACK / PREDICT" if used_prediction_only else "TERRAIN UPDATE"
        terrain_color = "#f39c12" if used_prediction_only else "#27ae60"
        gnss_status = "GNSS AVAILABLE" if gnss_available else "GNSS LOST"
        gnss_color = "#27ae60" if gnss_available else "#c0392b"

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
                title={"text": "Speed, m/s"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff6b6b"},
                    "bgcolor": "#1f2937",
                },
            )
        )
        figure.update_layout(
            template="plotly_dark",
            title="IMM Telemetry",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
            annotations=[
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.02,
                    "y": 0.25,
                    "showarrow": False,
                    "align": "left",
                    "text": (
                        f"Lat: {fix.lat:.6f}<br>"
                        f"Lon: {fix.lon:.6f}<br>"
                        f"Azimuth: {fix.azimuth_deg:.1f} deg<br>"
                        f"Mode: {fix.dominant_mode}<br>"
                        f"HDOP: {hdop:.1f} m<br>"
                        f"Nav mode: {nav_mode}<br>"
                        f"Window size: {selected_window_size} frames<br>"
                        f"GNSS: {gnss_status}<br>"
                        f"PSLR: {corr.pslr_db:.2f} dB<br>"
                        f"Ambiguous: {corr.is_ambiguous}<br>"
                        f"CRLB: {crlb_m:.1f} m<br>"
                        f"Terrain informative: {terrain_informative}"
                    ),
                    "font": {"size": 13},
                },
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.79,
                    "y": 0.24,
                    "showarrow": False,
                    "align": "center",
                    "text": terrain_status,
                    "font": {"size": 14, "color": "#f4f7fb"},
                    "bgcolor": terrain_color,
                    "bordercolor": "#f4f7fb",
                    "borderwidth": 1,
                    "borderpad": 6,
                },
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.79,
                    "y": 0.10,
                    "showarrow": False,
                    "align": "center",
                    "text": gnss_status,
                    "font": {"size": 14, "color": "#f4f7fb"},
                    "bgcolor": gnss_color,
                    "bordercolor": "#f4f7fb",
                    "borderwidth": 1,
                    "borderpad": 6,
                },
            ],
        )
        return figure
