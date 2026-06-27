"""Plotly Dash dashboard for TERRAIN NAVIGATOR."""

from __future__ import annotations

import math
import queue
import time
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, dcc, html, no_update
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as offline_plot

from correlator import CorrelationResult, build_heatmap
from imm_filter import IMMResult


def _meters_to_lat_deg(distance_m: float) -> float:
    """Convert local north/south metric offset into latitude degrees."""

    return float(distance_m) / 111_320.0


def _meters_to_lon_deg(distance_m: float, lat_deg: float) -> float:
    """Convert local east/west metric offset into longitude degrees."""

    lon_scale = max(math.cos(math.radians(lat_deg)), 1e-6)
    return float(distance_m) / (111_320.0 * lon_scale)


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


def create_probability_ellipse_traces(
    lat: float,
    lon: float,
    covariance: np.ndarray,
) -> list[go.Scatter]:
    """Create filled 1-sigma and 2-sigma uncertainty ellipses around the fix."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape[0] < 2 or covariance.shape[1] < 2:
        return []
    xy_cov_m = np.asarray(covariance[:2, :2], dtype=float)
    if not np.all(np.isfinite(xy_cov_m)):
        return []
    xy_cov_m = (xy_cov_m + xy_cov_m.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(xy_cov_m)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    if not np.all(np.isfinite(eigenvalues)):
        return []

    angles = np.linspace(0.0, 2.0 * math.pi, 80)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    traces: list[go.Scatter] = []
    for sigma_scale, fill_color, line_color, name in (
        (2.0, "rgba(255, 90, 95, 0.12)", "rgba(255, 90, 95, 0.55)", "Зона вероятности 2σ"),
        (1.0, "rgba(255, 140, 0, 0.18)", "rgba(255, 190, 60, 0.85)", "Зона вероятности 1σ"),
    ):
        axes_m = np.diag(np.sqrt(eigenvalues) * sigma_scale)
        offsets_m = eigenvectors @ axes_m @ unit_circle
        lon_offsets = [_meters_to_lon_deg(offsets_m[0, idx], lat) for idx in range(offsets_m.shape[1])]
        lat_offsets = [_meters_to_lat_deg(offsets_m[1, idx]) for idx in range(offsets_m.shape[1])]
        traces.append(
            go.Scatter(
                x=[lon + delta for delta in lon_offsets],
                y=[lat + delta for delta in lat_offsets],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line={"color": line_color, "width": 2},
                name=name,
                hovertemplate=(
                    f"{name}<br>"
                    + "Center lon=%{x:.6f}<br>"
                    + "Center lat=%{y:.6f}<extra></extra>"
                ),
            )
        )
    return traces


def build_velocity_annotation(fix: IMMResult) -> str:
    """Create a compact map annotation for direction and speed."""

    return (
        "Оценка БПЛА"
        + f"<br>Скорость: {fix.speed_mps:.1f} м/с"
        + f"<br>Курс: {fix.azimuth_deg:.1f}°"
        + f"<br>Режим: {fix.dominant_mode}"
    )


def export_flight_report(history: list[IMMResult], path: str) -> None:
    """Export a lightweight HTML report with the fused trajectory."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[item.lon for item in history],
            y=[item.lat for item in history],
            mode="lines+markers",
            name="Трек IMM",
            line={"color": "#2d9cdb", "width": 3},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="Отчет TERRAIN NAVIGATOR",
        xaxis_title="Долгота",
        yaxis_title="Широта",
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
                            "GNSS ВКЛ",
                            id="gnss-on-button",
                            n_clicks=0,
                            style={"padding": "10px 14px", "backgroundColor": "#27ae60", "color": "#ffffff", "border": "none"},
                        ),
                        html.Button(
                            "GNSS ВЫКЛ",
                            id="gnss-off-button",
                            n_clicks=0,
                            style={"padding": "10px 14px", "backgroundColor": "#c0392b", "color": "#ffffff", "border": "none"},
                        ),
                    ],
                ),
                html.Div(
                    id="control-status",
                    style={
                        "marginBottom": "12px",
                        "padding": "8px 12px",
                        "backgroundColor": "#17212b",
                        "border": "1px solid #31404f",
                        "borderRadius": "8px",
                    },
                    children="Управление GNSS готово",
                ),
                html.Div(
                    id="metrics-panel",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, minmax(140px, 1fr))",
                        "gap": "10px",
                        "marginBottom": "14px",
                    },
                ),
                dcc.Store(id="history-store", data={"history": []}),
                dcc.Interval(id="dashboard-interval", interval=100, n_intervals=0),
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
            Output("control-status", "children"),
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

        @self.app.callback(
            Output("metrics-panel", "children"),
            Input("history-store", "data"),
        )
        def _metrics_callback(data_store: dict[str, Any] | None) -> Any:
            if not data_store:
                return []
            latest_state = data_store.get("latest_state")
            if not isinstance(latest_state, dict):
                return []
            return self._build_metrics_panel(latest_state)

    def handle_gnss_button_click(self) -> str:
        """Handle a GNSS control button event by pushing a command to the control queue."""

        if self.control_queue is None:
            return "Управление GNSS недоступно"

        from dash import callback_context

        triggered = callback_context.triggered
        if not triggered:
            return "Нет команды GNSS"
        prop_id = triggered[0]["prop_id"].split(".", 1)[0]
        command = None
        if prop_id == "gnss-on-button":
            command = {"type": "set_gnss_enabled", "enabled": True}
        elif prop_id == "gnss-off-button":
            command = {"type": "set_gnss_enabled", "enabled": False}
        if command is None:
            return "Команда GNSS проигнорирована"
        self.send_control_command(command)
        return f"Поставлена команда: {prop_id}"

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
        now_monotonic_s = time.perf_counter()
        event_ingest_monotonic_s = float(state.get("event_ingest_monotonic_s", now_monotonic_s))
        pipeline_emitted_monotonic_s = float(state.get("pipeline_emitted_monotonic_s", now_monotonic_s))
        state["dashboard_latency_ms"] = max((now_monotonic_s - pipeline_emitted_monotonic_s) * 1000.0, 0.0)
        state["reaction_latency_ms"] = max((now_monotonic_s - event_ingest_monotonic_s) * 1000.0, 0.0)
        store["history"] = history[-500:]
        store["latest_state"] = self._summarize_state_for_store(state)

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
                    colorbar={"title": "Норм. корреляция"},
                ),
                go.Scatter(
                    x=[corr.best_offset_m],
                    y=[corr.best_azimuth_deg],
                    mode="markers",
                    marker={"color": "#ff5a5f", "symbol": "star", "size": 14},
                    name="Пик",
                ),
            ]
        )
        figure.update_layout(
            template="plotly_dark",
            title=f"Тепловая карта корреляции | пик={corr.peak_correlation:.3f} @ {corr.best_azimuth_deg:.0f}°",
            xaxis_title="Смещение, м",
            yaxis_title="Азимут, °",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        return figure

    def _build_map_figure(self, state: dict[str, Any], history: list[dict[str, Any]]) -> go.Figure:
        patch = np.asarray(state.get("dem_patch", np.zeros((2, 2), dtype=float)), dtype=float)
        transform = self._parse_dem_patch_transform(state.get("dem_patch_transform"))
        fix: IMMResult = state["fix"]
        truth = state.get("truth")
        figure = go.Figure()
        if patch.size > 0 and transform is not None:
            x_coords, y_coords = self._build_patch_axes(patch, transform)
        else:
            x_coords = None
            y_coords = None
        figure.add_trace(
            go.Heatmap(
                z=patch,
                x=x_coords,
                y=y_coords,
                colorscale="Earth",
                opacity=0.8,
                showscale=False,
                name="Фрагмент ЦМР",
            )
        )
        for ellipse_trace in create_probability_ellipse_traces(fix.lat, fix.lon, fix.covariance):
            figure.add_trace(ellipse_trace)
        figure.add_trace(
            go.Scatter(
                x=[entry["lon"] for entry in history],
                y=[entry["lat"] for entry in history],
                mode="lines+markers",
                line={"color": "#2d9cdb", "width": 3},
                marker={"size": 6},
                name="Траектория",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[fix.lon],
                y=[fix.lat],
                mode="markers",
                marker={"color": "#ff5a5f", "size": 11},
                name="Текущая позиция",
            )
        )
        if truth is not None:
            figure.add_trace(
                go.Scatter(
                    x=[truth["lon"]],
                    y=[truth["lat"]],
                    mode="markers",
                    marker={"color": "#27ae60", "size": 10, "symbol": "diamond"},
                    name="Истинная позиция",
                )
            )
        map_lon_span = 0.02
        map_lat_span = 0.02
        if x_coords is not None and len(x_coords) >= 2:
            map_lon_span = max(abs(float(x_coords[-1]) - float(x_coords[0])), 1e-4)
        if y_coords is not None and len(y_coords) >= 2:
            map_lat_span = max(abs(float(y_coords[-1]) - float(y_coords[0])), 1e-4)
        arrow_length_deg = max(min(max(map_lon_span, map_lat_span) * 0.12, 0.02), 0.0025)
        figure.update_layout(
            template="plotly_dark",
            title="Карта рельефа и траектория",
            xaxis_title="Долгота",
            yaxis_title="Широта",
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
            shapes=create_arrow_shape(fix.lat, fix.lon, fix.azimuth_deg, length_deg=arrow_length_deg),
            annotations=[
                {
                    "x": fix.lon,
                    "y": fix.lat,
                    "xref": "x",
                    "yref": "y",
                    "text": build_velocity_annotation(fix),
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 90,
                    "ay": -70,
                    "bgcolor": "rgba(16, 24, 32, 0.85)",
                    "bordercolor": "#5dade2",
                    "borderwidth": 1,
                    "font": {"size": 12, "color": "#f4f7fb"},
                }
            ],
        )
        return figure

    def _parse_dem_patch_transform(self, raw_transform: Any) -> tuple[float, float, float, float, float, float] | None:
        """Parse a serialized affine tuple for DEM patch plotting."""

        if not isinstance(raw_transform, (list, tuple)) or len(raw_transform) < 6:
            return None
        values = tuple(float(raw_transform[idx]) for idx in range(6))
        if not all(math.isfinite(value) for value in values):
            return None
        return values

    def _build_patch_axes(
        self,
        patch: np.ndarray,
        transform: tuple[float, float, float, float, float, float],
    ) -> tuple[list[float], list[float]]:
        """Build longitude/latitude axes for a DEM patch from its affine transform."""

        a, b, c, d, e, f = transform
        del b, d
        width = int(patch.shape[1])
        height = int(patch.shape[0])
        x_coords = [c + a * (col + 0.5) for col in range(width)]
        y_coords = [f + e * (row + 0.5) for row in range(height)]
        return x_coords, y_coords

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
                name="Измеренный профиль рельефа (h_meas)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x_axis[: best_ref.size],
                y=best_ref,
                mode="lines",
                line={"color": "#ff6b6b", "dash": "dot", "width": 3},
                name="Лучший эталон ЦМР (h_ref)",
            )
        )
        if best_ref.size == h_meas.size and h_meas.size > 0:
            residual = h_meas - best_ref
            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=residual,
                    mode="lines",
                    line={"color": "#f1c40f", "width": 2},
                    name="Остаток (h_meas - h_ref)",
                    opacity=0.75,
                )
            )
        figure.update_layout(
            template="plotly_dark",
            title="Профили рельефа",
            xaxis_title="Дистанция, м",
            yaxis_title="Высота, м",
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
        terrain_status = "ПРОГНОЗ / FALLBACK" if used_prediction_only else "ОБНОВЛЕНИЕ ПО РЕЛЬЕФУ"
        terrain_color = "#f39c12" if used_prediction_only else "#27ae60"
        gnss_status = "GNSS ДОСТУПЕН" if gnss_available else "GNSS ПОТЕРЯН"
        gnss_color = "#27ae60" if gnss_available else "#c0392b"

        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=["Зависание", "Маршрут", "Поворот"],
                y=fix.model_weights.tolist(),
                marker_color=["#4aa3df", "#27ae60", "#f39c12"],
                name="Веса режимов",
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
        figure.update_layout(
            template="plotly_dark",
            title="Навигационная телеметрия",
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
                        f"Широта: {fix.lat:.6f}<br>"
                        f"Долгота: {fix.lon:.6f}<br>"
                        f"Азимут: {fix.azimuth_deg:.1f}°<br>"
                        f"Режим: {fix.dominant_mode}<br>"
                        f"HDOP: {hdop:.1f} м<br>"
                        f"Режим навигации: {nav_mode}<br>"
                        f"Размер окна: {selected_window_size} кадров<br>"
                        f"GNSS: {gnss_status}<br>"
                        f"PSLR: {corr.pslr_db:.2f} dB<br>"
                        f"Неоднозначность: {corr.is_ambiguous}<br>"
                        f"CRLB: {crlb_m:.1f} м<br>"
                        f"Рельеф информативен: {terrain_informative}"
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

    def _build_metrics_panel(self, state: dict[str, Any]) -> list[html.Div]:
        """Build compact metric cards for the latest navigation state."""

        fix_state = state["fix"]
        corr_state = state["corr"]
        observability = dict(state.get("observability", {}))
        truth = state.get("truth")
        h_meas = np.asarray(state.get("h_meas", np.array([], dtype=float)), dtype=float)
        best_ref = np.asarray(state.get("ref", np.array([], dtype=float)), dtype=float)

        if isinstance(fix_state, IMMResult):
            speed_mps = float(fix_state.speed_mps)
        else:
            speed_mps = float(dict(fix_state).get("speed_mps", 0.0))

        if isinstance(corr_state, CorrelationResult):
            peak_correlation = float(corr_state.peak_correlation)
            confidence = float(corr_state.confidence)
            best_azimuth_deg = float(corr_state.best_azimuth_deg)
            best_offset_m = float(corr_state.best_offset_m)
            best_offset_subsample_m = float(corr_state.best_offset_subsample_m)
            pslr_db = float(corr_state.pslr_db)
            ambiguity_peak_count = int(corr_state.ambiguity_peak_count)
            corr_is_reliable = bool(corr_state.is_reliable)
            corr_is_ambiguous = bool(corr_state.is_ambiguous)
        else:
            corr_dict = dict(corr_state)
            peak_correlation = float(corr_dict.get("peak_correlation", 0.0))
            confidence = float(corr_dict.get("confidence", 0.0))
            best_azimuth_deg = float(corr_dict.get("best_azimuth_deg", 0.0))
            best_offset_m = float(corr_dict.get("best_offset_m", 0.0))
            best_offset_subsample_m = float(corr_dict.get("best_offset_subsample_m", 0.0))
            pslr_db = float(corr_dict.get("pslr_db", 0.0))
            ambiguity_peak_count = int(corr_dict.get("ambiguity_peak_count", 0))
            corr_is_reliable = bool(corr_dict.get("is_reliable", False))
            corr_is_ambiguous = bool(corr_dict.get("is_ambiguous", False))

        nav_mode = str(state.get("nav_mode", "n/a"))
        used_prediction_only = bool(state.get("used_prediction_only", False))
        fix_source = "Прогноз" if used_prediction_only else "Привязка по рельефу"
        runtime_stats = dict(state.get("runtime_stats", {}))
        reaction_latency_ms = float(state.get("reaction_latency_ms", float("nan")))
        pipeline_latency_ms = float(state.get("pipeline_latency_ms", float("nan")))
        dashboard_latency_ms = float(state.get("dashboard_latency_ms", float("nan")))
        state_queue_replacements = int(runtime_stats.get("state_queue_replacements", 0))
        frame_drop_count = int(runtime_stats.get("frame_drop_count", 0))
        latency_target_ok = np.isfinite(reaction_latency_ms) and reaction_latency_ms < 200.0
        integrity_status = str(state.get("integrity_status", "OK"))
        if frame_drop_count == 0 and state_queue_replacements == 0 and integrity_status == "OK":
            integrity_status = "OK / без потерь"

        residual_rmse = float("nan")
        if h_meas.size > 0 and best_ref.size == h_meas.size:
            residual_rmse = float(np.sqrt(np.nanmean((h_meas - best_ref) ** 2)))

        track_error_m = float("nan")
        if truth is not None:
            if isinstance(fix_state, IMMResult):
                lat_deg = float(fix_state.lat)
                lon_deg = float(fix_state.lon)
            else:
                fix_dict = dict(fix_state)
                lat_deg = float(fix_dict.get("lat", 0.0))
                lon_deg = float(fix_dict.get("lon", 0.0))
            lat_error_m = (lat_deg - float(truth["lat"])) * 111_320.0
            lon_scale = max(math.cos(math.radians(lat_deg)), 1e-6)
            lon_error_m = (lon_deg - float(truth["lon"])) * 111_320.0 * lon_scale
            track_error_m = float(math.hypot(lat_error_m, lon_error_m))

        metric_items = [
            ("Источник позиции", fix_source),
            ("Режим навигации", nav_mode),
            ("Корреляция надежна", "ДА" if corr_is_reliable else "НЕТ"),
            ("Есть неоднозначность", "ДА" if corr_is_ambiguous else "НЕТ"),
            ("Пик корреляции", f"{peak_correlation:.3f}"),
            ("Уверенность", f"{confidence:.3f}"),
            ("Лучший азимут", f"{best_azimuth_deg:.1f}°"),
            ("Лучшее смещение", f"{best_offset_m:.1f} м"),
            ("Субпиксельное смещение", f"{best_offset_subsample_m:.1f} м"),
            ("Скорость", f"{speed_mps:.2f} м/с"),
            ("Задержка реакции", _format_metric_value(reaction_latency_ms, "мс")),
            ("Цель по задержке", "< 200 мс" if latency_target_ok else "БОЛЬШЕ 200 мс"),
            ("Задержка пайплайна", _format_metric_value(pipeline_latency_ms, "мс")),
            ("Задержка дашборда", _format_metric_value(dashboard_latency_ms, "мс")),
            ("PSLR", f"{pslr_db:.2f} dB"),
            ("Число пиков", str(ambiguity_peak_count)),
            ("CRLB", _format_metric_value(observability.get("crlb_m"), "м")),
            ("Энергия градиента", _format_metric_value(observability.get("gradient_energy"), "")),
            ("Подсказка наблюдаемости", _format_metric_value(observability.get("efficiency_hint"), "")),
            ("Смещение рельефа", _format_metric_value(state.get("terrain_bias_m"), "м")),
            ("RMSE остатка", _format_metric_value(residual_rmse, "м")),
            ("Ошибка трека", _format_metric_value(track_error_m, "м")),
            ("Целостность", integrity_status),
            ("Потеряно кадров", str(frame_drop_count)),
            ("Замены состояний", str(state_queue_replacements)),
            ("Размер окна", f"{int(state.get('selected_window_size', 0))} кадров"),
            ("Состояние GNSS", "ВКЛ" if bool(state.get("gnss_available", True)) else "ВЫКЛ"),
        ]
        return [self._metric_card(label, value) for label, value in metric_items]

    def _summarize_state_for_store(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy-heavy runtime state into a lightweight serializable summary."""

        fix: IMMResult = state["fix"]
        corr: CorrelationResult = state["corr"]
        truth = state.get("truth")
        observability = dict(state.get("observability", {}))
        h_meas = np.asarray(state.get("h_meas", np.array([], dtype=float)), dtype=float)
        best_ref = np.asarray(state.get("ref", np.array([], dtype=float)), dtype=float)
        return {
            "fix": {
                "lat": float(fix.lat),
                "lon": float(fix.lon),
                "speed_mps": float(fix.speed_mps),
                "azimuth_deg": float(fix.azimuth_deg),
                "dominant_mode": str(fix.dominant_mode),
                "covariance": np.asarray(fix.covariance, dtype=float).tolist(),
            },
            "corr": {
                "peak_correlation": float(corr.peak_correlation),
                "confidence": float(corr.confidence),
                "best_azimuth_deg": float(corr.best_azimuth_deg),
                "best_offset_m": float(corr.best_offset_m),
                "best_offset_subsample_m": float(corr.best_offset_subsample_m),
                "pslr_db": float(corr.pslr_db),
                "ambiguity_peak_count": int(corr.ambiguity_peak_count),
            },
            "observability": observability,
            "terrain_bias_m": float(state.get("terrain_bias_m", 0.0)),
            "selected_window_size": int(state.get("selected_window_size", 0)),
            "gnss_available": bool(state.get("gnss_available", True)),
            "reaction_latency_ms": float(state.get("reaction_latency_ms", float("nan"))),
            "pipeline_latency_ms": float(state.get("pipeline_latency_ms", float("nan"))),
            "dashboard_latency_ms": float(state.get("dashboard_latency_ms", float("nan"))),
            "integrity_status": str(state.get("integrity_status", "OK")),
            "runtime_stats": dict(state.get("runtime_stats", {})),
            "truth": truth,
            "dem_patch_transform": state.get("dem_patch_transform"),
            "h_meas": h_meas.tolist(),
            "ref": best_ref.tolist(),
        }

    def _metric_card(self, label: str, value: str) -> html.Div:
        return html.Div(
            style={
                "backgroundColor": "#17212b",
                "border": "1px solid #31404f",
                "borderRadius": "10px",
                "padding": "10px 12px",
            },
            children=[
                html.Div(label, style={"fontSize": "12px", "color": "#8fa3b8", "marginBottom": "6px"}),
                html.Div(value, style={"fontSize": "18px", "fontWeight": "600"}),
            ],
        )


def _format_metric_value(value: Any, unit: str) -> str:
    """Format scalar dashboard metrics while handling NaN/inf cleanly."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    if unit:
        return f"{numeric:.2f} {unit}"
    return f"{numeric:.3f}"
