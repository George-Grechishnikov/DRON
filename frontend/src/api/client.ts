import type {
  DatasetLoadRequest,
  DatasetLoadResponse,
  HeatmapResponse,
  HealthResponse,
  LogsResponse,
  MetricsResponse,
  ProfilesResponse,
  ReplayActionResponse,
  ReplayStateResponse,
  TimelineResponse,
  TrajectoryResponse
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail ?? "Request failed");
  }
  return data as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  loadDataset: (payload: DatasetLoadRequest) =>
    request<DatasetLoadResponse>("/api/dataset/load", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  validateDataset: () => request("/api/dataset/validate"),
  startReplay: (speed: number) =>
    request<ReplayActionResponse>("/api/replay/start", {
      method: "POST",
      body: JSON.stringify({ speed })
    }),
  pauseReplay: () => request<ReplayActionResponse>("/api/replay/pause", { method: "POST", body: "{}" }),
  restartReplay: () => request<ReplayActionResponse>("/api/replay/restart", { method: "POST", body: "{}" }),
  stopReplay: () => request<ReplayActionResponse>("/api/replay/stop", { method: "POST", body: "{}" }),
  setSpeed: (speed: number) =>
    request<ReplayActionResponse>("/api/replay/set_speed", {
      method: "POST",
      body: JSON.stringify({ speed })
    }),
  forceGnssOff: () => request<ReplayActionResponse>("/api/gnss/force_off", { method: "POST", body: "{}" }),
  forceGnssOn: () => request<ReplayActionResponse>("/api/gnss/force_on", { method: "POST", body: "{}" }),
  state: () => request<ReplayStateResponse>("/api/state"),
  trajectory: () => request<TrajectoryResponse>("/api/trajectory"),
  profiles: () => request<ProfilesResponse>("/api/profiles"),
  heatmap: () => request<HeatmapResponse>("/api/correlation/heatmap"),
  timeline: () => request<TimelineResponse>("/api/timeline"),
  logs: () => request<LogsResponse>("/api/logs"),
  metrics: () => request<MetricsResponse>("/api/metrics")
};

export { API_BASE };
