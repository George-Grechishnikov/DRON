export type NavMode = "GNSS" | "TERRAIN_NAV" | "LOST" | "INIT" | "IDLE";

export type StatusLevel = "OK" | "WARNING" | "ERROR" | "PROCESSING" | "READY" | "NO_DATASET";

export interface HealthResponse {
  status: string;
  backend: string;
}

export interface DatasetLoadRequest {
  dem_path?: string;
  radar_data_path?: string;
  truth_path?: string;
  barometer_path?: string;
  config_path?: string;
}

export interface DatasetLoadResponse {
  loaded: boolean;
  radar_samples: number;
  truth_samples: number;
  barometer_samples: number;
  sample_rate_hz: number;
  duration_s: number;
  errors: string[];
}

export interface ReplayActionResponse {
  started?: boolean;
  paused?: boolean;
  stopped?: boolean;
  restarted?: boolean;
  updated?: boolean;
  processing?: boolean;
  sample_index?: number;
  speed?: number;
}

export interface PositionState {
  lat?: number | null;
  lon?: number | null;
  alt_msl?: number | null;
  heading_deg?: number | null;
  speed_mps?: number | null;
}

export interface ReplayStateResponse {
  timestamp: number;
  elapsed_time?: string | null;
  sample_index: number;
  total_samples: number;
  gnss_available: boolean;
  nav_mode: NavMode | string;
  sensors_status: StatusLevel | string;
  sample_rate_hz?: number | null;
  data_rate_hz?: number | null;
  correlation_score?: number | null;
  speed_mps?: number | null;
  heading_deg?: number | null;
  alt_msl?: number | null;
  radar_alt_m?: number | null;
  terrain_h?: number | null;
  position_error_3d_m?: number | null;
  position_error_2d_m?: number | null;
  distance_km?: number | null;
  truth?: PositionState | null;
  estimate?: PositionState | null;
  processing?: boolean | null;
  playing?: boolean | null;
  error?: string | null;
}

export interface TrajectoryPoint {
  timestamp: number;
  lat: number | null;
  lon: number | null;
}

export interface EventPoint {
  timestamp: number;
  type: string;
}

export interface TrajectoryResponse {
  truth: TrajectoryPoint[];
  estimated: TrajectoryPoint[];
  events: EventPoint[];
}

export interface HeatmapResponse {
  azimuths: number[];
  offsets: number[];
  values: number[][];
  best_azimuth?: number | null;
  best_offset?: number | null;
  best_score?: number | null;
}

export interface ProfilesResponse {
  time: Array<number | null>;
  baro_alt_m: Array<number | null>;
  dem_height_m: Array<number | null>;
  radar_alt_m: Array<number | null>;
  reconstructed_profile_m: Array<number | null>;
}

export interface TimelineSegment {
  start: number;
  end: number;
  mode: string;
}

export interface TimelineResponse {
  duration_s: number;
  current_time_s: number;
  segments: TimelineSegment[];
}

export interface MetricsResponse {
  total_flight_time_s: number;
  total_distance_m: number;
  average_speed_mps: number;
  max_speed_mps: number;
  mean_position_error_m?: number | null;
  max_position_error_m?: number | null;
  rmse_m?: number | null;
  cep50_m?: number | null;
  cep95_m?: number | null;
  average_correlation_score?: number | null;
  min_correlation_score?: number | null;
  time_in_gnss_s: number;
  time_in_terrain_nav_s: number;
  time_lost_s: number;
}

export interface LogEntry {
  time: string;
  level: string;
  event: string;
  details: string;
}

export interface LogsResponse {
  logs: LogEntry[];
}
