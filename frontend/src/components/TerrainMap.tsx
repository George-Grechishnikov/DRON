import type { EventPoint, ReplayStateResponse, TrajectoryResponse } from "../types";

interface TerrainMapProps {
  trajectory: TrajectoryResponse | null;
  state: ReplayStateResponse | null;
}

function projectPoints(points: Array<{ lat: number | null; lon: number | null }>) {
  const valid = points.filter((point) => point.lat != null && point.lon != null) as Array<{ lat: number; lon: number }>;
  if (!valid.length) {
    return { items: [], minLat: 0, maxLat: 1, minLon: 0, maxLon: 1 };
  }
  const lats = valid.map((point) => point.lat);
  const lons = valid.map((point) => point.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const latSpan = Math.max(maxLat - minLat, 0.0001);
  const lonSpan = Math.max(maxLon - minLon, 0.0001);
  return {
    minLat,
    maxLat,
    minLon,
    maxLon,
    items: valid.map((point) => ({
      x: 40 + ((point.lon - minLon) / lonSpan) * 620,
      y: 340 - ((point.lat - minLat) / latSpan) * 280
    }))
  };
}

function polyline(points: Array<{ x: number; y: number }>) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function findEvent(events: EventPoint[] | undefined, type: string) {
  return events?.find((event) => event.type === type) ?? null;
}

export function TerrainMap({ trajectory, state }: TerrainMapProps) {
  const truth = trajectory?.truth ?? [];
  const estimated = trajectory?.estimated ?? [];
  const events = trajectory?.events ?? [];
  const projection = projectPoints([...truth, ...estimated]);

  const currentPoint =
    state?.estimate?.lat != null && state?.estimate?.lon != null
      ? projectPoints([{ lat: state.estimate.lat, lon: state.estimate.lon }]).items[0]
      : null;
  const truthProjected = projectPoints(truth).items;
  const estimatedProjected = projectPoints(estimated).items;

  function eventMarker(type: string) {
    const event = findEvent(events, type);
    if (!event) return null;
    const source = truth.find((point) => Math.abs(point.timestamp - event.timestamp) < 0.001) ??
      estimated.find((point) => Math.abs(point.timestamp - event.timestamp) < 0.001);
    if (!source || source.lat == null || source.lon == null) return null;
    const point = {
      x: 40 + ((source.lon - projection.minLon) / Math.max(projection.maxLon - projection.minLon, 0.0001)) * 620,
      y: 340 - ((source.lat - projection.minLat) / Math.max(projection.maxLat - projection.minLat, 0.0001)) * 280
    };
    return { point, event };
  }

  const gnssLost = eventMarker("GNSS_LOST");
  const terrainNav = eventMarker("TERRAIN_NAV_START");
  const startPoint = truthProjected[0] ?? estimatedProjected[0] ?? null;

  return (
    <section className="rounded-xl border border-[#263442] bg-[#101821] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-[0.14em] text-[#E5EEF7]">КАРТА РЕЛЬЕФА</h2>
        <div className="flex items-center gap-2 text-xs text-[#95A3B5]">
          <button className="rounded-md border border-[#263442] px-2 py-1">+</button>
          <button className="rounded-md border border-[#263442] px-2 py-1">-</button>
        </div>
      </div>
      <div className="overflow-hidden rounded-xl border border-[#263442] bg-[#0b1118]">
        <svg viewBox="0 0 700 380" className="h-[420px] w-full">
          <defs>
            <linearGradient id="demGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#18222c" />
              <stop offset="50%" stopColor="#2a3742" />
              <stop offset="100%" stopColor="#101821" />
            </linearGradient>
            <pattern id="ridgePattern" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M0 30 Q10 10 20 20 T40 10" stroke="#415364" strokeWidth="1.2" fill="none" opacity="0.45" />
            </pattern>
          </defs>
          <rect x="0" y="0" width="700" height="380" fill="url(#demGradient)" />
          <rect x="0" y="0" width="700" height="380" fill="url(#ridgePattern)" opacity="0.5" />

          <g opacity="0.12" stroke="#6d7f92">
            {Array.from({ length: 8 }).map((_, index) => (
              <line key={`v-${index}`} x1={40 + index * 90} y1={20} x2={40 + index * 90} y2={340} />
            ))}
            {Array.from({ length: 5 }).map((_, index) => (
              <line key={`h-${index}`} x1={40} y1={20 + index * 80} x2={660} y2={20 + index * 80} />
            ))}
          </g>

          {truthProjected.length > 1 ? (
            <polyline points={polyline(truthProjected)} fill="none" stroke="#4ADE80" strokeWidth="4" strokeLinecap="round" />
          ) : null}
          {estimatedProjected.length > 1 ? (
            <polyline points={polyline(estimatedProjected)} fill="none" stroke="#3B82F6" strokeWidth="4" strokeLinecap="round" />
          ) : null}

          {startPoint ? (
            <>
              <circle cx={startPoint.x} cy={startPoint.y} r="7" fill="#ffffff" />
              <text x={startPoint.x + 10} y={startPoint.y - 10} fill="#E5EEF7" fontSize="12">СТАРТ</text>
            </>
          ) : null}
          {gnssLost ? (
            <>
              <circle cx={gnssLost.point.x} cy={gnssLost.point.y} r="8" fill="#EF4444" />
              <text x={gnssLost.point.x + 10} y={gnssLost.point.y - 10} fill="#EF4444" fontSize="12">ПОТЕРЯ GNSS</text>
            </>
          ) : null}
          {terrainNav ? (
            <>
              <polygon
                points={`${terrainNav.point.x},${terrainNav.point.y - 9} ${terrainNav.point.x + 9},${terrainNav.point.y} ${terrainNav.point.x},${terrainNav.point.y + 9} ${terrainNav.point.x - 9},${terrainNav.point.y}`}
                fill="#F97316"
              />
              <text x={terrainNav.point.x + 10} y={terrainNav.point.y - 10} fill="#F97316" fontSize="12">ПЕРЕХОД В TERRAIN_NAV</text>
            </>
          ) : null}
          {currentPoint ? (
            <>
              <circle cx={currentPoint.x} cy={currentPoint.y} r="9" fill="#FACC15" stroke="#111" strokeWidth="2" />
              <text x={currentPoint.x + 10} y={currentPoint.y + 4} fill="#FACC15" fontSize="12">ТЕКУЩАЯ ПОЗИЦИЯ</text>
            </>
          ) : null}

          <g transform="translate(28,22)">
            <rect x="0" y="0" width="150" height="110" rx="10" fill="#0d131a" opacity="0.86" stroke="#263442" />
            {[
              ["#4ADE80", "Truth trajectory"],
              ["#3B82F6", "Estimated trajectory"],
              ["#ffffff", "Старт"],
              ["#FACC15", "Текущая позиция"],
              ["#EF4444", "Потеря GNSS"],
              ["#F97316", "Переход в TERRAIN_NAV"]
            ].map(([color, label], index) => (
              <g key={label} transform={`translate(14, ${18 + index * 15})`}>
                <circle cx="0" cy="0" r="4" fill={color} />
                <text x="10" y="4" fill="#E5EEF7" fontSize="11">{label}</text>
              </g>
            ))}
          </g>

          <g transform="translate(618,24)">
            <circle cx="18" cy="18" r="18" fill="#0d131a" opacity="0.95" stroke="#263442" />
            <text x="18" y="12" textAnchor="middle" fill="#E5EEF7" fontSize="11">N</text>
            <path d="M18 16 L22 28 L18 24 L14 28 Z" fill="#FACC15" />
          </g>

          <g transform="translate(532,338)">
            <line x1="0" y1="0" x2="80" y2="0" stroke="#E5EEF7" strokeWidth="3" />
            <line x1="0" y1="-5" x2="0" y2="5" stroke="#E5EEF7" strokeWidth="2" />
            <line x1="80" y1="-5" x2="80" y2="5" stroke="#E5EEF7" strokeWidth="2" />
            <text x="40" y="-8" textAnchor="middle" fill="#95A3B5" fontSize="11">1 km</text>
          </g>
        </svg>
      </div>
    </section>
  );
}
