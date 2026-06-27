import { Expand, HelpCircle, Radar, Settings } from "lucide-react";
import { StatusBadge } from "./StatusBadge";

interface HeaderProps {
  gnssAvailable: boolean;
  navMode: string;
  sensorsStatus: string;
  sampleRateHz: number | null | undefined;
  correlation: number | null | undefined;
}

export function Header({ gnssAvailable, navMode, sensorsStatus, sampleRateHz, correlation }: HeaderProps) {
  return (
    <header className="flex flex-col gap-4 border-b border-[#263442] bg-[#101821] px-5 py-4 xl:flex-row xl:items-center xl:justify-between">
      <div className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#263442] bg-[#111923]">
          <Radar className="h-5 w-5 text-[#60a5fa]" />
        </div>
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-[#E5EEF7]">Симуляция полета БПЛА</h1>
          <p className="text-sm text-[#95A3B5]">Terrain Navigation System engineering cockpit</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge label="GNSS" value={gnssAvailable ? "ON" : "OFF"} tone={gnssAvailable ? "green" : "red"} />
        <StatusBadge
          label="Режим"
          value={navMode}
          tone={navMode === "TERRAIN_NAV" ? "blue" : navMode === "GNSS" ? "green" : navMode === "LOST" ? "red" : "neutral"}
        />
        <StatusBadge
          label="Сенсоры"
          value={sensorsStatus}
          tone={sensorsStatus === "OK" ? "green" : sensorsStatus === "WARNING" ? "orange" : sensorsStatus === "ERROR" ? "red" : "neutral"}
        />
        <StatusBadge label="Частота" value={sampleRateHz ? `${sampleRateHz.toFixed(1)} Гц` : "--"} tone="neutral" />
        <StatusBadge label="Корреляция" value={correlation != null ? correlation.toFixed(2) : "--"} tone="yellow" />
      </div>

      <div className="flex items-center gap-2 text-[#95A3B5]">
        <button className="rounded-lg border border-[#263442] bg-[#111923] p-2 hover:bg-[#162231]"><Settings className="h-4 w-4" /></button>
        <button className="rounded-lg border border-[#263442] bg-[#111923] p-2 hover:bg-[#162231]"><HelpCircle className="h-4 w-4" /></button>
        <button className="rounded-lg border border-[#263442] bg-[#111923] p-2 hover:bg-[#162231]"><Expand className="h-4 w-4" /></button>
      </div>
    </header>
  );
}
