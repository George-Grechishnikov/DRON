import { Pause, Play, RefreshCcw, ShieldOff, ShieldCheck, Upload } from "lucide-react";

interface SimulationControlsProps {
  configPath: string;
  demPath: string;
  radarPath: string;
  truthPath: string;
  barometerPath: string;
  speed: number;
  timestamp: number;
  duration: number;
  onConfigPathChange: (value: string) => void;
  onDemPathChange: (value: string) => void;
  onRadarPathChange: (value: string) => void;
  onTruthPathChange: (value: string) => void;
  onBarometerPathChange: (value: string) => void;
  onSpeedChange: (value: number) => void;
  onLoad: () => void;
  onStart: () => void;
  onPause: () => void;
  onRestart: () => void;
  onForceGnssOff: () => void;
  onForceGnssOn: () => void;
}

const speedOptions = [1, 2, 5, 10];

export function SimulationControls(props: SimulationControlsProps) {
  const {
    configPath,
    demPath,
    radarPath,
    truthPath,
    barometerPath,
    speed,
    timestamp,
    duration,
    onConfigPathChange,
    onDemPathChange,
    onRadarPathChange,
    onTruthPathChange,
    onBarometerPathChange,
    onSpeedChange,
    onLoad,
    onStart,
    onPause,
    onRestart,
    onForceGnssOff,
    onForceGnssOn
  } = props;

  return (
    <section className="rounded-xl border border-[#263442] bg-[#101821] p-4">
      <h2 className="mb-4 text-sm font-semibold tracking-[0.14em] text-[#E5EEF7]">УПРАВЛЕНИЕ СИМУЛЯЦИЕЙ</h2>

      <div className="grid gap-3">
        {[
          ["config.yaml", configPath, onConfigPathChange],
          ["DEM path", demPath, onDemPathChange],
          ["radar_data.nmea", radarPath, onRadarPathChange],
          ["truth.csv", truthPath, onTruthPathChange],
          ["barometer.csv", barometerPath, onBarometerPathChange]
        ].map(([label, value, handler]) => (
          <label key={label} className="grid gap-1">
            <span className="text-xs text-[#95A3B5]">{label}</span>
            <input
              className="min-h-10 rounded-lg border border-[#263442] bg-[#0d131a] px-3 text-sm text-[#E5EEF7] outline-none"
              value={value as string}
              onChange={(event) => (handler as (value: string) => void)(event.target.value)}
            />
          </label>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-[#3B82F6] px-3 text-sm font-medium text-white" onClick={onLoad}>
          <Upload className="h-4 w-4" /> Загрузить данные
        </button>
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-[#2f7f55] px-3 text-sm font-medium text-white" onClick={onStart}>
          <Play className="h-4 w-4" /> Старт Auto Demo
        </button>
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-[#263442] bg-[#111923] px-3 text-sm text-[#E5EEF7]" onClick={onPause}>
          <Pause className="h-4 w-4" /> Пауза
        </button>
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg border border-[#263442] bg-[#111923] px-3 text-sm text-[#E5EEF7]" onClick={onRestart}>
          <RefreshCcw className="h-4 w-4" /> Перезапуск
        </button>
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-[#7f1d1d] px-3 text-sm font-medium text-white" onClick={onForceGnssOff}>
          <ShieldOff className="h-4 w-4" /> GNSS OFF
        </button>
        <button className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-[#166534] px-3 text-sm font-medium text-white" onClick={onForceGnssOn}>
          <ShieldCheck className="h-4 w-4" /> GNSS ON
        </button>
      </div>

      <div className="mt-5">
        <div className="mb-2 text-xs text-[#95A3B5]">Скорость воспроизведения</div>
        <div className="flex flex-wrap gap-2">
          {speedOptions.map((option) => (
            <button
              key={option}
              className={`min-h-10 rounded-lg px-3 text-sm ${speed === option ? "bg-[#3B82F6] text-white" : "border border-[#263442] bg-[#111923] text-[#E5EEF7]"}`}
              onClick={() => onSpeedChange(option)}
            >
              {option}x
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between text-xs text-[#95A3B5]">
          <span>{timestamp.toFixed(1)} s</span>
          <span>{duration.toFixed(1)} s</span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(duration, 1)}
          value={Math.min(timestamp, Math.max(duration, 1))}
          readOnly
          className="h-2 w-full accent-[#3B82F6]"
        />
      </div>
    </section>
  );
}
