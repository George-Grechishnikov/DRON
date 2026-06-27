interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
}

export function MetricCard({ label, value, hint }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-[#263442] bg-[#101821] p-4">
      <div className="text-xs uppercase tracking-[0.12em] text-[#95A3B5]">{label}</div>
      <div className="mt-3 text-2xl font-semibold text-[#E5EEF7]">{value}</div>
      {hint ? <div className="mt-2 text-xs text-[#95A3B5]">{hint}</div> : null}
    </div>
  );
}
