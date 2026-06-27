import type { ReactNode } from "react";

interface StatusBadgeProps {
  label: string;
  value: string;
  tone?: "neutral" | "green" | "red" | "blue" | "orange" | "yellow";
  icon?: ReactNode;
}

const toneClass: Record<NonNullable<StatusBadgeProps["tone"]>, string> = {
  neutral: "border-[#263442] bg-[#111923] text-[#E5EEF7]",
  green: "border-[#285d3e] bg-[#112117] text-[#4ADE80]",
  red: "border-[#6b2326] bg-[#231214] text-[#EF4444]",
  blue: "border-[#244a8f] bg-[#111c30] text-[#60a5fa]",
  orange: "border-[#7b3c11] bg-[#28170d] text-[#F97316]",
  yellow: "border-[#74611a] bg-[#261f0c] text-[#FACC15]"
};

export function StatusBadge({ label, value, tone = "neutral", icon }: StatusBadgeProps) {
  return (
    <div className={`inline-flex min-h-9 items-center gap-2 rounded-lg border px-3 py-2 text-sm ${toneClass[tone]}`}>
      {icon}
      <span className="text-[#95A3B5]">{label}:</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
