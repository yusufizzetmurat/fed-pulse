import * as React from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import { cn } from "@/lib/utils";

export type SparklineTone = "up" | "down" | "neutral" | "primary";

interface SparklineProps {
  values: Array<number | null | undefined>;
  tone?: SparklineTone;
  className?: string;
  height?: number;
  labels?: string[];
  formatTooltip?: (value: number, label?: string) => string;
}

const TONE_VARS: Record<SparklineTone, string> = {
  up: "var(--up)",
  down: "var(--down)",
  neutral: "var(--neutral)",
  primary: "var(--primary)",
};

function inferTone(values: Array<number | null | undefined>, fallback: SparklineTone): SparklineTone {
  if (fallback !== "neutral") return fallback;
  const points = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (points.length < 2) return "neutral";
  const first = points[0];
  const last = points[points.length - 1];
  if (last > first) return "up";
  if (last < first) return "down";
  return "neutral";
}

export function Sparkline({
  values,
  tone = "neutral",
  className,
  height = 28,
  labels,
  formatTooltip,
}: SparklineProps) {
  const effectiveTone = inferTone(values, tone);
  const stroke = `hsl(${TONE_VARS[effectiveTone]})`;
  const data = React.useMemo(
    () =>
      values.map((value, index) => ({
        x: index,
        value: typeof value === "number" && Number.isFinite(value) ? value : null,
        label: labels?.[index] ?? "",
      })),
    [values, labels],
  );

  const hasSignal = data.some((row) => row.value != null);
  if (!hasSignal) {
    return (
      <div
        className={cn("flex items-center text-[10px] text-muted-foreground", className)}
        style={{ height }}
      >
        no data
      </div>
    );
  }

  const gradientId = React.useId();

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.25} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          {formatTooltip ? (
            <Tooltip
              cursor={{ stroke, strokeWidth: 0.75, strokeDasharray: "2 2" }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const point = payload[0].payload as { value: number; label: string };
                if (point.value == null) return null;
                return (
                  <div className="rounded-md border border-border bg-popover px-2 py-1 text-[11px] shadow-md">
                    <p className="numeric font-medium">{formatTooltip(point.value, point.label)}</p>
                  </div>
                );
              }}
            />
          ) : null}
          <Area
            type="monotone"
            dataKey="value"
            stroke={stroke}
            strokeWidth={1.25}
            fill={`url(#${gradientId})`}
            isAnimationActive={false}
            connectNulls
            dot={false}
            activeDot={{ r: 2, stroke, fill: stroke }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
