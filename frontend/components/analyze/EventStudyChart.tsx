import * as React from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import type { HistoryEventStudyResponse } from "@/lib/analyze/types";

interface EventStudyChartProps {
  data: HistoryEventStudyResponse | null;
  loading?: boolean;
  errorMessage?: string | null;
}

// Background tint for the predicted-regime band. Lifted off the same
// palette tokens the rest of the workspace uses for hawkish/dovish so a
// "high" band reads as risk-on red and "calm" as risk-off green.
const REGIME_FILL: Record<string, string> = {
  calm: "hsl(var(--up))",
  normal: "hsl(var(--neutral))",
  high: "hsl(var(--down))",
};

const TOOLTIP_STYLE: React.CSSProperties = {
  background: "hsl(var(--popover))",
  color: "hsl(var(--popover-foreground))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 12,
};

function regimeBadgeVariant(
  label: string | null | undefined,
): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "calm") return "dovish";
  if (label === "high") return "hawkish";
  if (label === "normal") return "neutral";
  return "outline";
}

function regimeLabel(label: string | null | undefined): string {
  if (!label) return "n/a";
  return label;
}

export function buildEventStudyHeadline(
  predicted: string | null | undefined,
  realized: string | null | undefined,
): string {
  return `predicted ${regimeLabel(predicted)}, realized ${regimeLabel(realized)}`;
}

export function EventStudyChart({ data, loading, errorMessage }: EventStudyChartProps) {
  const chartData = React.useMemo(() => {
    if (!data) return [];
    return data.forward_dates.map((date, idx) => ({
      day: idx + 1,
      date,
      close: data.forward_close[idx] ?? null,
      log_return: data.forward_log_returns[idx] ?? null,
    }));
  }, [data]);

  if (loading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Event study</CardTitle>
          <CardDescription>Loading forward 10-day price path…</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-72 w-full animate-pulse rounded-md bg-muted/40" />
        </CardContent>
      </Card>
    );
  }

  if (errorMessage) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Event study</CardTitle>
          <CardDescription>Forward 10-day market response.</CardDescription>
        </CardHeader>
        <CardContent>
          <EmptyState
            variant="inline"
            title="Could not load market path"
            description={errorMessage}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data || chartData.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Event study</CardTitle>
          <CardDescription>Forward 10-day market response.</CardDescription>
        </CardHeader>
        <CardContent>
          <EmptyState
            variant="inline"
            title="No forward bars yet"
            description="Yfinance returned no trading bars after the event date."
          />
        </CardContent>
      </Card>
    );
  }

  const closes = chartData.map((d) => d.close).filter((v): v is number => v != null);
  const closeMin = closes.length ? Math.min(...closes) : 0;
  const closeMax = closes.length ? Math.max(...closes) : 1;
  const closeSpan = Math.max(closeMax - closeMin, 1e-6);
  const yMin = closeMin - closeSpan * 0.08;
  const yMax = closeMax + closeSpan * 0.08;
  const firstDay = chartData[0]?.day ?? 1;
  const lastDay = chartData[chartData.length - 1]?.day ?? 10;
  const bandFill =
    REGIME_FILL[data.predicted_regime ?? ""] ?? "hsl(var(--muted-foreground))";
  const headline = buildEventStudyHeadline(data.predicted_regime, data.realized_regime);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="text-base">Event study</CardTitle>
            <CardDescription>
              {data.symbol} close, 10 trading days after {data.event_date}.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={regimeBadgeVariant(data.predicted_regime)}>
              predicted {regimeLabel(data.predicted_regime)}
            </Badge>
            <Badge variant={regimeBadgeVariant(data.realized_regime)}>
              realized {regimeLabel(data.realized_regime)}
            </Badge>
          </div>
        </div>
        <p className="mt-2 text-sm text-foreground">{headline}</p>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
            >
              <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="2 3" />
              <XAxis
                dataKey="day"
                type="number"
                domain={[firstDay, lastDay]}
                ticks={chartData.map((d) => d.day)}
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
                label={{
                  value: "trading day",
                  position: "insideBottom",
                  offset: -2,
                  style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                }}
              />
              <YAxis
                domain={[yMin, yMax]}
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
                tickFormatter={(value) => Number(value).toFixed(0)}
                label={{
                  value: "close",
                  angle: -90,
                  position: "insideLeft",
                  style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                }}
              />
              {data.predicted_regime ? (
                <ReferenceArea
                  x1={firstDay}
                  x2={lastDay}
                  y1={yMin}
                  y2={yMax}
                  fill={bandFill}
                  fillOpacity={0.12}
                  stroke={bandFill}
                  strokeOpacity={0.25}
                  ifOverflow="extendDomain"
                />
              ) : null}
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                cursor={{ strokeDasharray: "2 3" }}
                formatter={(value, name) => {
                  if (name === "close") {
                    return [Number(value).toFixed(2), "close"];
                  }
                  return [String(value), String(name)];
                }}
                labelFormatter={(label, payload) => {
                  const first = Array.isArray(payload) && payload[0];
                  const date =
                    first && typeof first === "object" && "payload" in first
                      ? (first as { payload?: { date?: string } }).payload?.date
                      : null;
                  return date ? `t+${label} · ${date}` : `t+${label}`;
                }}
              />
              <Line
                type="monotone"
                dataKey="close"
                stroke="hsl(var(--primary))"
                strokeWidth={1.5}
                dot={{ r: 2.5, fill: "hsl(var(--primary))" }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        {data.realized_vol_10d != null ? (
          <div className="mt-3 text-[11px] text-muted-foreground">
            Realised 10-day vol: {data.realized_vol_10d.toFixed(4)} (log-return std).
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
