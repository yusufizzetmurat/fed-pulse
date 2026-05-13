import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ChartRow } from "@/lib/analyze/derive";
import { formatDateTick, formatPrice, formatVol } from "@/lib/analyze/format";

interface ForecastChartProps {
  title: string;
  description: string;
  data: ChartRow[];
  kind: "close" | "volatility";
  splitTimestamp?: string;
  includeRealized: boolean;
  hasConfidence: boolean;
  confidenceLabel: string;
  yDomain?: [number, number];
}

const PALETTE = {
  close: {
    history: "hsl(var(--chart-1))",
    forecast: "hsl(var(--chart-2))",
    realized: "hsl(var(--chart-4))",
    band: "hsl(var(--chart-2) / 0.18)",
    bandEdge: "hsl(var(--chart-2) / 0.45)",
  },
  volatility: {
    history: "hsl(var(--chart-3))",
    forecast: "hsl(var(--chart-5))",
    realized: "hsl(var(--chart-4))",
    band: "hsl(var(--chart-5) / 0.18)",
    bandEdge: "hsl(var(--chart-5) / 0.45)",
  },
} as const;

function ChartTooltip({
  active,
  payload,
  label,
  kind,
  confidenceLabel,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartRow }>;
  label?: string;
  kind: "close" | "volatility";
  confidenceLabel: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload as ChartRow | undefined;
  if (!row) return null;
  const fmt = kind === "close" ? formatPrice : formatVol;
  const histLabel = kind === "close" ? "History close" : "History volatility";
  const realizedLabel = kind === "close" ? "Realized close" : "Realized volatility";

  return (
    <div className="rounded-md border border-border bg-popover px-3 py-2 text-xs shadow-md">
      <p className="font-medium text-foreground">{formatDateTick(label)}</p>
      <div className="mt-1 space-y-0.5 text-muted-foreground">
        {row.history != null ? <p>{histLabel}: {fmt(row.history)}</p> : null}
        {row.forecast != null ? <p>Forecast: {fmt(row.forecast)}</p> : null}
        {row.forecastLower != null && row.forecastUpper != null ? (
          <p>
            {confidenceLabel}: {fmt(row.forecastLower)} – {fmt(row.forecastUpper)}
          </p>
        ) : null}
        {row.realized != null ? <p>{realizedLabel}: {fmt(row.realized)}</p> : null}
      </div>
    </div>
  );
}

export function ForecastChart({
  title,
  description,
  data,
  kind,
  splitTimestamp,
  includeRealized,
  hasConfidence,
  confidenceLabel,
  yDomain,
}: ForecastChartProps) {
  const palette = PALETTE[kind];
  const tickFmt = kind === "close" ? formatPrice : formatVol;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ left: 4, right: 16, top: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatDateTick}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              />
              <YAxis
                domain={yDomain}
                tickFormatter={tickFmt}
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                width={64}
              />
              <Tooltip
                content={(props) => {
                  const { active, payload, label } = props as unknown as {
                    active?: boolean;
                    payload?: Array<{ payload: ChartRow }>;
                    label?: string;
                  };
                  return (
                    <ChartTooltip
                      active={active}
                      payload={payload}
                      label={label}
                      kind={kind}
                      confidenceLabel={confidenceLabel}
                    />
                  );
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: "hsl(var(--muted-foreground))" }} />
              {splitTimestamp ? (
                <ReferenceLine x={splitTimestamp} stroke="hsl(var(--border))" strokeDasharray="4 4" />
              ) : null}
              <Area
                type="monotone"
                dataKey="history"
                name="History"
                stroke={palette.history}
                fill={palette.history}
                fillOpacity={0.18}
                strokeWidth={2}
                isAnimationActive={false}
              />
              {hasConfidence ? (
                <>
                  <Area
                    type="monotone"
                    dataKey="forecastLower"
                    stackId="band"
                    stroke="none"
                    fill="transparent"
                    legendType="none"
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="forecastBand"
                    name={confidenceLabel}
                    stackId="band"
                    stroke="none"
                    fill={palette.band}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecastUpper"
                    stroke={palette.bandEdge}
                    strokeDasharray="4 4"
                    strokeWidth={1}
                    dot={false}
                    legendType="none"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecastLower"
                    stroke={palette.bandEdge}
                    strokeDasharray="4 4"
                    strokeWidth={1}
                    dot={false}
                    legendType="none"
                    isAnimationActive={false}
                  />
                </>
              ) : null}
              <Line
                type="monotone"
                dataKey="forecast"
                name="Forecast"
                stroke={palette.forecast}
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
              {includeRealized ? (
                <Line
                  type="monotone"
                  dataKey="realized"
                  name="Realized"
                  stroke={palette.realized}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              ) : null}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
