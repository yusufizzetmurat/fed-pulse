import * as React from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { toStance } from "@/lib/analyze/format";
import type { HistoryEntry } from "@/lib/analyze/types";

export interface HistoryTimelineRow extends HistoryEntry {
  realized_regime?: string | null;
  forward_realized_vol_10d?: number | null;
}

interface HistoryTimelineChartProps {
  rows: HistoryTimelineRow[];
}

interface ChartDatum {
  date: string;
  ts: number;
  stance: number;
  regime: string | null;
  vol: number | null;
  symbol: string;
  id: string;
}

const REGIME_COLOR: Record<string, string> = {
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

function stanceToScore(label: string): number {
  const stance = toStance(label);
  if (stance === "hawkish") return 1;
  if (stance === "dovish") return -1;
  return 0;
}

function readForwardVol(row: HistoryTimelineRow): number | null {
  const direct = row.forward_realized_vol_10d;
  if (typeof direct === "number" && Number.isFinite(direct)) return direct;
  return null;
}

export function HistoryTimelineChart({ rows }: HistoryTimelineChartProps) {
  const data = React.useMemo<ChartDatum[]>(() => {
    return rows
      .map((row) => {
        const ts = new Date(row.document_date).getTime();
        if (!Number.isFinite(ts)) return null;
        // ``sentiment_score`` is the winning-class confidence (always
        // in [0, 1]) and cannot drive a signed axis. Prefer the signed
        // ``stance_score = P(hawk) - P(dove)`` surfaced by the backend
        // and fall back to the ternary categorical mapping for rows
        // that pre-date the multi-axis block.
        const stance = typeof row.stance_score === "number" && Number.isFinite(row.stance_score)
          ? Math.max(-1, Math.min(1, row.stance_score))
          : stanceToScore(row.stance);
        return {
          date: row.document_date,
          ts,
          stance,
          regime: row.argmax_regime ?? null,
          vol: readForwardVol(row),
          symbol: row.symbol,
          id: row.id,
        } satisfies ChartDatum;
      })
      .filter((value): value is ChartDatum => value != null)
      .sort((a, b) => a.ts - b.ts);
  }, [rows]);

  const hasVol = data.some((d) => d.vol != null);

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Stance over time</CardTitle>
          <CardDescription>
            Stance score per run, coloured by regime. Forward realised vol overlay on the
            right axis when available.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <EmptyState
            variant="inline"
            title="No history yet"
            description="Use the Workspace to analyze a statement."
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Stance over time</CardTitle>
        <CardDescription>
          Stance score per run, coloured by regime. {hasVol ? "Forward 10-day realised vol on the right axis." : "No forward vol data on these rows."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
            >
              <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="2 3" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
              />
              <YAxis
                yAxisId="stance"
                domain={[-1, 1]}
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
                label={{
                  value: "stance",
                  angle: -90,
                  position: "insideLeft",
                  style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                }}
              />
              {hasVol ? (
                <YAxis
                  yAxisId="vol"
                  orientation="right"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  stroke="hsl(var(--border))"
                  label={{
                    value: "forward vol",
                    angle: 90,
                    position: "insideRight",
                    style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                  }}
                />
              ) : null}
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                cursor={{ strokeDasharray: "2 3" }}
                formatter={(value, name, ctx) => {
                  const d = ctx?.payload as ChartDatum | undefined;
                  if (!d) return [String(value), String(name)];
                  if (name === "vol") {
                    return [
                      d.vol != null ? d.vol.toFixed(4) : "—",
                      "forward vol",
                    ];
                  }
                  return [
                    `${d.stance.toFixed(2)} · ${d.regime ?? "—"} · ${d.symbol}`,
                    "stance",
                  ];
                }}
                labelFormatter={(label) => String(label)}
              />
              <Line
                yAxisId="stance"
                type="monotone"
                dataKey="stance"
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
              />
              <Scatter
                yAxisId="stance"
                dataKey="stance"
                isAnimationActive={false}
                shape={(props: unknown) => {
                  const p = props as { cx?: number; cy?: number; payload?: ChartDatum };
                  if (p.cx == null || p.cy == null || !p.payload) return <></>;
                  const fill = REGIME_COLOR[p.payload.regime ?? ""] ?? "hsl(var(--muted-foreground))";
                  return <circle cx={p.cx} cy={p.cy} r={4} fill={fill} />;
                }}
              />
              {hasVol ? (
                <Line
                  yAxisId="vol"
                  type="monotone"
                  dataKey="vol"
                  stroke="hsl(var(--primary))"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ) : null}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: REGIME_COLOR.calm }} />
            calm
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: REGIME_COLOR.normal }} />
            normal
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: REGIME_COLOR.high }} />
            high
          </span>
          {hasVol ? (
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-4" style={{ background: "hsl(var(--primary))" }} />
              forward vol (right)
            </span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}
