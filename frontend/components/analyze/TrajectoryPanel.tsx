import * as React from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { AlertTriangle, Compass, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export type TrajectoryStance = "hawkish" | "dovish" | "neutral";

export interface TrajectoryMarker {
  event_date: string;
  axis_stance: string | null;
  embedding_2d: [number, number];
}

export interface TrajectoryProjection {
  predicted_stance: string;
  class_probs: Record<string, number>;
  confidence_band?: string[] | null;
  conformal_alpha?: number | null;
}

export interface TrajectoryResponse {
  available: boolean;
  history: TrajectoryMarker[];
  projected_next: TrajectoryProjection | null;
  architecture?: string | null;
  encoder_alias?: string;
  history_length?: number;
  train_end?: string | null;
  as_of_date?: string;
  // #332 lift verdict. True iff the Transformer beats the strongest
  // naive baseline (previous_stance / rolling_majority / 1×16 LSTM)
  // by ≥ 5pp directional accuracy. Pre-#332 bundles default to false.
  lift_vs_baseline?: boolean;
  delta_dir_acc?: number | null;
  baseline_used?: string | null;
  // Non-fatal advisory from the backend — populated when the requested
  // as_of_date sits beyond the bundle's train_end so the caller can
  // flag that the projection extrapolates past the fold boundary.
  warning?: string | null;
}

interface TrajectoryPanelProps {
  apiBaseUrl: string;
  asOfDate: string; // ISO YYYY-MM-DD
  historyLength?: number;
}

const STANCE_COLOR: Record<TrajectoryStance, string> = {
  hawkish: "hsl(var(--down))", // red leaning — tighter policy
  dovish: "hsl(var(--up))", // green leaning — looser policy
  neutral: "hsl(var(--neutral))",
};

const STANCE_LABEL: Record<TrajectoryStance, string> = {
  hawkish: "Hawkish",
  dovish: "Dovish",
  neutral: "Neutral",
};

function toStance(value: string | null | undefined): TrajectoryStance {
  const v = (value ?? "").toLowerCase();
  if (v === "hawkish" || v === "dovish" || v === "neutral") return v;
  return "neutral";
}

interface ScatterRow {
  x: number;
  y: number;
  stance: TrajectoryStance;
  event_date: string;
  recency: number;
}

function recencyOpacity(row: ScatterRow, total: number): number {
  // More recent meetings render brighter; older meetings fade. Total
  // controls the slope so a longer history does not wash out the most
  // recent points. Always at least 0.35 so dots stay legible.
  if (total <= 1) return 1;
  const t = row.recency / Math.max(1, total - 1);
  return 0.35 + 0.65 * t;
}

const TOOLTIP_STYLE: React.CSSProperties = {
  background: "hsl(var(--popover))",
  color: "hsl(var(--popover-foreground))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 12,
};

export function TrajectoryPanel({
  apiBaseUrl,
  asOfDate,
  historyLength = 12,
}: TrajectoryPanelProps) {
  const [data, setData] = React.useState<TrajectoryResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Stabilise the request body across renders so the effect below
  // depends on a single object identity, not three primitives that
  // recompose on every parent re-render.
  const requestBody = React.useMemo(
    () => JSON.stringify({ as_of_date: asOfDate, history_length: historyLength }),
    [asOfDate, historyLength],
  );

  React.useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/analyze/trajectory`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: requestBody,
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = (await response.json()) as TrajectoryResponse;
        if (!controller.signal.aborted) {
          setData(payload);
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message || "Failed to load trajectory");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [apiBaseUrl, requestBody]);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <Compass className="h-3.5 w-3.5" />
            Hawkish / Dovish trajectory
          </CardDescription>
          <CardTitle>Loading…</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <Compass className="h-3.5 w-3.5" />
            Hawkish / Dovish trajectory
          </CardDescription>
          <CardTitle>Unavailable</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            variant="inline"
            title="Trajectory model isn't loaded."
            description={
              error
                ? `Check the Settings page for the active checkpoint. Backend reported: ${error}`
                : "Check the Settings page for the active checkpoint."
            }
          />
        </CardContent>
      </Card>
    );
  }

  if (!data.available || data.history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <Compass className="h-3.5 w-3.5" />
            Hawkish / Dovish trajectory
          </CardDescription>
          <CardTitle>Not trained on this host</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            variant="inline"
            title="Trajectory model isn't loaded."
            description={
              <span>
                Train the trajectory model against the corpus and load it from the Settings
                page to populate this panel.
              </span>
            }
          />
        </CardContent>
      </Card>
    );
  }

  const total = data.history.length;
  const rows: ScatterRow[] = data.history.map((m, idx) => ({
    x: m.embedding_2d[0],
    y: m.embedding_2d[1],
    stance: toStance(m.axis_stance),
    event_date: m.event_date,
    recency: idx,
  }));

  const projection = data.projected_next;
  const projectedStance = projection ? toStance(projection.predicted_stance) : "neutral";
  // Project the next anchor as the simple delta extension of the last
  // two real markers — visual heuristic only; the model's stance
  // probability stays the load-bearing claim.
  const projectedAnchor = (() => {
    if (rows.length === 0) return { x: 0, y: 0 };
    if (rows.length === 1) {
      return { x: rows[0].x + 0.1, y: rows[0].y };
    }
    const last = rows[rows.length - 1];
    const prev = rows[rows.length - 2];
    return { x: last.x + (last.x - prev.x), y: last.y + (last.y - prev.y) };
  })();

  const projectionPoint = projection
    ? [
        {
          x: projectedAnchor.x,
          y: projectedAnchor.y,
          stance: projectedStance,
          event_date: "projected next",
          recency: total,
        },
      ]
    : [];

  // One scatter per stance so we can colour the dots per stance without
  // a custom shape function (Recharts colour-by-row gets awkward).
  const byStance: Record<TrajectoryStance, ScatterRow[]> = {
    hawkish: rows.filter((r) => r.stance === "hawkish"),
    dovish: rows.filter((r) => r.stance === "dovish"),
    neutral: rows.filter((r) => r.stance === "neutral"),
  };

  const probEntries = projection
    ? (["hawkish", "dovish", "neutral"] as TrajectoryStance[]).map((s) => ({
        stance: s,
        prob: projection.class_probs[s] ?? 0,
      }))
    : [];

  const liftEstablished = data.lift_vs_baseline === true;
  const hasLiftSignal =
    data.lift_vs_baseline != null || data.delta_dir_acc != null || data.baseline_used != null;
  return (
    <Card className={cn(!liftEstablished && hasLiftSignal ? "opacity-90" : undefined)}>
      <CardHeader>
        <CardDescription className="flex items-center gap-1.5">
          <Compass className="h-3.5 w-3.5" />
          Hawkish / Dovish trajectory
        </CardDescription>
        <CardTitle className="flex items-center justify-between gap-2">
          <span>Last {total} meetings · {data.architecture ?? "lstm"}</span>
          <div className="flex flex-wrap items-center gap-2">
            {data.train_end ? (
              <Badge variant="outline" className="numeric text-[10px]">
                training ended · {data.train_end}
              </Badge>
            ) : null}
            {hasLiftSignal && !liftEstablished ? (
              <Badge
                variant="outline"
                className="text-muted-foreground text-[10px] uppercase tracking-wide"
                title={
                  data.delta_dir_acc != null && data.baseline_used
                    ? `Directional accuracy vs ${data.baseline_used}: ${(data.delta_dir_acc * 100).toFixed(1)}pp; needs at least 5pp to claim a lift.`
                    : "Directional accuracy matches the simple baseline within the lift threshold."
                }
              >
                matches simple-baseline accuracy
              </Badge>
            ) : null}
            {liftEstablished ? (
              <Badge variant="dovish" className="text-[10px] uppercase tracking-wide">
                +{((data.delta_dir_acc ?? 0) * 100).toFixed(1)}pp vs baseline
              </Badge>
            ) : null}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {data.warning ? (
          <div
            role="alert"
            data-testid="trajectory-warning"
            className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200"
          >
            <AlertTriangle className="mt-[1px] h-3.5 w-3.5 shrink-0" />
            <span>{data.warning}</span>
          </div>
        ) : null}
        <div className="h-72 min-h-[270px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 16, bottom: 12, left: 12 }}>
              <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="2 3" />
              <XAxis
                type="number"
                dataKey="x"
                name="component 1"
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
                tickFormatter={(v) => Number(v).toFixed(1)}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="component 2"
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                stroke="hsl(var(--border))"
                tickFormatter={(v) => Number(v).toFixed(1)}
              />
              <ZAxis type="number" range={[60, 60]} />
              <Tooltip
                cursor={{ strokeDasharray: "2 3" }}
                contentStyle={TOOLTIP_STYLE}
                formatter={(value, name, ctx) => {
                  const r = ctx?.payload as ScatterRow | undefined;
                  if (!r) return ["—", String(name)];
                  return [
                    `${r.event_date} · ${STANCE_LABEL[r.stance]}`,
                    "meeting",
                  ];
                }}
                labelFormatter={() => ""}
              />
              {(Object.keys(byStance) as TrajectoryStance[]).map((stance) => (
                <Scatter
                  key={stance}
                  name={STANCE_LABEL[stance]}
                  data={byStance[stance].map((row) => ({
                    ...row,
                    fillOpacity: recencyOpacity(row, total),
                  }))}
                  fill={STANCE_COLOR[stance]}
                  isAnimationActive={false}
                />
              ))}
              {projection ? (
                <Scatter
                  name="Projected next"
                  data={projectionPoint.map((row) => ({ ...row, fillOpacity: 0.85 }))}
                  fill={STANCE_COLOR[projectedStance]}
                  shape="star"
                  isAnimationActive={false}
                />
              ) : null}
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {projection ? (
          <div className="rounded-lg border border-border bg-muted/30 p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">
                  Projected next meeting: {STANCE_LABEL[projectedStance]}
                </span>
              </div>
              {projection.confidence_band && projection.confidence_band.length > 0 ? (
                <Badge variant="outline" className="text-[10px]">
                  prediction set · {projection.confidence_band.join(", ")}
                </Badge>
              ) : null}
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              {probEntries.map(({ stance, prob }) => (
                <div key={stance} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="capitalize">{STANCE_LABEL[stance]}</span>
                    <span className="numeric font-medium text-foreground">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={Math.round(prob * 100)} />
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Each dot is a past FOMC meeting, placed by a 2-D summary of the meeting text.
          Brighter dots are more recent; the star is the projected next meeting. The
          prediction set is the calibrated range of likely stances.
        </p>
      </CardContent>
    </Card>
  );
}
