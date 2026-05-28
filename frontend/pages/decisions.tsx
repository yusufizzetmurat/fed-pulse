import * as React from "react";
import Head from "next/head";
import { Gavel } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchNextFomcForecast, resolveApiBaseUrl } from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";
import type {
  NextFomcForecastResponse,
  NextFomcMeetingPrediction,
} from "@/lib/analyze/types";

const ORDINAL_LABELS: Record<string, string> = {
  cut_50: "Cut 50",
  cut_25: "Cut 25",
  hold: "Hold",
  hike_25: "Hike 25",
  hike_50: "Hike 50",
  hike_75: "Hike 75",
};

const ORDINAL_BPS: Record<string, number> = {
  cut_50: -50,
  cut_25: -25,
  hold: 0,
  hike_25: 25,
  hike_50: 50,
  hike_75: 75,
};

const PRIMARY_MODEL_PREFERENCE = ["ordinal_logit", "hist_gbt", "ois_baseline", "naive_carry"];

function pickPrimaryModel(modelNames: string[]): string {
  for (const candidate of PRIMARY_MODEL_PREFERENCE) {
    if (modelNames.includes(candidate)) return candidate;
  }
  return modelNames[0] ?? "";
}

function formatPercent(value: number | null | undefined, digits: number = 1): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value: number | null | undefined, digits: number = 3): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function ProbabilityBar({
  probabilities,
  ordinalClasses,
}: {
  probabilities: Record<string, number>;
  ordinalClasses: string[];
}) {
  // Compute the expected bp for the secondary line.
  const expectedBp = ordinalClasses.reduce((acc, cls) => {
    const prob = probabilities[cls] ?? 0;
    return acc + prob * (ORDINAL_BPS[cls] ?? 0);
  }, 0);
  return (
    <div className="space-y-2">
      <div className="flex h-7 w-full overflow-hidden rounded-md border border-border">
        {ordinalClasses.map((cls) => {
          const p = probabilities[cls] ?? 0;
          if (p <= 0) return null;
          const bp = ORDINAL_BPS[cls] ?? 0;
          const bg =
            bp < 0
              ? `rgba(56, 189, 248, ${0.2 + Math.abs(bp) / 90})`
              : bp > 0
              ? `rgba(244, 114, 182, ${0.2 + bp / 90})`
              : "rgba(148, 163, 184, 0.45)";
          return (
            <div
              key={cls}
              title={`${ORDINAL_LABELS[cls] ?? cls}: ${(p * 100).toFixed(1)}%`}
              style={{ width: `${p * 100}%`, backgroundColor: bg }}
            />
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        Expected move:{" "}
        <span className="font-mono">{expectedBp >= 0 ? "+" : ""}{expectedBp.toFixed(1)} bp</span>
      </p>
    </div>
  );
}

function HeadlineCard({
  data,
  primaryModel,
}: {
  data: NextFomcForecastResponse;
  primaryModel: string;
}) {
  const headline = data.headline;
  if (!headline) return null;
  const probabilities = headline.probabilities[primaryModel] ?? {};
  const predictedClass = headline.predicted_class[primaryModel] ?? "—";
  const baselineProbabilities = headline.probabilities["ois_baseline"] ?? null;
  const upcoming = data.upcoming_meeting;
  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle>Next meeting</CardTitle>
          {upcoming ? (
            <Badge variant="outline" className="font-mono text-xs">
              {upcoming.meeting_date}
              {upcoming.days_until != null ? ` · in ${upcoming.days_until}d` : ""}
            </Badge>
          ) : null}
        </div>
        <CardDescription>
          Model: <span className="font-mono">{primaryModel}</span>. Predicted:{" "}
          <span className="font-mono font-semibold">{ORDINAL_LABELS[predictedClass] ?? predictedClass}</span>
          {" · "}
          confidence{" "}
          <span className="font-mono">{formatPercent(probabilities[predictedClass])}</span>.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-1.5">
          <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {primaryModel}
          </h4>
          <ProbabilityBar probabilities={probabilities} ordinalClasses={data.ordinal_classes} />
        </div>
        {baselineProbabilities && primaryModel !== "ois_baseline" ? (
          <div className="space-y-1.5">
            <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              OIS-implied baseline
            </h4>
            <ProbabilityBar
              probabilities={baselineProbabilities}
              ordinalClasses={data.ordinal_classes}
            />
          </div>
        ) : null}
        <div className="grid grid-cols-3 gap-1.5 text-center sm:grid-cols-6">
          {data.ordinal_classes.map((cls) => (
            <div key={cls} className="rounded-md border border-border p-2">
              <p className="text-xs text-muted-foreground">{ORDINAL_LABELS[cls] ?? cls}</p>
              <p className="font-mono text-sm font-semibold">
                {formatPercent(probabilities[cls])}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function HistoryTable({
  history,
  primaryModel,
}: {
  history: NextFomcMeetingPrediction[];
  primaryModel: string;
}) {
  const recent = history.slice(-12).reverse();
  const resolved = recent.filter((entry) => entry.target_class != null);
  const hits = resolved.filter(
    (entry) => entry.predicted_class[primaryModel] === entry.target_class
  );
  const hitRate = resolved.length > 0 ? hits.length / resolved.length : null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Past 12 meetings</CardTitle>
        <CardDescription>
          {primaryModel} hit-rate over the last 12 meetings:{" "}
          <span className="font-mono">{formatPercent(hitRate)}</span>{" "}
          ({hits.length}/{resolved.length})
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        {recent.length === 0 ? (
          <p className="px-4 py-6 text-center text-sm text-muted-foreground">
            No forecast history yet. Train a checkpoint to populate this card.
          </p>
        ) : (
          <>
            {/* Mobile: stacked card-per-row to avoid horizontal scroll. */}
            <ul className="divide-y divide-border md:hidden">
              {recent.map((entry) => {
                const predicted = entry.predicted_class[primaryModel] ?? "—";
                const p = entry.probabilities[primaryModel]?.[predicted];
                const hit = entry.target_class != null && entry.target_class === predicted;
                return (
                  <li key={entry.target_event_date} className="space-y-1.5 px-4 py-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs">{entry.target_event_date}</span>
                      {entry.target_class == null ? (
                        <Badge variant="outline">pending</Badge>
                      ) : hit ? (
                        <Badge variant="hawkish">hit</Badge>
                      ) : (
                        <Badge variant="dovish">miss</Badge>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
                      <span className="text-muted-foreground">Predicted</span>
                      <span className="text-right">{ORDINAL_LABELS[predicted] ?? predicted}</span>
                      <span className="text-muted-foreground">P(predicted)</span>
                      <span className="text-right font-mono">{formatPercent(p)}</span>
                      <span className="text-muted-foreground">Realised</span>
                      <span className="text-right">
                        {entry.target_class ? ORDINAL_LABELS[entry.target_class] ?? entry.target_class : "—"}
                      </span>
                    </div>
                  </li>
                );
              })}
            </ul>
            <div className="hidden md:block">
              <table className="w-full text-sm">
                <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left">Meeting</th>
                    <th className="px-4 py-2 text-left">Realised</th>
                    <th className="px-4 py-2 text-left">Predicted</th>
                    <th className="px-4 py-2 text-right">P(predicted)</th>
                    <th className="px-4 py-2 text-center">Hit</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((entry) => {
                    const predicted = entry.predicted_class[primaryModel] ?? "—";
                    const p = entry.probabilities[primaryModel]?.[predicted];
                    const hit = entry.target_class != null && entry.target_class === predicted;
                    return (
                      <tr key={entry.target_event_date} className="border-b border-border last:border-0">
                        <td className="px-4 py-2 font-mono text-xs">{entry.target_event_date}</td>
                        <td className="px-4 py-2">
                          {entry.target_class ? ORDINAL_LABELS[entry.target_class] ?? entry.target_class : "—"}
                        </td>
                        <td className="px-4 py-2">{ORDINAL_LABELS[predicted] ?? predicted}</td>
                        <td className="px-4 py-2 text-right font-mono">{formatPercent(p)}</td>
                        <td className="px-4 py-2 text-center">
                          {entry.target_class == null ? (
                            <Badge variant="outline">pending</Badge>
                          ) : hit ? (
                            <Badge variant="hawkish">hit</Badge>
                          ) : (
                            <Badge variant="dovish">miss</Badge>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function MetricsTable({
  metrics,
}: {
  metrics: NextFomcForecastResponse["metrics_full_window"];
}) {
  const rows = Object.entries(metrics);
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Walk-forward metrics</CardTitle>
        </CardHeader>
        <CardContent className="py-6 text-center text-sm text-muted-foreground">
          No metrics available. Train a checkpoint to populate this card.
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Walk-forward metrics</CardTitle>
        <CardDescription>Leave-one-meeting-out walk-forward CV across the full window.</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollableTable>
          <table className="w-full min-w-[36rem] text-sm">
            <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-2 text-left">Model</th>
                <th className="px-4 py-2 text-right">n</th>
                <th className="px-4 py-2 text-right">Brier</th>
                <th className="px-4 py-2 text-right">Log-loss</th>
                <th className="px-4 py-2 text-right">Top-1 acc</th>
                <th className="px-4 py-2 text-right">Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([model, m]) => (
                <tr key={model} className="border-b border-border last:border-0">
                  <td className="px-4 py-2 font-mono">{model}</td>
                  <td className="px-4 py-2 text-right font-mono text-muted-foreground">{m.n}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(m.brier)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(m.log_loss)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatPercent(m.top1_accuracy)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(m.macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollableTable>
      </CardContent>
    </Card>
  );
}

function AttributionTable({
  rows,
}: {
  rows: NextFomcForecastResponse["feature_attribution"];
}) {
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Feature-family attribution</CardTitle>
        </CardHeader>
        <CardContent className="py-6 text-center text-sm text-muted-foreground">
          No attribution rows available. Train a checkpoint to populate this card.
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature-family attribution</CardTitle>
        <CardDescription>
          Walk-forward metrics on the ordinal_logit model with each feature subset.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollableTable>
          <table className="w-full min-w-[44rem] text-sm">
            <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-2 text-left">Subset</th>
                <th className="px-4 py-2 text-left">Families</th>
                <th className="px-4 py-2 text-right">#feat</th>
                <th className="px-4 py-2 text-right">Brier</th>
                <th className="px-4 py-2 text-right">Log-loss</th>
                <th className="px-4 py-2 text-right">Top-1 acc</th>
                <th className="px-4 py-2 text-right">Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.subset} className="border-b border-border last:border-0">
                  <td className="px-4 py-2 font-mono">{row.subset}</td>
                  <td className="px-4 py-2 text-xs text-muted-foreground">
                    {row.families.join(", ") || "—"}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                    {row.n_features ?? "—"}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(row.brier)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(row.log_loss)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatPercent(row.top1_accuracy)}</td>
                  <td className="px-4 py-2 text-right font-mono">{formatNumber(row.macro_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollableTable>
      </CardContent>
    </Card>
  );
}

/**
 * Horizontal scroll wrapper used for the wide metrics tables on
 * `/decisions`. A right-edge gradient hints at the off-screen columns
 * on narrow viewports; the gradient hides itself once the table fits
 * inside the container, so it never paints over a fully-visible table
 * on desktop. The container is keyboard-focusable so users can scroll
 * with arrow keys.
 */
function ScrollableTable({ children }: { children: React.ReactNode }) {
  const scrollerRef = React.useRef<HTMLDivElement | null>(null);
  const [showFade, setShowFade] = React.useState(false);

  React.useEffect(() => {
    const node = scrollerRef.current;
    if (!node) return;
    const update = () => {
      const overflow = node.scrollWidth - node.clientWidth;
      const room = overflow - node.scrollLeft;
      setShowFade(overflow > 4 && room > 4);
    };
    update();
    node.addEventListener("scroll", update, { passive: true });
    // ResizeObserver is missing in some test environments (jsdom). Fall
    // back to a window-resize listener so the fade still re-evaluates
    // on rotate / viewport change without crashing the page.
    let detach: () => void;
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(update);
      observer.observe(node);
      detach = () => observer.disconnect();
    } else {
      window.addEventListener("resize", update);
      detach = () => window.removeEventListener("resize", update);
    }
    return () => {
      node.removeEventListener("scroll", update);
      detach();
    };
  }, []);

  return (
    <div className="relative">
      <div
        ref={scrollerRef}
        role="region"
        aria-label="Scrollable table"
        tabIndex={0}
        className="overflow-x-auto focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {children}
      </div>
      {showFade ? (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-background to-transparent"
        />
      ) : null}
    </div>
  );
}

export default function DecisionsPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<NextFomcForecastResponse | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    fetchNextFomcForecast(apiBaseUrl)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) toast.error(errorMessage(err, "Failed to load decision forecast."));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const primaryModel = data ? pickPrimaryModel(data.model_names) : "";

  return (
    <>
      <Head>
        <title>Decisions — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="space-y-2">
            <h1 className="flex items-center gap-2 text-3xl font-semibold tracking-tight">
              <Gavel className="h-7 w-7 text-primary" />
              Decisions
            </h1>
            <p className="max-w-2xl text-muted-foreground">
              Next-FOMC rate-decision forecast as an ordinal class with the OIS-implied baseline alongside.
              Reads <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">data/artifacts/next_fomc/</code>.
            </p>
          </div>

          {loading ? (
            <div className="space-y-3">
              <Skeleton className="h-48 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          ) : data && data.available ? (
            <div className="space-y-6">
              <HeadlineCard data={data} primaryModel={primaryModel} />
              <Tabs defaultValue="history" className="w-full">
                <TabsList className="grid w-full max-w-md grid-cols-3">
                  <TabsTrigger value="history">History</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="attribution">Attribution</TabsTrigger>
                </TabsList>
                <TabsContent value="history">
                  <HistoryTable history={data.history} primaryModel={primaryModel} />
                </TabsContent>
                <TabsContent value="metrics" className="space-y-3">
                  <MetricsTable metrics={data.metrics_full_window} />
                  {Object.keys(data.metrics_ex_pandemic).length > 0 ? (
                    <>
                      <h3 className="text-sm font-medium text-muted-foreground">
                        Pandemic-excluded window
                      </h3>
                      <MetricsTable metrics={data.metrics_ex_pandemic} />
                    </>
                  ) : null}
                </TabsContent>
                <TabsContent value="attribution">
                  <AttributionTable rows={data.feature_attribution} />
                </TabsContent>
              </Tabs>
            </div>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>No forecast available.</CardTitle>
                <CardDescription>
                  Train a checkpoint to populate this page. The forecaster reads from{" "}
                  <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                    data/artifacts/next_fomc/
                  </code>
                  .
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>
                  Run{" "}
                  <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                    make next-fomc TRAINING_PACKAGE_ID=&lt;id&gt;
                  </code>{" "}
                  to publish a forecast.
                </p>
                {data?.upcoming_meeting ? (
                  <p>
                    Next scheduled meeting:{" "}
                    <span className="font-mono">{data.upcoming_meeting.meeting_date}</span>
                    {data.upcoming_meeting.days_until != null
                      ? ` (in ${data.upcoming_meeting.days_until} days)`
                      : ""}
                    .
                  </p>
                ) : null}
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
