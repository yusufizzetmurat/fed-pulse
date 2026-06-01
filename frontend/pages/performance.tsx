import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { Target, X } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";

import { ConfusionMatrix } from "@/components/analyze/ConfusionMatrix";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DataTable,
  type DataTableColumn,
} from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
import { KpiTile } from "@/components/ui/kpi-tile";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchClassificationBreakdown,
  fetchHistory,
  fetchHistoryRealizedBatch,
  fetchHistoryRun,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";
import {
  REGIME_CLASSES,
  aggregateRegimePerformance,
  buildRunRegimeRecord,
  proportionHalfWidth,
  type RunRegimeRecord,
} from "@/lib/analyze/performance";
import type {
  ClassificationBreakdownResponse,
  HistoryEntry,
  HistoryRealizedBatchResponse,
} from "@/lib/analyze/types";

const HISTORY_LIMIT = 100;

function formatPercent(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatScore(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

function regimeVariant(label: string | null | undefined): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "calm") return "dovish";
  if (label === "high") return "hawkish";
  if (label === "normal") return "neutral";
  return "outline";
}

function formatScoreWithCi(value: number | null, halfWidth: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (halfWidth == null) return value.toFixed(3);
  return `${value.toFixed(3)} ±${halfWidth.toFixed(3)}`;
}

const PERF_TOOLTIP_STYLE: React.CSSProperties = {
  background: "hsl(var(--popover))",
  color: "hsl(var(--popover-foreground))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 12,
};

export default function PerformancePage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [rows, setRows] = React.useState<RunRegimeRecord[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [totalRuns, setTotalRuns] = React.useState(0);
  const [breakdown, setBreakdown] = React.useState<ClassificationBreakdownResponse | null>(null);
  // Drill-down: when set, the per-asset chart, per-asset table, and
  // run-level table only show runs where this class was either
  // predicted as argmax or appeared as realised.
  const [classFilter, setClassFilter] = React.useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setLoading(true);
    (async () => {
      try {
        const [list, breakdownResponse] = await Promise.all([
          fetchHistory(apiBaseUrl, { limit: HISTORY_LIMIT }, signal),
          fetchClassificationBreakdown(apiBaseUrl, signal).catch(() => null),
        ]);
        if (signal.aborted) return;
        setTotalRuns(list.total);
        setBreakdown(breakdownResponse);
        // Realized labels fetched in one batched round trip; per-row
        // history-detail (payload) still needs a fan-out because the
        // persisted predicted_set list isn't carried on the summary.
        const ids = list.items.map((entry) => entry.id);
        const emptyBatch: HistoryRealizedBatchResponse = { items: {}, missing: ids };
        const [batch, detailResults] = await Promise.all([
          fetchHistoryRealizedBatch(apiBaseUrl, ids, signal).catch(() => emptyBatch),
          Promise.all(
            list.items.map((entry) =>
              fetchHistoryRun(apiBaseUrl, entry.id, signal).catch(() => null),
            ),
          ),
        ]);
        if (signal.aborted) return;
        const records = list.items.map((entry: HistoryEntry, idx) => {
          const realized = batch.items[entry.id] ?? null;
          const detail = detailResults[idx];
          return buildRunRegimeRecord({
            entry,
            realized,
            payload: (detail?.payload || null) as Record<string, unknown> | null,
          });
        });
        setRows(records);
      } catch (err) {
        if (!signal.aborted) {
          toast.error(errorMessage(err, "Failed to load performance data."));
        }
      } finally {
        if (!signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [apiBaseUrl]);

  const aggregate = React.useMemo(() => aggregateRegimePerformance(rows), [rows]);
  // Subset of rows that match the active class drill-down (or all rows
  // when no filter is set). The per-asset chart, per-asset table, and
  // run-level table all read from the filtered rows / aggregate.
  const filteredRows = React.useMemo(() => {
    if (!classFilter) return rows;
    return rows.filter(
      (row) => row.argmax === classFilter || row.realized === classFilter,
    );
  }, [rows, classFilter]);
  const filteredAggregate = React.useMemo(
    () => aggregateRegimePerformance(filteredRows),
    [filteredRows],
  );
  const breakdownAvailable = breakdown?.available === true;
  const headlineMacroF1 = breakdownAvailable
    ? breakdown?.macro_f1 ?? null
    : aggregate.macroF1;
  const headlineMacroRocAuc = breakdownAvailable ? breakdown?.macro_roc_auc ?? null : null;

  const symbolColumns: DataTableColumn<(typeof aggregate.bySymbol)[number]>[] = React.useMemo(
    () => [
      {
        key: "symbol",
        header: "Symbol",
        sortable: true,
        sortValue: (row) => row.symbol,
        render: (row) => row.symbol,
      },
      {
        key: "resolved",
        header: "Resolved",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.resolved,
        render: (row) => row.resolved,
      },
      {
        key: "argmaxAccuracy",
        header: "Top-pick accuracy",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.argmaxAccuracy ?? -1,
        render: (row) => formatPercent(row.argmaxAccuracy),
      },
      {
        key: "empiricalCoverage",
        header: "Actual coverage",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.empiricalCoverage ?? -1,
        render: (row) => formatPercent(row.empiricalCoverage),
      },
    ],
    [],
  );

  const runColumns: DataTableColumn<RunRegimeRecord>[] = React.useMemo(
    () => [
      {
        key: "document_date",
        header: "Date",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.document_date,
        render: (row) => row.document_date,
      },
      {
        key: "symbol",
        header: "Symbol",
        sortable: true,
        sortValue: (row) => row.symbol,
        render: (row) => row.symbol,
      },
      {
        key: "horizon",
        header: "H",
        render: (row) => <span className="text-muted-foreground">{row.horizon}</span>,
      },
      {
        key: "argmax",
        header: "Top pick",
        render: (row) =>
          row.argmax ? (
            <Badge variant={regimeVariant(row.argmax)} className="text-[10px] capitalize">
              {row.argmax}
            </Badge>
          ) : (
            <span className="text-muted-foreground">—</span>
          ),
      },
      {
        key: "argmaxProbability",
        header: "Probability",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.argmaxProbability ?? -1,
        render: (row) => formatPercent(row.argmaxProbability),
      },
      {
        key: "realized",
        header: "Actual",
        render: (row) =>
          row.realized ? (
            <Badge variant={regimeVariant(row.realized)} className="text-[10px] capitalize">
              {row.realized}
            </Badge>
          ) : (
            <span className="text-muted-foreground">pending</span>
          ),
      },
      {
        key: "argmaxHit",
        header: "Top-pick result",
        align: "center",
        render: (row) => {
          if (!row.argmax || !row.realized) {
            return <span className="text-muted-foreground">—</span>;
          }
          const hit = row.argmax === row.realized;
          return (
            <Badge variant={hit ? "dovish" : "hawkish"} className="text-[10px]">
              {hit ? "hit" : "miss"}
            </Badge>
          );
        },
      },
      {
        key: "setHit",
        header: "Prediction set",
        align: "center",
        render: (row) => {
          if (row.setHit == null) return <span className="text-muted-foreground">—</span>;
          return (
            <Badge variant={row.setHit ? "dovish" : "hawkish"} className="text-[10px]">
              {row.setHit ? "covered" : "missed"}
            </Badge>
          );
        },
      },
    ],
    [],
  );

  return (
    <>
      <Head>
        <title>Performance — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-5 py-6 focus:outline-none">
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">Performance</h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              How accurately the active model predicts each market regime. Per-class metrics
              show precision and recall by regime. The confusion matrix shows what the model
              predicted vs what actually happened. The per-asset breakdown shows accuracy by
              symbol.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <KpiTile
              label="Runs scanned"
              value={<span className="numeric">{rows.length}</span>}
              caption={totalRuns > rows.length ? `of ${totalRuns} total` : "all runs"}
              icon={<Target className="h-3.5 w-3.5" />}
            />
            <KpiTile
              label="Resolved"
              value={<span className="numeric">{aggregate.resolved}</span>}
              caption={
                aggregate.resolved < rows.length
                  ? `${rows.length - aggregate.resolved} still pending`
                  : "all in-window"
              }
              icon={<Target className="h-3.5 w-3.5" />}
            />
            <KpiTile
              label="Top-pick accuracy"
              value={<span className="numeric">{formatPercent(aggregate.argmaxAccuracy)}</span>}
              caption={`chance baseline ${formatPercent(1 / REGIME_CLASSES.length)}`}
              tone={
                aggregate.argmaxAccuracy != null && aggregate.argmaxAccuracy >= 1 / REGIME_CLASSES.length
                  ? "up"
                  : aggregate.argmaxAccuracy == null
                  ? "neutral"
                  : "down"
              }
            />
            <KpiTile
              label="Overall F1 score"
              value={<span className="numeric">{formatScore(headlineMacroF1)}</span>}
              caption={
                breakdownAvailable
                  ? "from training evaluation"
                  : aggregate.macroF1 != null
                  ? "computed from your history"
                  : aggregate.resolved > 0
                  ? "needs resolved runs in every regime"
                  : "needs resolved runs"
              }
              tone={headlineMacroF1 != null && headlineMacroF1 >= 0.4 ? "up" : "neutral"}
            />
            <KpiTile
              label="Actual coverage"
              value={<span className="numeric">{formatPercent(aggregate.empiricalCoverage)}</span>}
              caption="how often the realised regime fell inside the prediction set"
            />
            {headlineMacroRocAuc != null ? (
              <KpiTile
                label="Overall ROC-AUC"
                value={<span className="numeric">{formatScore(headlineMacroRocAuc)}</span>}
                caption="one-vs-rest, from training evaluation"
                tone={headlineMacroRocAuc >= 0.6 ? "up" : "neutral"}
              />
            ) : null}
          </div>

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : rows.length === 0 ? (
            <EmptyState
              title="No runs in history."
              description="Use the Workspace to analyze a statement and populate this view."
              action={
                <Button asChild size="sm" variant="outline">
                  <Link href="/">Open Workspace</Link>
                </Button>
              }
            />
          ) : (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Per-class metrics</CardTitle>
                  <CardDescription>
                    {breakdownAvailable
                      ? "From training-time evaluation — precision, recall, F1, ROC-AUC, PR-AUC."
                      : "Computed from your resolved history runs. Will switch to the training evaluation when one is published."}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-sm">
                    <thead className="border-b border-border bg-muted/30 text-[10px] uppercase tracking-wide text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left">Class</th>
                        <th className="px-4 py-2 text-right">Support</th>
                        <th className="px-4 py-2 text-right">Precision</th>
                        <th className="px-4 py-2 text-right">Recall</th>
                        <th className="px-4 py-2 text-right">F1</th>
                        {breakdownAvailable ? (
                          <>
                            <th className="px-4 py-2 text-right">ROC-AUC</th>
                            <th className="px-4 py-2 text-right">PR-AUC</th>
                          </>
                        ) : null}
                      </tr>
                    </thead>
                    <tbody>
                      {breakdownAvailable && breakdown?.per_class
                        ? breakdown.per_class.map((row, idx) => {
                            const label = breakdown.class_labels?.[row.class_id] ?? REGIME_CLASSES[row.class_id] ?? `class ${row.class_id}`;
                            const precisionHw = proportionHalfWidth(row.precision, row.support);
                            const recallHw = proportionHalfWidth(row.recall, row.support);
                            // F1 has no closed-form Wald SE; use the
                            // larger of precision / recall half-widths
                            // as a conservative band-width approximation.
                            const f1Hw =
                              precisionHw != null && recallHw != null
                                ? Math.max(precisionHw, recallHw)
                                : null;
                            const isActive = classFilter === label;
                            return (
                              <tr
                                key={`${row.class_id}-${idx}`}
                                className={`border-b border-border last:border-0 cursor-pointer hover:bg-accent/40 ${isActive ? "bg-accent/30" : ""}`}
                                onClick={() => setClassFilter(isActive ? null : label)}
                              >
                                <td className="px-4 py-2 capitalize">{label}</td>
                                <td className="numeric px-4 py-2 text-right">{row.support}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(row.precision, precisionHw)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(row.recall, recallHw)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(row.f1, f1Hw)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.roc_auc ?? null)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.pr_auc ?? null)}</td>
                              </tr>
                            );
                          })
                        : aggregate.perClass.map((entry) => {
                            const precisionHw = proportionHalfWidth(entry.precision, entry.support);
                            const recallHw = proportionHalfWidth(entry.recall, entry.support);
                            const f1Hw =
                              precisionHw != null && recallHw != null
                                ? Math.max(precisionHw, recallHw)
                                : null;
                            const isActive = classFilter === entry.klass;
                            return (
                              <tr
                                key={entry.klass}
                                className={`border-b border-border last:border-0 cursor-pointer hover:bg-accent/40 ${isActive ? "bg-accent/30" : ""}`}
                                onClick={() => setClassFilter(isActive ? null : entry.klass)}
                              >
                                <td className="px-4 py-2 capitalize">{entry.klass}</td>
                                <td className="numeric px-4 py-2 text-right">{entry.support}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(entry.precision, precisionHw)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(entry.recall, recallHw)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScoreWithCi(entry.f1, f1Hw)}</td>
                              </tr>
                            );
                          })}
                    </tbody>
                  </table>
                </CardContent>
                {breakdownAvailable && breakdown?.source ? (
                  <div className="border-t border-border bg-muted/20 px-4 py-2 text-[10px] text-muted-foreground">
                    Source: tier 7 (market + rich + NLP + xBank + LLM), 5 seeds x 4 folds, pooled walk-forward
                    {breakdown.source.training_package_id
                      ? ` · ${breakdown.source.training_package_id}`
                      : ""}
                    {" · "}
                    {new Date(breakdown.source.modified_at).toLocaleString(undefined, {
                      dateStyle: "short",
                      timeStyle: "short",
                    })}
                  </div>
                ) : null}
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Confusion matrix</CardTitle>
                  <CardDescription>
                    {breakdownAvailable
                      ? "From training-time evaluation — rows are the actual class, columns are the predicted top pick."
                      : "Computed from your resolved runs — rows are the realised regime, columns are the predicted top pick."}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {breakdownAvailable && breakdown?.confusion_matrix
                    ? (() => {
                        const labels = breakdown.class_labels?.length
                          ? breakdown.class_labels
                          : REGIME_CLASSES;
                        const matrixRows = breakdown.confusion_matrix.map((counts, idx) => ({
                          truth: labels[idx] ?? `class ${idx}`,
                          counts: Object.fromEntries(
                            counts.map((value, j) => [labels[j] ?? `class ${j}`, value]),
                          ),
                          total: counts.reduce((acc, n) => acc + n, 0),
                        }));
                        return (
                          <ConfusionMatrix
                            rows={matrixRows}
                            classes={labels}
                            onClassClick={(klass) =>
                              setClassFilter((prev) => (prev === klass ? null : klass))
                            }
                            activeClass={classFilter}
                          />
                        );
                      })()
                    : (
                      <ConfusionMatrix
                        rows={aggregate.confusion}
                        classes={REGIME_CLASSES}
                        onClassClick={(klass) =>
                          setClassFilter((prev) => (prev === klass ? null : klass))
                        }
                        activeClass={classFilter}
                      />
                    )}
                  {classFilter ? (
                    <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                      <span>
                        Filtered to runs where <span className="font-mono capitalize">{classFilter}</span> was predicted or actual.
                      </span>
                      <button
                        type="button"
                        onClick={() => setClassFilter(null)}
                        className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-0.5 hover:bg-accent/40"
                      >
                        <X className="h-3 w-3" aria-hidden="true" />
                        Clear
                      </button>
                    </div>
                  ) : null}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Per-asset breakdown</CardTitle>
                  <CardDescription>
                    Top-pick accuracy and actual coverage for every symbol with at least one
                    resolved run.{classFilter ? ` Filtered to ${classFilter}.` : ""}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4 p-0">
                  {(() => {
                    const chartRows = filteredAggregate.bySymbol
                      .filter((row) => row.argmaxAccuracy != null)
                      .map((row) => ({
                        symbol: row.symbol,
                        accuracy: row.argmaxAccuracy ?? 0,
                        resolved: row.resolved,
                      }))
                      .sort((a, b) => b.accuracy - a.accuracy);
                    if (chartRows.length === 0) return null;
                    return (
                      <div className="px-4 pt-4">
                        <div className="h-56 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={chartRows}
                              margin={{ top: 8, right: 16, bottom: 24, left: 0 }}
                            >
                              <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="2 3" />
                              <XAxis
                                dataKey="symbol"
                                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                                interval={0}
                                angle={-30}
                                textAnchor="end"
                              />
                              <YAxis
                                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                                domain={[0, 1]}
                                tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
                              />
                              <Tooltip
                                cursor={{ fill: "hsl(var(--muted) / 0.4)" }}
                                contentStyle={PERF_TOOLTIP_STYLE}
                                formatter={(value, _name, ctx) => {
                                  const d = ctx?.payload as { symbol: string; accuracy: number; resolved: number } | undefined;
                                  if (!d) return [String(value), "accuracy"];
                                  return [
                                    `${(d.accuracy * 100).toFixed(1)}% on ${d.resolved} run${d.resolved === 1 ? "" : "s"}`,
                                    "top-pick accuracy",
                                  ];
                                }}
                              />
                              <Bar dataKey="accuracy" isAnimationActive={false}>
                                {chartRows.map((d) => (
                                  <Cell
                                    key={d.symbol}
                                    fill={d.accuracy >= 1 / REGIME_CLASSES.length ? "hsl(var(--up))" : "hsl(var(--down))"}
                                  />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })()}
                  <DataTable
                    rows={filteredAggregate.bySymbol}
                    columns={symbolColumns}
                    rowKey={(row) => row.symbol}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Run-level detail</CardTitle>
                  <CardDescription>
                    Each scanned run with predicted argmax, calibrated probability, realised regime,
                    and set-membership coverage.{classFilter ? ` Filtered to ${classFilter}.` : ""}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <DataTable
                    rows={filteredRows}
                    columns={runColumns}
                    rowKey={(row) => row.id}
                    rowHref={(row) => `/history/${row.id}`}
                  />
                </CardContent>
              </Card>
            </>
          )}

          <div className="flex justify-end">
            <Button asChild variant="outline" size="sm">
              <Link href="/history">Back to history</Link>
            </Button>
          </div>
        </main>
      </div>
    </>
  );
}
