import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { Target } from "lucide-react";
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
import {
  REGIME_CLASSES,
  aggregateRegimePerformance,
  buildRunRegimeRecord,
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

export default function PerformancePage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [rows, setRows] = React.useState<RunRegimeRecord[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [totalRuns, setTotalRuns] = React.useState(0);
  const [breakdown, setBreakdown] = React.useState<ClassificationBreakdownResponse | null>(null);

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
          toast.error((err as Error).message || "Failed to load performance data.");
        }
      } finally {
        if (!signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [apiBaseUrl]);

  const aggregate = React.useMemo(() => aggregateRegimePerformance(rows), [rows]);
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
        header: "Argmax acc",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.argmaxAccuracy ?? -1,
        render: (row) => formatPercent(row.argmaxAccuracy),
      },
      {
        key: "empiricalCoverage",
        header: "Empirical coverage",
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
        header: "Argmax",
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
        header: "P(argmax)",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.argmaxProbability ?? -1,
        render: (row) => formatPercent(row.argmaxProbability),
      },
      {
        key: "realized",
        header: "Realized",
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
        header: "Argmax",
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
        header: "Set",
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
              Macro-F1, per-class precision / recall / F1, confusion matrix, and empirical conformal
              coverage on the resolved regime predictions. Realized regime is bucketed from the 10d
              forward vol path using the classifier&apos;s trained quantile cutoffs.
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
              label="Argmax accuracy"
              value={<span className="numeric">{formatPercent(aggregate.argmaxAccuracy)}</span>}
              caption={
                aggregate.argmaxAccuracy != null && aggregate.argmaxAccuracy >= 1 / REGIME_CLASSES.length
                  ? "above uniform baseline"
                  : "below uniform baseline"
              }
              tone={
                aggregate.argmaxAccuracy != null && aggregate.argmaxAccuracy >= 1 / REGIME_CLASSES.length
                  ? "up"
                  : aggregate.argmaxAccuracy == null
                  ? "neutral"
                  : "down"
              }
            />
            <KpiTile
              label="Macro-F1"
              value={<span className="numeric">{formatScore(headlineMacroF1)}</span>}
              caption={
                breakdownAvailable
                  ? "from training eval artifact"
                  : aggregate.macroF1 != null
                  ? "client aggregation across history"
                  : aggregate.resolved > 0
                  ? "needs resolved runs in every regime class"
                  : "needs resolved runs"
              }
              tone={headlineMacroF1 != null && headlineMacroF1 >= 0.4 ? "up" : "neutral"}
            />
            <KpiTile
              label="Empirical coverage"
              value={<span className="numeric">{formatPercent(aggregate.empiricalCoverage)}</span>}
              caption="realised regime inside the predicted set"
            />
            {headlineMacroRocAuc != null ? (
              <KpiTile
                label="Macro ROC-AUC"
                value={<span className="numeric">{formatScore(headlineMacroRocAuc)}</span>}
                caption="one-vs-rest, training eval"
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
              title="No runs in history"
              description="Submit analyses on the Workspace to populate this view."
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
                      ? "From the training-time classification breakdown — precision, recall, F1, ROC-AUC, PR-AUC."
                      : "Computed client-side from resolved history runs. Will switch to the training eval artifact when one is published."}
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
                            return (
                              <tr key={`${row.class_id}-${idx}`} className="border-b border-border last:border-0">
                                <td className="px-4 py-2 capitalize">{label}</td>
                                <td className="numeric px-4 py-2 text-right">{row.support}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.precision)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.recall)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.f1)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.roc_auc ?? null)}</td>
                                <td className="numeric px-4 py-2 text-right">{formatScore(row.pr_auc ?? null)}</td>
                              </tr>
                            );
                          })
                        : aggregate.perClass.map((entry) => (
                            <tr key={entry.klass} className="border-b border-border last:border-0">
                              <td className="px-4 py-2 capitalize">{entry.klass}</td>
                              <td className="numeric px-4 py-2 text-right">{entry.support}</td>
                              <td className="numeric px-4 py-2 text-right">{formatScore(entry.precision)}</td>
                              <td className="numeric px-4 py-2 text-right">{formatScore(entry.recall)}</td>
                              <td className="numeric px-4 py-2 text-right">{formatScore(entry.f1)}</td>
                            </tr>
                          ))}
                    </tbody>
                  </table>
                </CardContent>
                {breakdownAvailable && breakdown?.source ? (
                  <div className="border-t border-border bg-muted/20 px-4 py-2 text-[10px] text-muted-foreground">
                    Source: {breakdown.source.relative_path}
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
                      ? "From the training-time classification breakdown — rows are the true class, columns the predicted argmax."
                      : "Computed client-side from resolved runs — rows are realised regime, columns are predicted argmax."}
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
                        return <ConfusionMatrix rows={matrixRows} classes={labels} />;
                      })()
                    : <ConfusionMatrix rows={aggregate.confusion} classes={REGIME_CLASSES} />}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Per-asset breakdown</CardTitle>
                  <CardDescription>
                    Argmax accuracy and empirical coverage for every symbol with at least one
                    resolved run.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <DataTable
                    rows={aggregate.bySymbol}
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
                    and set-membership coverage.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <DataTable
                    rows={rows}
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
