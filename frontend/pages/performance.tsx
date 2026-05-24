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
  fetchHistory,
  fetchHistoryRealized,
  fetchHistoryRun,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import {
  REGIME_CLASSES,
  aggregateRegimePerformance,
  buildRunRegimeRecord,
  type RunRegimeRecord,
} from "@/lib/analyze/performance";
import type { HistoryEntry } from "@/lib/analyze/types";

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

  React.useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setLoading(true);
    (async () => {
      try {
        const list = await fetchHistory(apiBaseUrl, { limit: HISTORY_LIMIT }, signal);
        if (signal.aborted) return;
        setTotalRuns(list.total);
        const records = await Promise.all(
          list.items.map(async (entry: HistoryEntry) => {
            const [realized, detail] = await Promise.all([
              fetchHistoryRealized(apiBaseUrl, entry.id, signal).catch(() => null),
              fetchHistoryRun(apiBaseUrl, entry.id, signal).catch(() => null),
            ]);
            return buildRunRegimeRecord({
              entry,
              realized,
              payload: (detail?.payload || null) as Record<string, unknown> | null,
            });
          }),
        );
        if (!signal.aborted) setRows(records);
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
              value={<span className="numeric">{formatScore(aggregate.macroF1)}</span>}
              caption={
                aggregate.macroF1 != null
                  ? "unweighted mean of per-class F1"
                  : aggregate.resolved > 0
                  ? "needs resolved runs in every regime class"
                  : "needs resolved runs"
              }
              tone={aggregate.macroF1 != null && aggregate.macroF1 >= 0.4 ? "up" : "neutral"}
            />
            <KpiTile
              label="Empirical coverage"
              value={<span className="numeric">{formatPercent(aggregate.empiricalCoverage)}</span>}
              caption="realised regime inside the predicted set"
            />
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
                    Precision / recall / F1 for each calibrated class. Support is the count of
                    resolved runs whose realised regime matched the truth row.
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
                      </tr>
                    </thead>
                    <tbody>
                      {aggregate.perClass.map((entry) => (
                        <tr key={entry.klass} className="border-b border-border last:border-0">
                          <td className="px-4 py-2 capitalize">{entry.klass}</td>
                          <td className="numeric px-4 py-2 text-right">{entry.support}</td>
                          <td className="numeric px-4 py-2 text-right">
                            {formatScore(entry.precision)}
                          </td>
                          <td className="numeric px-4 py-2 text-right">{formatScore(entry.recall)}</td>
                          <td className="numeric px-4 py-2 text-right">{formatScore(entry.f1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Confusion matrix</CardTitle>
                  <CardDescription>
                    Rows are the realised regime, columns are the predicted argmax. Counts are
                    coloured by row share so misclassifications stand out at a glance.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ConfusionMatrix rows={aggregate.confusion} classes={REGIME_CLASSES} />
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
