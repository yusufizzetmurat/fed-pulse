import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { ArrowDownRight, ArrowUpRight, Target } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchHistory,
  fetchHistoryRealized,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { formatPrice } from "@/lib/analyze/format";
import {
  aggregatePerformance,
  computeRunPerformance,
  type RunPerformance,
} from "@/lib/analyze/performance";
import type { HistoryEntry } from "@/lib/analyze/types";

const HISTORY_LIMIT = 50;

function formatPercent(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export default function PerformancePage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [rows, setRows] = React.useState<RunPerformance[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [totalRuns, setTotalRuns] = React.useState(0);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const list = await fetchHistory(apiBaseUrl, { limit: HISTORY_LIMIT });
        if (cancelled) return;
        setTotalRuns(list.total);
        const resolvedRows: RunPerformance[] = await Promise.all(
          list.items.map(async (entry: HistoryEntry) => {
            try {
              const realized = await fetchHistoryRealized(apiBaseUrl, entry.id);
              const last = realized.close.length
                ? realized.close[realized.close.length - 1]
                : null;
              return computeRunPerformance(entry, last);
            } catch {
              return computeRunPerformance(entry, null);
            }
          })
        );
        if (!cancelled) setRows(resolvedRows);
      } catch (err) {
        if (!cancelled) {
          toast.error((err as Error).message || "Failed to load performance data.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const aggregate = React.useMemo(() => aggregatePerformance(rows), [rows]);

  return (
    <>
      <Head>
        <title>Performance — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">Performance</h1>
            <p className="max-w-2xl text-muted-foreground">
              Past predictions resolved against realized close. Direction = sign of (predicted − spot) vs sign of (realized − spot).
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            <SummaryCard
              title="Runs scanned"
              value={`${rows.length}`}
              caption={totalRuns > rows.length ? `of ${totalRuns} total` : "all runs"}
              icon={<Target className="h-4 w-4 text-muted-foreground" />}
            />
            <SummaryCard
              title="Resolved"
              value={`${aggregate.resolved}`}
              caption={aggregate.resolved < rows.length ? `${rows.length - aggregate.resolved} still pending` : "complete"}
              icon={<Target className="h-4 w-4 text-muted-foreground" />}
            />
            <SummaryCard
              title="Directional hit rate"
              value={formatPercent(aggregate.hitRate)}
              caption={aggregate.hitRate != null && aggregate.hitRate >= 0.5 ? "above coin-flip" : "below coin-flip"}
              icon={
                aggregate.hitRate != null && aggregate.hitRate >= 0.5 ? (
                  <ArrowUpRight className="h-4 w-4 text-emerald-500" />
                ) : (
                  <ArrowDownRight className="h-4 w-4 text-rose-500" />
                )
              }
            />
            <SummaryCard
              title="MAPE"
              value={formatPercent(aggregate.mape)}
              caption="mean absolute % error"
              icon={<Target className="h-4 w-4 text-muted-foreground" />}
            />
          </div>

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : rows.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                No runs in history yet — submit analyses on the Analyze page to populate this view.
              </CardContent>
            </Card>
          ) : (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Per-asset breakdown</CardTitle>
                  <CardDescription>Hit rate and MAPE for each symbol that has at least one resolved run.</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-sm">
                    <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left">Symbol</th>
                        <th className="px-4 py-2 text-right">Resolved</th>
                        <th className="px-4 py-2 text-right">Hit rate</th>
                        <th className="px-4 py-2 text-right">MAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {aggregate.bySymbol.map((entry) => (
                        <tr key={entry.symbol} className="border-b border-border last:border-0">
                          <td className="px-4 py-2 font-medium">{entry.symbol}</td>
                          <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                            {entry.resolved}
                          </td>
                          <td className="px-4 py-2 text-right font-mono">{formatPercent(entry.hitRate)}</td>
                          <td className="px-4 py-2 text-right font-mono">{formatPercent(entry.mape)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Run-level detail</CardTitle>
                  <CardDescription>Each scanned history run with predicted, spot, realized, and direction.</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-sm">
                    <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                      <tr>
                        <th className="px-4 py-2 text-left">Date</th>
                        <th className="px-4 py-2 text-left">Symbol</th>
                        <th className="px-4 py-2 text-left">Horizon</th>
                        <th className="px-4 py-2 text-right">Spot</th>
                        <th className="px-4 py-2 text-right">Predicted</th>
                        <th className="px-4 py-2 text-right">Realized</th>
                        <th className="px-4 py-2 text-right">% error</th>
                        <th className="px-4 py-2 text-center">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr key={row.id} className="border-b border-border last:border-0 hover:bg-muted/40">
                          <td className="px-4 py-2 font-mono text-xs">
                            <Link href={`/history/${row.id}`} className="hover:underline">
                              {row.document_date}
                            </Link>
                          </td>
                          <td className="px-4 py-2 font-medium">{row.symbol}</td>
                          <td className="px-4 py-2 text-muted-foreground">{row.horizon}</td>
                          <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                            {formatPrice(row.spot_close)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono">{formatPrice(row.predicted_close)}</td>
                          <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                            {formatPrice(row.realized_close)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono">{formatPercent(row.percent_error)}</td>
                          <td className="px-4 py-2 text-center">
                            {row.direction_correct == null ? (
                              <Badge variant="outline">—</Badge>
                            ) : row.direction_correct ? (
                              <Badge variant="hawkish">Hit</Badge>
                            ) : (
                              <Badge variant="dovish">Miss</Badge>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
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

function SummaryCard({
  title,
  value,
  caption,
  icon,
}: {
  title: string;
  value: string;
  caption: string;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="font-mono text-2xl font-semibold">{value}</div>
        <p className="text-xs text-muted-foreground">{caption}</p>
      </CardContent>
    </Card>
  );
}
