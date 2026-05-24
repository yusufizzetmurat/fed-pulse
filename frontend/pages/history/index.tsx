import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { ChevronRight, Trash2 } from "lucide-react";
import { toast } from "sonner";

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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  deleteHistoryRun,
  fetchHistory,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { formatPrice, stanceLabel, toStance } from "@/lib/analyze/format";
import type { HistoryEntry, HistoryQuery } from "@/lib/analyze/types";

const STANCE_OPTIONS = [
  { value: "any", label: "Any stance" },
  { value: "hawkish", label: "Hawkish" },
  { value: "neutral", label: "Neutral" },
  { value: "dovish", label: "Dovish" },
];

const HORIZON_OPTIONS = ["any", "1d", "3d", "5d", "10d"];

export default function HistoryPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [items, setItems] = React.useState<HistoryEntry[]>([]);
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [filters, setFilters] = React.useState<HistoryQuery>({ limit: 20, offset: 0 });

  const reload = React.useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetchHistory(apiBaseUrl, filters);
      setItems(result.items);
      setTotal(result.total);
    } catch (err) {
      const message = (err as Error).message || "Failed to load history.";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, filters]);

  React.useEffect(() => {
    reload();
  }, [reload]);

  const handleDelete = async (id: string) => {
    try {
      await deleteHistoryRun(apiBaseUrl, id);
      toast.success("Run deleted");
      await reload();
    } catch (err) {
      toast.error((err as Error).message || "Delete failed");
    }
  };

  const patchFilter = (delta: Partial<HistoryQuery>) =>
    setFilters((value) => ({ ...value, offset: 0, ...delta }));

  return (
    <>
      <Head>
        <title>History — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">History</h1>
            <p className="max-w-2xl text-muted-foreground">
              Past analyses. Filter by asset, horizon, or stance; click an entry to drill in.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Filters</CardTitle>
              <CardDescription>{total} total run{total === 1 ? "" : "s"}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-4">
                <div className="space-y-1">
                  <Label htmlFor="filter-symbol">Symbol</Label>
                  <Input
                    id="filter-symbol"
                    placeholder="e.g. ^GSPC"
                    value={filters.symbol ?? ""}
                    onChange={(event) =>
                      patchFilter({ symbol: event.target.value || undefined })
                    }
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="filter-horizon">Horizon</Label>
                  <Select
                    value={filters.horizon ?? "any"}
                    onValueChange={(value) =>
                      patchFilter({ horizon: value === "any" ? undefined : value })
                    }
                  >
                    <SelectTrigger id="filter-horizon">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {HORIZON_OPTIONS.map((option) => (
                        <SelectItem key={option} value={option}>
                          {option === "any" ? "Any horizon" : option}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="filter-stance">Stance</Label>
                  <Select
                    value={filters.stance ?? "any"}
                    onValueChange={(value) =>
                      patchFilter({ stance: value === "any" ? undefined : value })
                    }
                  >
                    <SelectTrigger id="filter-stance">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {STANCE_OPTIONS.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="filter-date">Document date</Label>
                  <Input
                    id="filter-date"
                    type="date"
                    value={filters.document_date ?? ""}
                    onChange={(event) =>
                      patchFilter({ document_date: event.target.value || undefined })
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : items.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                No runs yet — submit an analysis to populate the history.
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-0">
                <table className="w-full text-sm">
                  <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2 text-left">Date</th>
                      <th className="px-4 py-2 text-left">Symbol</th>
                      <th className="px-4 py-2 text-left">Horizon</th>
                      <th className="px-4 py-2 text-left">Stance</th>
                      <th className="px-4 py-2 text-right">Predicted close</th>
                      <th className="px-4 py-2 text-right">Spot</th>
                      <th className="px-4 py-2" aria-label="actions" />
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((row) => {
                      const stance = toStance(row.stance);
                      const stanceVariant =
                        stance === "hawkish"
                          ? "hawkish"
                          : stance === "dovish"
                          ? "dovish"
                          : stance === "neutral"
                          ? "neutral"
                          : "outline";
                      return (
                        <tr key={row.id} className="border-b border-border last:border-0 hover:bg-muted/40">
                          <td className="px-4 py-2 font-mono text-xs">
                            <Link href={`/history/${row.id}`} className="hover:underline">
                              {row.document_date}
                            </Link>
                          </td>
                          <td className="px-4 py-2 font-medium">{row.symbol}</td>
                          <td className="px-4 py-2 text-muted-foreground">{row.horizon}</td>
                          <td className="px-4 py-2">
                            <Badge variant={stanceVariant}>{stanceLabel(stance)}</Badge>
                          </td>
                          <td className="px-4 py-2 text-right font-mono">
                            {formatPrice(row.predicted_close ?? null)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                            {formatPrice(row.current_close ?? null)}
                          </td>
                          <td className="px-4 py-2 text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Button asChild variant="ghost" size="icon" aria-label={`Open run on ${row.document_date}`}>
                                <Link href={`/history/${row.id}`}>
                                  <ChevronRight className="h-4 w-4" aria-hidden="true" />
                                </Link>
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                aria-label={`Delete run on ${row.document_date}`}
                                onClick={() => handleDelete(row.id)}
                              >
                                <Trash2 className="h-4 w-4" aria-hidden="true" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
