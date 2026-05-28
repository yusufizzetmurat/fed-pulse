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
import { DataTable, type DataTableColumn } from "@/components/ui/data-table";
import { EmptyState } from "@/components/ui/empty-state";
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
  fetchHistoryRealizedBatch,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import type { HistoryEntry, HistoryQuery } from "@/lib/analyze/types";

const STANCE_OPTIONS = [
  { value: "any", label: "Any stance" },
  { value: "hawkish", label: "Hawkish" },
  { value: "neutral", label: "Neutral" },
  { value: "dovish", label: "Dovish" },
];

const REGIME_OPTIONS = [
  { value: "any", label: "Any regime" },
  { value: "calm", label: "Calm" },
  { value: "normal", label: "Normal" },
  { value: "high", label: "High" },
];

const HORIZON_OPTIONS = ["any", "1d", "3d", "5d", "10d"];

function regimeVariant(label: string | null | undefined): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "calm") return "dovish";
  if (label === "high") return "hawkish";
  if (label === "normal") return "neutral";
  return "outline";
}

interface RowWithRealized extends HistoryEntry {
  realized_regime?: string | null;
}

export default function HistoryPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [items, setItems] = React.useState<RowWithRealized[]>([]);
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [filters, setFilters] = React.useState<HistoryQuery>({ limit: 50, offset: 0 });
  const [regimeFilter, setRegimeFilter] = React.useState<string>("any");

  // Bump this version to force a refetch (e.g. after a delete) without
  // rebuilding the filters object. The effect owns the AbortController
  // so cleanup actually runs when React tears it down — an async
  // useCallback cannot hand the cleanup back to React.
  const [reloadVersion, setReloadVersion] = React.useState(0);
  const [search, setSearch] = React.useState("");
  const reload = React.useCallback(() => {
    setReloadVersion((value) => value + 1);
  }, []);

  React.useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setLoading(true);
    (async () => {
      try {
        const result = await fetchHistory(apiBaseUrl, filters, signal);
        if (signal.aborted) return;
        setItems(result.items.map((row) => ({ ...row, realized_regime: null })));
        setTotal(result.total);
        // One batched round trip replaces the N per-row fetches the page
        // used to fan out. Backend caps the batch at 50 ids; the page
        // limit defaults to 50 so the call fits in one request. Failures
        // are best-effort — aborted requests and yfinance hiccups leave
        // the realized column on "pending" rather than nuking the table.
        const ids = result.items.map((row) => row.id);
        try {
          const batch = await fetchHistoryRealizedBatch(apiBaseUrl, ids, signal);
          if (signal.aborted) return;
          setItems((prev) =>
            prev.map((entry) => {
              const realized = batch.items[entry.id];
              if (!realized) return entry;
              return { ...entry, realized_regime: realized.realized_regime ?? null };
            }),
          );
        } catch {
          // Best-effort: realized column stays "pending" on batch failure.
        }
      } catch (err) {
        if (!signal.aborted) {
          toast.error((err as Error).message || "Failed to load history.");
        }
      } finally {
        if (!signal.aborted) setLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, filters, reloadVersion]);

  const handleDelete = React.useCallback(
    async (id: string) => {
      try {
        await deleteHistoryRun(apiBaseUrl, id);
        toast.success("Run deleted");
        reload();
      } catch (err) {
        toast.error((err as Error).message || "Delete failed");
      }
    },
    [apiBaseUrl, reload],
  );

  const patchFilter = (delta: Partial<HistoryQuery>) =>
    setFilters((value) => ({ ...value, offset: 0, ...delta }));

  const visibleRows = React.useMemo(() => {
    const regimeFiltered =
      regimeFilter === "any" ? items : items.filter((row) => row.argmax_regime === regimeFilter);
    const needle = search.trim().toLowerCase();
    if (!needle) return regimeFiltered;
    return regimeFiltered.filter((row) => {
      const haystack = [
        row.id,
        row.document_date,
        row.stance,
        row.symbol,
        row.horizon,
        row.argmax_regime ?? "",
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [items, regimeFilter, search]);

  const columns = React.useMemo<DataTableColumn<RowWithRealized>[]>(
    () => [
      {
        key: "document_date",
        header: "Date",
        align: "left",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.document_date,
        render: (row) => row.document_date,
      },
      {
        key: "symbol",
        header: "Symbol",
        align: "left",
        sortable: true,
        sortValue: (row) => row.symbol,
        render: (row) => row.symbol,
      },
      {
        key: "horizon",
        header: "H",
        align: "left",
        render: (row) => <span className="text-muted-foreground">{row.horizon}</span>,
      },
      {
        key: "stance",
        header: "Stance",
        render: (row) => {
          const stance = toStance(row.stance);
          const variant =
            stance === "hawkish"
              ? "hawkish"
              : stance === "dovish"
              ? "dovish"
              : stance === "neutral"
              ? "neutral"
              : "outline";
          return (
            <Badge variant={variant} className="text-[10px]">
              {stanceLabel(stance)}
            </Badge>
          );
        },
      },
      {
        key: "argmax_regime",
        header: "Regime",
        render: (row) =>
          row.argmax_regime ? (
            <Badge variant={regimeVariant(row.argmax_regime)} className="text-[10px] capitalize">
              {row.argmax_regime}
            </Badge>
          ) : (
            <span className="text-muted-foreground">—</span>
          ),
      },
      {
        key: "argmax_probability",
        header: "Probability",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.argmax_probability ?? -1,
        render: (row) =>
          row.argmax_probability != null
            ? `${(row.argmax_probability * 100).toFixed(1)}%`
            : "—",
      },
      {
        key: "regime_set_size",
        header: "Set size",
        align: "right",
        numeric: true,
        sortable: true,
        sortValue: (row) => row.regime_set_size ?? -1,
        render: (row) => (row.regime_set_size != null ? row.regime_set_size.toString() : "—"),
      },
      {
        key: "realized_regime",
        header: "Actual",
        render: (row) =>
          row.realized_regime ? (
            <Badge variant={regimeVariant(row.realized_regime)} className="text-[10px] capitalize">
              {row.realized_regime}
            </Badge>
          ) : (
            <span className="text-muted-foreground">pending</span>
          ),
      },
      {
        key: "hit",
        header: "Hit",
        align: "center",
        render: (row) => {
          if (!row.argmax_regime || !row.realized_regime) {
            return <span className="text-muted-foreground">—</span>;
          }
          const hit = row.argmax_regime === row.realized_regime;
          return (
            <Badge variant={hit ? "dovish" : "hawkish"} className="text-[10px]">
              {hit ? "hit" : "miss"}
            </Badge>
          );
        },
      },
      {
        key: "actions",
        header: <span className="sr-only">Actions</span>,
        align: "right",
        render: (row) => (
          <div
            className="flex items-center justify-end gap-0.5"
            onClick={(event) => event.stopPropagation()}
          >
            <Button asChild variant="ghost" size="icon" aria-label={`Open run on ${row.document_date}`}>
              <Link href={`/history/${row.id}`}>
                <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              aria-label={`Delete run on ${row.document_date}`}
              onClick={() => handleDelete(row.id)}
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          </div>
        ),
      },
    ],
    [handleDelete],
  );

  return (
    <>
      <Head>
        <title>History — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-5 py-6 focus:outline-none">
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">History</h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Past regime predictions. Realized regime is bucketed from the post-event 10d-forward vol path
              against the classifier&apos;s trained quantile cutoffs.
            </p>
          </div>

          <div className="space-y-1">
            <Label htmlFor="history-search">Search</Label>
            <Input
              id="history-search"
              type="search"
              placeholder="Filter by run id, date, stance, or symbol…"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Filters</CardTitle>
              <CardDescription>
                {total} total run{total === 1 ? "" : "s"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-5">
                <div className="space-y-1">
                  <Label htmlFor="filter-symbol">Symbol</Label>
                  <Input
                    id="filter-symbol"
                    placeholder="e.g. ^GSPC"
                    value={filters.symbol ?? ""}
                    onChange={(event) => patchFilter({ symbol: event.target.value || undefined })}
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
                  <Label htmlFor="filter-regime">Regime</Label>
                  <Select value={regimeFilter} onValueChange={setRegimeFilter}>
                    <SelectTrigger id="filter-regime">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {REGIME_OPTIONS.map((option) => (
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
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : visibleRows.length === 0 ? (
            <EmptyState
              title="No runs match these filters"
              description="Submit an analysis from the Workspace to populate the history, or relax the filters."
              action={
                <Button asChild size="sm" variant="outline">
                  <Link href="/">Open Workspace</Link>
                </Button>
              }
            />
          ) : (
            <Card>
              <CardContent className="p-0">
                <DataTable
                  rows={visibleRows}
                  columns={columns}
                  rowKey={(row) => row.id}
                  rowHref={(row) => `/history/${row.id}`}
                />
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
