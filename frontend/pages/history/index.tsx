import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { ChevronRight, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { HistoryTimelineChart, type HistoryTimelineRow } from "@/components/analyze/HistoryTimelineChart";
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
import { Skeleton } from "@/components/ui/skeleton";
import {
  deleteHistoryRun,
  fetchHistory,
  fetchHistoryRealizedBatch,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import { useSymbols } from "@/lib/analyze/useSymbols";
import type { HistoryEntry, Stance } from "@/lib/analyze/types";

const STANCE_VALUES: Stance[] = ["hawkish", "neutral", "dovish"];
const REGIME_VALUES = ["calm", "normal", "high"] as const;
const DEFAULT_PAGE_SIZE = 50;
const PAGE_SIZE_CAP = 200;

function parsePageSize(value: string | string[] | undefined): number {
  if (!value) return DEFAULT_PAGE_SIZE;
  const raw = Array.isArray(value) ? value[0] : value;
  const n = Number.parseInt(String(raw), 10);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_PAGE_SIZE;
  return Math.min(n, PAGE_SIZE_CAP);
}

type RegimeValue = (typeof REGIME_VALUES)[number];

function regimeVariant(label: string | null | undefined): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "calm") return "dovish";
  if (label === "high") return "hawkish";
  if (label === "normal") return "neutral";
  return "outline";
}

interface RowWithRealized extends HistoryEntry {
  realized_regime?: string | null;
}

interface Filters {
  stances: Set<Stance>;
  regimes: Set<RegimeValue>;
  symbols: Set<string>;
  variants: Set<string>;
  dateStart: string;
  dateEnd: string;
}

function parseSet<T extends string>(value: string | string[] | undefined, allowed: readonly T[]): Set<T> {
  if (!value) return new Set<T>();
  const list = Array.isArray(value) ? value : value.split(",");
  const set = new Set<T>();
  for (const item of list) {
    const trimmed = String(item).trim();
    if (allowed.includes(trimmed as T)) set.add(trimmed as T);
  }
  return set;
}

function parseFreeSet(value: string | string[] | undefined): Set<string> {
  if (!value) return new Set<string>();
  const list = Array.isArray(value) ? value : value.split(",");
  return new Set(list.map((v) => String(v).trim()).filter((v) => v.length > 0));
}

function parseDate(value: string | string[] | undefined): string {
  if (!value) return "";
  if (Array.isArray(value)) return value[0] ?? "";
  return value;
}

function readFilters(query: Record<string, string | string[] | undefined>): Filters {
  return {
    stances: parseSet(query.stance, STANCE_VALUES),
    regimes: parseSet(query.regime, REGIME_VALUES),
    symbols: parseFreeSet(query.symbol),
    variants: parseFreeSet(query.variant),
    dateStart: parseDate(query.start),
    dateEnd: parseDate(query.end),
  };
}

function serialiseFilters(
  filters: Filters,
  search: string,
  pageSize: number,
): Record<string, string> {
  const out: Record<string, string> = {};
  if (filters.stances.size > 0) out.stance = [...filters.stances].join(",");
  if (filters.regimes.size > 0) out.regime = [...filters.regimes].join(",");
  if (filters.symbols.size > 0) out.symbol = [...filters.symbols].join(",");
  if (filters.variants.size > 0) out.variant = [...filters.variants].join(",");
  if (filters.dateStart) out.start = filters.dateStart;
  if (filters.dateEnd) out.end = filters.dateEnd;
  if (search) out.q = search;
  if (pageSize !== DEFAULT_PAGE_SIZE) out.size = String(pageSize);
  return out;
}

function emptyFilters(): Filters {
  return {
    stances: new Set(),
    regimes: new Set(),
    symbols: new Set(),
    variants: new Set(),
    dateStart: "",
    dateEnd: "",
  };
}

function MultiToggle<T extends string>({
  label,
  options,
  selected,
  onToggle,
  variant,
}: {
  label: string;
  options: readonly { value: T; label: string }[];
  selected: Set<T>;
  onToggle: (value: T) => void;
  variant?: (value: T) => "hawkish" | "dovish" | "neutral" | "outline";
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => {
          const active = selected.has(opt.value);
          const tone = variant ? variant(opt.value) : "outline";
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onToggle(opt.value)}
              className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-full"
            >
              <Badge
                variant={active ? tone : "outline"}
                className={`cursor-pointer text-[10px] ${active ? "" : "opacity-60"}`}
              >
                {opt.label}
              </Badge>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function HistoryPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const router = useRouter();
  const { symbols: symbolUniverse } = useSymbols();
  const [items, setItems] = React.useState<RowWithRealized[]>([]);
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(false);

  const [filters, setFilters] = React.useState<Filters>(emptyFilters);
  const [search, setSearch] = React.useState("");
  const [pageSize, setPageSize] = React.useState<number>(DEFAULT_PAGE_SIZE);
  const [loadingMore, setLoadingMore] = React.useState(false);
  const [reloadVersion, setReloadVersion] = React.useState(0);
  const reload = React.useCallback(() => {
    setReloadVersion((value) => value + 1);
  }, []);

  // Hydrate filters and page size from URL once router is ready. Only
  // runs on mount / first ready; subsequent edits flow through pushUrl()
  // and don't re-hydrate (avoiding the round-trip loop).
  const hydratedRef = React.useRef(false);
  React.useEffect(() => {
    if (!router.isReady || hydratedRef.current) return;
    hydratedRef.current = true;
    setFilters(readFilters(router.query as Record<string, string | string[] | undefined>));
    const q = router.query.q;
    setSearch(typeof q === "string" ? q : "");
    setPageSize(parsePageSize(router.query.size));
  }, [router.isReady, router.query]);

  // Mirror filter state to URL.
  const pushUrl = React.useCallback(
    (nextFilters: Filters, nextSearch: string, nextSize: number) => {
      const params = serialiseFilters(nextFilters, nextSearch, nextSize);
      router.replace({ pathname: "/history", query: params }, undefined, { shallow: true });
    },
    [router],
  );

  const hydrateRealized = React.useCallback(
    async (rows: HistoryEntry[], signal: AbortSignal) => {
      if (rows.length === 0) return;
      try {
        const batch = await fetchHistoryRealizedBatch(
          apiBaseUrl,
          rows.map((row) => row.id),
          signal,
        );
        if (signal.aborted) return;
        const lookup = batch.items;
        setItems((prev) =>
          prev.map((entry) => {
            const realized = lookup[entry.id];
            if (!realized) return entry;
            return { ...entry, realized_regime: realized.realized_regime ?? null };
          }),
        );
      } catch {
        // Best-effort.
      }
    },
    [apiBaseUrl],
  );

  React.useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    setLoading(true);
    (async () => {
      try {
        const result = await fetchHistory(apiBaseUrl, { limit: pageSize, offset: 0 }, signal);
        if (signal.aborted) return;
        setItems(result.items.map((row) => ({ ...row, realized_regime: null })));
        setTotal(result.total);
        await hydrateRealized(result.items, signal);
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
  }, [apiBaseUrl, reloadVersion, pageSize, hydrateRealized]);

  const handleShowMore = React.useCallback(async () => {
    if (loadingMore) return;
    const controller = new AbortController();
    const { signal } = controller;
    setLoadingMore(true);
    try {
      const result = await fetchHistory(
        apiBaseUrl,
        { limit: pageSize, offset: items.length },
        signal,
      );
      if (signal.aborted) return;
      const fresh = result.items.map((row) => ({ ...row, realized_regime: null }));
      setItems((prev) => [...prev, ...fresh]);
      setTotal(result.total);
      await hydrateRealized(result.items, signal);
    } catch (err) {
      if (!signal.aborted) {
        toast.error((err as Error).message || "Failed to load more history.");
      }
    } finally {
      if (!signal.aborted) setLoadingMore(false);
    }
  }, [apiBaseUrl, hydrateRealized, items.length, loadingMore, pageSize]);

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

  // Collect the available variant / symbol options from the loaded rows.
  // Symbols also include the static universe so filters work pre-load.
  const symbolOptions = React.useMemo(() => {
    const set = new Set<string>(symbolUniverse.map((s) => s.symbol));
    for (const row of items) set.add(row.symbol);
    return [...set].sort();
  }, [items, symbolUniverse]);

  const variantOptions = React.useMemo(() => {
    const set = new Set<string>();
    for (const row of items) {
      if (row.forecast_mode) set.add(row.forecast_mode);
    }
    return [...set].sort();
  }, [items]);

  const visibleRows = React.useMemo(() => {
    const needle = search.trim().toLowerCase();
    return items.filter((row) => {
      if (filters.stances.size > 0 && !filters.stances.has(toStance(row.stance))) return false;
      if (filters.regimes.size > 0) {
        if (!row.argmax_regime || !filters.regimes.has(row.argmax_regime as RegimeValue)) return false;
      }
      if (filters.symbols.size > 0 && !filters.symbols.has(row.symbol)) return false;
      if (filters.variants.size > 0 && !filters.variants.has(row.forecast_mode)) return false;
      if (filters.dateStart && row.document_date < filters.dateStart) return false;
      if (filters.dateEnd && row.document_date > filters.dateEnd) return false;
      if (needle) {
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
        if (!haystack.includes(needle)) return false;
      }
      return true;
    });
  }, [items, filters, search]);

  const patchFilters = React.useCallback(
    (delta: Partial<Filters>) => {
      setFilters((prev) => {
        const next = { ...prev, ...delta } as Filters;
        pushUrl(next, search, pageSize);
        return next;
      });
    },
    [pageSize, pushUrl, search],
  );

  const toggleInSet = <T extends string>(set: Set<T>, value: T): Set<T> => {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    return next;
  };

  const handleResetFilters = () => {
    const next = emptyFilters();
    setFilters(next);
    setSearch("");
    pushUrl(next, "", pageSize);
  };

  // Aggregate hit-rate from the realized data already fetched. Counts a
  // row as resolved when both the predicted argmax and the realized
  // regime are present; hits are exact matches between the two.
  const hitStats = React.useMemo(() => {
    let resolved = 0;
    let hits = 0;
    for (const row of visibleRows) {
      if (!row.argmax_regime || !row.realized_regime) continue;
      resolved += 1;
      if (row.argmax_regime === row.realized_regime) hits += 1;
    }
    return { resolved, hits };
  }, [visibleRows]);

  const hitRatePct =
    hitStats.resolved > 0 ? (hitStats.hits / hitStats.resolved) * 100 : null;

  const hasMore = items.length < total;

  const filtersActive =
    filters.stances.size > 0 ||
    filters.regimes.size > 0 ||
    filters.symbols.size > 0 ||
    filters.variants.size > 0 ||
    !!filters.dateStart ||
    !!filters.dateEnd ||
    !!search;

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
            <p className="max-w-2xl text-xs text-muted-foreground">
              The text-neutral residual-fusion canonical collapses to a single argmax
              (typically Calm) when the gate is closed, so the live hit-rate is low by
              construction. Treat the number as the cost of running the late-fusion
              second opinion alongside HAR-tercile, not as the regime model&apos;s
              underlying quality. The HAR-tercile baseline above the workspace fold
              is the headline forecast.
            </p>
          </div>

          <div
            className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground"
            aria-label="History hit-rate summary"
          >
            <span>
              Shown <span className="numeric text-foreground">{visibleRows.length}</span>
            </span>
            <span aria-hidden="true">·</span>
            <span>
              resolved <span className="numeric text-foreground">{hitStats.resolved}</span>
            </span>
            <span aria-hidden="true">·</span>
            <span>
              hit-rate{" "}
              <span className="numeric text-foreground">
                {hitRatePct == null ? "—" : `${hitRatePct.toFixed(1)}%`}
              </span>
            </span>
          </div>

          <HistoryTimelineChart rows={visibleRows as HistoryTimelineRow[]} />

          <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
            <aside className="space-y-4 rounded-md border border-border p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">Filters</p>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleResetFilters}
                  disabled={!filtersActive}
                  className="h-7 text-[11px]"
                >
                  Reset
                </Button>
              </div>

              <MultiToggle
                label="Stance"
                options={[
                  { value: "hawkish" as Stance, label: "Hawkish" },
                  { value: "neutral" as Stance, label: "Neutral" },
                  { value: "dovish" as Stance, label: "Dovish" },
                ]}
                selected={filters.stances}
                onToggle={(value) => patchFilters({ stances: toggleInSet(filters.stances, value) })}
                variant={(value) =>
                  value === "hawkish" ? "hawkish" : value === "dovish" ? "dovish" : "neutral"
                }
              />

              <MultiToggle
                label="Regime"
                options={[
                  { value: "calm" as RegimeValue, label: "Calm" },
                  { value: "normal" as RegimeValue, label: "Normal" },
                  { value: "high" as RegimeValue, label: "High" },
                ]}
                selected={filters.regimes}
                onToggle={(value) => patchFilters({ regimes: toggleInSet(filters.regimes, value) })}
                variant={(value) =>
                  value === "high" ? "hawkish" : value === "calm" ? "dovish" : "neutral"
                }
              />

              <div className="space-y-1.5">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">
                  Date range
                </Label>
                <div className="grid grid-cols-2 gap-1.5">
                  <Input
                    type="date"
                    value={filters.dateStart}
                    onChange={(e) => patchFilters({ dateStart: e.target.value })}
                    aria-label="Start date"
                  />
                  <Input
                    type="date"
                    value={filters.dateEnd}
                    onChange={(e) => patchFilters({ dateEnd: e.target.value })}
                    aria-label="End date"
                  />
                </div>
              </div>

              {symbolOptions.length > 0 ? (
                <div className="space-y-1.5">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">
                    Symbol
                  </Label>
                  <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto">
                    {symbolOptions.map((sym) => {
                      const active = filters.symbols.has(sym);
                      return (
                        <button
                          key={sym}
                          type="button"
                          onClick={() => patchFilters({ symbols: toggleInSet(filters.symbols, sym) })}
                          className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-full"
                        >
                          <Badge
                            variant={active ? "outline" : "outline"}
                            className={`cursor-pointer text-[10px] font-mono ${active ? "border-primary text-foreground" : "opacity-60"}`}
                          >
                            {sym}
                          </Badge>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {variantOptions.length > 0 ? (
                <div className="space-y-1.5">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">
                    Model variant
                  </Label>
                  <div className="flex flex-wrap gap-1.5">
                    {variantOptions.map((v) => {
                      const active = filters.variants.has(v);
                      return (
                        <button
                          key={v}
                          type="button"
                          onClick={() => patchFilters({ variants: toggleInSet(filters.variants, v) })}
                          className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-full"
                        >
                          <Badge
                            variant="outline"
                            className={`cursor-pointer text-[10px] font-mono ${active ? "border-primary text-foreground" : "opacity-60"}`}
                          >
                            {v}
                          </Badge>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </aside>

            <div className="space-y-3">
              <div className="space-y-1">
                <Label htmlFor="history-search">Search</Label>
                <Input
                  id="history-search"
                  type="search"
                  placeholder="Filter by run id, date, stance, or symbol…"
                  value={search}
                  onChange={(event) => {
                    const next = event.target.value;
                    setSearch(next);
                    pushUrl(filters, next, pageSize);
                  }}
                />
              </div>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Runs</CardTitle>
                  <CardDescription>
                    {visibleRows.length} shown · {total} total run{total === 1 ? "" : "s"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  {loading ? (
                    <div className="space-y-2 p-4">
                      <Skeleton className="h-10 w-full" />
                      <Skeleton className="h-10 w-full" />
                      <Skeleton className="h-10 w-full" />
                    </div>
                  ) : visibleRows.length === 0 ? (
                    <div className="p-4">
                      <EmptyState
                        title={total === 0 ? "No history yet." : "No runs match these filters."}
                        description={
                          total === 0
                            ? "Use the Workspace to analyze a statement."
                            : "Relax the filters or use the Workspace to analyze a new statement."
                        }
                        action={
                          <Button asChild size="sm" variant="outline">
                            <Link href="/">Open Workspace</Link>
                          </Button>
                        }
                      />
                    </div>
                  ) : (
                    <DataTable
                      rows={visibleRows}
                      columns={columns}
                      rowKey={(row) => row.id}
                      rowHref={(row) => `/history/${row.id}`}
                    />
                  )}
                </CardContent>
              </Card>

              {!loading && hasMore ? (
                <div className="flex items-center justify-center pt-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleShowMore}
                    disabled={loadingMore}
                  >
                    {loadingMore
                      ? "Loading…"
                      : `Show more (${items.length} of ${total})`}
                  </Button>
                </div>
              ) : null}
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
