import * as React from "react";

import {
  fetchEvaluationCoverage,
  fetchFomcCalendar,
  fetchHistory,
  fetchSymbols,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { SYMBOL_OPTIONS } from "@/lib/analyze/constants";
import type {
  EvaluationCoverageResponse,
  FomcCalendarResponse,
  HistoryList,
  SymbolDescriptor,
  SymbolListResponse,
} from "@/lib/analyze/types";

// Workspace + status-bar previously each fetched /symbols,
// /fomc/calendar, /evaluation/coverage and the recent /history slice
// on mount. With multiple consumers and StrictMode the triplet
// produced 3x request fan-outs on first paint. This context fetches
// each endpoint once per (baseUrl[, symbol]) and exposes the result
// to every consumer.

const STATIC_SYMBOL_FALLBACK: SymbolDescriptor[] = SYMBOL_OPTIONS.map((entry) => ({
  symbol: entry.value,
  name: entry.label.replace(/\s*\([^)]*\)\s*$/, "").trim() || entry.value,
  category: "Asset",
  default_horizon: "10d",
}));

interface SymbolsState {
  symbols: SymbolDescriptor[];
  loading: boolean;
  error: string | null;
}

interface CalendarState {
  data: FomcCalendarResponse | null;
  loading: boolean;
  error: string | null;
}

interface CoverageState {
  data: EvaluationCoverageResponse | null;
  loading: boolean;
  error: string | null;
}

interface RecentHistoryState {
  data: HistoryList | null;
  loading: boolean;
  error: string | null;
}

interface SharedContextValue {
  apiBaseUrl: string;
  symbols: SymbolsState;
  calendar: CalendarState;
  coverageMap: Record<string, CoverageState>;
  recentHistoryMap: Record<string, RecentHistoryState>;
  ensureCoverage: (symbol: string | undefined) => void;
  ensureRecentHistory: (symbol: string | undefined, limit: number) => void;
}

const SharedContext = React.createContext<SharedContextValue | null>(null);

const DEFAULT_RECENT_HISTORY_LIMIT = 12;

const LOADING_COVERAGE: CoverageState = { data: null, loading: true, error: null };
const LOADING_RECENT_HISTORY: RecentHistoryState = {
  data: null,
  loading: true,
  error: null,
};

export function SymbolCalendarProvider({ children }: { children: React.ReactNode }) {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);

  const [symbols, setSymbols] = React.useState<SymbolsState>({
    symbols: STATIC_SYMBOL_FALLBACK,
    loading: true,
    error: null,
  });
  const [calendar, setCalendar] = React.useState<CalendarState>({
    data: null,
    loading: true,
    error: null,
  });
  const [coverageMap, setCoverageMap] = React.useState<Record<string, CoverageState>>({});
  const [recentHistoryMap, setRecentHistoryMap] = React.useState<
    Record<string, RecentHistoryState>
  >({});

  // Guard against StrictMode double-invoke + cross-consumer races by
  // tracking which keys are already in flight. The ref survives the
  // effect cleanup so the second strict-mode pass does not re-issue.
  const inFlight = React.useRef<Set<string>>(new Set());

  React.useEffect(() => {
    const key = `symbols:${apiBaseUrl}`;
    if (inFlight.current.has(key)) return;
    inFlight.current.add(key);
    const controller = new AbortController();
    fetchSymbols(apiBaseUrl, controller.signal)
      .then((response: SymbolListResponse) => {
        if (controller.signal.aborted) return;
        const list =
          response.symbols.length > 0 ? response.symbols : STATIC_SYMBOL_FALLBACK;
        setSymbols({ symbols: list, loading: false, error: null });
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setSymbols({
          symbols: STATIC_SYMBOL_FALLBACK,
          loading: false,
          error: (err as Error).message || "Failed to load symbols.",
        });
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl]);

  React.useEffect(() => {
    const key = `calendar:${apiBaseUrl}`;
    if (inFlight.current.has(key)) return;
    inFlight.current.add(key);
    const controller = new AbortController();
    fetchFomcCalendar(apiBaseUrl, { upcoming_limit: 4, past_limit: 0 }, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setCalendar({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setCalendar({
          data: null,
          loading: false,
          error: (err as Error).message || "calendar fetch failed",
        });
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl]);

  const ensureCoverage = React.useCallback(
    (symbol: string | undefined) => {
      const symKey = symbol ?? "";
      const key = `coverage:${apiBaseUrl}:${symKey}`;
      if (inFlight.current.has(key)) return;
      inFlight.current.add(key);
      setCoverageMap((prev) =>
        prev[symKey] ? prev : { ...prev, [symKey]: LOADING_COVERAGE },
      );
      fetchEvaluationCoverage(apiBaseUrl, { symbol })
        .then((data) => {
          setCoverageMap((prev) => ({
            ...prev,
            [symKey]: { data, loading: false, error: null },
          }));
        })
        .catch((err) => {
          setCoverageMap((prev) => ({
            ...prev,
            [symKey]: {
              data: null,
              loading: false,
              error: (err as Error).message || "coverage fetch failed",
            },
          }));
        });
    },
    [apiBaseUrl],
  );

  const ensureRecentHistory = React.useCallback(
    (symbol: string | undefined, limit: number) => {
      const symKey = symbol ?? "";
      const mapKey = `${symKey}|${limit}`;
      const key = `history:${apiBaseUrl}:${mapKey}`;
      if (inFlight.current.has(key)) return;
      inFlight.current.add(key);
      setRecentHistoryMap((prev) =>
        prev[mapKey] ? prev : { ...prev, [mapKey]: LOADING_RECENT_HISTORY },
      );
      fetchHistory(apiBaseUrl, { symbol, limit })
        .then((data) => {
          setRecentHistoryMap((prev) => ({
            ...prev,
            [mapKey]: { data, loading: false, error: null },
          }));
        })
        .catch((err) => {
          setRecentHistoryMap((prev) => ({
            ...prev,
            [mapKey]: {
              data: null,
              loading: false,
              error: (err as Error).message || "history fetch failed",
            },
          }));
        });
    },
    [apiBaseUrl],
  );

  const value = React.useMemo<SharedContextValue>(
    () => ({
      apiBaseUrl,
      symbols,
      calendar,
      coverageMap,
      recentHistoryMap,
      ensureCoverage,
      ensureRecentHistory,
    }),
    [
      apiBaseUrl,
      symbols,
      calendar,
      coverageMap,
      recentHistoryMap,
      ensureCoverage,
      ensureRecentHistory,
    ],
  );

  return <SharedContext.Provider value={value}>{children}</SharedContext.Provider>;
}

export function useSharedContext(): SharedContextValue {
  const ctx = React.useContext(SharedContext);
  if (!ctx) {
    throw new Error("useSharedContext must be used within a SymbolCalendarProvider");
  }
  return ctx;
}

export function useSharedSymbols(): SymbolsState {
  return useSharedContext().symbols;
}

export function useSharedCalendar(): CalendarState {
  return useSharedContext().calendar;
}

export function useSharedCoverage(symbol: string | undefined): CoverageState {
  const ctx = useSharedContext();
  const symKey = symbol ?? "";
  React.useEffect(() => {
    ctx.ensureCoverage(symbol);
  }, [ctx, symbol]);
  return ctx.coverageMap[symKey] ?? LOADING_COVERAGE;
}

export function useSharedRecentHistory(
  symbol: string | undefined,
  limit: number = DEFAULT_RECENT_HISTORY_LIMIT,
): RecentHistoryState {
  const ctx = useSharedContext();
  const symKey = symbol ?? "";
  const mapKey = `${symKey}|${limit}`;
  React.useEffect(() => {
    ctx.ensureRecentHistory(symbol, limit);
  }, [ctx, symbol, limit]);
  return ctx.recentHistoryMap[mapKey] ?? LOADING_RECENT_HISTORY;
}
