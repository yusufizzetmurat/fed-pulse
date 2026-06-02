import * as React from "react";

import {
  fetchEvaluationCoverage,
  fetchFomcCalendar,
  fetchHarBaselines,
  fetchHistory,
  fetchSymbols,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { SYMBOL_OPTIONS } from "@/lib/analyze/constants";
import type {
  EvaluationCoverageResponse,
  FomcCalendarResponse,
  HarTercileBaselineResponse,
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

interface HarBaselinesState {
  data: HarTercileBaselineResponse | null;
  loading: boolean;
  error: string | null;
}

interface SharedContextValue {
  apiBaseUrl: string;
  symbols: SymbolsState;
  calendar: CalendarState;
  coverageMap: Record<string, CoverageState>;
  recentHistoryMap: Record<string, RecentHistoryState>;
  harBaselinesMap: Record<string, HarBaselinesState>;
  ensureCoverage: (symbol: string | undefined) => void;
  ensureRecentHistory: (symbol: string | undefined, limit: number) => void;
  refreshRecentHistory: (symbol: string | undefined, limit: number) => void;
  ensureHarBaselines: (symbol: string | undefined) => void;
}

const SharedContext = React.createContext<SharedContextValue | null>(null);

const DEFAULT_RECENT_HISTORY_LIMIT = 12;

const LOADING_COVERAGE: CoverageState = { data: null, loading: true, error: null };
const LOADING_RECENT_HISTORY: RecentHistoryState = {
  data: null,
  loading: true,
  error: null,
};
const LOADING_HAR_BASELINES: HarBaselinesState = {
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
  const [harBaselinesMap, setHarBaselinesMap] = React.useState<
    Record<string, HarBaselinesState>
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
      // Release the guard on unmount so React StrictMode's intentional
      // double-mount (or a future apiBaseUrl change) re-fires the fetch
      // and lands a fresh result. Without this the first mount's
      // controller is aborted before its .then() can call setSymbols and
      // the second mount short-circuits at the guard, leaving symbols
      // stuck in its initial loading state for the rest of the session.
      inFlight.current.delete(key);
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
      // Same StrictMode-double-mount fix as the symbols effect above:
      // release the guard so the second mount actually fires the fetch
      // and lands a fresh result on calendar state. Without this the
      // upcoming-FOMC chip stays blank on every non-Workspace page.
      inFlight.current.delete(key);
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

  const ensureHarBaselines = React.useCallback(
    (symbol: string | undefined) => {
      const symKey = symbol ?? "";
      const key = `har_baselines:${apiBaseUrl}:${symKey}`;
      if (inFlight.current.has(key)) return;
      inFlight.current.add(key);
      setHarBaselinesMap((prev) =>
        prev[symKey] ? prev : { ...prev, [symKey]: LOADING_HAR_BASELINES },
      );
      fetchHarBaselines(apiBaseUrl, symbol ?? "")
        .then((data) => {
          setHarBaselinesMap((prev) => ({
            ...prev,
            [symKey]: { data, loading: false, error: null },
          }));
          // Only clear the in-flight guard on success so a symbol that
          // has already returned can still skip the duplicate fetch on
          // re-render. On error the guard stays set for the session
          // lifetime to prevent an effect that re-runs on every state
          // change from thrashing the endpoint; the user can recover
          // by changing the symbol or reloading the page.
          inFlight.current.delete(key);
        })
        .catch((err) => {
          setHarBaselinesMap((prev) => ({
            ...prev,
            [symKey]: {
              data: null,
              loading: false,
              error: (err as Error).message || "HAR baselines fetch failed",
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
          // Mirror ensureHarBaselines: only clear the in-flight guard on
          // success so a follow-up refresh (e.g., after /analyze persists
          // a new row) can re-fetch instead of short-circuiting at the
          // dedupe gate. On error the guard stays set for the session
          // lifetime to coalesce retries and prevent an effect that
          // re-runs on every state change from thrashing the endpoint;
          // the user can recover by changing the symbol or reloading.
          inFlight.current.delete(key);
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
          // Intentionally do NOT delete the in-flight guard here; see the
          // success-path comment above for the rationale.
        });
    },
    [apiBaseUrl],
  );

  const refreshRecentHistory = React.useCallback(
    (symbol: string | undefined, limit: number) => {
      const symKey = symbol ?? "";
      const mapKey = `${symKey}|${limit}`;
      const key = `history:${apiBaseUrl}:${mapKey}`;
      inFlight.current.delete(key);
      ensureRecentHistory(symbol, limit);
    },
    [apiBaseUrl, ensureRecentHistory],
  );

  const value = React.useMemo<SharedContextValue>(
    () => ({
      apiBaseUrl,
      symbols,
      calendar,
      coverageMap,
      recentHistoryMap,
      harBaselinesMap,
      ensureCoverage,
      ensureRecentHistory,
      refreshRecentHistory,
      ensureHarBaselines,
    }),
    [
      apiBaseUrl,
      symbols,
      calendar,
      coverageMap,
      recentHistoryMap,
      harBaselinesMap,
      ensureCoverage,
      ensureRecentHistory,
      refreshRecentHistory,
      ensureHarBaselines,
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
  const ctx = React.useContext(SharedContext);
  // Provider presence is fixed per mount, so the hooks below run in a
  // stable order. Production paths always have the provider and the
  // fetch effect is a no-op; tests render standalone and use this
  // fallback to avoid wrapping every test in SymbolCalendarProvider.
  const apiBaseUrl = React.useMemo(
    () => (ctx ? ctx.apiBaseUrl : resolveApiBaseUrl()),
    [ctx],
  );
  const [fallback, setFallback] = React.useState<SymbolsState>({
    symbols: STATIC_SYMBOL_FALLBACK,
    loading: true,
    error: null,
  });
  React.useEffect(() => {
    if (ctx) return;
    const controller = new AbortController();
    fetchSymbols(apiBaseUrl, controller.signal)
      .then((response: SymbolListResponse) => {
        if (controller.signal.aborted) return;
        const list =
          response.symbols.length > 0 ? response.symbols : STATIC_SYMBOL_FALLBACK;
        setFallback({ symbols: list, loading: false, error: null });
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setFallback({
          symbols: STATIC_SYMBOL_FALLBACK,
          loading: false,
          error: (err as Error).message || "Failed to load symbols.",
        });
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, ctx]);
  return ctx ? ctx.symbols : fallback;
}

export function useSharedCalendar(): CalendarState {
  return useSharedContext().calendar;
}

export function useSharedCoverage(symbol: string | undefined): CoverageState {
  const ctx = useSharedContext();
  const symKey = symbol ?? "";
  const { ensureCoverage } = ctx;
  React.useEffect(() => {
    ensureCoverage(symbol);
    // Pull the stable callback out of ctx so the effect re-runs only when
    // symbol actually changes, not on every provider re-render — matches
    // the pattern used by useHarBaselines below.
  }, [ensureCoverage, symbol]);
  return ctx.coverageMap[symKey] ?? LOADING_COVERAGE;
}

export function useHarBaselines(symbol: string | undefined): HarBaselinesState {
  const ctx = useSharedContext();
  const symKey = symbol ?? "";
  const { ensureHarBaselines } = ctx;
  React.useEffect(() => {
    ensureHarBaselines(symbol);
    // ensureHarBaselines is a stable useCallback in the provider; pulling
    // it out of `ctx` lets the effect re-run only when the symbol
    // actually changes, instead of every time the provider re-renders.
  }, [ensureHarBaselines, symbol]);
  return ctx.harBaselinesMap[symKey] ?? LOADING_HAR_BASELINES;
}

export function useSharedRecentHistory(
  symbol: string | undefined,
  limit: number = DEFAULT_RECENT_HISTORY_LIMIT,
): RecentHistoryState {
  const ctx = useSharedContext();
  const symKey = symbol ?? "";
  const mapKey = `${symKey}|${limit}`;
  const { ensureRecentHistory } = ctx;
  React.useEffect(() => {
    ensureRecentHistory(symbol, limit);
    // Pull the stable callback out of ctx so the effect re-runs only on
    // (symbol, limit) change. Putting `ctx` in deps thrashes the endpoint
    // because every fetch updates state inside the provider, which gives
    // ctx a new reference, which re-fires the effect, which re-fetches.
  }, [ensureRecentHistory, symbol, limit]);
  return ctx.recentHistoryMap[mapKey] ?? LOADING_RECENT_HISTORY;
}
