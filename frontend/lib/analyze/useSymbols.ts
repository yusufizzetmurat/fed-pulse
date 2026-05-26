import * as React from "react";

import { fetchSymbols, resolveApiBaseUrl } from "@/lib/analyze/api";
import { SYMBOL_OPTIONS } from "@/lib/analyze/constants";
import type { SymbolDescriptor } from "@/lib/analyze/types";

interface UseSymbolsState {
  symbols: SymbolDescriptor[];
  loading: boolean;
  error: string | null;
}

const STATIC_FALLBACK: SymbolDescriptor[] = SYMBOL_OPTIONS.map((entry) => ({
  symbol: entry.value,
  name: entry.label.replace(/\s*\([^)]*\)\s*$/, "").trim() || entry.value,
  category: "Asset",
  default_horizon: "10d",
}));

/**
 * Live symbol universe for the workspace asset picker. Falls back to the
 * static constants list when /symbols is unreachable so the form keeps
 * working in offline / cold-start dev environments.
 */
export function useSymbols(): UseSymbolsState {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [state, setState] = React.useState<UseSymbolsState>({
    symbols: STATIC_FALLBACK,
    loading: true,
    error: null,
  });

  React.useEffect(() => {
    const controller = new AbortController();
    fetchSymbols(apiBaseUrl, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        const symbols = response.symbols.length > 0 ? response.symbols : STATIC_FALLBACK;
        setState({ symbols, loading: false, error: null });
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setState({
          symbols: STATIC_FALLBACK,
          loading: false,
          error: (err as Error).message || "Failed to load symbols.",
        });
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl]);

  return state;
}
