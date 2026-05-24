import * as React from "react";

import { fetchEvaluationCoverage } from "@/lib/analyze/api";
import type { EvaluationCoverageResponse } from "@/lib/analyze/types";

interface UseEvaluationCoverageOptions {
  symbol?: string;
  lookbackRuns?: number;
}

interface UseEvaluationCoverageState {
  data: EvaluationCoverageResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * Polls /evaluation/coverage so the workspace can surface a "Nominal X% /
 * Empirical Y%" chip alongside the regime headline. Failures are
 * intentionally silent — the chip is informational and the workspace
 * should not flash an error toast when the aggregation has nothing to
 * show yet. Backend caches its own aggregation for 5 minutes; the hook
 * does not poll.
 */
export function useEvaluationCoverage(
  apiBaseUrl: string,
  options: UseEvaluationCoverageOptions = {},
): UseEvaluationCoverageState {
  const { symbol, lookbackRuns } = options;
  const [state, setState] = React.useState<UseEvaluationCoverageState>({
    data: null,
    loading: true,
    error: null,
  });

  React.useEffect(() => {
    const controller = new AbortController();
    setState((prev) => ({ ...prev, loading: true }));
    fetchEvaluationCoverage(
      apiBaseUrl,
      { symbol, lookback_runs: lookbackRuns },
      controller.signal,
    )
      .then((data) => {
        if (controller.signal.aborted) return;
        setState({ data, loading: false, error: null });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const message = err instanceof Error ? err.message : "coverage fetch failed";
        setState({ data: null, loading: false, error: message });
      });
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, symbol, lookbackRuns]);

  return state;
}
