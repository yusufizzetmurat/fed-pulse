import axios from "axios";
import type {
  AnalogsRequest,
  AnalogsResponse,
  AnalyzeRequest,
  AnalyzeResult,
  BacktestPositionEntry,
  BacktestResponse,
  ClassificationBreakdownResponse,
  DocumentDetailResponse,
  EvaluationCoverageResponse,
  ExpectedVolumeForecastResponse,
  FomcCalendarResponse,
  FuturesConsensusResponse,
  HarTercileBacktestResponse,
  HarTercileBaselineResponse,
  HistoryDetail,
  HistoryEventStudyResponse,
  HistoryList,
  HistoryQuery,
  HistoryRealizedBatchResponse,
  HistoryRealizedResponse,
  MarketReactionPanelResponse,
  MonetaryPolicySurpriseResponse,
  NextFomcForecastResponse,
  RealizedVolForecastResponse,
  ResearchArtifactsResponse,
  ResearchRegistryResponse,
  RvBacktestResponse,
  SemanticDiffResponse,
  SettingsCheckpointsResponse,
  StanceContextResponse,
  SymbolListResponse,
  TrainJobState,
  TrainJobSummary,
  TrainJobsListResponse,
} from "./types";

export function resolveApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  // Production builds set NEXT_PUBLIC_API_URL to the public origin
  // (https://fedpulse.yusufizzetmurat.com). Caddy reverse-proxies
  // `/api/*` to the backend container after stripping the `/api`
  // prefix, so the browser-side axios call must include that prefix
  // for the production origin only. Local dev keeps the literal
  // localhost:8000 backend URL and the existing rewrite-from-compose
  // hostname guard.
  if (typeof window !== "undefined" && raw.includes("://backend:")) {
    return raw.replace("://backend:", "://localhost:");
  }
  try {
    const parsed = new URL(raw);
    if (parsed.protocol === "https:" && parsed.pathname === "/") {
      // ``https://host`` -> ``https://host/api``
      return raw.replace(/\/?$/, "") + "/api";
    }
  } catch {
    // Not a parseable URL — fall through to the raw value.
  }
  return raw;
}

export async function postAnalyze(
  baseUrl: string,
  request: AnalyzeRequest
): Promise<AnalyzeResult> {
  const response = await axios.post(`${baseUrl}/analyze`, request);
  if (!response.data) {
    // Silent empty-body fallbacks were the hardest class of "panel is
    // blank" bug to debug; log here so the network tab plus console
    // tell the same story.
    // eslint-disable-next-line no-console
    console.warn("postAnalyze: empty response body, returning {} fallback");
    return {} as AnalyzeResult;
  }
  return response.data as AnalyzeResult;
}

export async function postAnalyzeMarket(
  baseUrl: string,
  request: AnalyzeRequest
): Promise<MarketReactionPanelResponse> {
  const response = await axios.post(`${baseUrl}/analyze/market`, request);
  return (response.data || {
    rates: [],
    vol_regime: null,
    encoder_alias: null,
    checkpoint_path: null,
  }) as MarketReactionPanelResponse;
}

const ANALOGS_BUNDLE_ABSENT: AnalogsResponse = {
  analogs: [],
  index_size: 0,
  encoder_alias: "",
};

export async function postAnalyzeAnalogs(
  baseUrl: string,
  request: AnalogsRequest,
  signal?: AbortSignal,
): Promise<AnalogsResponse> {
  const response = await axios.post(`${baseUrl}/analyze/analogs`, request, { signal });
  return (response.data ?? ANALOGS_BUNDLE_ABSENT) as AnalogsResponse;
}

export async function fetchTrainJob(baseUrl: string, jobId: string): Promise<TrainJobState> {
  const response = await axios.get(`${baseUrl}/train-jobs/${jobId}`);
  const state = (response.data || {}) as TrainJobState;
  return { ...state, job_id: jobId };
}

export async function fetchHistory(
  baseUrl: string,
  query: HistoryQuery = {},
  signal?: AbortSignal,
): Promise<HistoryList> {
  const response = await axios.get(`${baseUrl}/history`, { params: query, signal });
  return response.data as HistoryList;
}

export async function fetchRecentStanceScores(
  baseUrl: string,
  params: {
    symbol: string;
    horizon?: string;
    n?: number;
    excludeRunId?: string;
  },
  signal?: AbortSignal,
): Promise<StanceContextResponse> {
  const response = await axios.get(`${baseUrl}/history/recent-stance-scores`, {
    params: {
      symbol: params.symbol,
      horizon: params.horizon,
      n: params.n,
      exclude_run_id: params.excludeRunId,
    },
    signal,
  });
  return response.data as StanceContextResponse;
}

export async function fetchHistoryRun(
  baseUrl: string,
  runId: string,
  signal?: AbortSignal,
): Promise<HistoryDetail> {
  const response = await axios.get(`${baseUrl}/history/${runId}`, { signal });
  return response.data as HistoryDetail;
}

export async function deleteHistoryRun(baseUrl: string, runId: string): Promise<void> {
  await axios.delete(`${baseUrl}/history/${runId}`);
}

export async function fetchHistoryRealized(
  baseUrl: string,
  runId: string,
  signal?: AbortSignal,
): Promise<HistoryRealizedResponse> {
  const response = await axios.get(`${baseUrl}/history/${runId}/realized`, { signal });
  return response.data as HistoryRealizedResponse;
}

export async function fetchHistoryEventStudy(
  baseUrl: string,
  runId: string,
  signal?: AbortSignal,
): Promise<HistoryEventStudyResponse> {
  const response = await axios.get(`${baseUrl}/history/${runId}/event-study`, { signal });
  return response.data as HistoryEventStudyResponse;
}

// Batched companion to ``fetchHistoryRealized``. The /history list page
// used to fan out N parallel per-row requests; this collapses them to
// one round trip. Deleted runs and yfinance failures land under
// ``missing`` so a single broken row does not nuke the table render.
//
// The backend caps ``ids`` at 50 per call (see ``/history-realized``);
// chunk locally so the caller can pass any number of ids without
// hitting a 422.
const HISTORY_REALIZED_CHUNK = 50;

export async function fetchHistoryRealizedBatch(
  baseUrl: string,
  runIds: readonly string[],
  signal?: AbortSignal,
): Promise<HistoryRealizedBatchResponse> {
  if (runIds.length === 0) {
    return { items: {}, missing: [] };
  }
  const chunks: string[][] = [];
  for (let i = 0; i < runIds.length; i += HISTORY_REALIZED_CHUNK) {
    chunks.push(runIds.slice(i, i + HISTORY_REALIZED_CHUNK) as string[]);
  }
  // ``allSettled`` so a transient failure on one chunk does not blank the
  // whole table — the surviving chunks' rows still render and the failed
  // chunk's ids fall under ``missing`` so the caller can show them as
  // unresolved.
  const settled = await Promise.allSettled(
    chunks.map((chunk) =>
      axios
        .get(`${baseUrl}/history-realized`, {
          params: { ids: chunk.join(",") },
          signal,
        })
        .then((r) => r.data as HistoryRealizedBatchResponse),
    ),
  );
  const items: HistoryRealizedBatchResponse["items"] = {};
  const missing: string[] = [];
  settled.forEach((result, idx) => {
    if (result.status === "fulfilled") {
      Object.assign(items, result.value.items);
      missing.push(...result.value.missing);
    } else {
      missing.push(...chunks[idx]);
    }
  });
  return { items, missing };
}

export async function fetchEvaluationCoverage(
  baseUrl: string,
  params?: { symbol?: string; lookback_runs?: number },
  signal?: AbortSignal,
): Promise<EvaluationCoverageResponse> {
  const response = await axios.get(`${baseUrl}/evaluation/coverage`, { params, signal });
  return response.data as EvaluationCoverageResponse;
}

export async function fetchClassificationBreakdown(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<ClassificationBreakdownResponse> {
  const response = await axios.get(`${baseUrl}/evaluation/classification-breakdown`, { signal });
  return response.data as ClassificationBreakdownResponse;
}

// Pair-helper for the /compare page. There is no dedicated backend endpoint;
// compare just fans out two history-detail reads in parallel. Kept separate
// from `fetchHistoryRun` so call sites that need the pair can swap to a
// future server-side endpoint without touching the page.
export async function compare(
  baseUrl: string,
  runIdA: string,
  runIdB: string,
): Promise<{ a: HistoryDetail; b: HistoryDetail }> {
  const [a, b] = await Promise.all([
    fetchHistoryRun(baseUrl, runIdA),
    fetchHistoryRun(baseUrl, runIdB),
  ]);
  return { a, b };
}

export async function fetchSymbols(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<SymbolListResponse> {
  const response = await axios.get(`${baseUrl}/symbols`, { signal });
  return response.data as SymbolListResponse;
}

export async function fetchSettingsCheckpoints(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<SettingsCheckpointsResponse> {
  const response = await axios.get(`${baseUrl}/settings/checkpoints`, { signal });
  return response.data as SettingsCheckpointsResponse;
}

export async function fetchFomcCalendar(
  baseUrl: string,
  params?: { upcoming_limit?: number; past_limit?: number; as_of?: string },
  signal?: AbortSignal,
): Promise<FomcCalendarResponse> {
  const response = await axios.get(`${baseUrl}/fomc/calendar`, { params, signal });
  return response.data as FomcCalendarResponse;
}

// Path-based document viewer fetcher. The backend 404s when the row
// isn't on disk; the page renders a tailored not-found state off the
// nullable return rather than threading an axios error through the
// generic toast path. Anything other than 404 propagates so the page
// can surface a 500 banner.
export async function fetchDocumentDetail(
  baseUrl: string,
  type: string,
  date: string,
  signal?: AbortSignal,
): Promise<DocumentDetailResponse | null> {
  try {
    const response = await axios.get(
      `${baseUrl}/documents/${encodeURIComponent(type)}/${encodeURIComponent(date)}`,
      { signal },
    );
    return response.data as DocumentDetailResponse;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return null;
    }
    throw err;
  }
}

export async function fetchResearchArtifacts(
  baseUrl: string
): Promise<ResearchArtifactsResponse> {
  const response = await axios.get(`${baseUrl}/research/artifacts`);
  return response.data as ResearchArtifactsResponse;
}

export async function fetchTrainJobs(
  baseUrl: string,
  params?: { status?: string; limit?: number; offset?: number }
): Promise<TrainJobsListResponse> {
  const response = await axios.get(`${baseUrl}/train-jobs`, { params });
  return response.data as TrainJobsListResponse;
}

export async function fetchTrainJobDetail(
  baseUrl: string,
  jobId: string
): Promise<TrainJobState & TrainJobSummary> {
  const response = await axios.get(`${baseUrl}/train-jobs/${jobId}`);
  return response.data as TrainJobState & TrainJobSummary;
}

export async function fetchNextFomcForecast(
  baseUrl: string
): Promise<NextFomcForecastResponse> {
  const response = await axios.get(`${baseUrl}/forecasts/next-fomc`);
  return response.data as NextFomcForecastResponse;
}

export async function fetchRealizedVolForecast(
  baseUrl: string,
  symbol: string = "^GSPC",
  signal?: AbortSignal,
): Promise<RealizedVolForecastResponse> {
  const response = await axios.get(`${baseUrl}/forecast/realized-vol`, {
    params: { symbol },
    signal,
  });
  return response.data as RealizedVolForecastResponse;
}

// Workspace-spine expected-volume forecast. Backend returns 503 when
// the HAR-volume artifact cannot be loaded or the volume history is
// insufficient. The fetcher re-throws the axios error so the caller's
// existing ``errorMessage`` path renders the "Model unavailable" copy
// (matched on the 503 status) instead of folding to a bare ``null``
// that leaves the card stuck on the ambiguous retry placeholder.
export async function fetchExpectedVolumeForecast(
  baseUrl: string,
  symbol: string = "^GSPC",
  signal?: AbortSignal,
): Promise<ExpectedVolumeForecastResponse> {
  const response = await axios.get(`${baseUrl}/forecast/abnormal-volume`, {
    params: { symbol },
    signal,
  });
  return response.data as ExpectedVolumeForecastResponse;
}


export async function fetchResearchRegistry(
  baseUrl: string,
  options?: { surface?: "dual" | "cls"; includeRejected?: boolean }
): Promise<ResearchRegistryResponse> {
  const params: Record<string, string | boolean> = {};
  if (options?.surface) params.surface = options.surface;
  if (options?.includeRejected) params.include_rejected = true;
  const response = await axios.get(`${baseUrl}/research/registry`, { params });
  return response.data as ResearchRegistryResponse;
}

// Workspace-spine MP-surprise chip. The backend returns 503 when the
// parquet artifact is missing; callers translate that into the chip's
// "unavailable" placeholder rather than surfacing a generic error.
export async function fetchLatestMpSurprise(
  baseUrl: string,
  signal?: AbortSignal,
): Promise<MonetaryPolicySurpriseResponse | null> {
  try {
    const response = await axios.get(`${baseUrl}/fomc/latest-mp-surprise`, {
      signal,
    });
    return response.data as MonetaryPolicySurpriseResponse;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 503) {
      return null;
    }
    throw err;
  }
}


// Workspace-spine FRED futures-consensus descriptive panel. The
// endpoint pulls the short-end DGS Treasury proxy and the current
// fed-funds target band off FRED; the backend returns 503 when FRED
// is unreachable or the FOMC calendar has no upcoming meeting on or
// after the as-of date. Callers translate the 503 into the panel's
// "unavailable" placeholder rather than surfacing a generic error.
export async function fetchFuturesConsensus(
  baseUrl: string,
  options?: { asOf?: string; signal?: AbortSignal },
): Promise<FuturesConsensusResponse | null> {
  const params: Record<string, string> = {};
  if (options?.asOf) params.as_of = options.asOf;
  try {
    const response = await axios.get(`${baseUrl}/fomc/futures-consensus`, {
      params,
      signal: options?.signal,
    });
    return response.data as FuturesConsensusResponse;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 503) {
      return null;
    }
    throw err;
  }
}


// Workspace-spine semantic-diff descriptive panel. The backend POST
// accepts the pasted statement body + its ISO date; the strict-prior
// statement is loaded server-side off the on-disk statements JSON.
// Cold-start (no strict-prior available) returns empty span and
// topic lists with an explanatory summary — the panel renders the
// banner-only mode in that case.
export async function fetchSemanticDiff(
  baseUrl: string,
  body: { current_date: string; current_text: string },
  signal?: AbortSignal,
): Promise<SemanticDiffResponse> {
  const response = await axios.post(`${baseUrl}/fomc/semantic-diff`, body, {
    signal,
  });
  return response.data as SemanticDiffResponse;
}


export async function fetchHarBaselines(
  baseUrl: string,
  symbol: string,
  signal?: AbortSignal,
): Promise<HarTercileBaselineResponse> {
  const response = await axios.get(`${baseUrl}/forecast/regime/baselines`, {
    params: { symbol },
    signal,
  });
  return response.data as HarTercileBaselineResponse;
}


// HAR-tercile backtest fetcher. The backend returns 503 only on a
// downstream artifact-load failure; the panel's empty-state covers the
// "no resolved runs yet" branch off ``rows.length === 0``, so a 503
// folds into ``null`` to match the parity contract shared with the
// other workspace-spine fetchers.
export async function fetchHarTercileBacktest(
  baseUrl: string,
  symbol: string,
  limit: number = 10,
  signal?: AbortSignal,
): Promise<HarTercileBacktestResponse | null> {
  try {
    const response = await axios.get(`${baseUrl}/forecast/har-tercile-backtest`, {
      params: { symbol, limit },
      signal,
    });
    return response.data as HarTercileBacktestResponse;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 503) {
      return null;
    }
    throw err;
  }
}

// QLIKE-RV backtest fetcher. Mirrors ``fetchHarTercileBacktest``: 503
// (model / history unavailable) folds into ``null`` so the panel
// renders the tailored "unavailable" placeholder rather than the
// generic error path. 400 (non-^GSPC symbol) propagates as an axios
// error so the caller can surface a tailored toast.
export async function fetchRvBacktest(
  baseUrl: string,
  symbol: string,
  limit: number = 10,
  signal?: AbortSignal,
): Promise<RvBacktestResponse | null> {
  try {
    const response = await axios.get(`${baseUrl}/forecast/rv-backtest`, {
      params: { symbol, limit },
      signal,
    });
    return response.data as RvBacktestResponse;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 503) {
      return null;
    }
    throw err;
  }
}

export async function postResearchBacktest(
  baseUrl: string,
  body: {
    positions: BacktestPositionEntry[];
    symbol?: string;
    horizon_days?: number;
  }
): Promise<BacktestResponse> {
  const response = await axios.post(`${baseUrl}/research/backtest`, body);
  return response.data as BacktestResponse;
}
