import axios from "axios";
import type {
  AnalogsRequest,
  AnalogsResponse,
  AnalyzeRequest,
  AnalyzeResult,
  BacktestPositionEntry,
  BacktestResponse,
  ClassificationBreakdownResponse,
  EvaluationCoverageResponse,
  FomcCalendarResponse,
  HistoryDetail,
  HistoryList,
  HistoryQuery,
  HistoryRealizedBatchResponse,
  HistoryRealizedResponse,
  MarketReactionPanelResponse,
  NextFomcForecastResponse,
  ResearchArtifactsResponse,
  ResearchRegistryResponse,
  SettingsCheckpointsResponse,
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
  return (response.data || {}) as AnalyzeResult;
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
