import axios from "axios";
import type {
  AnalyzeRequest,
  AnalyzeResult,
  FomcCalendarResponse,
  HistoryDetail,
  HistoryList,
  HistoryQuery,
  HistoryRealizedResponse,
  NextFomcForecastResponse,
  ResearchArtifactsResponse,
  TrainJobState,
  TrainJobSummary,
  TrainJobsListResponse,
} from "./types";

export function resolveApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  if (typeof window !== "undefined" && raw.includes("://backend:")) {
    return raw.replace("://backend:", "://localhost:");
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

export async function fetchTrainJob(baseUrl: string, jobId: string): Promise<TrainJobState> {
  const response = await axios.get(`${baseUrl}/train-jobs/${jobId}`);
  const state = (response.data || {}) as TrainJobState;
  return { ...state, job_id: jobId };
}

export async function fetchHistory(
  baseUrl: string,
  query: HistoryQuery = {}
): Promise<HistoryList> {
  const response = await axios.get(`${baseUrl}/history`, { params: query });
  return response.data as HistoryList;
}

export async function fetchHistoryRun(
  baseUrl: string,
  runId: string
): Promise<HistoryDetail> {
  const response = await axios.get(`${baseUrl}/history/${runId}`);
  return response.data as HistoryDetail;
}

export async function deleteHistoryRun(baseUrl: string, runId: string): Promise<void> {
  await axios.delete(`${baseUrl}/history/${runId}`);
}

export async function fetchHistoryRealized(
  baseUrl: string,
  runId: string
): Promise<HistoryRealizedResponse> {
  const response = await axios.get(`${baseUrl}/history/${runId}/realized`);
  return response.data as HistoryRealizedResponse;
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

export async function fetchFomcCalendar(
  baseUrl: string,
  params?: { upcoming_limit?: number; past_limit?: number; as_of?: string }
): Promise<FomcCalendarResponse> {
  const response = await axios.get(`${baseUrl}/fomc/calendar`, { params });
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
