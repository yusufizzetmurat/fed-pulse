import axios from "axios";
import type { AnalyzeRequest, AnalyzeResult, TrainJobState } from "./types";

export function resolveApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  if (typeof window !== "undefined" && raw.includes("://backend:")) {
    return raw.replace("://backend:", "://localhost:");
  }
  return raw;
}

export type AnalyzeResponse =
  | { mode: "result"; result: AnalyzeResult }
  | { mode: "queued"; job: TrainJobState };

export async function postAnalyze(
  baseUrl: string,
  request: AnalyzeRequest
): Promise<AnalyzeResponse> {
  const response = await axios.post(`${baseUrl}/analyze`, request);
  const data = response.data || {};
  if (request.forecast_mode === "real_train") {
    const jobId = (data as { job_id?: string }).job_id;
    if (!jobId) throw new Error("Real Train did not return a job id.");
    return {
      mode: "queued",
      job: {
        job_id: jobId,
        status: (data as { status?: TrainJobState["status"] }).status || "queued",
        message: (data as { message?: string }).message || "Real Train queued.",
        error: null,
      },
    };
  }
  return { mode: "result", result: data as AnalyzeResult };
}

export async function fetchTrainJob(baseUrl: string, jobId: string): Promise<TrainJobState> {
  const response = await axios.get(`${baseUrl}/train-jobs/${jobId}`);
  const state = (response.data || {}) as TrainJobState;
  return { ...state, job_id: jobId };
}
