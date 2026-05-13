export type ForecastMode = "fast" | "quick_train" | "real_train";
export type Horizon = "1d" | "3d" | "5d" | "10d";
export type SymbolValue = string;

export interface AnalyzeRequest {
  text: string;
  date: string;
  symbol: SymbolValue;
  forecast_mode: ForecastMode;
  horizon: Horizon;
  include_realized: boolean;
}

export interface SentimentResponse {
  label?: string | null;
  score?: number | null;
}

export interface PredictionResponse {
  close?: number | null;
  volatility?: number | null;
  horizon?: string | null;
}

export interface MarketResponse {
  symbol?: string;
  requested_date?: string;
  date_used?: string;
  close?: number | null;
  volatility_5d?: number | null;
}

export interface ChunkAttentionResponse {
  weights?: number[];
  decay_coeffs?: number[];
  chunk_previews?: string[];
  chunk_count?: number;
  lambda_value?: number;
}

export interface ModelDiagnosticsResponse {
  hidden_size?: number;
  num_layers?: number;
  dropout?: number;
  combined_rmse?: number;
  checkpoint_loaded?: boolean;
  runtime_mode?: string;
  chunk_attention?: ChunkAttentionResponse | null;
}

export interface SeriesResponse {
  timestamps?: string[];
  history_close?: number[];
  history_volatility?: number[];
  forecast_timestamps?: string[];
  forecast_close?: number[];
  forecast_close_lower?: number[];
  forecast_close_upper?: number[];
  forecast_volatility?: number[];
  forecast_volatility_lower?: number[];
  forecast_volatility_upper?: number[];
  realized_timestamps?: string[];
  realized_close?: number[];
  realized_volatility?: number[];
  forecast_confidence_level?: number;
  volatility_scale?: { suggested_ymin?: number; suggested_ymax?: number };
}

export interface AnalyzeResult {
  sentiment?: SentimentResponse;
  prediction?: PredictionResponse;
  market?: MarketResponse;
  model?: ModelDiagnosticsResponse;
  series?: SeriesResponse;
}

export interface TrainJobState {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  message?: string;
  error?: string | null;
  result?: AnalyzeResult | null;
}

export type Stance = "hawkish" | "dovish" | "neutral" | "unknown";
