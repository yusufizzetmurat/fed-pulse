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

export type StanceAxis = "hawkish" | "dovish" | "neutral";
export type CertaintyAxis = "tentative" | "measured" | "decisive";

export interface MultiAxisStance {
  label: StanceAxis;
  confidence: number;
  distribution?: Partial<Record<StanceAxis, number>>;
}

export interface MultiAxisFactor {
  value: number;
  confidence: number;
  range?: [number, number];
}

export interface MultiAxisCertainty {
  label: CertaintyAxis;
  confidence: number;
}

export interface MultiAxisTopic {
  primary: string;
  confidence: number;
  secondary?: string[];
}

export interface MultiAxisResponse {
  stance?: MultiAxisStance;
  factor?: MultiAxisFactor;
  certainty?: MultiAxisCertainty;
  topic?: MultiAxisTopic;
}

export interface XaiTokenAttribution {
  token: string;
  weight: number;
}

export interface XaiSentence {
  text: string;
  score: number;
  topTokens: XaiTokenAttribution[];
}

export interface XaiResponse {
  sentences: XaiSentence[];
  method?: string;
}

export interface CredibilityResponse {
  driftScore: number;
  driftTrend?: number[];
  realizedVsStatedGap?: number;
  marketImpliedGap?: number;
  monthsSinceReversal?: number;
}

export interface AnalyzeResult {
  sentiment?: SentimentResponse;
  prediction?: PredictionResponse;
  market?: MarketResponse;
  model?: ModelDiagnosticsResponse;
  series?: SeriesResponse;
  multi_axis?: MultiAxisResponse;
  xai?: XaiResponse;
  credibility?: CredibilityResponse;
}

export interface TrainJobState {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  message?: string;
  error?: string | null;
  result?: AnalyzeResult | null;
}

export type Stance = "hawkish" | "dovish" | "neutral" | "unknown";

export interface HistoryEntry {
  id: string;
  created_at: string;
  symbol: string;
  document_date: string;
  horizon: string;
  forecast_mode: string;
  stance: string;
  sentiment_score?: number | null;
  predicted_close?: number | null;
  current_close?: number | null;
  predicted_volatility?: number | null;
  text_excerpt?: string | null;
}

export interface HistoryDetail extends HistoryEntry {
  payload: Record<string, unknown>;
}

export interface HistoryList {
  items: HistoryEntry[];
  total: number;
  limit: number;
  offset: number;
}

export interface HistoryQuery {
  symbol?: string;
  horizon?: string;
  stance?: string;
  document_date?: string;
  limit?: number;
  offset?: number;
}

export interface FomcMeeting {
  meeting_date: string;
  meeting_type: string;
  statement_release_date?: string | null;
  minutes_release_date?: string | null;
  notes?: string | null;
}

export interface FomcCalendarResponse {
  past: FomcMeeting[];
  upcoming: FomcMeeting[];
}
