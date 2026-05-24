export type ForecastMode = "fast" | "quick_train" | "real_train";
export type Horizon = "1d" | "3d" | "5d" | "10d";
export type SymbolValue = string;

export interface AnalyzeRequest {
  text: string;
  date: string;
  symbol: SymbolValue;
  // forecast_mode is retired from the frontend (#265). Field is optional
  // so older history rows still type-check; new requests omit it and the
  // backend defaults to "fast".
  forecast_mode?: ForecastMode;
  horizon: Horizon;
  include_realized: boolean;
  include_xai?: boolean;
  // Counterfactual: 0-based indices of sentences to drop from the text
  // before running the pipeline. Empty / omitted = no mask.
  mask_sentence_indices?: number[];
}

export interface SentimentResponse {
  label?: string | null;
  score?: number | null;
  ood_energy?: number | null;
  ood_threshold?: number | null;
  is_in_distribution?: boolean | null;
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
  // Encoder alias backing the multi-axis classifier (e.g.
  // "finbert_fed_adjacent"). Absent when no multi-axis checkpoint is loaded.
  encoder_key?: string | null;
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
  forecast_band_source?: "gaussian_z" | "conformal" | null;
  conformal_coverage?: number | null;
}

export interface HistoryRealizedResponse {
  run_id: string;
  symbol: string;
  document_date: string;
  horizon: string;
  timestamps: string[];
  close: number[];
  volatility: number[];
  realized_regime?: string | null;
}

export type StanceAxis = "hawkish" | "dovish" | "neutral";
export type CertaintyAxis = "certain" | "uncertain" | "neutral";
export type TopicAxis = "macro" | "forward_guidance" | "market_reaction" | "other";

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
  distribution?: Partial<Record<CertaintyAxis, number>>;
}

export interface MultiAxisTopic {
  label: TopicAxis | string;
  confidence: number;
  distribution?: Partial<Record<string, number>>;
  // Back-compat aliases that older fixtures use. ``primary`` mirrors
  // ``label`` and ``secondary`` lists alternate topics the model
  // considered. New code should prefer ``label`` + ``distribution``.
  primary?: string;
  secondary?: string[];
}

export interface MultiAxisResponse {
  stance: MultiAxisStance | null;
  factor: MultiAxisFactor | null;
  certainty: MultiAxisCertainty | null;
  topic: MultiAxisTopic | null;
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
  drift_score: number;
  drift_trend?: number[];
  realized_vs_stated_gap?: number;
  market_implied_gap?: number;
  months_since_reversal?: number;
}

export interface RegimeClassificationResponse {
  predicted_set: string[];
  set_label: string;
  set_size: number;
  coverage: number;
  distribution: Record<string, number>;
  argmax_class: string;
}

export interface AnalyzeResult {
  sentiment?: SentimentResponse;
  prediction?: PredictionResponse;
  market?: MarketResponse;
  model?: ModelDiagnosticsResponse;
  series?: SeriesResponse;
  multi_axis?: MultiAxisResponse;
  regime_classification?: RegimeClassificationResponse | null;
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
  argmax_regime?: string | null;
  argmax_probability?: number | null;
  regime_set_size?: number | null;
}

export interface HistoryDetail extends HistoryEntry {
  payload: Record<string, unknown>;
}

export interface SymbolDescriptor {
  symbol: string;
  name: string;
  category: string;
  default_horizon: string;
}

export interface SymbolListResponse {
  symbols: SymbolDescriptor[];
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

// ---------------------------------------------------------------------------
// Research dashboard (Phase 8 multi-page expansion)
// ---------------------------------------------------------------------------

export interface ArtifactFile {
  relative_path: string;
  size_bytes: number;
  modified_at: string;
  suffix: string;
}

export interface EncoderBakeoffRow {
  encoder_key: string;
  checkpoint: string;
  seeds: number[];
  macro_f1_values: number[];
  macro_f1_mean: number;
  macro_f1_ci_low: number | null;
  macro_f1_ci_high: number | null;
  weighted_f1_mean: number | null;
  accuracy_mean: number | null;
  cohen_kappa: number | null;
}

export interface EncoderBakeoffSection {
  available: boolean;
  coverage: number | null;
  rows: EncoderBakeoffRow[];
  source_files: string[];
}

export interface TransferMatrixCell {
  source: string;
  target: string;
  metric: number;
}

export interface CrossBankTransferSection {
  available: boolean;
  metric_name: string;
  sources: string[];
  targets: string[];
  cells: TransferMatrixCell[];
  source_files: string[];
}

export interface ResearchArtifactsResponse {
  artifacts_root: string;
  sections: Record<string, ArtifactFile[]>;
  encoder_bakeoff: EncoderBakeoffSection;
  cross_bank_transfer: CrossBankTransferSection;
}

// ---------------------------------------------------------------------------
// Training dashboard
// ---------------------------------------------------------------------------

export type TrainJobStatus = "queued" | "running" | "succeeded" | "failed";

export interface TrainJobSummary {
  job_id: string;
  status: TrainJobStatus | string;
  symbol: string | null;
  date: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  history_length: number | null;
  error: string | null;
}

export interface TrainJobsListResponse {
  items: TrainJobSummary[];
  total: number;
  limit: number;
  offset: number;
}

// ---------------------------------------------------------------------------
// Decisions dashboard
// ---------------------------------------------------------------------------

export type OrdinalDecisionClass =
  | "cut_50"
  | "cut_25"
  | "hold"
  | "hike_25"
  | "hike_50"
  | "hike_75";

export interface NextFomcMeetingPrediction {
  target_event_date: string;
  target_as_of_ts: string;
  target_class: string | null;
  n_train_rows: number;
  probabilities: Record<string, Record<string, number>>;
  predicted_class: Record<string, string>;
}

export interface NextFomcModelMetrics {
  n: number;
  brier: number | null;
  log_loss: number | null;
  top1_accuracy: number | null;
  macro_f1: number | null;
  confusion_matrix: Record<string, Record<string, number>>;
}

export interface NextFomcAttributionRow {
  subset: string;
  families: string[];
  n_features: number | null;
  n: number | null;
  brier: number | null;
  log_loss: number | null;
  top1_accuracy: number | null;
  macro_f1: number | null;
}

export interface NextFomcUpcomingMeeting {
  meeting_date: string;
  meeting_type: string;
  statement_release_date: string | null;
  days_until: number | null;
}

export interface NextFomcForecastResponse {
  available: boolean;
  artifacts_dir: string;
  ordinal_classes: string[];
  model_names: string[];
  upcoming_meeting: NextFomcUpcomingMeeting | null;
  headline: NextFomcMeetingPrediction | null;
  history: NextFomcMeetingPrediction[];
  metrics_full_window: Record<string, NextFomcModelMetrics>;
  metrics_ex_pandemic: Record<string, NextFomcModelMetrics>;
  feature_attribution: NextFomcAttributionRow[];
  summary: Record<string, number>;
}
