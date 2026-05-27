// Issue #336 dead-code sweep: the quick_train / real_train members are
// retired (the backend dropped both adaptation paths in #265). The
// union is kept narrow to the only runtime mode the /analyze handler
// still accepts. Persisted history rows can still carry stale strings
// via the `forecast_mode` column on HistoryEntry, which is typed as a
// plain `string` so the listing renders rows from before the sweep.
export type ForecastMode = "fast";
export type Horizon = "1d" | "3d" | "5d" | "10d";
export type SymbolValue = string;

export interface AnalyzeRequest {
  text: string;
  date: string;
  symbol: SymbolValue;
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

export interface HistoryRealizedBatchResponse {
  items: Record<string, HistoryRealizedResponse>;
  missing: string[];
}

export interface EvaluationCoverageResponse {
  nominal: number | null;
  empirical: number | null;
  sample_size: number;
  runs_total: number;
  computed_at: string;
}

export interface ClassificationBreakdownClass {
  class_id: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  roc_auc?: number | null;
  pr_auc?: number | null;
}

export interface ClassificationBreakdownSource {
  relative_path: string;
  training_package_id?: string | null;
  checkpoint_path?: string | null;
  modified_at: string;
}

export interface ClassificationBreakdownResponse {
  available: boolean;
  confusion_matrix?: number[][] | null;
  per_class?: ClassificationBreakdownClass[] | null;
  macro_f1?: number | null;
  macro_precision?: number | null;
  macro_recall?: number | null;
  macro_roc_auc?: number | null;
  macro_pr_auc?: number | null;
  weighted_f1?: number | null;
  n_classes?: number | null;
  class_labels?: string[] | null;
  source?: ClassificationBreakdownSource | null;
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
  // Backend emits null when the gap can't be computed yet (missing
  // realized series, missing DFF cache). The KPI tile reads `== null`
  // so the runtime path tolerates undefined too — the union here just
  // makes the type honest about both possibilities.
  realized_vs_stated_gap?: number | null;
  market_implied_gap?: number | null;
  months_since_reversal?: number | null;
}

export interface RegimeClassificationResponse {
  predicted_set: string[];
  set_label: string;
  set_size: number;
  coverage: number;
  distribution: Record<string, number>;
  argmax_class: string;
}

export type RatesHeadName = "2y" | "5y" | "terminal";
export type RatesDirectionalBucket = "easing" | "neutral" | "tightening";

export interface RatesReactionCard {
  head: RatesHeadName;
  point_bps: number;
  lower_bps: number | null;
  upper_bps: number | null;
  coverage: number | null;
  // #317 finding #10: nullable when the checkpoint exposes no aux
  // classifier for this head. The card renders an "aux classifier
  // unavailable" badge in that case rather than fabricating an
  // argmax over a fake uniform distribution.
  directional_bucket: RatesDirectionalBucket | null;
  bucket_probabilities: Partial<Record<RatesDirectionalBucket, number>> | null;
  // #317 finding #3: calibrated APS prediction set per head when the
  // conformal sidecar carries the per-head softmax_quantile.
  predicted_set: RatesDirectionalBucket[] | null;
}

export interface VolRegimeReactionCard {
  log_rv_point: number | null;
  log_rv_lower: number | null;
  log_rv_upper: number | null;
  regime_label: string;
  regime_probabilities: Record<string, number>;
  predicted_set: string[];
  coverage: number | null;
}

export interface MarketReactionPanelResponse {
  rates: RatesReactionCard[];
  vol_regime: VolRegimeReactionCard | null;
  encoder_alias: string | null;
  checkpoint_path: string | null;
}

export type AnalogVolRegime = "calm" | "normal" | "high";

export interface AnalogsRequest {
  text: string;
  k?: number;
  as_of_date?: string | null;
}

export interface AnalogCard {
  event_date: string;
  similarity: number;
  axis_stance: string | null;
  // UI-only bucket — the backend deliberately withholds the raw
  // ``forward_realized_vol_10d`` target so this label is the only
  // post-event signal available. Never feed back into a model.
  subsequent_vol_regime: AnalogVolRegime | null;
  excerpt: string;
}

export interface AnalogsResponse {
  analogs: AnalogCard[];
  index_size: number;
  encoder_alias: string;
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

export interface SettingsCheckpoint {
  filename: string;
  relative_path: string;
  role: string;
  size_bytes: number;
  modified_at: string;
  is_active: boolean;
  output_mode?: string | null;
  encoder_alias?: string | null;
  conformal_sidecar_present?: boolean | null;
  // #342: inference contract surfaces. ``required_kwargs`` mirrors the
  // sidecar; ``supplied_at_inference`` maps each declared kwarg to a
  // boolean for the live serving wiring. Empty / undefined when the
  // checkpoint pre-dates the #341 contract — ``inference_contract_status``
  // discriminates ``"sidecar_absent"`` (legacy) from ``"present"``.
  required_kwargs?: string[];
  supplied_at_inference?: Record<string, boolean>;
  inference_contract_status?: string | null;
}

export interface SettingsCheckpointsResponse {
  models_dir: string;
  checkpoints: SettingsCheckpoint[];
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
