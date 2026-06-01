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

export interface HistoryEventStudyResponse {
  event_date: string;
  symbol: string;
  forward_dates: string[];
  forward_close: number[];
  forward_log_returns: number[];
  realized_vol_10d?: number | null;
  predicted_regime?: string | null;
  realized_regime?: string | null;
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

export interface MultiAxisResponse {
  stance: MultiAxisStance | null;
  factor: MultiAxisFactor | null;
  certainty: MultiAxisCertainty | null;
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

export interface XaiFeatureFamilyAttribution {
  family: string;
  magnitude: number;
  signed: number;
}

export interface XaiPanelAttribution {
  panel: string;
  target: string;
  families: XaiFeatureFamilyAttribution[];
  n_steps: number;
  unavailable: boolean;
  reason: string | null;
}

export interface XaiResponse {
  sentences: XaiSentence[];
  method?: string;
  // #297: per-panel integrated-gradients attribution. Populated only
  // when `include_xai=true` on the request AND the active checkpoint
  // surfaces at least one panel that admits explanation.
  panels?: XaiPanelAttribution[];
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

// #304 sibling block: dual-head regression output on log(forward
// realized vol). Surfaced alongside the classification card so the
// "show details" toggle on the regime panel can render the
// continuous prediction + 90% conformal interval without re-parsing
// it out of RegimeClassificationResponse. Null on a classification-
// only checkpoint where the regression head is not mounted.
export interface RegimeRegressionResponse {
  log_rv_point: number;
  log_rv_lower: number | null;
  log_rv_upper: number | null;
  coverage: number | null;
}

export interface RegimeClassificationResponse {
  predicted_set: string[];
  set_label: string;
  set_size: number;
  coverage: number;
  distribution: Record<string, number>;
  argmax_class: string;
  // #322 / #338: dual-head regression branch on standardised log(RV).
  // Null when the active checkpoint mounts the classifier only — the
  // UI then falls back to the classifier surface as the primary read.
  log_rv_point?: number | null;
  log_rv_lower?: number | null;
  log_rv_upper?: number | null;
  // Declares which head produced argmax_class: "regression" means the
  // 3-class label was bucketed UI-side from log_rv_point against the
  // active checkpoint's vol_regime_quantiles cutoffs; "classification"
  // means the label came from the 3-class softmax head's argmax.
  bucket_source?: "regression" | "classification";
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

// #446 mechanical policy decision extracted from the statement text.
// Pure regex / keyword pass on the backend — every field is nullable
// so a non-policy excerpt (press conference Q&A, scraping miss)
// surfaces as a card with every field null. Units mirror the
// backend's `PolicyActionCard`: target_range_*_bp in basis points,
// change_magnitude_bp signed.
export type PolicyChangeDirection = "hike" | "hold" | "cut";
export type BalanceSheetState = "expansion" | "tapering" | "runoff";

export interface PolicyActionResponse {
  target_range_low_bp: number | null;
  target_range_high_bp: number | null;
  change_direction: PolicyChangeDirection | null;
  change_magnitude_bp: number | null;
  balance_sheet_state: BalanceSheetState | null;
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
  // #299: realized S&P 500 close-to-close % returns over 5 / 20 trading
  // days starting the day after event_date. Market-data overlay (NOT a
  // training label); null when the historical window is sparse.
  subsequent_close_pct_5d: number | null;
  subsequent_close_pct_20d: number | null;
  excerpt: string;
}

export interface AnalogsResponse {
  analogs: AnalogCard[];
  index_size: number;
  encoder_alias: string;
}

// Strict-prior diff against the previous FOMC statement (#443). Each
// entry is a span the diff produced; the consumer renders inserted
// spans in green and deleted spans struck-through in red. Both arrays
// are optional — older backends omit the field entirely.
export interface StatementDeltaSpan {
  text: string;
}

export interface AnalyzeResult {
  sentiment?: SentimentResponse;
  prediction?: PredictionResponse;
  market?: MarketResponse;
  model?: ModelDiagnosticsResponse;
  series?: SeriesResponse;
  multi_axis?: MultiAxisResponse;
  regime_classification?: RegimeClassificationResponse | null;
  // #443 strict-prior diff vs the previous FOMC statement.
  statement_delta_inserted?: StatementDeltaSpan[] | null;
  statement_delta_deleted?: StatementDeltaSpan[] | null;
  // #444 vote tally.
  votes_for?: number | null;
  votes_against?: number | null;
  dissent_direction?: "hawkish" | "dovish" | null;
  // #450 press-conference Q&A indicator.
  has_press_conf?: 0 | 1 | null;
  // #304 dual-head regression sibling block.
  regime_regression?: RegimeRegressionResponse | null;
  // #293 rates-reaction list. One entry per mounted rates head
  // (2y / 5y / terminal). Null on legacy single-head checkpoints.
  // An empty list rides when the heads are mounted but the per-event
  // forward produced no rows.
  rates_reaction?: RatesReactionCard[] | null;
  // #446 mechanical policy decision extracted from the statement
  // text. Null when the request body carried no text or the
  // extractor degraded; never raises.
  policy_action?: PolicyActionResponse | null;
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

// #299: quant-facing research registry (§6.41 manifest)
export interface ResearchRegistryBaseline {
  label: string;
  dual_f1: number | null;
  cls_f1: number | null;
  regression_f1: number | null;
}

export interface ResearchRegistryRow {
  encoder_alias: string;
  encoder_display: string;
  dual_f1: number | null;
  cls_f1: number | null;
  regression_f1: number | null;
  delta_dual: number | null;
  delta_cls: number | null;
  is_winner: boolean;
  checkpoint_relpath: string | null;
  cache_uri: string | null;
  notes: string;
}

export interface ResearchRegistryResponse {
  available: boolean;
  surface: "dual" | "cls";
  baseline: ResearchRegistryBaseline | null;
  rows: ResearchRegistryRow[];
  rejected_count: number;
  training_package_id: string;
  head: string;
  seeds: number[];
  source_wiki_section: string;
}

// #299 PR-B — stance-directional backtest engine
export type BacktestPosition = -1 | 0 | 1;

export interface BacktestPositionEntry {
  date: string;
  position: BacktestPosition;
}

export interface BacktestTradeRow {
  date: string;
  position: number;
  forward_return_pct: number | null;
  strategy_return_pct: number | null;
}

export interface RealizedVolHorizonForecast {
  h: number;
  point: number;
  band_lo_80: number;
  band_hi_80: number;
  band_lo_90: number;
  band_hi_90: number;
  qlike_model: number | null;
  qlike_har: number | null;
  coverage_empirical_90: number | null;
}

export interface RealizedVolForecastResponse {
  symbol: string;
  horizons: RealizedVolHorizonForecast[];
  history: number[];
  history_dates: string[];
  model_revision: string;
}

export interface BacktestResponse {
  trades: BacktestTradeRow[];
  n_trades: number;
  sharpe: number | null;
  hit_rate: number | null;
  max_dd_pct: number | null;
  cum_return_pct: number | null;
  benchmark_cum_pct: number | null;
  alpha_cum_pct: number | null;
  horizon_days: number;
  symbol: string;
}

// HAR-tercile regime baseline served from
// GET /forecast/regime/baselines?symbol=^GSPC. HAR-tercile is the
// Workspace's primary regime headline on the 3-class forward-RV
// classification task; the late-fusion card becomes a "second
// opinion" alongside it. The numeric ``predicted_rv`` is the
// model's point estimate of forward realized variance — annualized
// vol % is derived UI-side.
export type HarTercileLabel = "low" | "medium" | "high";

export interface HarTercileHorizon {
  // Trading-day horizon. The product surfaces 1 / 5 / 22 as
  // "1 day" / "1 week" / "1 month".
  h: number;
  tercile: HarTercileLabel;
  tercile_probs: Record<HarTercileLabel, number>;
  predicted_rv: number;
  macro_f1: number;
  macro_f1_source: string;
}

export interface HarTercileBaselineResponse {
  symbol: string;
  horizons: HarTercileHorizon[];
  source_wiki_section: string;
}

// HAR-tercile backtest served from
// GET /forecast/har-tercile-backtest?symbol=^GSPC. One row per
// persisted analyze run carrying the predicted tercile + the
// realized tercile resolved from the forward 10-trading-day market
// history. ``correct`` is null for rows whose forward window has not
// yet closed.
export interface HarTercileBacktestRow {
  event_date: string;
  predicted_tercile: HarTercileLabel;
  predicted_prob: number;
  realized_tercile: HarTercileLabel | null;
  realized_rv: number | null;
  correct: boolean | null;
}

export interface HarAccuracyMetrics {
  total_runs: number;
  resolved_runs: number;
  accuracy_overall: number | null;
  per_tercile_hit_rate: Partial<Record<HarTercileLabel, number>>;
}

export interface HarTercileBacktestResponse {
  symbol: string;
  horizon: number;
  rows: HarTercileBacktestRow[];
  metrics: HarAccuracyMetrics;
  generated_at: string;
}

// Workspace-spine bundle: shared response types matching the
// backend Pydantic models in backend/app/schemas.py.
//
// SPINE separation:
//   - ExpectedVolumeForecastResponse is the only forecast surface
//     in this bundle (HAR over market data only).
//   - MonetaryPolicySurpriseResponse, FuturesConsensusResponse and
//     SemanticDiffResponse are descriptive panels (text- or
//     realized-derived) and never feed forecasts.
export interface ExpectedVolumeHorizonForecast {
  h: number;
  point_log_residual: number;
  point_pct_vs_baseline: number;
  band_lo_80: number;
  band_hi_80: number;
  band_lo_90: number;
  band_hi_90: number;
  r2_har: number | null;
  calendar_adjusted: boolean;
}

export interface ExpectedVolumeForecastResponse {
  symbol: string;
  horizons: ExpectedVolumeHorizonForecast[];
  model_revision: string;
  generated_at: string;
}

export type MonetaryPolicySurpriseDirection =
  | "hawkish"
  | "dovish"
  | "no_surprise";

export interface MonetaryPolicySurpriseResponse {
  event_date: string;
  mp_surprise_level_bps: number;
  direction: MonetaryPolicySurpriseDirection;
  magnitude_bps: number;
  is_intermeeting: boolean;
  ff_target_prior_bps?: number | null;
}

export interface FuturesConsensusHorizon {
  horizon_label: string;
  implied_rate_bps: number;
  change_vs_current_bps: number;
  probability_hike: number;
  probability_cut: number;
  probability_pause: number;
}

export interface FuturesConsensusResponse {
  meeting_date: string;
  generated_at: string;
  current_target_lo_bps: number;
  current_target_hi_bps: number;
  horizons: FuturesConsensusHorizon[];
  methodology: string;
  data_source: string;
}

export type SemanticDiffSpanKind =
  | "unchanged"
  | "added"
  | "removed"
  | "substituted";

export interface SemanticDiffSpan {
  kind: SemanticDiffSpanKind;
  text: string;
  paired_text?: string | null;
}

export interface SemanticDiffTopic {
  topic: string;
  prior_emphasis: number;
  current_emphasis: number;
  delta: number;
  sample_phrases: string[];
}

export interface SemanticDiffResponse {
  current_date: string;
  prior_date: string;
  token_spans: SemanticDiffSpan[];
  topic_deltas: SemanticDiffTopic[];
  summary: string;
}
