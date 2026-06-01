from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_STRICT_REQUEST_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)
# Response models stay open to extras so the OpenAPI snapshot does not churn;
# `frozen` still blocks mutation after construction.
_FORBID_FROZEN_CONFIG = ConfigDict(frozen=True)
# #99 strict response config: enables Pydantic v2 strict mode so the
# numeric fields refuse cross-type coercion at construction time.
# Concretely it rejects:
#   - float -> int field   (a numpy.float64 leak into lookback_days)
#   - str   -> any numeric (string concat artefacts)
#   - bool  -> any numeric (True/False misuse)
# Pydantic v2 strict_float still accepts a bare ``int`` (treated as a
# lossless promotion), so a numpy.int64 leak into close/volatility
# is NOT caught here -- the guard is asymmetric across the numeric
# directions. It also accepts numpy.float64 against a float field
# because numpy.float64 is a subclass of Python float. Decimal and
# Fraction are likewise silently coerced for float fields in practice
# (pydantic/pydantic#11131) despite the docs implying otherwise. The
# value-add is the directional rejections above; the asymmetry is
# documented so the next audit pass knows what gap remains.
# Applied to the two leaf-level numeric response models whose
# service-layer builders have been audited end-to-end
# (MarketDataResponse and PredictionResponse).
#
# Scope caveat (Pydantic v2 semantics): strict=True is model-local.
# When a model with strict=True is populated as a nested field of a
# non-strict outer model (e.g. AnalyzeResponse), the outer model's
# non-strict coercion governs the validation pass and the nested
# strict guard does NOT re-fire on field values coming from the
# outer dict. Strict therefore catches numpy at direct-construction
# sites (services that build the model by name, tests that
# round-trip it, fixture factories) but not at FastAPI's
# response-serialisation boundary when the outer AnalyzeResponse is
# still _FORBID_FROZEN_CONFIG. The follow-up #99 PR that flips
# AnalyzeResponse to strict will close that hole; the leaf-level
# strict here still adds value for the direct-construction path,
# which is where the service builders actually live.
#
# Remaining response models (SentimentResponse, ChunkAttentionDiagnostics,
# ModelDiagnostics, XaiResponse, HistoryEntry, AnalyzeResponse) wait
# on matching audit passes -- this PR is the first half of the #99
# rollout.
_STRICT_RESPONSE_CONFIG = ConfigDict(strict=True, frozen=True)


class AnalyzeRequest(BaseModel):
    model_config = _STRICT_REQUEST_CONFIG

    text: str = Field(..., min_length=1, description="FOMC statement text")
    date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    symbol: str = Field("^GSPC", description="Market ticker, e.g. ^GSPC or DX-Y.NYB")
    horizon: str = Field("3d", description="Forecast horizon label")
    include_realized: bool = Field(
        False,
        description="When true and date is in the past, include realized forward series overlay.",
    )
    include_xai: bool = Field(
        False,
        description="When true, return per-sentence + per-token XAI attribution alongside the forecast.",
    )
    mask_sentence_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Counterfactual: 0-based indices of sentences to drop from the "
            "text before running the pipeline. Sentences are split using the "
            "same tokenizer that produces xai.sentences. Empty list = no mask."
        ),
    )


class SentimentResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    label: str
    score: float
    raw: list[dict[str, float | str]]
    # Energy-based out-of-distribution detection (Liu et al. 2020). Populated
    # only when a calibration manifest exists alongside the checkpoint at
    # forecaster_best.ood.json. ood_energy = -T * logsumexp(logits / T)
    # averaged across chunks per the manifest's aggregation rule.
    # is_in_distribution is True when ood_energy <= ood_threshold.
    ood_energy: float | None = None
    ood_threshold: float | None = None
    is_in_distribution: bool | None = None


class MarketDataResponse(BaseModel):
    model_config = _STRICT_RESPONSE_CONFIG

    symbol: str
    requested_date: str
    date_used: str
    lookback_days: int
    close: float
    volatility_5d: float


class PredictionResponse(BaseModel):
    model_config = _STRICT_RESPONSE_CONFIG

    close: float
    volatility: float
    horizon: str


class ChunkAttentionDiagnostics(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    chunk_count: int
    weights: list[float]
    decay_coeffs: list[float]
    chunk_previews: list[str]
    lambda_value: float


class ModelDiagnosticsResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    checkpoint_path: str
    checkpoint_exists: bool
    checkpoint_loaded: bool
    runtime_mode: str
    hidden_size: int
    num_layers: int
    dropout: float
    head_hidden_size: int
    close_scale: float
    sequence_length: int
    best_loss: float | None = None
    combined_rmse: float | None = None
    adaptation_epochs_completed: int | None = None
    adaptation_best_epoch: int | None = None
    adaptation_loss: float | None = None
    adaptation_combined_rmse: float | None = None
    decay_rate: float | None = None
    chunk_attention: ChunkAttentionDiagnostics | None = None
    # Encoder alias backing the multi-axis classifier (e.g.
    # "finbert_fed_adjacent"). None when no multi-axis checkpoint is loaded;
    # surfaced for the workspace status bar and pipeline trace.
    encoder_key: str | None = None


class ForecastSeriesResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    timestamps: list[str]
    history_close: list[float]
    history_volatility: list[float]
    forecast_timestamps: list[str]
    forecast_close: list[float]
    forecast_close_lower: list[float]
    forecast_close_upper: list[float]
    forecast_volatility: list[float]
    forecast_volatility_lower: list[float]
    forecast_volatility_upper: list[float]
    forecast_confidence_level: float
    realized_timestamps: list[str] | None = None
    realized_close: list[float] | None = None
    realized_volatility: list[float] | None = None
    volatility_scale: dict[str, float]
    forecast_band_source: str = Field(
        default="gaussian_z",
        description="Source of the forecast bands: 'gaussian_z' (z-score) or 'conformal'.",
    )
    conformal_coverage: float | None = Field(
        default=None,
        description="Nominal coverage of conformal bands when forecast_band_source='conformal'.",
    )


class XaiTokenAttribution(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    token: str
    weight: float


class XaiSentenceAttribution(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    text: str
    score: float
    topTokens: list[XaiTokenAttribution] = Field(default_factory=list)


class XaiFeatureFamilyAttribution(BaseModel):
    """One feature-family bar on a panel attribution chart (#297).

    Emitted under :class:`XaiPanelAttribution.families`. ``magnitude`` is
    the L1 attribution sum across all features in the family (always
    non-negative); ``signed`` is the sum-with-sign so the frontend can
    colour the bar by direction.
    """

    model_config = _FORBID_FROZEN_CONFIG

    family: str
    magnitude: float
    signed: float


class XaiPanelAttribution(BaseModel):
    """Integrated-gradients attribution for one /analyze panel (#297).

    Returned under :class:`XaiResponse.panels`. One entry per active
    panel (regime, rates_2y/5y/terminal, trajectory). ``unavailable``
    flips to True when the panel cannot be explained (panel not active
    on the checkpoint, kwarg mismatch, runtime error); the frontend
    then renders the "explanation unavailable" badge rather than an
    empty bar chart.
    """

    model_config = _FORBID_FROZEN_CONFIG

    panel: str
    target: str
    families: list[XaiFeatureFamilyAttribution] = Field(default_factory=list)
    n_steps: int = Field(
        default=0,
        description="Integrated-gradients integration step count used to compute this panel's attribution.",
    )
    unavailable: bool = False
    reason: str | None = None


class XaiResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    method: str = "keyword_salience_v1"
    sentences: list[XaiSentenceAttribution] = Field(default_factory=list)
    # #297: per-panel integrated-gradients attribution. Populated when
    # ``include_xai=true`` on the request AND the active checkpoint
    # surfaces at least one panel that can be explained. Per-panel
    # ``unavailable`` flags carry the structured reason when the panel
    # is present on the checkpoint but the attribution call degrades.
    panels: list[XaiPanelAttribution] = Field(default_factory=list)


class CredibilityResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    drift_score: float
    realized_vs_stated_gap: float
    market_implied_gap: float
    months_since_reversal: int


class MultiAxisStanceCard(BaseModel):
    """Stance prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="hawkish | dovish | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class softmax probability over hawkish/dovish/neutral.",
    )


class MultiAxisTimeCard(BaseModel):
    """Forward-looking horizon classification from the multi-task head.

    Two classes: ``forward looking`` (the statement references future
    actions or expectations) vs ``not forward looking`` (backward-looking
    or current-state only). Sourced from the gtfintechlab ``time_label``
    column; trained on ~5 992 labelled sentences.
    """

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="forward looking | not forward looking")
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(default_factory=dict)


class MultiAxisCertaintyCard(BaseModel):
    """Certainty prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="certain | uncertain | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(default_factory=dict)


class MultiAxisBlock(BaseModel):
    """Multi-task head per-axis predictions surfaced on /analyze (#78).

    Active axes: stance / certainty / time. The factor axis (GSS
    market-derived regression target) was retired — text cannot predict it
    and the training pool had 0% coverage. Topic was retired in ADR 0044.
    """

    model_config = _FORBID_FROZEN_CONFIG

    stance: MultiAxisStanceCard
    certainty: MultiAxisCertaintyCard | None = None
    time: MultiAxisTimeCard | None = None


class RatesReactionCard(BaseModel):
    """Per-rates-head market-reaction card surfaced on /analyze/market (#293).

    The card carries the model's point prediction in basis points for one
    rates head (``2y`` / ``5y`` / ``terminal``), the symmetric conformal
    band derived from the val-fitted residual quantile, the directional
    bucket (``easing`` / ``neutral`` / ``tightening``) the auxiliary
    classifier emits, and the per-class probability distribution so the
    frontend can render a probability bar alongside the headline number.
    """

    model_config = _FORBID_FROZEN_CONFIG

    head: str = Field(..., description="Head short name: 2y | 5y | terminal")
    point_bps: float = Field(..., description="Point prediction in basis points.")
    lower_bps: float | None = Field(
        default=None,
        description="Conformal band lower bound in bps. None when no calibration sidecar is present.",
    )
    upper_bps: float | None = Field(
        default=None,
        description="Conformal band upper bound in bps. None when no calibration sidecar is present.",
    )
    coverage: float | None = Field(
        default=None,
        description="Nominal conformal coverage (1 - alpha). None when bands are absent.",
    )
    # #317 finding #10: when the checkpoint has no aux classifier
    # mounted (regression-only mode or unbuilt cls head) the response
    # must clearly say "not available" rather than emitting a fake
    # argmax over uniform probabilities. Nullable fields let the
    # frontend render a "no model evidence" badge instead.
    directional_bucket: str | None = Field(
        default=None,
        description=(
            "easing | neutral | tightening (argmax of the auxiliary "
            "classifier). None when no aux classifier is mounted."
        ),
    )
    bucket_probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-bucket softmax probabilities over "
            "(easing, neutral, tightening). None when no aux "
            "classifier is mounted."
        ),
    )
    # #317 finding #3: calibrated APS prediction set per rates head
    # (subset of {easing, neutral, tightening}). None when no
    # ``rates_softmax_quantiles`` sidecar is present.
    predicted_set: list[str] | None = Field(
        default=None,
        description=(
            "Calibrated APS prediction set per rates head. None when "
            "no conformal sidecar is present."
        ),
    )


class VolRegimeReactionCard(BaseModel):
    """Vol-regime card mirroring the rates card shape on /analyze/market (#293).

    Carries the dual-head log(RV) regression prediction (or ``None`` when
    the active checkpoint mounts only the classifier), the regime
    classification distribution, and the calibrated APS prediction set
    derived from the existing softmax_quantile.
    """

    model_config = _FORBID_FROZEN_CONFIG

    log_rv_point: float | None = Field(
        default=None,
        description=(
            "Standardised log(forward realized vol) prediction from "
            "the dual-head regression branch. None on classification-only "
            "checkpoints."
        ),
    )
    log_rv_lower: float | None = None
    log_rv_upper: float | None = None
    regime_label: str = Field(
        ..., description="Argmax regime label: calm | normal | high.",
    )
    regime_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class softmax probabilities over the regime classes.",
    )
    predicted_set: list[str] = Field(
        default_factory=list,
        description="Calibrated APS prediction set (list of regime labels).",
    )
    coverage: float | None = None


class MarketReactionPanel(BaseModel):
    """Bundle of four reaction cards (#293).

    Returned by ``POST /analyze/market``. Includes one card per mounted
    rates head plus the existing vol-regime card. Heads not mounted on
    the active checkpoint emit an empty ``rates`` list and ``None`` for
    ``vol_regime`` so the frontend can render a graceful empty state.
    """

    model_config = _FORBID_FROZEN_CONFIG

    rates: list[RatesReactionCard] = Field(default_factory=list)
    vol_regime: VolRegimeReactionCard | None = None
    encoder_alias: str | None = None
    checkpoint_path: str | None = None


class RegimeClassificationCard(BaseModel):
    """Calibrated prediction-set output from the vol-regime classifier (#216).

    Surfaces the conformal APS set on the /analyze response. ``predicted_set``
    holds the class labels the calibrated threshold admits at the
    ``coverage`` level; ``set_label`` is the UI-friendly bracketed string;
    ``distribution`` is the raw per-class softmax for the inference row so
    the frontend can render a bar chart alongside the set chips.

    Populated only when the active checkpoint is classification-mode AND
    a sibling ``.conformal.json`` manifest with ``softmax_quantile`` exists.
    Pre-#216 regression-only checkpoints leave the field as ``None`` on
    the response and the legacy uncalibrated max-softmax stays on the
    ``sentiment`` block.
    """

    model_config = _FORBID_FROZEN_CONFIG

    predicted_set: list[str]
    set_label: str
    set_size: int
    coverage: float
    distribution: dict[str, float]
    argmax_class: str
    log_rv_point: float | None = Field(
        default=None,
        description=(
            "Regression-head point prediction in standardised log(forward "
            "realized vol) space, or None on a classification-only checkpoint. "
            "See ADR 0015 / #322."
        ),
    )
    log_rv_lower: float | None = Field(
        default=None,
        description=(
            "80% conformal-band lower bound around log_rv_point (matches the "
            "existing close/vol band convention). None when no conformal "
            "manifest is on disk or the regression head is not active."
        ),
    )
    log_rv_upper: float | None = Field(default=None, description="See log_rv_lower; upper bound.")
    bucket_source: Literal["regression", "classification"] = Field(
        default="classification",
        description=(
            "Declares which head produced argmax_class: 'regression' means "
            "the 3-class label was bucketed UI-side from log_rv_point against "
            "the active checkpoint's vol_regime_quantiles cutoffs (see "
            "app.services.regime_bucketing); 'classification' means the "
            "label came from the 3-class softmax head's argmax."
        ),
    )


class RegimeRegressionCard(BaseModel):
    """Regression-head sibling block on the /analyze response (#304).

    The classification card stays the headline; this block carries the
    same dual-head regression output (``log_rv_point`` + symmetric 90%
    conformal band) as a standalone surface so a downstream consumer
    can read the continuous prediction without parsing it out of the
    classification card. Populated only when the active checkpoint
    mounts the regression head (``head_mode`` in ``regression`` /
    ``dual``) AND ``build_regime_classification_card`` returned a card
    whose ``log_rv_point`` is non-null; otherwise the field stays
    ``None`` on the response.

    Units mirror :class:`RegimeClassificationCard`: ``log_rv_point`` is
    in standardised log(forward realized vol) space (the per-fold
    train-slice standardiser the dual-head trainer fits, see
    ``log_rv_scaler`` on the run summary). ``log_rv_lower`` /
    ``log_rv_upper`` are the symmetric conformal interval at
    ``coverage`` nominal coverage; on a checkpoint without a conformal
    sidecar the bounds collapse to ``None`` even when the point
    estimate is populated.
    """

    model_config = _FORBID_FROZEN_CONFIG

    log_rv_point: float
    log_rv_lower: float | None = None
    log_rv_upper: float | None = None
    coverage: float | None = Field(
        default=None,
        description=(
            "Nominal coverage on the conformal interval (0.9 by default; "
            "matches the manifest's nominal_coverage when the sidecar "
            "carries one). None when the regression head is mounted but "
            "no conformal manifest is on disk."
        ),
    )


class PolicyActionCard(BaseModel):
    """Mechanical policy decision extracted from the statement text (#446).

    Sibling of :class:`RegimeClassificationCard` on the /analyze
    response. Pure extraction surface — no model inference, no
    calibration. The four fields mirror the
    :class:`app.services.policy_action_extractor.PolicyAction`
    dataclass and are all optional so a statement that names no target
    range (press conference Q&A, scraping miss, non-policy text) still
    serialises as a card with every field ``None``.

    Units: ``target_range_low_bp`` / ``target_range_high_bp`` are in
    basis points (3.50% → 350). ``change_magnitude_bp`` is signed
    (positive on a hike, negative on a cut, zero on a hold). The
    frontend renders the colour by ``change_direction``: hike = red,
    cut = green, hold = neutral.
    """

    model_config = _FORBID_FROZEN_CONFIG

    target_range_low_bp: int | None = Field(
        default=None,
        description=(
            "Lower bound of the named target range, in basis points "
            "(e.g. 350 for a 3.50% lower bound). None when no target "
            "range is named in the text."
        ),
    )
    target_range_high_bp: int | None = Field(
        default=None,
        description="Upper bound of the named target range, in basis points.",
    )
    change_direction: Literal["hike", "hold", "cut"] | None = Field(
        default=None,
        description=(
            "Verb-derived direction of the action. None when no policy "
            "verb is named (e.g. press-conference Q&A) and no prior "
            "midpoint was provided to the extractor."
        ),
    )
    change_magnitude_bp: int | None = Field(
        default=None,
        description=(
            "Signed change in basis points relative to the prior "
            "meeting (positive on a hike, negative on a cut, zero on a "
            "hold). Pulled from in-prose magnitude phrases ('by 25 "
            "basis points', 'by 1/4 percentage point') first; falls "
            "back to ``this_mid - prior_mid`` when the caller supplied "
            "a prior midpoint."
        ),
    )
    balance_sheet_state: Literal["expansion", "tapering", "runoff"] | None = Field(
        default=None,
        description=(
            "Balance-sheet posture extracted from the paragraph that "
            "names balance-sheet operations. None when the paragraph "
            "is absent or carries no posture-defining keyword."
        ),
    )


class InferenceStatusSurface(BaseModel):
    """Structured error surface for the per-card inference helpers (#341).

    Sibling of :class:`RegimeClassificationCard` on the /analyze
    response. Populated when the card-build helper degrades through
    one of three structured branches:

    - ``not_classification_mode`` -- the active checkpoint emits no
      regime card by contract. Legitimate; UI renders the card as
      absent.
    - ``inference_kwarg_missing`` -- the serving call site fed (or
      omitted) a kwarg the checkpoint did not declare in its
      inference contract sidecar. Operator-facing bug.
    - ``unexpected_exception`` -- anything else; ``exception_class``
      carries the class name and ``detail`` the message so the
      operator can grep for it without parsing logs.
    """

    model_config = _FORBID_FROZEN_CONFIG

    status: str
    missing_kwarg: str | None = None
    exception_class: str | None = None
    detail: str | None = None


class AnalyzeResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    sentiment: SentimentResponse
    prediction: PredictionResponse
    market: MarketDataResponse
    model: ModelDiagnosticsResponse
    series: ForecastSeriesResponse
    xai: XaiResponse | None = None
    credibility: CredibilityResponse | None = None
    multi_axis: MultiAxisBlock | None = None
    regime_classification: RegimeClassificationCard | None = None
    # #304 regression sibling. Populated when the active checkpoint
    # mounts the dual-head regression head AND the classification card
    # carried a non-null ``log_rv_point``. The classification card
    # stays the headline; this block surfaces the continuous
    # prediction + 90% conformal interval as a standalone read for the
    # frontend's "show details" toggle on the regime panel.
    regime_regression: RegimeRegressionCard | None = None
    # #292 rates-reaction cards. One card per mounted rates head
    # (2y / 5y / terminal) carrying the bps point + conformal interval
    # plus the optional directional bucket / APS prediction set when
    # ``--rates-classification-heads`` was on. ``None`` on a legacy
    # single-head checkpoint or on a regression-output run; an empty
    # list when the heads exist but the per-event forward produced no
    # rows. Hooked by #293's MarketReactionPanel.
    rates_reaction: list[RatesReactionCard] | None = None
    # #446 mechanical policy decision extracted from the statement
    # text. Pure regex / keyword pass — no model inference. None when
    # the request carries no statement text (defensive; the schema
    # requires text but the extractor wrapper still short-circuits on
    # an empty body). Populated for any statement that names a target
    # range; balance-sheet posture rides off the balance-sheet
    # paragraph when present.
    policy_action: PolicyActionCard | None = None
    # #341 sibling status surface so an operator can grep the JSON
    # response for the structured error branch when the regime card
    # degrades. Mutually exclusive with ``regime_classification``
    # being populated -- either the card lands, or this field carries
    # the structured reason.
    regime_classification_status: InferenceStatusSurface | None = None


class HistoryEntry(BaseModel):
    id: str
    created_at: str
    symbol: str
    document_date: str
    horizon: str
    forecast_mode: str
    stance: str
    sentiment_score: float | None = None
    predicted_close: float | None = None
    current_close: float | None = None
    predicted_volatility: float | None = None
    text_excerpt: str | None = None
    # Regime summary extracted from the persisted payload. None when the row
    # came from a regression-mode checkpoint or pre-dated the regime head.
    argmax_regime: str | None = None
    argmax_probability: float | None = None
    regime_set_size: int | None = None


class HistoryDetail(HistoryEntry):
    payload: dict[str, Any]


class HistoryList(BaseModel):
    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


class HistoryRealizedResponse(BaseModel):
    run_id: str
    symbol: str
    document_date: str
    horizon: str
    timestamps: list[str]
    close: list[float]
    volatility: list[float]
    # Realized regime label derived from the post-event 10d-forward vol
    # path, bucketed against the classifier's trained quantile cutoffs
    # when those cutoffs are accessible. None when the cutoffs are not
    # available on this host (regression-only checkpoint, cold start).
    realized_regime: str | None = None


class HistoryRealizedBatchResponse(BaseModel):
    """Batched realized payloads keyed by run_id. Missing runs (deleted,
    yfinance failure) come back under ``missing`` so the caller can render
    a partial result instead of failing the whole page."""

    items: dict[str, HistoryRealizedResponse]
    missing: list[str]


class HistoryEventStudyResponse(BaseModel):
    """Forward 10-trading-day price path anchored on the event date.

    Backs the event-study chart on /history/[id]: the realised close path
    plus the bucketed realised regime label so the headline can read
    "predicted X, realized Y".
    """

    event_date: str
    symbol: str
    forward_dates: list[str]
    forward_close: list[float]
    forward_log_returns: list[float]
    realized_vol_10d: float | None = None
    predicted_regime: str | None = None
    realized_regime: str | None = None


class EvaluationCoverageResponse(BaseModel):
    """Empirical conformal coverage aggregated across recent history runs.

    ``nominal`` is the conformal target the active model was calibrated
    to (read off the most-recent run that carries
    ``series.forecast_confidence_level``). ``empirical`` is the fraction
    of runs whose realized regime label fell inside the predicted set.
    Both are None when no qualifying runs exist."""

    model_config = _FORBID_FROZEN_CONFIG

    nominal: float | None = None
    empirical: float | None = None
    sample_size: int
    runs_total: int
    computed_at: str


class ClassificationBreakdownClass(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    class_id: int
    precision: float
    recall: float
    f1: float
    support: int
    roc_auc: float | None = None
    pr_auc: float | None = None


class ClassificationBreakdownSource(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    relative_path: str
    training_package_id: str | None = None
    checkpoint_path: str | None = None
    modified_at: str


class ClassificationBreakdownResponse(BaseModel):
    """The richer eval emitted by app/evaluation/classification_breakdown.py.

    Surfaces the freshest ``best_trial.summary.metrics.classification_breakdown``
    block written under ``data/artifacts/regime_*``. ``available`` is
    false when no qualifying artifact exists — the UI then falls back to
    its client-side aggregation across history rows."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool
    confusion_matrix: list[list[int]] | None = None
    per_class: list[ClassificationBreakdownClass] | None = None
    macro_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_roc_auc: float | None = None
    macro_pr_auc: float | None = None
    weighted_f1: float | None = None
    n_classes: int | None = None
    # Frontend assumes the canonical {calm, normal, high} ordering for
    # 3-class regime models. ``class_labels`` is filled when the source
    # artifact carries it; otherwise the field is None and the UI uses
    # its defaults.
    class_labels: list[str] | None = None
    source: ClassificationBreakdownSource | None = None


class SymbolDescriptor(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    name: str
    category: str
    default_horizon: str


class SymbolListResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    symbols: list[SymbolDescriptor]


class SettingsCheckpoint(BaseModel):
    """One file under ``backend/models/`` surfaced on the settings page.

    Inventory only; nothing here mutates the running singleton. ``role``
    is inferred from the filename (forecaster / multi_axis / lora /
    calibration) and ``is_active`` flags the file the active service is
    currently loaded from. The diagnostic fields (output_mode,
    encoder_alias, conformal_sidecar_present) only populate on the
    active forecaster + active multi-axis entries.

    #342: ``required_kwargs`` mirrors the inference-contract sidecar
    (empty list when no sidecar — pre-#341 legacy artefact).
    ``supplied_at_inference`` maps each declared kwarg to the live
    serving wiring; mismatches drive the red-badge surface on the
    settings page. ``inference_contract_status`` is ``"sidecar_absent"``
    for legacy checkpoints, otherwise ``"present"``.
    """

    model_config = _FORBID_FROZEN_CONFIG

    filename: str
    relative_path: str
    role: str
    size_bytes: int
    modified_at: str
    is_active: bool = False
    output_mode: str | None = None
    encoder_alias: str | None = None
    conformal_sidecar_present: bool | None = None
    required_kwargs: list[str] = Field(default_factory=list)
    supplied_at_inference: dict[str, bool] = Field(default_factory=dict)
    inference_contract_status: str | None = None


class SettingsCheckpointsResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    models_dir: str
    checkpoints: list[SettingsCheckpoint]


class FomcMeetingResponse(BaseModel):
    meeting_date: str
    meeting_type: str
    statement_release_date: str | None = None
    minutes_release_date: str | None = None
    notes: str | None = None
    statement_available: bool = False
    minutes_available: bool = False
    press_conference_available: bool = False


class FomcCalendarResponse(BaseModel):
    past: list[FomcMeetingResponse]
    upcoming: list[FomcMeetingResponse]


class DocumentParseUrlRequest(BaseModel):
    url: str


class DocumentParseResponse(BaseModel):
    text: str
    char_count: int
    source_kind: str
    source_metadata: dict[str, str]


# ---------------------------------------------------------------------------
# Research / training / decisions endpoints (Phase 8 multi-page expansion)
# ---------------------------------------------------------------------------


class ArtifactFile(BaseModel):
    """One file under ``data/artifacts/<section>/``."""

    model_config = _FORBID_FROZEN_CONFIG

    relative_path: str = Field(..., description="Path relative to data/artifacts/")
    size_bytes: int
    modified_at: str = Field(..., description="ISO-8601 mtime, UTC.")
    suffix: str = Field(..., description="File extension including the dot, e.g. .json")


class EncoderBakeoffRow(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    encoder_key: str
    checkpoint: str
    seeds: list[int]
    macro_f1_values: list[float]
    macro_f1_mean: float
    macro_f1_ci_low: float | None = None
    macro_f1_ci_high: float | None = None
    weighted_f1_mean: float | None = None
    accuracy_mean: float | None = None
    cohen_kappa: float | None = None


class EncoderBakeoffSection(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    available: bool
    coverage: float | None = None
    rows: list[EncoderBakeoffRow] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)


class TransferMatrixCell(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    source: str
    target: str
    metric: float


class CrossBankTransferSection(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    available: bool
    metric_name: str = "macro_f1"
    sources: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    cells: list[TransferMatrixCell] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)


class ResearchArtifactsResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    artifacts_root: str
    sections: dict[str, list[ArtifactFile]]
    encoder_bakeoff: EncoderBakeoffSection
    cross_bank_transfer: CrossBankTransferSection


class NextFomcMeetingPrediction(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    target_event_date: str
    target_as_of_ts: str
    target_class: str | None = Field(
        default=None,
        description="Realised class, when known. None for the next-scheduled meeting.",
    )
    n_train_rows: int
    probabilities: dict[str, dict[str, float]]
    predicted_class: dict[str, str]


class NextFomcModelMetrics(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    n: int
    brier: float | None = None
    log_loss: float | None = None
    top1_accuracy: float | None = None
    macro_f1: float | None = None
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)


class NextFomcAttributionRow(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    subset: str
    families: list[str]
    n_features: int | None = None
    n: int | None = None
    brier: float | None = None
    log_loss: float | None = None
    top1_accuracy: float | None = None
    macro_f1: float | None = None


class NextFomcUpcomingMeeting(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    meeting_date: str
    meeting_type: str
    statement_release_date: str | None = None
    days_until: int | None = None


class NextFomcForecastResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    available: bool
    artifacts_dir: str
    ordinal_classes: list[str]
    model_names: list[str] = Field(default_factory=list)
    upcoming_meeting: NextFomcUpcomingMeeting | None = None
    headline: NextFomcMeetingPrediction | None = None
    history: list[NextFomcMeetingPrediction] = Field(default_factory=list)
    metrics_full_window: dict[str, NextFomcModelMetrics] = Field(default_factory=dict)
    metrics_ex_pandemic: dict[str, NextFomcModelMetrics] = Field(default_factory=dict)
    feature_attribution: list[NextFomcAttributionRow] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Historical analog retrieval (#294 — /analyze/analogs)
# ---------------------------------------------------------------------------


class AnalogsRequest(BaseModel):
    """Query a fine-tuned retrieval encoder for past FOMC statements that
    sound like ``text``. Returns top-``k`` analogs ordered by cosine
    similarity together with a coarse post-event volatility regime label
    so the caller can show "what regime followed each analog" without
    exposing the underlying supervised target value."""

    model_config = _STRICT_REQUEST_CONFIG

    text: str = Field(..., min_length=1, description="Statement text to match against past FOMC statements.")
    k: int = Field(default=5, ge=1, le=20, description="Number of analogs to return (1-20).")
    as_of_date: date | None = Field(
        default=None,
        description=(
            "Strict-backward walk-forward boundary: only analogs with "
            "``event_date < as_of_date`` are eligible. Default is no "
            "boundary; for ML feature use always pass a date so the "
            "retrieval set does not leak future statements into the "
            "training fold."
        ),
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _coerce_as_of_date(cls, value: Any) -> Any:
        """Accept ISO ``YYYY-MM-DD`` strings under strict mode.

        Pydantic strict mode rejects string -> date coercion by default,
        but the JSON wire format only carries strings; this validator
        parses the ISO form so callers do not need to pre-coerce the
        value before posting.
        """

        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"as_of_date must be ISO YYYY-MM-DD, got {value!r}"
                ) from exc
        return value


class AnalogCard(BaseModel):
    """One historical analog returned by /analyze/analogs.

    The card deliberately does NOT carry the raw
    ``forward_realized_vol_10d`` target the project trains on — that
    value is the supervised label and surfacing it as an API field
    would tempt downstream consumers to feed it back into a model.
    Instead the card exposes ``subsequent_vol_regime``, a coarse
    ``calm`` / ``normal`` / ``high`` bucket pinned to the
    ``VOL_REGIME_BUCKET_EDGES`` constant in ``app.retrieval.index``
    (held-out 2000-2015 reference distribution). The bucket label is a
    UI-only marker; treat it as informative for humans, not as a
    feature for a downstream model.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO date of the historical FOMC statement.")
    similarity: float = Field(..., description="Cosine similarity in [-1, 1] vs. the query embedding.")
    axis_stance: str | None = Field(
        default=None,
        description="Stored stance label for the analog statement (hawkish / dovish / neutral). None when absent.",
    )
    subsequent_vol_regime: Literal["calm", "normal", "high"] | None = Field(
        default=None,
        description=(
            "Coarse post-event 10-day realised vol bucket — UI-only label, "
            "NOT a model feature. Do not feed back into a downstream model."
        ),
    )
    # #299 — realized S&P forward returns measured from the event-day
    # close (Bloomberg / FactSet convention). The denominator is the
    # close on ``event_date`` (or the nearest prior trading day) and
    # the numerator is the close ``N`` trading days forward of that
    # anchor. These are MARKET DATA OVERLAYS (yfinance ^GSPC), not
    # training labels — safe to surface even when the supervised
    # ``subsequent_vol_regime`` bucket is intentionally suppressed.
    # None when the historical market data is unavailable.
    subsequent_close_pct_5d: float | None = Field(
        default=None,
        description="S&P 500 close-to-close % return over the 5 trading days following the event-day close.",
    )
    subsequent_close_pct_20d: float | None = Field(
        default=None,
        description="S&P 500 close-to-close % return over the 20 trading days following the event-day close.",
    )
    excerpt: str = Field(..., description="First ~280 characters of the analog statement.")


class AnalogsResponse(BaseModel):
    """Result envelope for /analyze/analogs."""

    model_config = _FORBID_FROZEN_CONFIG

    analogs: list[AnalogCard] = Field(default_factory=list)
    index_size: int = Field(..., description="Total number of past statements in the loaded retrieval index.")
    encoder_alias: str = Field(..., description="Registry alias of the encoder used to embed the query.")


# ---------------------------------------------------------------------------
# Hawkish/dovish trajectory model (#296 — /analyze/trajectory)
# ---------------------------------------------------------------------------


class TrajectoryRequest(BaseModel):
    """Project the FOMC stance trajectory as of ``as_of_date``.

    Strict-backward by construction: the history slice only considers
    meetings whose ``event_date <= as_of_date``. ``history_length``
    caps the number of recent meetings rendered in the panel chart.
    Defaults to 12 (~1.5 years of FOMC meetings) per the §3 Panel 4
    spec.
    """

    model_config = _STRICT_REQUEST_CONFIG

    as_of_date: date = Field(
        ..., description="As-of date for the trajectory projection (YYYY-MM-DD)."
    )
    history_length: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Number of past meetings to surface in the panel (1-60).",
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _coerce_as_of_date(cls, value: Any) -> Any:
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"as_of_date must be ISO YYYY-MM-DD, got {value!r}"
                ) from exc
        return value


class TrajectoryMarker(BaseModel):
    """One past FOMC meeting rendered as a semantic marker on the panel chart."""

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str = Field(..., description="ISO date of the historical FOMC meeting.")
    axis_stance: str | None = Field(
        default=None,
        description=(
            "Stored stance label (hawkish / dovish / neutral) for the meeting. "
            "None when the panel was trained against a corpus without labels."
        ),
    )
    embedding_2d: tuple[float, float] = Field(
        ...,
        description=(
            "PCA / UMAP projection of the meeting's encoder embedding to "
            "2-D space, used as the marker's (x, y) coordinates."
        ),
    )


class TrajectoryProjection(BaseModel):
    """Next-meeting projection — predicted class + calibrated confidence band."""

    model_config = _FORBID_FROZEN_CONFIG

    predicted_stance: str = Field(
        ..., description="Argmax over the next-meeting stance distribution."
    )
    class_probs: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-class probability over hawkish / dovish / neutral. Sums "
            "to 1.0 after defensive renormalisation."
        ),
    )
    confidence_band: list[str] | None = Field(
        default=None,
        description=(
            "APS-calibrated stance set (Romano et al. 2020). Empty when no "
            "conformal sidecar shipped with the bundle; the UI then "
            "renders the argmax marker without the confidence ring."
        ),
    )
    conformal_alpha: float | None = Field(
        default=None,
        description="Conformal mis-coverage level applied to confidence_band.",
    )


class TrajectoryResponse(BaseModel):
    """Result envelope for /analyze/trajectory."""

    model_config = _FORBID_FROZEN_CONFIG

    available: bool = Field(
        ..., description="False when no trajectory bundle is loaded on this host."
    )
    history: list[TrajectoryMarker] = Field(default_factory=list)
    projected_next: TrajectoryProjection | None = None
    architecture: str | None = Field(
        default=None,
        description="lstm | transformer — which arm produced the projection.",
    )
    encoder_alias: str = Field(
        default="",
        description="Registry alias of the encoder backing the meeting embeddings.",
    )
    history_length: int = Field(
        default=0,
        description="How many past meetings were considered (after truncation).",
    )
    train_end: str | None = Field(
        default=None,
        description="Walk-forward boundary from the bundle manifest.",
    )
    as_of_date: str = Field(
        default="",
        description="Echo of the request as_of_date (ISO YYYY-MM-DD).",
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Optional non-fatal advisory — set when ``as_of_date`` is "
            "beyond the bundle's ``train_end`` so the caller can flag "
            "that the projection extrapolates beyond the fold."
        ),
    )
    # Lift-vs-baseline badge fields (#332). Surface the verdict on
    # whether the Transformer arm beats the strongest naive baseline
    # (previous_stance / rolling_majority / small-LSTM) by >= 5pp
    # directional accuracy on the canonical fold protocol. All three
    # fields default to None / False so a bundle trained before #332
    # remains back-compatible.
    lift_vs_baseline: bool = Field(
        default=False,
        description=(
            "True iff the Transformer arm beats the strongest naive "
            "baseline (previous_stance / rolling_majority(3) / "
            "small-LSTM) by >= 5pp directional accuracy on the "
            "canonical fold protocol. False when the holdout slice is "
            "empty, when the bundle predates #332, or when the lift "
            "did not clear the threshold."
        ),
    )
    delta_dir_acc: float | None = Field(
        default=None,
        description=(
            "Transformer directional accuracy minus the strongest "
            "naive baseline's directional accuracy. None when no "
            "baseline comparison is available."
        ),
    )
    baseline_used: str | None = Field(
        default=None,
        description=(
            "Name of the strongest naive baseline the lift verdict "
            "compared against. None when no baseline comparison is "
            "available."
        ),
    )


class ResearchRegistryBaseline(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    label: str
    dual_f1: float | None = None
    cls_f1: float | None = None
    regression_f1: float | None = None


class ResearchRegistryRow(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    encoder_alias: str
    encoder_display: str
    dual_f1: float | None = None
    cls_f1: float | None = None
    regression_f1: float | None = None
    delta_dual: float | None = Field(
        default=None,
        description="Δ macro-F1 on the dual-head surface vs no-text baseline.",
    )
    delta_cls: float | None = Field(
        default=None,
        description="Δ macro-F1 on the classification-only surface vs no-text baseline.",
    )
    is_winner: bool = Field(
        default=False,
        description="True iff the active surfaces Δ is >= 0 (non-negative lift).",
    )
    checkpoint_relpath: str | None = None
    cache_uri: str | None = Field(
        default=None,
        description="hf:// URI of the shareable embedding cache parquet, if published.",
    )
    notes: str = ""


class ResearchRegistryResponse(BaseModel):
    """Quant-facing encoder registry response (§6.41 manifest).

    Filtered by default to non-negative Δ on the requested surface so
    the dashboard does not surface negative-lift encoders. Use
    ?include_rejected=true to see the full table including nulls and
    negatives.
    """

    model_config = _FORBID_FROZEN_CONFIG

    available: bool
    surface: Literal["dual", "cls"]
    baseline: ResearchRegistryBaseline | None = None
    rows: list[ResearchRegistryRow] = Field(default_factory=list)
    rejected_count: int = 0
    training_package_id: str = ""
    head: str = ""
    seeds: list[int] = Field(default_factory=list)
    source_wiki_section: str = ""


# #299 PR-B — stance-directional backtest engine

class BacktestPositionEntry(BaseModel):
    """One {date, position} signal in the backtest request."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    date: str = Field(..., description="ISO date YYYY-MM-DD of the signal.")
    position: int = Field(..., description="Position in {-1, 0, 1}. Hawkish=-1, neutral=0, dovish=+1.")


class BacktestRequest(BaseModel):
    """Request body for POST /research/backtest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    positions: list[BacktestPositionEntry] = Field(
        ..., min_length=1, description="At least one signal entry."
    )
    symbol: str = Field("^GSPC", description="Market ticker for the strategy backtest.")
    horizon_days: int = Field(
        5,
        ge=1,
        le=60,
        description="Forward holding period in trading days.",
    )


class BacktestTradeRow(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    date: str
    position: int
    forward_return_pct: float | None = None
    strategy_return_pct: float | None = None


class BacktestResponse(BaseModel):
    """Aggregate backtest metrics for the quant terminal."""

    model_config = _FORBID_FROZEN_CONFIG

    trades: list[BacktestTradeRow] = Field(default_factory=list)
    n_trades: int
    sharpe: float | None = None
    hit_rate: float | None = None
    max_dd_pct: float | None = None
    cum_return_pct: float | None = None
    benchmark_cum_pct: float | None = None
    alpha_cum_pct: float | None = None
    horizon_days: int
    symbol: str


class RealizedVolHorizonForecast(BaseModel):
    """Banded RV forecast for one horizon (1, 5, or 22 trading days).

    ``point`` and the four ``band_*`` numbers are RV (variance) units, not
    log-RV. ``qlike_model`` / ``qlike_har`` are the pooled walk-forward
    QLIKE losses (lower is better); the card surfaces the gain as a
    beat-HAR badge. ``coverage_empirical_90`` is the prospective empirical
    coverage of the 90% conformal band, for the calibration chip.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int
    point: float
    band_lo_80: float
    band_hi_80: float
    band_lo_90: float
    band_hi_90: float
    qlike_model: float | None = None
    qlike_har: float | None = None
    coverage_empirical_90: float | None = None


class RealizedVolHistoricalBand(BaseModel):
    """Single walk-forward h=1 conformal band aligned to a realized day.

    Renders behind the realized sparkline so the card shows the band
    actually covered each day's outcome.
    """

    model_config = _FORBID_FROZEN_CONFIG

    date: str
    band_lo_80: float
    band_hi_80: float
    realized_rv: float | None = None


class RealizedVolForecastResponse(BaseModel):
    """Multi-horizon QLIKE-DLq forecast plus last-60d realized history."""

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    horizons: list[RealizedVolHorizonForecast]
    history: list[float] = Field(default_factory=list)
    history_dates: list[str] = Field(default_factory=list)
    model_revision: str
    historical_bands: list[RealizedVolHistoricalBand] | None = None


class HarTercileHorizon(BaseModel):
    """HAR-tercile baseline classification for one forecast horizon.

    ``predicted_rv`` is HAR's OLS point in realized-variance units.
    ``tercile`` is the argmax bucket against the q33 / q67 cutoffs the
    response also returns; ``tercile_probs`` is the Gaussian-CDF mass
    triple in log-RV space (sums to 1.0). ``macro_f1`` is wired through
    from wiki section 20 — the published HAR-tercile pooled macro-F1 on
    the canonical 5-fold expanding walk-forward (0.687 / 0.685 / 0.654
    at h=1 / 5 / 22). The serving path does not recompute this number;
    it surfaces the offline-measured one so the frontend can render an
    honest chip alongside the live point forecast.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int
    predicted_rv: float
    tercile: Literal["low", "medium", "high"]
    tercile_probs: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class probability over (low, medium, high). Sums to 1.0.",
    )
    macro_f1: float = Field(
        ...,
        description=(
            "Pooled macro-F1 for the HAR-tercile baseline at this horizon, "
            "read off wiki section 20 (Gated_Fusion_InfoNCE_Comprehensive_Null, "
            "Result 2). Not recomputed on the serving path."
        ),
    )
    macro_f1_source: str = Field(
        ...,
        description="Citation for the macro_f1 number (wiki section + result block).",
    )
    qlike_model: float | None = Field(
        default=None,
        description=(
            "Pooled walk-forward QLIKE loss for the QLIKE-DLq ensemble at "
            "this horizon (lower is better). None when the eval sidecar is "
            "missing."
        ),
    )
    qlike_har: float | None = Field(
        default=None,
        description="Pooled walk-forward QLIKE loss for HAR-OLS at this horizon.",
    )


class HarTercileBaselineResponse(BaseModel):
    """Multi-horizon HAR-tercile regime baseline for the headline regime card.

    Per wiki section 20, HAR-tercile is the strongest forward-vol-regime
    classifier on the canonical fold protocol (beats market-only and the
    text+market fused arm at h=1 and h=22; null at h=5). The frontend
    renders this response as the headline regime card and demotes the
    text+market fused card to a "second opinion" with explicit
    weaker-baseline disclosure.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    horizons: list[HarTercileHorizon]
    cutoffs_q33: float = Field(
        ...,
        description=(
            "Lower tercile cutoff on realized variance used to bucket the "
            "HAR point forecast. Derived from the supplied RV history when "
            "the artifact does not pin a per-horizon train-slice cutoff."
        ),
    )
    cutoffs_q67: float = Field(
        ...,
        description="Upper tercile cutoff on realized variance.",
    )
    model_revision: str
    generated_at: str


# Workspace-spine bundle: shared response models for the four
# feature steps that build on top of this foundation. The feature
# steps fill the service-layer builders that populate these
# numbers; this module only owns the wire shape so the frontend
# types and the OpenAPI snapshot can settle ahead of the wiring.
#
# SPINE separation: ExpectedVolumeHorizonForecast /
# ExpectedVolumeForecastResponse are the only forecast surface in
# this bundle (market data only). MonetaryPolicySurpriseResponse,
# FuturesConsensusResponse and SemanticDiffResponse are descriptive
# panels (text- or realized-derived) and never feed forecasts.
class ExpectedVolumeHorizonForecast(BaseModel):
    """HAR-based forecast of expected log-residual trading volume for
    one horizon. ``point_log_residual`` is the model's point estimate
    in log-volume residual space (after calendar adjustment when the
    flag is set); ``point_pct_vs_baseline`` is the same number
    expressed as a % deviation from the rolling calendar-adjusted
    baseline so the card can render a human-readable headline.
    ``r2_har`` is the offline pooled walk-forward R^2 of the HAR
    volume head at this horizon, surfaced for the calibration chip.
    """

    model_config = _FORBID_FROZEN_CONFIG

    h: int
    point_log_residual: float
    point_pct_vs_baseline: float
    band_lo_80: float
    band_hi_80: float
    band_lo_90: float
    band_hi_90: float
    r2_har: float | None = None
    calendar_adjusted: bool


class ExpectedVolumeForecastResponse(BaseModel):
    """Multi-horizon HAR-volume forecast for the Expected Volume card.
    Market-data-only forecast; never wired to text features.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    horizons: list[ExpectedVolumeHorizonForecast]
    model_revision: str
    generated_at: str


class MonetaryPolicySurpriseResponse(BaseModel):
    """Monetary-policy surprise (descriptive panel, not a forecast input).

    ``mp_surprise_level_bps`` is the realized rate-path surprise in
    basis points relative to the pre-meeting fed-funds futures
    consensus. ``direction`` is the discrete sign bucket the panel
    renders; ``no_surprise`` covers the inside-the-band cases.
    ``is_intermeeting`` flags off-cycle actions where the consensus
    baseline is constructed differently.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str
    mp_surprise_level_bps: float
    direction: Literal["hawkish", "dovish", "no_surprise"]
    magnitude_bps: float
    is_intermeeting: bool
    ff_target_prior_bps: float | None = None


class FuturesConsensusHorizon(BaseModel):
    """One horizon of the fed-funds futures implied-path consensus.

    Probabilities are derived from the implied-rate distribution and
    bucketed against the current target band; they sum to 1.0 across
    hike / cut / pause.
    """

    model_config = _FORBID_FROZEN_CONFIG

    horizon_label: str
    implied_rate_bps: float
    change_vs_current_bps: float
    probability_hike: float
    probability_cut: float
    probability_pause: float


class FuturesConsensusResponse(BaseModel):
    """FRED / CME-derived futures consensus panel.

    Descriptive only — the rate-path expectations chart reads off
    realized futures prices and never feeds the forecast cards.
    """

    model_config = _FORBID_FROZEN_CONFIG

    meeting_date: str
    generated_at: str
    current_target_lo_bps: float
    current_target_hi_bps: float
    horizons: list[FuturesConsensusHorizon]
    methodology: str
    data_source: str


class SemanticDiffSpan(BaseModel):
    """One token-aligned span of the current-vs-prior statement diff.

    ``kind`` is the alignment bucket; ``paired_text`` carries the
    matched span on the opposite side for ``substituted`` (and
    optionally for ``added`` / ``removed`` if the aligner emitted a
    near-match neighbour).
    """

    model_config = _FORBID_FROZEN_CONFIG

    kind: Literal["unchanged", "added", "removed", "substituted"]
    text: str
    paired_text: str | None = None


class SemanticDiffTopic(BaseModel):
    """Topic-level emphasis delta across the two statements.

    ``prior_emphasis`` and ``current_emphasis`` are the topic-share
    masses in [0, 1]; ``delta`` = current - prior. ``sample_phrases``
    are the highest-loading n-grams the panel surfaces alongside the
    bar.
    """

    model_config = _FORBID_FROZEN_CONFIG

    topic: str
    prior_emphasis: float
    current_emphasis: float
    delta: float
    sample_phrases: list[str] = Field(default_factory=list)


class SemanticDiffRequest(BaseModel):
    """Inbound body for ``POST /fomc/semantic-diff``.

    ``current_date`` selects the strict-prior FOMC statement off disk;
    ``current_text`` is the pasted body the panel diffs against that
    prior. Both fields are required — the cold-start case is still a
    valid call, the service just returns an empty span list when no
    strict-prior is on file for the supplied date.
    """

    model_config = _STRICT_REQUEST_CONFIG

    current_date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    current_text: str = Field(..., min_length=1, description="FOMC statement text to diff")


class SemanticDiffResponse(BaseModel):
    """Semantic diff between the current statement and its prior.

    Descriptive panel — the spans and topic deltas are post-hoc
    explanations of the realized text change and never feed the
    forecast surface.
    """

    model_config = _FORBID_FROZEN_CONFIG

    current_date: str
    prior_date: str
    token_spans: list[SemanticDiffSpan]
    topic_deltas: list[SemanticDiffTopic]
    summary: str


class HarTercileBacktestRow(BaseModel):
    """One resolved (or pending) row in the HAR-tercile backtest table.

    A row carries the persisted predicted tercile (read off the analyze
    payload's ``regime_classification`` card) and, when the forward
    window has elapsed, the realized tercile bucketed off the same
    cutoffs that produced the prediction. ``correct`` is None for rows
    whose forward window has not yet closed.
    """

    model_config = _FORBID_FROZEN_CONFIG

    event_date: str
    predicted_tercile: str
    predicted_prob: float
    realized_tercile: str | None = None
    realized_rv: float | None = None
    correct: bool | None = None


class HarAccuracyMetrics(BaseModel):
    """Aggregate accuracy KPIs across the HAR-tercile backtest rows.

    ``total_runs`` is the number of rows in the window, regardless of
    whether their forward window has resolved. ``resolved_runs`` counts
    rows whose realized tercile could be derived. ``accuracy_overall``
    is the hit rate across resolved rows only; ``per_tercile_hit_rate``
    keys are the predicted-tercile labels and values are per-label hit
    rates (denominator = resolved rows whose prediction was that label).
    """

    model_config = _FORBID_FROZEN_CONFIG

    total_runs: int
    resolved_runs: int
    accuracy_overall: float | None = None
    per_tercile_hit_rate: dict[str, float] = Field(default_factory=dict)


class HarTercileBacktestResponse(BaseModel):
    """Response wire shape for ``GET /forecast/har-tercile-backtest``.

    Surfaces the last N persisted ^GSPC analyze runs with their stored
    HAR-tercile prediction and the realized tercile derived from
    forward market history. Drives the HarAccuracyPanel card.
    """

    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    horizon: int
    rows: list[HarTercileBacktestRow]
    metrics: HarAccuracyMetrics
    generated_at: str
