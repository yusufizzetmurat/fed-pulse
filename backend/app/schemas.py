from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_STRICT_REQUEST_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)
# Response models stay open to extras so the OpenAPI snapshot does not churn;
# `frozen` still blocks mutation after construction.
_FORBID_FROZEN_CONFIG = ConfigDict(frozen=True)


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
    model_config = _FORBID_FROZEN_CONFIG

    symbol: str
    requested_date: str
    date_used: str
    lookback_days: int
    close: float
    volatility_5d: float


class PredictionResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

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


class XaiResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    method: str = "keyword_salience_v1"
    sentences: list[XaiSentenceAttribution] = Field(default_factory=list)


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


class MultiAxisFactorCard(BaseModel):
    """Forward-guidance factor regression in [-1, 1].

    Positive values lean hawkish, negative values lean dovish. Sourced
    from the multi-task head's tanh-bounded regression branch.
    Confidence reflects training-time coverage and is left to the
    caller to calibrate; absent supervision, the field is None.
    """

    model_config = _FORBID_FROZEN_CONFIG

    value: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class MultiAxisCertaintyCard(BaseModel):
    """Certainty prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="certain | uncertain | neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(default_factory=dict)


class MultiAxisTopicCard(BaseModel):
    """Topic prediction from the multi-task head."""

    model_config = _FORBID_FROZEN_CONFIG

    label: str = Field(..., description="macro | forward_guidance | market_reaction | other")
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: dict[str, float] = Field(default_factory=dict)


class MultiAxisBlock(BaseModel):
    """Multi-task head per-axis predictions surfaced on /analyze (#78).

    The four axes mirror the multi-task head's four output branches.
    Stance reuses the canonical 3-class classifier (also exposed on
    the legacy ``sentiment`` field for back-compat); the other three
    branches are populated for the first time with this block. Axes
    whose checkpoint was trained on very few labels are flagged as
    low-confidence; the frontend renders a muted card in that case.
    """

    model_config = _FORBID_FROZEN_CONFIG

    stance: MultiAxisStanceCard
    factor: MultiAxisFactorCard | None = None
    certainty: MultiAxisCertaintyCard | None = None
    topic: MultiAxisTopicCard | None = None


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
