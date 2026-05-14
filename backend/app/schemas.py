from pydantic import BaseModel, ConfigDict, Field


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
    forecast_mode: str = Field("fast", description="Forecast mode: fast, quick_train, or real_train")
    include_realized: bool = Field(
        False,
        description="When true and date is in the past, include realized forward series overlay.",
    )
    include_xai: bool = Field(
        False,
        description="When true, return per-sentence + per-token XAI attribution alongside the forecast.",
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


class AnalyzeResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    sentiment: SentimentResponse
    prediction: PredictionResponse
    market: MarketDataResponse
    model: ModelDiagnosticsResponse
    series: ForecastSeriesResponse
    xai: XaiResponse | None = None


class TrainJobAcceptedResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    status: str = "queued"
    job_id: str
    message: str


class TrainJobStatusResponse(BaseModel):
    model_config = _FORBID_FROZEN_CONFIG

    job_id: str
    status: str
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: AnalyzeResponse | None = None


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


class HistoryDetail(HistoryEntry):
    payload: dict


class HistoryList(BaseModel):
    items: list[HistoryEntry]
    total: int
    limit: int
    offset: int


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
