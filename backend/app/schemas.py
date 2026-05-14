from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="FOMC statement text")
    date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    symbol: str = Field("^GSPC", description="Market ticker, e.g. ^GSPC or DX-Y.NYB")
    horizon: str = Field("3d", description="Forecast horizon label")
    forecast_mode: str = Field("fast", description="Forecast mode: fast, quick_train, or real_train")
    include_realized: bool = Field(
        False,
        description="When true and date is in the past, include realized forward series overlay.",
    )


class SentimentResponse(BaseModel):
    label: str
    score: float
    raw: list[dict[str, float | str]]


class MarketDataResponse(BaseModel):
    symbol: str
    requested_date: str
    date_used: str
    lookback_days: int
    close: float
    volatility_5d: float


class PredictionResponse(BaseModel):
    close: float
    volatility: float
    horizon: str


class ChunkAttentionDiagnostics(BaseModel):
    chunk_count: int
    weights: list[float]
    decay_coeffs: list[float]
    chunk_previews: list[str]
    lambda_value: float


class ModelDiagnosticsResponse(BaseModel):
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


class AnalyzeResponse(BaseModel):
    sentiment: SentimentResponse
    prediction: PredictionResponse
    market: MarketDataResponse
    model: ModelDiagnosticsResponse
    series: ForecastSeriesResponse


class TrainJobAcceptedResponse(BaseModel):
    status: str = "queued"
    job_id: str
    message: str


class TrainJobStatusResponse(BaseModel):
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
