from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="FOMC statement text")
    date: str = Field(..., description="Document date in ISO format: YYYY-MM-DD")
    symbol: str = Field("^GSPC", description="Market ticker, e.g. ^GSPC or DX-Y.NYB")
    horizon: str = Field("3d", description="Forecast horizon label")


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
    volatility: float
    horizon: str


class AnalyzeResponse(BaseModel):
    sentiment: SentimentResponse
    prediction: PredictionResponse
    market: MarketDataResponse
