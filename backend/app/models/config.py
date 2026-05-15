from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.config import DATA_DIR, MODEL_CHECKPOINT_DIR

# v2 reference: 20-day lookback over daily bars. v1 used 5 (sub-week)
# which was too short for the recurrent core to learn temporal structure.
# Tests/regression callers reference this constant; on-disk checkpoints
# persist their training-time value via training/checkpoint.py.
SEQUENCE_LENGTH = 20
FEATURE_SIZE = 6  # [sentiment_score, market_close, market_volatility, close_change_pct, volatility_change, elapsed_time]
SENTIMENT_FEATURE_INDEX = 0
ELAPSED_TIME_FEATURE_INDEX = 5
FORECAST_CONFIDENCE_LEVEL = 0.8
CONFIDENCE_Z_SCORE = 1.2816  # Approximate central 80% interval
DEFAULT_CLOSE_SCALE = 10000.0
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.15
DEFAULT_HEAD_HIDDEN_SIZE = 32
DEFAULT_INITIAL_DECAY_RATE = 1.5
DEFAULT_CHUNK_DECAY_RATE = 1.0 / 30.0
DEFAULT_CHUNK_EMBEDDING_SIZE = 768
DEFAULT_CHUNK_PROJECTION_DIM = 8
CREDIBILITY_FEATURE_DIM = 4

DEFAULT_DATA_DIR = DATA_DIR
MODELS_DIR = MODEL_CHECKPOINT_DIR
BEST_MODEL_PATH = MODELS_DIR / "forecaster_best.pt"


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = FEATURE_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE
    initial_decay_rate: float = DEFAULT_INITIAL_DECAY_RATE
    text_channel: str = "scalar"
    embedding_adapter_dim: int = 128
    credibility_features: bool = False

    @classmethod
    def from_model(cls, model: "Any") -> "ModelConfig":
        return cls(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout,
            head_hidden_size=model.head_hidden_size,
            initial_decay_rate=model.initial_decay_rate,
            text_channel=getattr(model, "text_channel", "scalar"),
            embedding_adapter_dim=getattr(model, "chunk_projection_dim", 128) or 128,
            credibility_features=bool(getattr(model, "credibility_features", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureVector:
    date: str
    sentiment_score: float
    market_close: float
    market_volatility: float
    close_change_pct: float = 0.0
    volatility_change: float = 0.0
    elapsed_time: float = 0.0
    text_embedding: list[float] | None = None

    @classmethod
    def from_market_state(
        cls,
        *,
        date: str,
        sentiment_score: float,
        market_close: float,
        market_volatility: float,
        previous_close: float | None = None,
        previous_volatility: float | None = None,
        elapsed_time: float = 0.0,
        text_embedding: list[float] | None = None,
    ) -> "FeatureVector":
        close_change_pct = 0.0
        if previous_close is not None and abs(previous_close) > 1e-12:
            close_change_pct = (float(market_close) - float(previous_close)) / float(previous_close)

        volatility_change = 0.0
        if previous_volatility is not None:
            volatility_change = float(market_volatility) - float(previous_volatility)

        return cls(
            date=date,
            sentiment_score=float(sentiment_score),
            market_close=float(market_close),
            market_volatility=float(market_volatility),
            close_change_pct=float(close_change_pct),
            volatility_change=float(volatility_change),
            elapsed_time=float(elapsed_time),
            text_embedding=list(text_embedding) if text_embedding is not None else None,
        )

    def as_list(self, close_scale: float = DEFAULT_CLOSE_SCALE) -> list[float]:
        return [
            float(self.sentiment_score),
            float(self.market_close) / close_scale,
            float(self.market_volatility),
            max(min(float(self.close_change_pct), 1.0), -1.0),
            max(min(float(self.volatility_change), 1.0), -1.0),
            float(self.elapsed_time) / 30.0,
        ]


def build_lookback_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    """Pad-front (with the oldest vector) or truncate to a fixed lookback window."""

    items = list(vectors)
    if not items:
        items = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items
