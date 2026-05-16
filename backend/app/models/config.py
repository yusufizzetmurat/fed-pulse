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

# Rich-feature input space (PR-#173 onward). The training-package loader
# joins the four feature families produced under Phase 8 onto the
# per-bar feature vector:
#
# - 4 credibility fields (drift_score, realized_vs_stated_gap,
#   market_implied_gap, months_since_reversal) -- direct off the
#   events.parquet row.
# - 15 linguistic features (8 LDA topic shares + 6 hand-crafted
#   densities + pivot_distance) -- joined on text_hash from
#   linguistic_features.parquet.
# - 4 MP-surprise fields (mp_surprise_level, mp_surprise_path_factor,
#   fed_info_factor, mp_is_intermeeting) -- joined on event_date from
#   mp_surprises.parquet. ``mp_is_intermeeting`` is the boolean
#   ``is_intermeeting`` field encoded as 0.0 / 1.0.
# - 6 multi-axis fields (axis_factor, axis_certainty, axis_time, each
#   with a paired *_missing flag) -- direct off the events.parquet row.
#
# The event-level signal is broadcast to every bar of the 20-day prior
# window plus the appended event-day target frame, so every bar in a
# supervised window carries the same rich-feature row.
#
# Per-bar slice ordering (deterministic; documented on
# ``FeatureVector`` below):
#
#   [0:6]    market features (existing FEATURE_SIZE slice)
#   [6:10]   credibility 4-vector
#   [10:25]  linguistic 15-vector
#   [25:29]  MP-surprise 4-vector
#   [29:35]  multi-axis 6-vector (3 values + 3 missing flags)
#
# ``RICH_FEATURE_SIZE`` is the constant downstream model factories /
# CLI use to widen the input projection when ``rich_features=True``.
RICH_CREDIBILITY_DIM = 4
RICH_LINGUISTIC_DIM = 15
RICH_MP_SURPRISE_DIM = 4
RICH_MULTI_AXIS_DIM = 6
RICH_EXTRA_FEATURE_SIZE = (
    RICH_CREDIBILITY_DIM
    + RICH_LINGUISTIC_DIM
    + RICH_MP_SURPRISE_DIM
    + RICH_MULTI_AXIS_DIM
)
RICH_FEATURE_SIZE = FEATURE_SIZE + RICH_EXTRA_FEATURE_SIZE

# Slice offsets inside the rich vector. Used by the per-family
# ablation path on the loader to zero an individual family without
# changing the per-bar feature size; a downstream sweep can then
# measure per-family lift while keeping the model input shape
# constant.
RICH_MARKET_SLICE = slice(0, FEATURE_SIZE)
RICH_CREDIBILITY_SLICE = slice(
    FEATURE_SIZE, FEATURE_SIZE + RICH_CREDIBILITY_DIM
)
RICH_LINGUISTIC_SLICE = slice(
    RICH_CREDIBILITY_SLICE.stop,
    RICH_CREDIBILITY_SLICE.stop + RICH_LINGUISTIC_DIM,
)
RICH_MP_SURPRISE_SLICE = slice(
    RICH_LINGUISTIC_SLICE.stop,
    RICH_LINGUISTIC_SLICE.stop + RICH_MP_SURPRISE_DIM,
)
RICH_MULTI_AXIS_SLICE = slice(
    RICH_MP_SURPRISE_SLICE.stop,
    RICH_MP_SURPRISE_SLICE.stop + RICH_MULTI_AXIS_DIM,
)

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


FORECASTER_ARCHITECTURES: tuple[str, ...] = (
    "lstm",
    "lstm_attn",
    "gru",
    "tcn",
    "transformer",
    "dlinear",
    "informer",
    "tft",
)


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
    architecture: str = "lstm"

    @classmethod
    def from_model(cls, model: "Any") -> "ModelConfig":
        architecture = getattr(model, "model_type", None) or "lstm"
        if architecture not in FORECASTER_ARCHITECTURES:
            architecture = "lstm"
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
            architecture=str(architecture),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureVector:
    """Per-bar feature row consumed by the forecaster.

    The 6 market fields (``sentiment_score`` through ``elapsed_time``)
    are the legacy ``FEATURE_SIZE`` input; ``as_list`` emits exactly
    that slice and is back-compat with every pre-PR-#173 inference and
    training path.

    The trailing fields carry the rich-feature input (``RICH_FEATURE_SIZE
    = 35``) added in PR #173. They are populated by
    ``app.training.loaders.load_training_sequences_from_package`` when
    ``rich_features=True``; on the legacy path they stay at their
    documented defaults so ``as_list`` and ``as_rich_list`` agree on
    the 6 market positions. ``as_rich_list`` emits the full 35-dim
    layout in the order documented at the module-level slice
    constants:

    - positions ``[0:6]`` -- market features.
    - positions ``[6:10]`` -- credibility 4-vector
      (``credibility_drift_score`` / ``credibility_realized_vs_stated_gap``
      / ``credibility_market_implied_gap`` /
      ``credibility_months_since_reversal``).
    - positions ``[10:25]`` -- 15-dim linguistic vector
      (8 LDA topic shares + 6 hand-crafted densities +
      ``pivot_distance``), in the same field order as
      ``app.features.linguistic.LinguisticVector``.
    - positions ``[25:29]`` -- MP-surprise 4-vector
      (``mp_surprise_level`` / ``mp_surprise_path_factor`` /
      ``fed_info_factor`` / ``mp_is_intermeeting``).
    - positions ``[29:35]`` -- 6-dim multi-axis vector
      (``axis_factor`` / ``axis_factor_missing`` /
      ``axis_certainty`` / ``axis_certainty_missing`` /
      ``axis_time`` / ``axis_time_missing``). NaN inputs collapse to
      ``0.0`` and the paired ``*_missing`` flag flips to ``1.0`` so
      the model can tell "no signal" apart from "neutral signal".
    """

    date: str
    sentiment_score: float
    market_close: float
    market_volatility: float
    close_change_pct: float = 0.0
    volatility_change: float = 0.0
    elapsed_time: float = 0.0
    text_embedding: list[float] | None = None
    # Rich-feature payload (PR #173). Default values match
    # "all-zero / no-signal" so a FeatureVector built via the legacy
    # constructors round-trips ``as_rich_list`` to the existing
    # ``as_list`` plus zero-padding. The loader sets ``rich_payload``
    # to ``True`` after populating the trailing fields; the tensor
    # builder dispatches on that flag.
    credibility_drift_score: float = 0.0
    credibility_realized_vs_stated_gap: float = 0.0
    credibility_market_implied_gap: float = 0.0
    credibility_months_since_reversal: float = 0.0
    linguistic_features: list[float] | None = None
    mp_surprise_level: float = 0.0
    mp_surprise_path_factor: float = 0.0
    fed_info_factor: float = 0.0
    mp_is_intermeeting: float = 0.0
    axis_factor: float = 0.0
    axis_factor_missing: float = 1.0
    axis_certainty: float = 0.0
    axis_certainty_missing: float = 1.0
    axis_time: float = 0.0
    axis_time_missing: float = 1.0
    rich_payload: bool = False

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

    def as_rich_list(self, close_scale: float = DEFAULT_CLOSE_SCALE) -> list[float]:
        """Emit the full 35-dim per-bar feature vector.

        Layout matches the slice constants at the top of this module
        and the docstring on :class:`FeatureVector`. The first six
        positions are byte-identical to :meth:`as_list` so models
        widened to ``RICH_FEATURE_SIZE`` still see the legacy market
        signal in positions ``[0:6]``.
        """

        market = self.as_list(close_scale=close_scale)
        credibility = [
            float(self.credibility_drift_score),
            float(self.credibility_realized_vs_stated_gap),
            float(self.credibility_market_implied_gap),
            float(self.credibility_months_since_reversal),
        ]
        linguistic_source = self.linguistic_features or []
        linguistic = [float(v) for v in linguistic_source[:RICH_LINGUISTIC_DIM]]
        if len(linguistic) < RICH_LINGUISTIC_DIM:
            linguistic = linguistic + [0.0] * (RICH_LINGUISTIC_DIM - len(linguistic))
        mp_surprise = [
            float(self.mp_surprise_level),
            float(self.mp_surprise_path_factor),
            float(self.fed_info_factor),
            float(self.mp_is_intermeeting),
        ]
        multi_axis = [
            float(self.axis_factor),
            float(self.axis_factor_missing),
            float(self.axis_certainty),
            float(self.axis_certainty_missing),
            float(self.axis_time),
            float(self.axis_time_missing),
        ]
        return market + credibility + linguistic + mp_surprise + multi_axis


def build_lookback_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    """Pad-front (with the oldest vector) or truncate to a fixed lookback window."""

    items = list(vectors)
    if not items:
        items = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items
