from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

SEQUENCE_LENGTH = 5
FEATURE_SIZE = 3  # [sentiment_score, market_close, market_volatility]


class ForecasterModel(nn.Module):
    def __init__(self, input_size: int = FEATURE_SIZE, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step)


@dataclass
class FeatureVector:
    date: str
    sentiment_score: float
    market_close: float
    market_volatility: float

    def as_list(self, close_scale: float = 10000.0) -> list[float]:
        # Normalize close price to stable range for MVP inference.
        return [
            float(self.sentiment_score),
            float(self.market_close) / close_scale,
            float(self.market_volatility),
        ]


_model: ForecasterModel | None = None
_model_lock = threading.Lock()


def _get_model() -> ForecasterModel:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            model = ForecasterModel()
            model.eval()
            _model = model
    return _model


def _parse_horizon_steps(horizon: str) -> int:
    if horizon.endswith("d") and horizon[:-1].isdigit():
        return max(1, int(horizon[:-1]))
    return 3


def parse_horizon_steps(horizon: str) -> int:
    return _parse_horizon_steps(horizon)


def build_last5_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    items = list(vectors)
    if not items:
        items = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items


def _train_quick(model: ForecasterModel, vectors: list[FeatureVector], epochs: int = 18) -> ForecasterModel:
    if len(vectors) < SEQUENCE_LENGTH + 1:
        return model

    work_model = copy.deepcopy(model)
    work_model.train()
    optimizer = torch.optim.Adam(work_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    sequences: list[list[list[float]]] = []
    targets: list[list[float]] = []
    for idx in range(SEQUENCE_LENGTH, len(vectors)):
        window = vectors[idx - SEQUENCE_LENGTH : idx]
        target = vectors[idx]
        sequences.append([item.as_list() for item in window])
        targets.append(
            [
                min(max(target.market_close / 10000.0, 0.0), 1.0),
                min(max(target.market_volatility, 0.0), 1.0),
            ]
        )

    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = work_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    work_model.eval()
    return work_model


def _predict_next_point(model: ForecasterModel, sequence: list[FeatureVector]) -> tuple[float, float]:
    x = torch.tensor([[item.as_list() for item in sequence]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x).squeeze(0)
    pred_close = float(out[0].item()) * 10000.0
    pred_vol = float(out[1].item())
    return pred_close, pred_vol


def forecast_quantitative_series(
    vectors: list[FeatureVector],
    forecast_mode: str = "fast",
    horizon: str = "3d",
) -> dict[str, object]:
    if not vectors:
        vectors = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    base_model = _get_model()
    model = _train_quick(base_model, vectors) if forecast_mode == "quick_train" else base_model

    history_vectors = vectors[-30:]
    history_timestamps = [item.date for item in history_vectors]
    history_close = [float(item.market_close) for item in history_vectors]
    history_vol = [float(item.market_volatility) for item in history_vectors]

    steps = _parse_horizon_steps(horizon)
    rolling = history_vectors[-SEQUENCE_LENGTH:]
    forecast_close: list[float] = []
    forecast_vol: list[float] = []
    forecast_timestamps: list[str] = []

    last_date = history_timestamps[-1] if history_timestamps else ""
    for step in range(steps):
        fixed_sequence = build_last5_sequence(rolling)
        next_close, next_vol = _predict_next_point(model, fixed_sequence)

        next_date_label = f"{last_date}+{step + 1}" if last_date else f"t+{step + 1}"
        next_vector = FeatureVector(
            date=next_date_label,
            sentiment_score=float(fixed_sequence[-1].sentiment_score),
            market_close=next_close,
            market_volatility=next_vol,
        )
        rolling = (rolling + [next_vector])[-SEQUENCE_LENGTH:]

        forecast_timestamps.append(next_date_label)
        forecast_close.append(next_close)
        forecast_vol.append(next_vol)

    vol_values = [*history_vol, *forecast_vol]
    if vol_values:
        vol_min = min(vol_values)
        vol_max = max(vol_values)
        spread = max(vol_max - vol_min, 1e-6)
        vol_scale = {
            "suggested_ymin": max(0.0, vol_min - spread * 0.15),
            "suggested_ymax": vol_max + spread * 0.15,
        }
    else:
        vol_scale = {"suggested_ymin": 0.0, "suggested_ymax": 1.0}

    return {
        "prediction": {
            "close": float(forecast_close[-1]),
            "volatility": float(forecast_vol[-1]),
            "horizon": horizon,
        },
        "series": {
            "timestamps": history_timestamps,
            "history_close": history_close,
            "history_volatility": history_vol,
            "forecast_timestamps": forecast_timestamps,
            "forecast_close": forecast_close,
            "forecast_volatility": forecast_vol,
            "volatility_scale": vol_scale,
        },
    }
