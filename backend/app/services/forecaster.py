from __future__ import annotations

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
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step)


@dataclass
class FeatureVector:
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


def build_last5_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    items = list(vectors)
    if not items:
        items = [FeatureVector(sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items


def predict_volatility(sequence: list[FeatureVector], horizon: str = "3d") -> dict[str, float | str]:
    model = _get_model()
    fixed_sequence = build_last5_sequence(sequence)

    tensor = torch.tensor(
        [[vector.as_list() for vector in fixed_sequence]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        prediction = model(tensor).squeeze().item()

    return {"volatility": float(prediction), "horizon": horizon}
