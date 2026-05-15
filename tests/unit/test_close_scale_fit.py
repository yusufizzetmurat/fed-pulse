"""Per-fold close-scale fitting + checkpoint persistence.

The forecaster used to scale close prices by a global ``DEFAULT_CLOSE_SCALE
= 10000`` constant. Real assets trade across orders of magnitude (treasury
yields near 5, crypto in the tens of thousands, FX pairs near 1), so the
constant tilted the loss surface whenever the actual scale diverged. The
training loop now fits the scale on each fold's training rows; the fitted
value rides on the checkpoint so inference recovers the correct magnitude.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import (  # noqa: E402
    DEFAULT_CLOSE_SCALE,
    FeatureVector,
    ModelConfig,
)
from app.services.forecaster import train_model  # noqa: E402
from app.training.loaders import (  # noqa: E402
    _build_training_tensors,
    fit_close_scale,
)


def _vectors(n: int, close_at: float = 4000.0) -> list[FeatureVector]:
    out: list[FeatureVector] = []
    for i in range(n):
        sentiment = math.sin(i * 0.41) * 0.6 + 0.1
        close = close_at + 12.0 * i + 5.0 * math.sin(i * 0.3)
        vol = 0.012 + 0.003 * math.sin(i * 0.7 + 1.1)
        prev_close = (
            close_at + 12.0 * (i - 1) + 5.0 * math.sin((i - 1) * 0.3) if i else close
        )
        prev_vol = 0.012 + 0.003 * math.sin((i - 1) * 0.7 + 1.1) if i else vol
        out.append(
            FeatureVector.from_market_state(
                date=f"2024-01-{i + 1:02d}",
                sentiment_score=sentiment,
                market_close=close,
                market_volatility=vol,
                previous_close=prev_close,
                previous_volatility=prev_vol,
                elapsed_time=float(i % 30),
            )
        )
    return out


def test_fit_close_scale_returns_mean_of_target_closes() -> None:
    vectors = _vectors(50, close_at=5000.0)
    scale = fit_close_scale([vectors])
    # The fit walks the same windows as the tensor builder — it ignores the
    # first SEQUENCE_LENGTH rows because those become only lookback frames,
    # never y rows. The mean over the remaining target rows must land in
    # the same ballpark as the input close range.
    assert 4500.0 < scale < 5500.0


def test_fit_close_scale_is_deterministic() -> None:
    vectors = _vectors(50, close_at=200.0)
    first = fit_close_scale([vectors])
    second = fit_close_scale([vectors])
    # Bit-identical determinism: the fit is a deterministic mean, not a
    # stochastic estimator. The regression test asserts the wider training
    # loop stays byte-identical across reruns; this micro-test isolates the
    # scaler so a future refactor cannot quietly add randomness.
    assert first == second


def test_fit_close_scale_falls_back_when_no_positive_closes() -> None:
    # Empty input -> fallback constant. Same with all-zero closes.
    assert fit_close_scale([]) == float(DEFAULT_CLOSE_SCALE)
    zero_vectors = [
        FeatureVector(
            date=f"2024-01-{i + 1:02d}",
            sentiment_score=0.0,
            market_close=0.0,
            market_volatility=0.0,
        )
        for i in range(50)
    ]
    assert fit_close_scale([zero_vectors]) == float(DEFAULT_CLOSE_SCALE)


def test_build_training_tensors_returns_fitted_close_scale() -> None:
    vectors = _vectors(50, close_at=3000.0)
    x, y, scale = _build_training_tensors([vectors])
    assert x is not None
    assert y is not None
    expected = fit_close_scale([vectors])
    assert scale == expected
    # Targets are stored as `close / scale`, so multiplying back must
    # recover the original close magnitude (modulo float).
    recovered = float(y[0, 0]) * scale
    assert 2500.0 < recovered < 3500.0


def test_train_model_persists_fitted_close_scale_in_checkpoint(tmp_path: Path) -> None:
    """End-to-end: a checkpoint round-trip must carry the fitted scaler.

    Without this, inference uses the legacy constant and predictions land
    an order of magnitude off whenever the training asset trades far from
    5000.
    """

    vectors = _vectors(60, close_at=120.0)  # Treasury-like magnitude.
    checkpoint = tmp_path / "scaler_roundtrip.pt"
    result = train_model(
        vectors=vectors,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        validation_split=0.2,
        early_stopping_patience=2,
        save_checkpoint=True,
        checkpoint_path=checkpoint,
        device="cpu",
        seed=11,
        model_config=ModelConfig(
            input_size=6, hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=8
        ),
    )
    assert result.summary.checkpoint_saved is True
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "close_scale" in payload
    persisted = float(payload["close_scale"])
    expected = fit_close_scale([vectors])
    assert persisted == pytest.approx(expected, rel=1e-6, abs=1e-8)
    # Sanity: the persisted scale tracks the input magnitude, not the
    # legacy 10000 constant.
    assert persisted != float(DEFAULT_CLOSE_SCALE)
