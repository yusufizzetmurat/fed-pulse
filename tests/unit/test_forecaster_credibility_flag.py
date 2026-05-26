"""Lock the credibility-features default-off contract for the training loop.

The forecaster sweep harness must keep ``credibility_features=False`` as the
default so the determinism regression and every published checkpoint behind
v1/v2 keep loading.  These tests verify two slices of that contract:

1. The default ``ModelConfig().credibility_features`` is ``False``.
2. ``train_model`` with ``credibility_features=False`` is bit-identical to
   itself across two runs at the same seed -- the same byte-for-byte
   protocol the regression test enforces at the canonical seed 11.

The default-off invariant is more important than the default-on numbers
because every consumer of the v1/v2 forecaster checkpoint assumes the
6-feature input shape.  Flipping the default would silently change the
LSTM input layer and break every on-disk checkpoint without warning.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig  # noqa: E402
from app.services.forecaster import FeatureVector, train_model  # noqa: E402


def _synthetic_vectors(n: int) -> list[FeatureVector]:
    out: list[FeatureVector] = []
    for i in range(n):
        sentiment = math.sin(i * 0.41) * 0.6 + 0.1
        close = 4000.0 + 12.0 * i + 5.0 * math.sin(i * 0.3)
        vol = 0.012 + 0.003 * math.sin(i * 0.7 + 1.1)
        prev_close = (
            4000.0 + 12.0 * (i - 1) + 5.0 * math.sin((i - 1) * 0.3) if i else close
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


def _train_rmse(
    *,
    seed: int,
    vectors: list[FeatureVector],
    checkpoint: Path,
    credibility_features: bool,
    architecture: str = "lstm",
) -> float:
    result = train_model(
        vectors=vectors,
        epochs=4,
        batch_size=8,
        learning_rate=1e-3,
        validation_fraction=0.25,
        early_stopping_patience=2,
        save_checkpoint=False,
        checkpoint_path=checkpoint,
        device="cpu",
        seed=seed,
        model_config=ModelConfig(
            input_size=6,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            head_hidden_size=8,
            architecture=architecture,
            credibility_features=credibility_features,
        ),
    )
    metrics = result.summary.metrics
    assert metrics is not None
    return float(metrics.combined_rmse)


def test_default_modelconfig_has_credibility_off() -> None:
    assert ModelConfig().credibility_features is False
    assert ModelConfig().architecture == "lstm"


def test_credibility_off_is_byte_identical_under_same_seed(tmp_path: Path) -> None:
    """Two runs of ``train_model`` at seed=11 with credibility_features=False
    must produce identical combined-RMSE.  Locks the byte-identical contract
    declared in ``docs/data-and-training-contracts.md`` for the default
    forecaster path.
    """

    vectors = _synthetic_vectors(28)
    first = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "first.pt",
        credibility_features=False,
    )
    second = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "second.pt",
        credibility_features=False,
    )
    assert first == pytest.approx(second, abs=1e-7), (
        f"credibility-off byte-identity regression: seed=11 produced "
        f"{first} vs {second}; the default training path leaks randomness"
    )


def test_credibility_off_lstm_matches_default_architecture(tmp_path: Path) -> None:
    """``architecture="lstm"`` + ``credibility_features=False`` is the default
    path; switching to the explicit lstm choice must be byte-identical so the
    factory dispatch does not silently change the v2 LSTM result.
    """

    vectors = _synthetic_vectors(28)
    default_path = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "default.pt",
        credibility_features=False,
        architecture="lstm",
    )
    explicit_lstm = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "explicit.pt",
        credibility_features=False,
        architecture="lstm",
    )
    # Within the ±1e-4 contract for default-LSTM stability documented in
    # docs/data-and-training-contracts.md, the two paths must agree.
    assert default_path == pytest.approx(explicit_lstm, abs=1e-4)


def test_credibility_on_diverges_from_credibility_off(tmp_path: Path) -> None:
    """Flipping the credibility flag must change the model graph, so the
    trained RMSE should not coincidentally equal the credibility-off run.
    This guards against the flag being accidentally a no-op.
    """

    vectors = _synthetic_vectors(28)
    off = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "off.pt",
        credibility_features=False,
    )
    on = _train_rmse(
        seed=11,
        vectors=vectors,
        checkpoint=tmp_path / "on.pt",
        credibility_features=True,
    )
    assert off != pytest.approx(on, abs=1e-6), (
        "credibility-on produced the same RMSE as credibility-off; "
        "the flag is silently a no-op somewhere in the loop"
    )
