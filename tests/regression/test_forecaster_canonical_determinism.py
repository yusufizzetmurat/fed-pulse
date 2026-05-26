"""Canonical determinism guard for the post-#322 / post-#336 split.

Replaces the pre-#322 v1 byte-identity contract with a meaningful
determinism regression on the current canonical config. The pinned
config carries the regression-canonical head_mode (ADR 0015 / #322)
plus the research-vs-serving split landed in #336: research-class
construction at seed=11 must produce the same loss two runs in a row,
and the promotion path into the serving class must preserve the
relevant forward outputs byte-for-byte.

The classification-head path (regime card surface) is also covered so
the regime classification card flow off /analyze keeps working through
the split.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig  # noqa: E402
from app.models.factory import (  # noqa: E402
    build_research_forecaster,
    build_serving_forecaster,
)
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
        prev_vol = (
            0.012 + 0.003 * math.sin((i - 1) * 0.7 + 1.1) if i else vol
        )
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


def _canonical_config() -> ModelConfig:
    """Post-#322 canonical config: regression-output, lstm core,
    head_mode='regression' (ADR 0015), no rates heads, no text path."""
    return ModelConfig(
        input_size=6,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        head_hidden_size=8,
        architecture="lstm",
        output_mode="regression",
        head_mode="regression",
    )


def _train_canonical_combined_rmse(
    seed: int, vectors: list[FeatureVector], checkpoint: Path
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
        model_config=_canonical_config(),
    )
    metrics = result.summary.metrics
    assert metrics is not None
    return float(metrics.combined_rmse)


def test_canonical_config_training_loss_is_deterministic_at_seed_11(
    tmp_path: Path,
) -> None:
    """Two runs of the canonical config at seed=11 must agree to 1e-8."""
    vectors = _synthetic_vectors(28)
    first = _train_canonical_combined_rmse(
        seed=11, vectors=vectors, checkpoint=tmp_path / "first.pt"
    )
    second = _train_canonical_combined_rmse(
        seed=11, vectors=vectors, checkpoint=tmp_path / "second.pt"
    )
    assert first == pytest.approx(second, abs=1e-8), (
        f"canonical-determinism regression: seed=11 produced two different "
        f"combined RMSE values ({first} vs {second}); a non-deterministic "
        f"path leaked into the post-#322 regression-canonical config."
    )


def test_research_and_serving_models_forward_identically_under_same_seed() -> None:
    """Research + serving classes constructed from the same config and
    seeded identically must emit byte-identical outputs on the same
    inputs.

    The #336 split moved the input-prep onto a shared helper; this is
    the contract that protects the helper extraction. The classes carry
    the same backbone, the same head construction (under
    output_mode='regression'), and the same state_dict key layout, so
    on identical RNG state the forward output must agree exactly.
    """
    config = _canonical_config()

    torch.manual_seed(11)
    research = build_research_forecaster(config)
    research.eval()

    torch.manual_seed(11)
    serving = build_serving_forecaster(config)
    serving.eval()

    x = torch.randn(2, 5, 6)
    with torch.no_grad():
        research_out = research(x)
        serving_out = serving(x)

    assert research_out.shape == serving_out.shape == (2, 2)
    assert torch.allclose(research_out, serving_out, atol=0.0, rtol=0.0), (
        "#336 split broke output parity between research + serving "
        "forecasters on the canonical regression-output config; "
        "the input-prep extraction or the head wiring drifted."
    )


def test_regime_card_classification_forward_is_deterministic() -> None:
    """The regime / market-reaction card surface (#322) drives
    ``forward_multi_task`` on the serving class. Pin determinism on
    that path so regime card output stays reproducible across runs.
    """
    cls_config = ModelConfig(
        input_size=6,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        head_hidden_size=8,
        architecture="lstm",
        output_mode="classification",
        head_mode="regression",
        vol_regime_quantiles=(-2.0, -1.0),
    )
    torch.manual_seed(29)
    serving_a = build_serving_forecaster(cls_config)
    serving_a.eval()
    torch.manual_seed(29)
    serving_b = build_serving_forecaster(cls_config)
    serving_b.eval()

    x = torch.randn(3, 5, 6)
    with torch.no_grad():
        out_a = serving_a.forward_multi_task(x)
        out_b = serving_b.forward_multi_task(x)

    assert set(out_a.keys()) == set(out_b.keys())
    for key in out_a:
        assert torch.allclose(out_a[key], out_b[key], atol=0.0, rtol=0.0), (
            f"regime card forward_multi_task drifted on key {key!r} "
            "between two identically-seeded serving instances; "
            "the regime card surface is non-deterministic."
        )
    # The classification surface must expose `stance` (regime axis)
    # and `log_rv` (the regression-canonical scalar the UI buckets).
    assert "stance" in out_a
    assert "log_rv" in out_a
