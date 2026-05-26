"""Derived-text-features ablation wiring (#309).

The forecaster head currently reads two text paths in parallel:
(a) the encoder-pooled embedding, and (b) lossy per-bar derived
features (``sentiment_score``, multi-axis stance / certainty /
topic). The #309 ablation toggles path (b) off so the three-way
comparison (baseline / ablation / replacement) can quantify whether
the derived features carry forecaster-relevant signal over the
document-level encoder path.

These tests pin the wiring at three layers:

- ``ModelConfig`` carries the new ``use_derived_text_features`` field,
- the CLI exposes ``--use-derived-text-features`` (and the
  ``--no-derived-text-features`` convenience alias),
- the training loop's ``_zero_derived_text_features`` helper zeros the
  right slots without breaking the model's input shape.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FeatureVector, ModelConfig
from app.training.loop import _zero_derived_text_features, train_model


_TRAIN_FORECASTER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "app" / "train_forecaster.py"
)


# ---------------------------------------------------------------------------
# ModelConfig + CLI surface


def test_model_config_carries_use_derived_text_features_field() -> None:
    config = ModelConfig()
    assert config.use_derived_text_features is True


def test_model_config_round_trips_use_derived_text_features() -> None:
    from app.models.factory import build_forecaster

    config = ModelConfig(use_derived_text_features=False)
    model = build_forecaster(config)
    rebuilt = ModelConfig.from_model(model)
    assert rebuilt.use_derived_text_features is False


def test_cli_exposes_use_derived_text_features_flag() -> None:
    """The CLI must expose the on/off toggle so the runner can flip it."""

    if not _TRAIN_FORECASTER_PATH.exists():
        pytest.skip("train_forecaster.py not visible from this test path")
    source = _TRAIN_FORECASTER_PATH.read_text(encoding="utf-8")
    assert "\"--use-derived-text-features\"" in source
    assert "\"--no-derived-text-features\"" in source
    assert "dest=\"use_derived_text_features\"" in source


# ---------------------------------------------------------------------------
# _zero_derived_text_features helper


def _make_rich_tensor(n: int = 4, seq_len: int = 20, feat_dim: int = 35) -> torch.Tensor:
    """Per-bar 35-dim tensor with documented slot layout."""

    return torch.arange(
        n * seq_len * feat_dim, dtype=torch.float32
    ).reshape(n, seq_len, feat_dim) + 1.0  # nothing zero-valued by accident


def test_zero_derived_text_features_zeros_sentiment_slot() -> None:
    x = _make_rich_tensor()
    zeroed, _aux = _zero_derived_text_features(x, None)
    assert zeroed is not None
    assert torch.all(zeroed[..., 0] == 0.0)
    # Other market features stay intact.
    assert torch.all(zeroed[..., 1] == x[..., 1])
    assert torch.all(zeroed[..., 5] == x[..., 5])


def test_zero_derived_text_features_zeros_full_derived_family_on_80dim() -> None:
    """All derived-text slices [0], [10:25], [25:29], [29:35], [45:80] are zeroed."""

    x = _make_rich_tensor(feat_dim=80)
    zeroed, _aux = _zero_derived_text_features(x, None)
    assert zeroed is not None
    # Every derived-text slot is zero.
    assert torch.all(zeroed[..., 0] == 0.0)
    assert torch.all(zeroed[..., 10:25] == 0.0)
    assert torch.all(zeroed[..., 25:29] == 0.0)
    assert torch.all(zeroed[..., 29:35] == 0.0)
    assert torch.all(zeroed[..., 45:80] == 0.0)
    # Non-derived slices stay intact: market (1..6) + credibility (6:10) +
    # realized_vol (35:37) + cross_asset (37:45).
    assert torch.all(zeroed[..., 1:6] == x[..., 1:6])
    assert torch.all(zeroed[..., 6:10] == x[..., 6:10])
    assert torch.all(zeroed[..., 35:37] == x[..., 35:37])
    assert torch.all(zeroed[..., 37:45] == x[..., 37:45])


def test_zero_derived_text_features_zeros_multi_axis_slot_on_rich_tensor() -> None:
    """The 35-dim rich tensor zeros every applicable slice up to [29:35]."""

    x = _make_rich_tensor()
    zeroed, _aux = _zero_derived_text_features(x, None)
    assert zeroed is not None
    # Sentiment + linguistic + MP-surprise + multi-axis blocks all zeroed.
    assert torch.all(zeroed[..., 0] == 0.0)
    assert torch.all(zeroed[..., 10:25] == 0.0)
    assert torch.all(zeroed[..., 25:29] == 0.0)
    assert torch.all(zeroed[..., 29:35] == 0.0)
    # Credibility block stays intact (not derived from text).
    assert torch.all(zeroed[..., 6:10] == x[..., 6:10])


def test_zero_derived_text_features_legacy_6dim_skips_multi_axis() -> None:
    """Legacy 6-feature tensors short-circuit the multi-axis slot zeroing."""

    x = torch.ones((4, 20, 6))
    zeroed, _aux = _zero_derived_text_features(x, None)
    assert zeroed is not None
    # Sentiment slot still zeroed; the rest untouched.
    assert torch.all(zeroed[..., 0] == 0.0)
    assert torch.all(zeroed[..., 1:] == 1.0)


def test_zero_derived_text_features_masks_multi_task_aux() -> None:
    """The factor / certainty / topic masks must all collapse to False."""

    n = 4
    aux = {
        "factor": torch.randn(n),
        "factor_mask": torch.ones(n, dtype=torch.bool),
        "certainty": torch.zeros(n, dtype=torch.long),
        "certainty_mask": torch.ones(n, dtype=torch.bool),
        "topic": torch.zeros(n, dtype=torch.long),
        "topic_mask": torch.ones(n, dtype=torch.bool),
    }
    _x, new_aux = _zero_derived_text_features(None, aux)
    assert new_aux is not None
    assert not new_aux["factor_mask"].any()
    assert not new_aux["certainty_mask"].any()
    assert not new_aux["topic_mask"].any()
    # Target tensors are NOT touched -- only the masks drop to False.
    assert torch.equal(new_aux["factor"], aux["factor"])


def test_zero_derived_text_features_handles_none_inputs() -> None:
    """Both None inputs -> both None outputs."""

    x, aux = _zero_derived_text_features(None, None)
    assert x is None
    assert aux is None


# ---------------------------------------------------------------------------
# End-to-end smoke through train_model


def _dummy_feature_vector(*, vol: float, day: int) -> FeatureVector:
    return FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
    )


def _make_groups(n: int = 40) -> list[list[FeatureVector]]:
    return [
        [
            _dummy_feature_vector(day=i + 1, vol=0.01 + 0.001 * i)
            for i in range(n)
        ]
    ]


def test_derived_features_off_does_not_break_training() -> None:
    """The forecaster must still train when derived features are off."""

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        use_derived_text_features=False,
        hidden_size=16,
        head_hidden_size=8,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=_make_groups(),
        val_sequence_groups=_make_groups(),
        test_sequence_groups=_make_groups(),
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    assert result.summary.metrics is not None
    assert result.summary.metrics.regime_f1_macro is not None


def test_derived_features_default_on_runs_normally() -> None:
    """Back-compat: default ``True`` reproduces the pre-#309 behaviour."""

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=_make_groups(),
        val_sequence_groups=_make_groups(),
        test_sequence_groups=_make_groups(),
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    # use_derived_text_features should round-trip as True on the model.
    rebuilt = ModelConfig.from_model(result.model)
    assert rebuilt.use_derived_text_features is True
