"""Dual-head methodology wiring (#304).

The vol-regime classifier head is byte-identical to the pre-#304
path under ``head_mode='classification'`` (default). ``regression``
mounts a log(RV) MSE head only and drops the CE contribution.
``dual`` keeps both heads and trains the joint
``(1 - alpha) * CE + alpha * MSE`` loss.

These tests pin the wiring at three layers:

- module construction (regression_head mounts only on the right modes),
- forward_multi_task (the log_rv branch appears alongside the four
  classification axes),
- training loop integration (a single epoch produces a finite loss in
  each of the three configurations and the per-trial
  EvaluationMetrics surface carries the regression numbers).
"""

from __future__ import annotations

import datetime as _dt

import pytest

torch = pytest.importorskip("torch")

from app.evaluation.metrics import TrainingResult
from app.models.config import FeatureVector, ModelConfig
from app.models.factory import build_forecaster
from app.training.loop import (
    _combine_dual_head_loss,
    _maybe_add_dual_head_loss,
    train_model,
)


# ---------------------------------------------------------------------------
# Module-level construction


def _build_model(head_mode: str) -> "torch.nn.Module":
    config = ModelConfig(
        output_mode="classification",
        head_mode=head_mode,
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    return build_forecaster(config)


def test_head_mode_classification_does_not_mount_regression_head() -> None:
    """Default head_mode keeps the pre-#304 path byte-identical."""

    model = _build_model("classification")
    assert getattr(model, "head_mode", None) == "classification"
    assert getattr(model, "regression_head", None) is None


def test_head_mode_regression_mounts_regression_head() -> None:
    model = _build_model("regression")
    assert model.head_mode == "regression"
    assert model.regression_head is not None


def test_head_mode_dual_mounts_regression_head() -> None:
    model = _build_model("dual")
    assert model.head_mode == "dual"
    assert model.regression_head is not None


def test_unknown_head_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown head_mode"):
        ModelConfig(
            output_mode="classification",
            head_mode="not_a_mode",
            n_classes=3,
        )
        # Construction surfaces the error via the ForecasterModel kwarg
        # check; ModelConfig itself accepts the string so the error
        # fires only at build_forecaster time.
        build_forecaster(
            ModelConfig(
                output_mode="classification",
                head_mode="not_a_mode",
                n_classes=3,
            )
        )


# ---------------------------------------------------------------------------
# forward_multi_task contract


def test_forward_multi_task_emits_log_rv_branch_on_dual_head() -> None:
    """The log_rv branch must appear alongside the 4 classification axes."""

    model = _build_model("dual")
    x = torch.zeros((4, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert set(out.keys()) == {"stance", "factor", "certainty", "topic", "log_rv"}
    assert out["log_rv"].shape == (4,)


def test_forward_multi_task_omits_log_rv_on_classification_only() -> None:
    """Default head_mode must keep the existing 4-axis dict shape."""

    model = _build_model("classification")
    x = torch.zeros((4, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert set(out.keys()) == {"stance", "factor", "certainty", "topic"}


# ---------------------------------------------------------------------------
# Loss-helper unit tests (pure functions, no training)


def test_combine_dual_head_loss_classification_returns_ce_unchanged() -> None:
    ce = torch.tensor(0.7, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3, requires_grad=True)}
    out = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict=logits,
        batch_log_rv=None,
        head_mode="classification",
        regression_alpha=0.5,
    )
    assert torch.equal(out, ce)


def test_combine_dual_head_loss_regression_returns_mse_only() -> None:
    ce = torch.tensor(0.7, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4, requires_grad=True)}
    target = torch.full((4,), 1.0)
    out = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict=logits,
        batch_log_rv=target,
        head_mode="regression",
        regression_alpha=0.5,
    )
    # MSE of (0, 1) over 4 rows = 1.0; CE term dropped.
    assert torch.isclose(out, torch.tensor(1.0), atol=1e-6)


def test_combine_dual_head_loss_dual_mixes_ce_and_mse() -> None:
    ce = torch.tensor(0.4, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4, requires_grad=True)}
    target = torch.full((4,), 2.0)
    alpha = 0.25
    out = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict=logits,
        batch_log_rv=target,
        head_mode="dual",
        regression_alpha=alpha,
    )
    # MSE = 4.0; expected = (1 - 0.25) * 0.4 + 0.25 * 4.0 = 0.3 + 1.0 = 1.3
    assert torch.isclose(out, torch.tensor(1.3), atol=1e-6)


def test_combine_dual_head_loss_missing_log_rv_target_raises() -> None:
    ce = torch.tensor(0.5, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4)}
    with pytest.raises(RuntimeError, match="log_rv target tensor"):
        _combine_dual_head_loss(
            ce_loss=ce,
            logits_dict=logits,
            batch_log_rv=None,
            head_mode="dual",
            regression_alpha=0.5,
        )


def test_maybe_add_dual_head_loss_classification_no_op() -> None:
    """Multi-task helper is a no-op when head_mode='classification'."""

    base = torch.tensor(0.6, requires_grad=True)
    out = _maybe_add_dual_head_loss(
        base,
        logits_dict={"stance": torch.zeros(4, 3)},
        batch_log_rv=None,
        head_mode="classification",
        regression_alpha=0.5,
    )
    assert torch.equal(out, base)


# ---------------------------------------------------------------------------
# End-to-end integration: one epoch through train_model


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


def _make_walk_forward_groups(n: int = 40) -> list[list[FeatureVector]]:
    """Synthetic walk-forward fold with monotonic forward-vol coverage."""

    return [
        [
            _dummy_feature_vector(day=i + 1, vol=0.01 + 0.001 * i)
            for i in range(n)
        ]
    ]


def _run_one_epoch(head_mode: str) -> TrainingResult:
    groups = _make_walk_forward_groups()
    config = ModelConfig(
        output_mode="classification",
        head_mode=head_mode,
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    return train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )


def test_head_mode_classification_produces_finite_loss() -> None:
    """Default head_mode runs end-to-end (back-compat regression contract)."""

    result = _run_one_epoch("classification")
    assert result.summary.epochs_completed == 1
    assert result.summary.metrics is not None
    # Classification path leaves regression metrics at None (no head mounted).
    assert result.summary.metrics.regression_rmse_log_rv is None
    assert result.summary.metrics.regression_loss is None


def test_head_mode_regression_populates_log_rv_metrics() -> None:
    """head_mode='regression' surfaces RMSE / MAE / loss on log(RV)."""

    result = _run_one_epoch("regression")
    metrics = result.summary.metrics
    assert metrics is not None
    assert metrics.regression_rmse_log_rv is not None
    assert metrics.regression_mae_log_rv is not None
    assert metrics.regression_loss is not None
    import math

    assert math.isfinite(metrics.regression_loss)
    assert math.isfinite(metrics.regression_rmse_log_rv)


def test_head_mode_dual_populates_both_classification_and_regression() -> None:
    """head_mode='dual' keeps the regime macro-F1 surface and adds log(RV)."""

    result = _run_one_epoch("dual")
    metrics = result.summary.metrics
    assert metrics is not None
    assert metrics.regime_f1_macro is not None
    assert metrics.regression_rmse_log_rv is not None
    assert metrics.regression_mae_log_rv is not None
    assert metrics.regression_loss is not None


def test_head_mode_dual_persists_on_round_tripped_config() -> None:
    """ModelConfig.from_model must round-trip the new fields."""

    result = _run_one_epoch("dual")
    rebuilt = ModelConfig.from_model(result.model)
    assert rebuilt.head_mode == "dual"
    # regression_alpha picked up from the saved attribute on the module.
    assert rebuilt.regression_alpha == pytest.approx(0.5)


def test_dual_head_requires_classification_output_mode() -> None:
    """head_mode in {regression, dual} only makes sense for classification."""

    config = ModelConfig(output_mode="regression", head_mode="dual")
    with pytest.raises(ValueError, match="output_mode='classification'"):
        train_model(
            model_config=config,
            train_sequence_groups=_make_walk_forward_groups(),
            val_sequence_groups=_make_walk_forward_groups(),
            test_sequence_groups=_make_walk_forward_groups(),
            epochs=1,
            save_checkpoint=False,
            use_compile=False,
            use_amp=False,
        )
