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
import math

import pytest

torch = pytest.importorskip("torch")

from app.evaluation.metrics import TrainingResult
from app.models.config import FeatureVector, ModelConfig
from app.models.factory import build_forecaster
from app.training.loop import (
    _build_partition_log_rv_target,
    _combine_dual_head_loss,
    _is_finite_positive_forward_vol,
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
    """The log_rv branch must appear alongside the 3 classification axes
    (topic was retired in ADR 0044)."""

    model = _build_model("dual")
    x = torch.zeros((4, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert set(out.keys()) == {"stance", "certainty", "time", "log_rv"}
    assert out["log_rv"].shape == (4,)


def test_forward_multi_task_omits_log_rv_on_classification_only() -> None:
    """Default head_mode must keep the existing 3-axis dict shape."""

    model = _build_model("classification")
    x = torch.zeros((4, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert set(out.keys()) == {"stance", "certainty", "time"}


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


def test_combine_dual_head_loss_missing_log_rv_target_soft_demotes_to_ce() -> None:
    """ADR 0015 (#322) flipped ``head_mode`` to regression by default;
    fixtures and datasets that lack ``forward_realized_vol_10d`` rows
    now inherit the regression objective without supplying the target.
    The helper must soft-demote to CE-only on that batch rather than
    failing the run."""

    ce = torch.tensor(0.5, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4)}
    out = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict=logits,
        batch_log_rv=None,
        head_mode="dual",
        regression_alpha=0.5,
    )
    assert torch.equal(out, ce)


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


def test_dual_head_soft_demotes_on_incompatible_output_mode(capsys) -> None:
    """ADR 0015 (#322) flipped the ``head_mode`` default to ``regression``,
    so the close/vol regression path (``output_mode='regression'``) now
    inherits the new default and would otherwise fail the run. The
    documented contract on ``ModelConfig.head_mode`` ("regression-output
    mode (close, vol) ignores ``head_mode`` entirely") forces a soft
    demotion: training completes, the run's effective head mode collapses
    to classification, and a single diagnostic line is emitted on
    stdout."""

    config = ModelConfig(output_mode="regression", head_mode="dual")
    result = train_model(
        model_config=config,
        train_sequence_groups=_make_walk_forward_groups(),
        val_sequence_groups=_make_walk_forward_groups(),
        test_sequence_groups=_make_walk_forward_groups(),
        epochs=1,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    captured = capsys.readouterr()
    assert "ignored on output_mode='regression'" in captured.out


# ---------------------------------------------------------------------------
# Additional fix-up coverage (review findings #2, #6, #7, #13, #14).


def test_combine_dual_head_loss_dual_alpha_zero_returns_ce_unchanged() -> None:
    """alpha=0 must short-circuit and equal head_mode='classification'."""

    ce = torch.tensor(0.7, requires_grad=True)
    out_dual = _combine_dual_head_loss(
        ce_loss=ce,
        # No log_rv key — the dual + alpha=0 path must not need it.
        logits_dict={"stance": torch.zeros(4, 3)},
        batch_log_rv=None,
        head_mode="dual",
        regression_alpha=0.0,
    )
    out_cls = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict={"stance": torch.zeros(4, 3)},
        batch_log_rv=None,
        head_mode="classification",
        regression_alpha=0.0,
    )
    assert torch.equal(out_dual, ce)
    assert torch.equal(out_cls, ce)


def test_combine_dual_head_loss_dual_alpha_one_returns_mse_only() -> None:
    """alpha=1 must collapse the dual path to MSE only."""

    ce = torch.tensor(0.4, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4, requires_grad=True)}
    target = torch.full((4,), 3.0)
    out = _combine_dual_head_loss(
        ce_loss=ce,
        logits_dict=logits,
        batch_log_rv=target,
        head_mode="dual",
        regression_alpha=1.0,
    )
    assert torch.isclose(out, torch.tensor(9.0), atol=1e-6)


def test_maybe_add_dual_head_loss_regression_keeps_multi_task_loss() -> None:
    """The multi-task helper must NOT discard the per-axis loss under regression."""

    base = torch.tensor(0.6, requires_grad=True)
    logits = {"stance": torch.zeros(4, 3), "log_rv": torch.zeros(4, requires_grad=True)}
    target = torch.full((4,), 2.0)
    out = _maybe_add_dual_head_loss(
        base,
        logits_dict=logits,
        batch_log_rv=target,
        head_mode="regression",
        regression_alpha=0.5,
    )
    # MSE = 4.0; expected = 0.6 + 4.0 = 4.6 (per-axis loss is preserved).
    assert torch.isclose(out, torch.tensor(4.6), atol=1e-6)


def test_maybe_add_dual_head_loss_dual_alpha_zero_is_noop() -> None:
    """alpha=0 must keep the multi-task loss byte-identical."""

    base = torch.tensor(0.6, requires_grad=True)
    out = _maybe_add_dual_head_loss(
        base,
        logits_dict={"stance": torch.zeros(4, 3)},
        batch_log_rv=None,
        head_mode="dual",
        regression_alpha=0.0,
    )
    assert torch.equal(out, base)


# ---------------------------------------------------------------------------
# _build_partition_log_rv_target row alignment + finite guard


def _group_with_target_vol(target_vol: float | None) -> list[FeatureVector]:
    """Build a single-target sequence whose target row carries ``target_vol``."""

    vectors: list[FeatureVector] = []
    for i in range(20):
        vectors.append(_dummy_feature_vector(day=i + 1, vol=0.01))
    target = _dummy_feature_vector(day=21, vol=0.0)
    target.forward_realized_vol_10d = target_vol  # type: ignore[assignment]
    vectors.append(target)
    return vectors


def test_log_rv_target_aligned_with_x_y() -> None:
    """A group with a null leading target must drop from log_rv too."""

    quantiles = (0.012, 0.018)
    groups: list[list[FeatureVector]] = [
        _group_with_target_vol(0.015),  # kept
        _group_with_target_vol(None),  # dropped (leading target null)
        _group_with_target_vol(0.025),  # kept
    ]
    log_rv, scaler = _build_partition_log_rv_target(
        groups, vol_regime_quantiles=quantiles
    )
    assert log_rv is not None
    # Two surviving groups, each with one target row.
    assert log_rv.shape == (2,)
    # Standardiser fitted on the train slice => mean ~ 0, std ~ 1.
    assert scaler is not None
    assert log_rv.mean().abs() < 1e-5
    assert torch.isfinite(log_rv).all()


def test_log_rv_target_handles_zero_inf_negative() -> None:
    """Non-finite + non-positive forward-vol rows must be filtered."""

    assert _is_finite_positive_forward_vol(None) is False
    assert _is_finite_positive_forward_vol(0.0) is False
    assert _is_finite_positive_forward_vol(-0.01) is False
    assert _is_finite_positive_forward_vol(float("inf")) is False
    assert _is_finite_positive_forward_vol(float("nan")) is False
    assert _is_finite_positive_forward_vol(0.01) is True

    quantiles = (0.012, 0.018)
    groups: list[list[FeatureVector]] = [
        _group_with_target_vol(0.015),  # kept
        _group_with_target_vol(0.0),  # rejected (non-positive)
        _group_with_target_vol(float("inf")),  # rejected (non-finite)
        _group_with_target_vol(0.020),  # kept
    ]
    log_rv, _scaler = _build_partition_log_rv_target(
        groups, vol_regime_quantiles=quantiles
    )
    assert log_rv is not None
    assert log_rv.shape == (2,)
    assert torch.isfinite(log_rv).all()


def test_log_rv_target_val_test_reuse_train_scaler() -> None:
    """val/test partitions must standardise using the train-fitted scaler."""

    quantiles = (0.012, 0.018)
    train_groups = [_group_with_target_vol(0.015), _group_with_target_vol(0.025)]
    val_groups = [_group_with_target_vol(0.020)]
    train_log_rv, train_scaler = _build_partition_log_rv_target(
        train_groups, vol_regime_quantiles=quantiles
    )
    assert train_scaler is not None
    val_log_rv, val_scaler_echo = _build_partition_log_rv_target(
        val_groups, vol_regime_quantiles=quantiles, log_rv_scaler=train_scaler
    )
    assert val_log_rv is not None
    # The echoed scaler must match the train scaler exactly.
    assert val_scaler_echo == train_scaler
    # Manually invert and verify the val value: log(0.020) standardised
    # under (train_mean, train_std).
    expected = (math.log(0.020) - train_scaler[0]) / train_scaler[1]
    assert torch.isclose(val_log_rv[0], torch.tensor(expected, dtype=torch.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# Forward path (fix #4): regression head reachable via model(x)


def test_forward_populates_last_multi_task_log_rv_under_dual() -> None:
    """Standard forward must stash the regression prediction on _last_multi_task."""

    model = _build_model("dual")
    x = torch.zeros((3, 20, model.input_size))
    out = model(x)
    assert out.shape == (3, 3)  # stance logits
    cached = getattr(model, "_last_multi_task", None)
    assert cached is not None
    assert "log_rv" in cached
    assert cached["log_rv"].shape == (3,)


def test_forward_classification_only_has_no_log_rv_in_cache() -> None:
    """head_mode='classification' keeps the cache four-axis only."""

    model = _build_model("classification")
    x = torch.zeros((3, 20, model.input_size))
    _ = model(x)
    cached = getattr(model, "_last_multi_task", {})
    assert "log_rv" not in cached


# ---------------------------------------------------------------------------
# Legacy 80/20 path (fix #6): head_mode='dual' must not crash


def _legacy_groups(n: int = 40) -> list[list[FeatureVector]]:
    """Sequence groups long enough for the legacy 80/20 chronological split."""

    return [
        [
            _dummy_feature_vector(day=i + 1, vol=0.01 + 0.0005 * i)
            for i in range(n)
        ]
    ]


def test_legacy_80_20_path_supports_dual_head() -> None:
    """The legacy single-list path must train under head_mode='dual'."""

    config = ModelConfig(
        output_mode="classification",
        head_mode="dual",
        regression_alpha=0.5,
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    result = train_model(
        model_config=config,
        sequence_groups=_legacy_groups(),
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
    metrics = result.summary.metrics
    assert metrics is not None
    assert metrics.regression_rmse_log_rv is not None
    assert math.isfinite(metrics.regression_rmse_log_rv)
