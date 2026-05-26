"""Rates-complex heads loss + training-loop wiring (#292).

The three rates heads (``2y`` / ``5y`` / ``terminal``) mount on the
shared encoder alongside the existing vol-regime classifier. The loss
helper mixes the per-head MSE on the standardised bps target with the
auxiliary CE on the per-fold tertile label. These tests pin the wiring
at three layers:

- module construction (per-head regression + classifier mount only when
  the head is in the active set);
- :func:`_compute_rates_loss` math (regression / classification / dual
  mode + alpha boundary equivalences);
- end-to-end training-loop integration (one epoch produces a finite
  loss with active rates heads and the per-trial EvaluationMetrics
  surface carries the per-head MAE-bps panel).
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

torch = pytest.importorskip("torch")

from app.evaluation.metrics import TrainingResult  # noqa: E402
from app.models.config import FeatureVector, ModelConfig  # noqa: E402
from app.models.factory import build_forecaster  # noqa: E402
from app.models.rates_heads import (  # noqa: E402
    RATES_HEAD_NAMES,
    RATES_HEAD_N_CLASSES,
    resolve_rates_heads,
)
from app.training.loop import (  # noqa: E402
    RatesHeadPartitionBundle,
    RatesPartitionTensors,
    _compute_rates_loss,
    train_model,
)
from app.training.rates_targets import (  # noqa: E402
    build_partition_rates_targets,
    fit_rates_scaler,
    inverse_standardise_bps,
)


# ---------------------------------------------------------------------------
# Resolver helpers


def test_resolve_rates_heads_default_returns_empty() -> None:
    assert resolve_rates_heads(None) == ()
    assert resolve_rates_heads("none") == ()


def test_resolve_rates_heads_all() -> None:
    assert resolve_rates_heads("all") == RATES_HEAD_NAMES


def test_resolve_rates_heads_singleton() -> None:
    assert resolve_rates_heads("2y") == ("2y",)
    assert resolve_rates_heads("terminal") == ("terminal",)


def test_resolve_rates_heads_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        resolve_rates_heads("3y")


# ---------------------------------------------------------------------------
# Module construction


def _build_model(
    *,
    rates_heads: tuple[str, ...],
    head_mode: str = "classification",
) -> "torch.nn.Module":
    config = ModelConfig(
        output_mode="classification",
        head_mode=head_mode,
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=rates_heads,
    )
    return build_forecaster(config)


def test_default_run_does_not_mount_rates_heads() -> None:
    model = _build_model(rates_heads=())
    assert tuple(model.rates_heads_active) == ()
    assert len(model.rates_regression_heads) == 0
    assert len(model.rates_classification_heads) == 0


def test_active_rates_heads_mount_regression_and_classifier() -> None:
    model = _build_model(rates_heads=("2y", "terminal"))
    assert tuple(model.rates_heads_active) == ("2y", "terminal")
    assert set(model.rates_regression_heads.keys()) == {"2y", "terminal"}
    assert set(model.rates_classification_heads.keys()) == {"2y", "terminal"}


def test_unknown_rates_head_raises() -> None:
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("invalid",),
    )
    with pytest.raises(ValueError, match="Unknown rates head"):
        build_forecaster(config)


def test_forward_multi_task_emits_per_head_bps_and_logits() -> None:
    model = _build_model(rates_heads=("2y", "5y"))
    x = torch.zeros((3, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert "rates_2y_bps" in out
    assert "rates_5y_bps" in out
    assert "rates_2y_cls_logits" in out
    assert "rates_5y_cls_logits" in out
    assert out["rates_2y_bps"].shape == (3,)
    assert out["rates_5y_cls_logits"].shape == (3, RATES_HEAD_N_CLASSES)


# ---------------------------------------------------------------------------
# Loss helper math


def _make_bundle(
    *,
    bps: list[float],
    cls: list[int],
    mask_bps: list[bool] | None = None,
    mask_cls: list[bool] | None = None,
) -> RatesHeadPartitionBundle:
    n = len(bps)
    if mask_bps is None:
        mask_bps = [True] * n
    if mask_cls is None:
        mask_cls = [True] * n
    return RatesHeadPartitionBundle(
        bps_target=torch.tensor(bps, dtype=torch.float32),
        bps_mask=torch.tensor(mask_bps, dtype=torch.bool),
        cls_target=torch.tensor(cls, dtype=torch.int64),
        cls_mask=torch.tensor(mask_cls, dtype=torch.bool),
    )


def test_compute_rates_loss_regression_only_returns_mse() -> None:
    logits = {
        "rates_2y_bps": torch.zeros(4, requires_grad=True),
        "rates_2y_cls_logits": torch.zeros(4, 3),
    }
    targets = RatesPartitionTensors(
        per_head={"2y": _make_bundle(bps=[1.0, 1.0, 1.0, 1.0], cls=[0, 0, 0, 0])}
    )
    loss = _compute_rates_loss(
        logits_dict=logits,
        rates_targets=targets,
        head_names=("2y",),
        rates_head_mode="regression",
        rates_alpha=0.5,
    )
    assert loss is not None
    # MSE of (0, 1) = 1.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_compute_rates_loss_dual_mixes_terms() -> None:
    logits = {
        "rates_2y_bps": torch.zeros(4, requires_grad=True),
        "rates_2y_cls_logits": torch.zeros(4, 3, requires_grad=True),
    }
    targets = RatesPartitionTensors(
        per_head={"2y": _make_bundle(bps=[2.0] * 4, cls=[0] * 4)}
    )
    loss = _compute_rates_loss(
        logits_dict=logits,
        rates_targets=targets,
        head_names=("2y",),
        rates_head_mode="dual",
        rates_alpha=0.25,
    )
    assert loss is not None
    # MSE = 4.0; CE with uniform logits = log(3) ~ 1.0986
    # expected = 0.25 * 4.0 + 0.75 * 1.0986 = 1.0 + 0.82397 = 1.82397
    expected = 0.25 * 4.0 + 0.75 * math.log(3.0)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-4)


def test_compute_rates_loss_no_heads_returns_none() -> None:
    logits = {"rates_2y_bps": torch.zeros(4)}
    out = _compute_rates_loss(
        logits_dict=logits,
        rates_targets=None,
        head_names=(),
        rates_head_mode="regression",
        rates_alpha=0.5,
    )
    assert out is None


def test_compute_rates_loss_masked_rows_drop_contribution() -> None:
    """Rows whose bps_mask is False must contribute zero to the MSE."""

    logits = {
        "rates_2y_bps": torch.ones(4, requires_grad=True),
    }
    targets = RatesPartitionTensors(
        per_head={
            "2y": _make_bundle(
                bps=[0.0, 0.0, 0.0, 0.0],
                cls=[0, 0, 0, 0],
                mask_bps=[True, True, False, False],
                mask_cls=[False, False, False, False],
            )
        }
    )
    loss = _compute_rates_loss(
        logits_dict=logits,
        rates_targets=targets,
        head_names=("2y",),
        rates_head_mode="regression",
        rates_alpha=0.5,
    )
    assert loss is not None
    # Only 2 rows kept; each (1 - 0)^2 = 1.0; mean = 1.0.
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Per-fold scalers + standardisation round-trip


def test_fit_rates_scaler_unit_variance() -> None:
    scaler = fit_rates_scaler([-1.0, 1.0])
    assert scaler.mean == pytest.approx(0.0)
    assert scaler.std == pytest.approx(1.0)


def test_inverse_standardise_recovers_raw() -> None:
    scaler = fit_rates_scaler([10.0, 20.0, 30.0, 40.0])
    raw = 25.0
    standardised = (raw - scaler.mean) / scaler.std
    recovered = inverse_standardise_bps(standardised, scaler)
    assert math.isclose(recovered, raw, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end training-loop integration


def _dummy_feature_vector(
    *,
    day: int,
    vol: float,
    bps_2y: float = 0.0,
    bps_5y: float = 0.0,
    bps_terminal: float = 0.0,
) -> FeatureVector:
    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
    )
    fv.target_yield_2y_change_5d = bps_2y
    fv.target_yield_5y_change_5d = bps_5y
    fv.target_terminal_rate_change_5d = bps_terminal
    return fv


def _make_walk_forward_groups(n: int = 40) -> list[list[FeatureVector]]:
    return [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                bps_2y=float(2 * i - 20),
                bps_5y=float(1.5 * i - 15),
                bps_terminal=float(i - 10),
            )
            for i in range(n)
        ]
    ]


def test_rates_heads_end_to_end_one_epoch_produces_finite_loss() -> None:
    groups = _make_walk_forward_groups()
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("2y", "5y", "terminal"),
        rates_head_mode="regression",
        rates_alpha=0.5,
    )
    result: TrainingResult = train_model(
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
    assert result.summary.epochs_completed == 1
    metrics = result.summary.metrics
    assert metrics is not None
    assert metrics.rates_metrics is not None
    assert set(metrics.rates_metrics.keys()) == {"2y", "5y", "terminal"}
    for _head_name, payload in metrics.rates_metrics.items():
        assert payload["n_rows"] > 0
        mae = payload["mae_bps"]
        assert mae is not None
        assert math.isfinite(mae["point"])


def test_rates_heads_dual_mode_uses_both_terms() -> None:
    """``dual`` mode runs and emits finite loss with both heads active."""

    groups = _make_walk_forward_groups()
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("2y",),
        rates_head_mode="dual",
        rates_alpha=0.5,
    )
    result = train_model(
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
    assert result.summary.epochs_completed == 1
    payload = result.summary.metrics.rates_metrics["2y"]
    assert math.isfinite(payload["mae_bps"]["point"])


def test_rates_heads_scaler_persists_on_summary() -> None:
    groups = _make_walk_forward_groups()
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        rates_heads=("2y", "5y"),
    )
    result = train_model(
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
    assert result.summary.rates_scalers is not None
    assert set(result.summary.rates_scalers.keys()) == {"2y", "5y"}
    for payload in result.summary.rates_scalers.values():
        assert "mean" in payload
        assert "std" in payload
    assert result.summary.rates_quantile_edges is not None
    assert set(result.summary.rates_quantile_edges.keys()) == {"2y", "5y"}


def test_rates_heads_require_classification_output_mode() -> None:
    config = ModelConfig(
        output_mode="regression",
        rates_heads=("2y",),
    )
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


# ---------------------------------------------------------------------------
# build_partition_rates_targets row alignment


def test_build_partition_rates_targets_aligns_with_y() -> None:
    """Per-head tensors must agree on row count with each other."""

    groups = _make_walk_forward_groups(n=30)
    bps_t, bps_m, cls_t, cls_m, scalers, edges = build_partition_rates_targets(
        groups,
        head_names=("2y", "5y", "terminal"),
    )
    assert set(bps_t.keys()) == {"2y", "5y", "terminal"}
    lengths = {name: int(t.shape[0]) for name, t in bps_t.items()}
    assert len(set(lengths.values())) == 1, (
        f"per-head row counts diverge: {lengths}"
    )
    # Train-fit scalers must be ``RatesHeadScaler``.
    for name in ("2y", "5y", "terminal"):
        assert scalers[name].std > 0.0
    for name in ("2y", "5y", "terminal"):
        assert math.isfinite(edges[name].lower)
        assert math.isfinite(edges[name].upper)


def test_build_partition_rates_targets_reuses_train_scaler() -> None:
    train_groups = _make_walk_forward_groups(n=30)
    val_groups = _make_walk_forward_groups(n=20)
    _, _, _, _, train_scalers, train_edges = build_partition_rates_targets(
        train_groups, head_names=("2y",)
    )
    _, _, _, _, val_scalers, val_edges = build_partition_rates_targets(
        val_groups,
        head_names=("2y",),
        scalers=train_scalers,
        edges_by_head=train_edges,
    )
    # Echoed back unchanged so val/test reuse the train-fitted scaler.
    assert val_scalers["2y"] == train_scalers["2y"]
    assert val_edges["2y"].lower == train_edges["2y"].lower
