"""Multi-target heads on the shared encoder (#292 acceptance).

The active surface mounts three heads off one encoder pass: the
vol-regime classifier (existing) plus the per-rates regression heads
(``2y`` / ``terminal``). This file pins the shape contract the AC asks
for: head mount, forward dict keys, the new ``rates_aux_classification``
opt-in gate, the ``--targets`` CLI resolver, and the
``_coerce_payload_config`` round-trip on the new field.

The deeper loss-side wiring (per-batch mask, alpha boundary, end-to-end
train_model finite loss) is covered by ``test_rates_heads_loss.py``;
this file focuses on the multi-head schema-level guarantees the issue
contract pins.
"""

from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig  # noqa: E402
from app.models.factory import build_forecaster  # noqa: E402
from app.models.rates_heads import RATES_HEAD_N_CLASSES  # noqa: E402
from app.train_forecaster import _resolve_rates_heads_from_args  # noqa: E402
from app.training.checkpoint import _coerce_payload_config  # noqa: E402


# ---------------------------------------------------------------------------
# Three-head mount on a single encoder


def _build_model(
    *,
    rates_heads: tuple[str, ...],
    rates_aux_classification: bool = False,
    head_mode: str = "classification",
) -> torch.nn.Module:
    return build_forecaster(
        ModelConfig(
            output_mode="classification",
            head_mode=head_mode,
            n_classes=3,
            hidden_size=16,
            head_hidden_size=8,
            rates_heads=rates_heads,
            rates_aux_classification=rates_aux_classification,
        )
    )


def test_two_rates_regression_heads_mount_alongside_regime() -> None:
    """A single checkpoint carries regime + 2y + terminal regression heads."""

    model = _build_model(rates_heads=("2y", "terminal"))
    # The regime classifier rides on ``model.head`` (MultiTaskHead);
    # the two rates regression heads ride on the ModuleDict.
    assert hasattr(model, "head")
    assert tuple(model.rates_heads_active) == ("2y", "terminal")
    assert set(model.rates_regression_heads.keys()) == {"2y", "terminal"}
    # Default aux-off: no classification heads land alongside the
    # regression heads.
    assert len(model.rates_classification_heads) == 0
    assert model.rates_aux_classification is False


def test_aux_classification_opt_in_mounts_paired_classifier() -> None:
    model = _build_model(
        rates_heads=("2y", "terminal"),
        rates_aux_classification=True,
    )
    assert set(model.rates_classification_heads.keys()) == {"2y", "terminal"}
    assert model.rates_aux_classification is True


def test_forward_multi_task_emits_one_bps_key_per_active_head() -> None:
    model = _build_model(rates_heads=("2y", "terminal"))
    x = torch.zeros((2, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert "rates_2y_bps" in out
    assert "rates_terminal_bps" in out
    # Aux off: the cls logits keys must NOT appear.
    assert "rates_2y_cls_logits" not in out
    assert "rates_terminal_cls_logits" not in out
    assert out["rates_2y_bps"].shape == (2,)
    assert out["rates_terminal_bps"].shape == (2,)


def test_forward_multi_task_emits_cls_logits_when_aux_on() -> None:
    model = _build_model(
        rates_heads=("2y", "terminal"), rates_aux_classification=True
    )
    x = torch.zeros((2, 20, model.input_size))
    out = model.forward_multi_task(x)
    assert "rates_2y_cls_logits" in out
    assert "rates_terminal_cls_logits" in out
    assert out["rates_2y_cls_logits"].shape == (2, RATES_HEAD_N_CLASSES)


def test_gradient_reaches_each_regression_head_independently() -> None:
    """A loss summed across both rates regression heads pushes a gradient
    into each per-head linear stack, not just the shared encoder."""

    model = _build_model(rates_heads=("2y", "terminal"))
    x = torch.zeros((4, 20, model.input_size))
    out = model.forward_multi_task(x)
    loss = (out["rates_2y_bps"] ** 2).mean() + (out["rates_terminal_bps"] ** 2).mean()
    loss.backward()
    # Each head's final linear must have received a non-zero gradient on
    # at least one parameter (the bias is the simplest check).
    head_2y_final = model.rates_regression_heads["2y"][-1]
    head_terminal_final = model.rates_regression_heads["terminal"][-1]
    assert head_2y_final.bias.grad is not None
    assert head_terminal_final.bias.grad is not None
    # Gradients must differ between the heads -- otherwise we'd know the
    # two heads collapsed onto a single shared weight, which would defeat
    # the per-head calibration contract.
    assert not torch.allclose(
        head_2y_final.bias.grad, head_terminal_final.bias.grad, atol=1e-12
    )


# ---------------------------------------------------------------------------
# --targets CLI resolver


def test_targets_flag_resolves_to_rates_head_tuple() -> None:
    args = argparse.Namespace(targets="regime,rates_2y,rates_terminal")
    assert _resolve_rates_heads_from_args(args) == ("2y", "terminal")


def test_targets_flag_regime_only_resolves_to_empty_tuple() -> None:
    """``regime`` alone implies no rates heads mount."""

    args = argparse.Namespace(targets="regime")
    assert _resolve_rates_heads_from_args(args) == ()


def test_targets_flag_unknown_token_raises() -> None:
    args = argparse.Namespace(targets="rates_30y")
    with pytest.raises(ValueError, match="unknown target"):
        _resolve_rates_heads_from_args(args)


def test_targets_flag_overrides_rates_heads_alias() -> None:
    """When both flags ride on the namespace, ``--targets`` wins."""

    args = argparse.Namespace(targets="rates_2y", rates_heads="all")
    assert _resolve_rates_heads_from_args(args) == ("2y",)


def test_rates_heads_alias_still_works_without_targets() -> None:
    args = argparse.Namespace(targets=None, rates_heads="all")
    # Resolves through resolve_rates_heads; the all keyword returns the
    # full ordered tuple.
    assert _resolve_rates_heads_from_args(args) == ("2y", "5y", "terminal")


# ---------------------------------------------------------------------------
# Aux-classification mode invariants


def test_dual_rates_mode_without_aux_classifier_rejected() -> None:
    """``rates_head_mode='dual'`` with aux off is a misconfiguration."""

    with pytest.raises(ValueError, match="rates-classification-heads"):
        build_forecaster(
            ModelConfig(
                output_mode="classification",
                n_classes=3,
                hidden_size=16,
                head_hidden_size=8,
                rates_heads=("2y",),
                rates_aux_classification=False,
                rates_head_mode="dual",
            )
        )


def test_regression_rates_mode_with_aux_off_is_legal() -> None:
    """The default rates path (regression mode, aux off) must build."""

    model = _build_model(rates_heads=("2y",))
    assert tuple(model.rates_heads_active) == ("2y",)
    assert len(model.rates_classification_heads) == 0


# ---------------------------------------------------------------------------
# Checkpoint round-trip on the new aux flag


def test_coerce_payload_config_carries_aux_flag() -> None:
    """A serialised dict checkpoint rehydrates with the aux flag intact."""

    payload = {
        "model_config": {
            "rates_heads": ["2y", "terminal"],
            "rates_head_mode": "regression",
            "rates_aux_classification": True,
            "rates_alpha": 0.5,
        }
    }
    config = _coerce_payload_config(payload)
    assert tuple(config.rates_heads) == ("2y", "terminal")
    assert config.rates_aux_classification is True


def test_coerce_payload_config_defaults_aux_flag_false_on_pre_292() -> None:
    """A pre-#292 checkpoint without the new key rehydrates with aux off."""

    payload = {"model_config": {"rates_heads": ["2y"]}}
    config = _coerce_payload_config(payload)
    assert config.rates_aux_classification is False


def test_model_config_from_model_round_trips_aux_flag() -> None:
    """The ``from_model`` reflection picks the aux flag off the built module."""

    model = _build_model(
        rates_heads=("2y", "terminal"), rates_aux_classification=True
    )
    reflected = ModelConfig.from_model(model)
    assert reflected.rates_aux_classification is True
    assert set(reflected.rates_heads) == {"2y", "terminal"}
