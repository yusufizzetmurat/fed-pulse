"""Cover the multi-task-loss config + factory plumbing (#273 foundation).

The wider plumbing — threading per-axis target tensors through the
DataLoader and swapping CrossEntropy for MultiTaskLoss in the training
step — lands as a follow-up. This test file pins the configuration
surface so the follow-up has a stable contract to build against.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig
from app.models.factory import build_forecaster
from app.training.loop import _fit_axis_class_weights_from_mask


def test_default_modelconfig_keeps_multi_task_loss_off() -> None:
    """The byte-identity contract on every existing classification
    run depends on this default — flipping it would change the loss
    surface and break the determinism regression."""

    cfg = ModelConfig()
    assert cfg.multi_task_loss is False
    assert cfg.multi_task_lambda_stance == 1.0
    assert cfg.multi_task_lambda_certainty == 0.3
    assert cfg.multi_task_lambda_time == 0.3


def test_factory_stashes_multi_task_flags_on_built_model() -> None:
    """``ModelConfig.from_model`` reads these attributes back when
    serialising a checkpoint summary; without them the run summary
    would record the wrong loss config."""

    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        multi_task_loss=True,
        multi_task_lambda_stance=0.5,
        multi_task_lambda_certainty=0.4,
        multi_task_lambda_time=0.2,
    )
    model = build_forecaster(cfg)
    assert bool(getattr(model, "multi_task_loss")) is True
    assert float(getattr(model, "multi_task_lambda_stance")) == 0.5
    assert float(getattr(model, "multi_task_lambda_certainty")) == 0.4
    assert float(getattr(model, "multi_task_lambda_time")) == 0.2


def test_modelconfig_from_model_round_trips_multi_task_fields() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        multi_task_loss=True,
        multi_task_lambda_stance=0.7,
    )
    model = build_forecaster(cfg)
    restored = ModelConfig.from_model(model)
    assert restored.multi_task_loss is True
    assert restored.multi_task_lambda_stance == 0.7


def test_fit_axis_class_weights_handles_empty_mask() -> None:
    """An axis whose mask is all-False (sparse-label case) must return
    uniform weights so the masked MultiTaskLoss does not blow up
    when the cross-entropy receives a length-N_classes weight vector."""

    targets = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.tensor([False, False, False, False, False])
    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes=3)
    assert weights.shape == (3,)
    assert torch.allclose(weights, torch.ones(3))


def test_fit_axis_class_weights_inverts_frequency() -> None:
    """A rare class should weight more than a common one; weights
    normalise to sum to ``n_classes`` so an evenly-distributed axis
    reads ~1.0 per class."""

    # Class 0 has 8 rows, class 1 has 2 rows, class 2 has 0 rows
    targets = torch.tensor([0] * 8 + [1] * 2 + [0])  # 9th 0
    # Mask: keep the 10 rows but drop the 11th
    mask = torch.tensor([True] * 10 + [False])
    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes=3)
    assert weights.shape == (3,)
    # Rare class (1) outweighs common class (0); empty class (2)
    # gets the largest weight via the smoothing floor.
    assert weights[2] > weights[1]
    assert weights[1] > weights[0]
    # Normalised to sum to n_classes
    assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-5)


def test_fit_axis_class_weights_clips_out_of_range_labels() -> None:
    """Defensive contract: targets outside [0, n_classes) should be
    silently dropped from the count so a malformed row does not
    crash the loss construction."""

    targets = torch.tensor([0, 5, 1, -1, 2])
    mask = torch.tensor([True, True, True, True, True])
    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes=3)
    assert weights.shape == (3,)
    assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-5)


# ---------------------------------------------------------------------------
# #470 ordinal CE on the regime head
# ---------------------------------------------------------------------------


def test_default_modelconfig_keeps_regime_loss_mode_at_ce() -> None:
    """Default-off is the byte-identity lock for every pre-#470 run."""

    assert ModelConfig().regime_loss_mode == "ce"


def test_modelconfig_round_trips_regime_loss_mode() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        regime_loss_mode="ordinal_ce",
    )
    model = build_forecaster(cfg)
    assert str(model.regime_loss_mode) == "ordinal_ce"
    restored = ModelConfig.from_model(model)
    assert restored.regime_loss_mode == "ordinal_ce"


def test_modelconfig_rejects_unknown_regime_loss_mode() -> None:
    from app.training.loss import MultiTaskLoss

    with pytest.raises(ValueError, match="regime_loss_mode"):
        MultiTaskLoss(regime_loss_mode="cosine")


def test_ordinal_cross_entropy_hand_computed_expected_value() -> None:
    """Hand-computed expected value: bin-distance scales standard CE.

    Two rows with confident predictions of class 0; targets are 0 and 2
    respectively, so the row weights are ``1 + 0 = 1`` and
    ``1 + |2 - 0| = 3``. The mean ordinal CE is therefore
    ``(1 * ce_row0 + 3 * ce_row2) / 2`` while standard CE would have
    been ``(ce_row0 + ce_row2) / 2``. The far-bin miss carries 3x the
    weight of an exact match.
    """

    from torch.nn import functional as F

    from app.training.loss import ordinal_cross_entropy

    # Class 0 confident logits: argmax is 0 on both rows.
    logits = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    targets = torch.tensor([0, 2])

    standard_per_row = F.cross_entropy(logits, targets, reduction="none")
    assert standard_per_row.shape == (2,)
    expected = (
        (1.0 + 0.0) * float(standard_per_row[0])
        + (1.0 + 2.0) * float(standard_per_row[1])
    ) / 2.0
    actual = ordinal_cross_entropy(logits, targets)
    assert torch.isclose(actual, torch.tensor(expected), atol=1e-6)


def test_ordinal_cross_entropy_collapses_to_ce_when_all_rows_correct() -> None:
    """A perfectly correct batch carries distance 0 and the floor of 1,
    so the ordinal-CE return equals standard-CE byte-identically."""

    from torch.nn import functional as F

    from app.training.loss import ordinal_cross_entropy

    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    targets = torch.tensor([0, 1, 2])
    assert torch.allclose(
        ordinal_cross_entropy(logits, targets),
        F.cross_entropy(logits, targets),
        atol=1e-6,
    )


def test_multi_task_loss_ordinal_ce_changes_stance_loss_value() -> None:
    """Routing the stance axis through ordinal CE produces a strictly
    larger total than standard CE when at least one row is mispredicted
    by more than one bin -- the load-bearing behaviour the variant is
    introduced to deliver."""

    from app.training.loss import MultiTaskLoss

    logits = {
        "stance": torch.tensor(
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], requires_grad=True
        ),
        "certainty": torch.zeros(2, 3, requires_grad=True),
        "time": torch.zeros(2, 2, requires_grad=True),
    }
    targets = {
        "stance": torch.tensor([0, 2]),
        "certainty": torch.tensor([0, 0]),
        "time": torch.tensor([0, 0]),
    }
    masks = {
        "stance_mask": torch.tensor([True, True]),
        "certainty_mask": torch.tensor([False, False]),
        "time_mask": torch.tensor([False, False]),
    }

    ce_total, ce_breakdown = MultiTaskLoss(regime_loss_mode="ce")(
        logits, targets, masks
    )
    ord_total, ord_breakdown = MultiTaskLoss(regime_loss_mode="ordinal_ce")(
        logits, targets, masks
    )
    assert ord_total.detach().item() > ce_total.detach().item()
    assert ord_breakdown["stance"].item() > ce_breakdown["stance"].item()


def test_run_dual_head_runner_threads_regime_loss_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch", reason="train_model import path needs torch")
    import sys
    from types import SimpleNamespace

    from app.training import loaders as loaders_module
    from app.training import loop as loop_module
    from scripts import run_dual_head_comparison as runner

    captured: dict[str, list[dict[str, Any]]] = {"train_calls": []}

    class _StubSplit:
        fold_id = "fold_001"
        protocol = "walk_forward"
        train: list[Any] = []
        val: list[Any] = []
        test: list[Any] = []

    def _fake_load_walk_forward_split(**_kwargs: Any) -> _StubSplit:
        return _StubSplit()

    def _fake_train_model(**kwargs: Any) -> SimpleNamespace:
        captured["train_calls"].append(kwargs)
        return SimpleNamespace(
            summary=SimpleNamespace(
                test_metrics=SimpleNamespace(
                    regime_f1_macro=0.5,
                    regime_accuracy=0.5,
                    regime_loss=1.0,
                    regression_rmse_log_rv=1.0,
                    regression_mae_log_rv=0.8,
                    regression_loss=1.0,
                )
            )
        )

    monkeypatch.setattr(
        loaders_module, "load_walk_forward_split", _fake_load_walk_forward_split
    )
    monkeypatch.setattr(loop_module, "train_model", _fake_train_model)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
        ],
    )
    args = runner._parse_args()
    assert args.regime_loss == "ce"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--regime-loss",
            "ordinal_ce",
        ],
    )
    args = runner._parse_args()
    assert args.regime_loss == "ordinal_ce"

    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
        regime_loss="ordinal_ce",
    )
    assert captured["train_calls"], "runner did not call train_model"
    model_config = captured["train_calls"][0]["model_config"]
    assert model_config.regime_loss_mode == "ordinal_ce"
