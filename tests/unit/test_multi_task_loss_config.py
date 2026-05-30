"""Cover the multi-task-loss config + factory plumbing (#273 foundation).

The wider plumbing — threading per-axis target tensors through the
DataLoader and swapping CrossEntropy for MultiTaskLoss in the training
step — lands as a follow-up. This test file pins the configuration
surface so the follow-up has a stable contract to build against.
"""

from __future__ import annotations

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
    assert cfg.multi_task_lambda_factor == 0.3
    assert cfg.multi_task_lambda_certainty == 0.3


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
        multi_task_lambda_factor=0.2,
        multi_task_lambda_certainty=0.4,
    )
    model = build_forecaster(cfg)
    assert bool(getattr(model, "multi_task_loss")) is True
    assert float(getattr(model, "multi_task_lambda_stance")) == 0.5
    assert float(getattr(model, "multi_task_lambda_factor")) == 0.2
    assert float(getattr(model, "multi_task_lambda_certainty")) == 0.4


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
