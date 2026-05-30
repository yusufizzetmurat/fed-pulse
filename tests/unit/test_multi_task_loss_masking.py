"""Per-axis masking on the MultiTaskLoss (#78).

The two sparse axes (factor / certainty) carry labels on
fewer than 5% of supervised rows. A naive un-masked loss would train
those branches against a synthetic placeholder index — locking them
on a meaningless mean. The masked-loss contract is the load-bearing
behaviour the head depends on; pin it here.
"""

from __future__ import annotations

import torch


def _zero_logits_batch(batch: int = 4) -> dict[str, torch.Tensor]:
    return {
        "stance": torch.zeros(batch, 3, requires_grad=True),
        "factor": torch.zeros(batch, requires_grad=True),
        "certainty": torch.zeros(batch, 3, requires_grad=True),
    }


def test_axis_with_all_false_mask_contributes_zero_loss() -> None:
    from app.training.loss import MultiTaskLoss

    loss_fn = MultiTaskLoss()
    logits = _zero_logits_batch(batch=4)
    targets = {
        "stance": torch.tensor([0, 1, 2, 0]),
        "factor": torch.zeros(4),
        "certainty": torch.tensor([0, 0, 0, 0]),
    }
    masks = {
        "stance_mask": torch.tensor([True, True, True, True]),
        "factor_mask": torch.tensor([False, False, False, False]),
        "certainty_mask": torch.tensor([False, False, False, False]),
    }
    total, breakdown = loss_fn(logits, targets, masks)
    assert float(breakdown["factor"]) == 0.0
    assert float(breakdown["certainty"]) == 0.0
    # Stance is the only contributing axis so the total equals its lambda * loss.
    assert float(total) > 0.0


def test_masked_classification_matches_filtered_cross_entropy() -> None:
    """The masked branch must agree with a manual CE over the masked
    subset; this is the load-bearing equivalence behind the masking
    decision."""

    from app.training.loss import MultiTaskLoss
    from torch.nn import functional as F

    torch.manual_seed(7)
    loss_fn = MultiTaskLoss(lambda_factor=0.0, lambda_certainty=0.0)
    logits = {
        "stance": torch.randn(8, 3, requires_grad=True),
        "factor": torch.zeros(8, requires_grad=True),
        "certainty": torch.zeros(8, 3, requires_grad=True),
    }
    targets = {
        "stance": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        "factor": torch.zeros(8),
        "certainty": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
    }
    mask = torch.tensor([True, False, True, True, False, True, True, False])
    masks = {
        "stance_mask": mask,
        "factor_mask": torch.zeros(8, dtype=torch.bool),
        "certainty_mask": torch.zeros(8, dtype=torch.bool),
    }
    total, _ = loss_fn(logits, targets, masks)

    expected = F.cross_entropy(logits["stance"][mask], targets["stance"][mask])
    assert torch.allclose(total, expected, atol=1e-6)


def test_total_loss_carries_gradients_back_to_logits() -> None:
    """Even with all-False masks on two axes, the total loss must
    flow gradients back to the contributing axis's logits. A
    detached zero would prevent the optimiser from training the
    stance branch on its own."""

    from app.training.loss import MultiTaskLoss

    loss_fn = MultiTaskLoss()
    logits = _zero_logits_batch(batch=4)
    targets = {
        "stance": torch.tensor([0, 1, 2, 0]),
        "factor": torch.zeros(4),
        "certainty": torch.zeros(4, dtype=torch.long),
    }
    masks = {
        "stance_mask": torch.tensor([True, True, True, True]),
        "factor_mask": torch.zeros(4, dtype=torch.bool),
        "certainty_mask": torch.zeros(4, dtype=torch.bool),
    }
    total, _ = loss_fn(logits, targets, masks)
    total.backward()
    assert logits["stance"].grad is not None
    assert torch.any(logits["stance"].grad != 0.0)
