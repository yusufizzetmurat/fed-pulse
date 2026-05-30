"""Cover the MultiTaskHead module (#78).

The head shape is what the inference path and the /analyze response
serialiser depend on. Pin the per-branch shapes + value bounds so a
future refactor does not silently change the wire format.
"""

from __future__ import annotations

import torch


def test_multi_task_head_emits_three_branches_with_expected_shapes() -> None:
    """Post-ADR-0044: topic branch retired; head emits stance / factor / certainty."""
    from app.models.multi_task_head import MultiTaskHead

    head = MultiTaskHead(
        hidden_size=32,
        head_hidden_size=16,
        dropout=0.0,
    )
    pooled = torch.zeros(4, 32)
    out = head(pooled)

    assert set(out.keys()) == {"stance", "factor", "certainty"}
    assert out["stance"].shape == (4, 3)
    assert out["factor"].shape == (4,)
    assert out["certainty"].shape == (4, 3)


def test_factor_branch_stays_in_minus_one_to_one_range() -> None:
    """The factor regressor applies tanh so the emit value lives in
    ``[-1, 1]``. Without that bound an unsupervised row could blow
    out the scale of the regression target and confuse the
    downstream RobustScaler."""

    from app.models.multi_task_head import MultiTaskHead

    torch.manual_seed(0)
    head = MultiTaskHead(hidden_size=8, head_hidden_size=8, dropout=0.0)
    # Force a large activation through the stem so an unbounded head
    # would saturate the regression branch above 1.0.
    pooled = torch.randn(16, 8) * 50.0
    out = head(pooled)
    assert torch.all(out["factor"] >= -1.0)
    assert torch.all(out["factor"] <= 1.0)


def test_classification_branches_are_raw_logits_not_softmax() -> None:
    """CrossEntropyLoss applies log_softmax internally; the head must
    not pre-softmax. Confirm by asserting the row sums vary (raw
    logits) instead of summing to 1 (softmax)."""

    from app.models.multi_task_head import MultiTaskHead

    torch.manual_seed(1)
    head = MultiTaskHead(hidden_size=8, head_hidden_size=8, dropout=0.0)
    pooled = torch.randn(4, 8)
    out = head(pooled)
    stance_sum = out["stance"].sum(dim=-1)
    # Probabilities summing to 1.0 would mean the head pre-softmaxed.
    assert not torch.allclose(stance_sum, torch.ones_like(stance_sum), atol=1e-4)
