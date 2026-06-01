"""Cover the MultiTaskHead module (#78).

The head shape is what the inference path and the /analyze response
serialiser depend on. Pin the per-branch shapes + value bounds so a
future refactor does not silently change the wire format.
"""

from __future__ import annotations

import torch


def test_multi_task_head_emits_three_branches_with_expected_shapes() -> None:
    """Active axes: stance / certainty / time. Factor was retired (text
    cannot predict the GSS target); topic was retired in ADR 0044."""
    from app.models.multi_task_head import MultiTaskHead

    head = MultiTaskHead(
        hidden_size=32,
        head_hidden_size=16,
        dropout=0.0,
    )
    pooled = torch.zeros(4, 32)
    out = head(pooled)

    assert set(out.keys()) == {"stance", "certainty", "time"}
    assert out["stance"].shape == (4, 3)
    assert out["certainty"].shape == (4, 3)
    assert out["time"].shape == (4, 2)


def test_time_branch_softmaxes_to_a_two_class_distribution() -> None:
    """The time branch emits 2-class raw logits; a softmax over them
    yields probabilities in ``[0, 1]`` that sum to 1 per row."""

    from app.models.multi_task_head import MultiTaskHead

    torch.manual_seed(0)
    head = MultiTaskHead(hidden_size=8, head_hidden_size=8, dropout=0.0)
    pooled = torch.randn(16, 8) * 50.0
    out = head(pooled)
    probs = torch.softmax(out["time"], dim=-1)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_classification_branches_are_raw_logits_not_softmax() -> None:
    """CrossEntropyLoss applies log_softmax internally; the head must
    not pre-softmax. Confirm by asserting the row sums vary (raw
    logits) instead of summing to 1 (softmax)."""

    from app.models.multi_task_head import MultiTaskHead

    torch.manual_seed(1)
    head = MultiTaskHead(hidden_size=8, head_hidden_size=8, dropout=0.0)
    pooled = torch.randn(4, 8)
    out = head(pooled)
    assert set(out.keys()) == {"stance", "certainty", "time"}
    stance_sum = out["stance"].sum(dim=-1)
    # Probabilities summing to 1.0 would mean the head pre-softmaxed.
    assert not torch.allclose(stance_sum, torch.ones_like(stance_sum), atol=1e-4)
