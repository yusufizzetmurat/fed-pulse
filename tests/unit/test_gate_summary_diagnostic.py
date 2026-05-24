"""Cover the gate-distribution diagnostic on the multi-modal eval path (#235).

When the model is a MultiModalForecasterModel, the eval pass captures
the per-row gate tensor and reduces it into a summary dict that
attaches to the returned EvaluationMetrics. The thesis appendix
reads ``mean`` (modality lean) and ``mean_per_class`` (regime-
conditional gate drift) off this dict.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.training.loop import _summarise_gate


def test_summarise_gate_returns_none_when_no_gate_chunks() -> None:
    """Legacy single-modal eval path collects no gate values; the
    summary must stay None so the EvaluationMetrics contract on
    pre-#235 callers is unchanged."""

    out = _summarise_gate([], true_classes=torch.empty(0, dtype=torch.long), n_classes=3)
    assert out is None


def test_summarise_gate_computes_overall_mean_and_dim_means() -> None:
    """Two batches of (B=2, latent_dim=4) gate values; the summary
    must average across the (N=4, latent_dim=4) stack."""

    gate_a = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    gate_b = torch.tensor([[0.9, 1.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.5]])
    out = _summarise_gate(
        [gate_a, gate_b],
        true_classes=torch.tensor([0, 1, 2, 1]),
        n_classes=3,
    )
    assert out is not None
    assert out["n_rows"] == 4
    # Mean over (4, 4) tensor: sum = 8.0, mean = 0.5
    assert abs(out["mean"] - 0.5) < 1e-6
    assert len(out["mean_per_dim"]) == 4


def test_summarise_gate_per_class_mean_handles_empty_class() -> None:
    """Classes with zero rows in the eval partition (e.g. wf_fold_4's
    missing calm class on legacy data) must report None for that
    slot rather than crashing on a div-by-zero."""

    gate = torch.tensor([[0.7, 0.7], [0.3, 0.3]])
    out = _summarise_gate(
        [gate],
        true_classes=torch.tensor([0, 2]),  # class 1 has 0 rows
        n_classes=3,
    )
    assert out is not None
    assert out["mean_per_class"][1] is None
    # Class 0 = single row [0.7, 0.7] → mean 0.7; class 2 = [0.3, 0.3] → 0.3
    assert out["mean_per_class"][0] is not None
    assert abs(out["mean_per_class"][0] - 0.7) < 1e-6
    assert out["mean_per_class"][2] is not None
    assert abs(out["mean_per_class"][2] - 0.3) < 1e-6


def test_summarise_gate_skips_per_class_when_classes_misaligned() -> None:
    """When the true-class tensor length disagrees with the gate
    row count (defensive contract; should not happen in practice),
    the per-class breakdown stays at None entries but the overall
    mean still surfaces."""

    gate = torch.tensor([[0.5, 0.5], [0.5, 0.5]])  # N=2
    out = _summarise_gate(
        [gate],
        true_classes=torch.tensor([0]),  # N=1 — mismatch
        n_classes=3,
    )
    assert out is not None
    assert out["mean"] == pytest.approx(0.5)
    assert all(v is None for v in out["mean_per_class"])
