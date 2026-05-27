"""Per-axis class weights fit on the train slice only (#273).

The multi-task loss path constructs a separate ``CrossEntropyLoss``
weight tensor for stance / certainty / topic. The weights must be
fitted strictly on the train partition's masked rows -- val or test
labels must never inform the fit -- so the eval metrics are not
inflated by class-prior leakage from the held-out partitions. This
mirrors the existing single-task ``fit_class_weights`` protocol the
loop has used since #206 for the stance head.
"""

from __future__ import annotations

import torch

from app.training.loop import _fit_axis_class_weights_from_mask


def test_axis_weights_only_see_masked_train_rows() -> None:
    """Rows with mask=False contribute zero to the per-axis counts.

    Construct a partition where the masked-out rows would heavily skew
    the class prior if the helper ignored the mask. The fitted weights
    must come from the masked-IN rows only.
    """

    # Masked-in distribution: balanced 1:1:1 across 3 classes (6 rows).
    # Masked-out distribution: 100x class 0 (would dominate without mask).
    n_classes = 3
    masked_in_targets = [0, 0, 1, 1, 2, 2]
    masked_out_targets = [0] * 100
    targets = torch.tensor(masked_in_targets + masked_out_targets, dtype=torch.long)
    mask = torch.tensor(
        [True] * len(masked_in_targets) + [False] * len(masked_out_targets),
        dtype=torch.bool,
    )

    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes)

    # Balanced inverse-frequency on equal counts collapses to a
    # near-uniform tensor that sums to n_classes (the helper's
    # normalisation contract). A leaky implementation would emit
    # something heavily skewed toward downweighting class 0.
    assert weights.shape == (n_classes,)
    assert weights.sum().item() == _approx(float(n_classes))
    # Uniform within ~1% on this balanced subset; smoothing keeps it
    # away from a hard equality but well below any leak-driven skew.
    for w in weights.tolist():
        assert abs(w - 1.0) < 0.05, weights


def test_axis_weights_empty_mask_returns_uniform() -> None:
    """An axis with zero supervised rows yields a uniform fallback.

    Topic on FOMC-only training has 0% label coverage upstream; the
    weight fit must still return a well-defined length-n_classes tensor
    so MultiTaskLoss can construct a CrossEntropyLoss for that axis.
    """

    n_classes = 4
    targets = torch.zeros(10, dtype=torch.long)
    mask = torch.zeros(10, dtype=torch.bool)  # all False

    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes)

    assert weights.shape == (n_classes,)
    assert torch.allclose(weights, torch.ones(n_classes))


def test_axis_weights_rare_class_gets_higher_weight() -> None:
    """Inverse-frequency: a rare class gets a larger weight than a common one.

    The standard contract the stance ``fit_class_weights`` carries: the
    minority class should pull more gradient per row than the majority.
    """

    n_classes = 3
    # Heavily imbalanced: 90 rows of class 0, 9 of class 1, 1 of class 2.
    targets = torch.tensor([0] * 90 + [1] * 9 + [2], dtype=torch.long)
    mask = torch.ones(100, dtype=torch.bool)

    weights = _fit_axis_class_weights_from_mask(targets, mask, n_classes)

    # The minority class (2) must outweigh class 1 must outweigh
    # the majority class (0).
    assert weights[2].item() > weights[1].item() > weights[0].item()


def _approx(expected: float, *, rel: float = 1e-4) -> object:
    """Tiny inline approx so the test file stays free of pytest imports
    beyond the assertion expression itself.
    """

    class _Approx:
        def __eq__(self, other: object) -> bool:
            try:
                return abs(float(other) - expected) <= rel * max(1.0, abs(expected))
            except (TypeError, ValueError):
                return False

        def __repr__(self) -> str:  # pragma: no cover - diag only
            return f"approx({expected}, rel={rel})"

    return _Approx()
