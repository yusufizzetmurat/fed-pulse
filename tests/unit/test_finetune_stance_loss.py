"""Unit tests for the Lead-1 loss-function knobs on the stance trainer.

Targets ``_class_weights`` (inverse-frequency vs class-balanced
effective-number) and ``_focal_loss`` (γ modulator on CE). The trainer
loop itself is not exercised here — too heavy, requires a real
encoder checkpoint — but the loss factory determines the gradient
signal the loop applies, so getting it right matters for the
hold-vs-cut resolution gap the validity study flagged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.data.finetune_stance import _class_weights, _focal_loss


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


def test_class_weights_inverse_freq_matches_legacy(device) -> None:
    """``ce`` and ``focal`` modes must keep the pre-Lead-1 inverse-frequency
    weighting (so an unchanged baseline retrain produces the same loss).
    """

    counts = np.array([1000, 200, 50], dtype=np.float64)
    w_ce = _class_weights(counts, mode="ce", cb_beta=0.0, device=device)
    w_focal = _class_weights(counts, mode="focal", cb_beta=0.0, device=device)
    # Both must produce the same legacy ratio; normalised to max=1.
    raw = counts.sum() / (len(counts) * counts)
    expected = raw / raw.max()
    np.testing.assert_allclose(w_ce.cpu().numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(w_focal.cpu().numpy(), expected, atol=1e-6)


def test_class_weights_balanced_smooths_rare_class_gradient(device) -> None:
    """``ce_balanced`` at β → 1 is a *regularised* alternative to naive
    inverse-frequency: the rare-class weight saturates instead of
    growing inversely with count. The motivation for Lead 1 is not "hit
    rare classes harder" — that's what inverse-frequency already does
    — but "stabilise the rare-class signal" so the encoder converges
    on robust hold-vs-cut features rather than over-fitting cut-class
    noise. Lock in the saturation property here.
    """

    counts = np.array([1000, 200, 50], dtype=np.float64)
    w_inv = _class_weights(counts, mode="ce", cb_beta=0.0, device=device)
    w_bal = _class_weights(counts, mode="ce_balanced", cb_beta=0.999, device=device)
    # Rare class is index 2. After max-normalisation both produce
    # rare_weight == 1.0; the diagnostic is the DOMINANT-class weight,
    # which is HIGHER under class-balanced because the rare-class
    # weight saturates instead of running to 20x dominant.
    assert float(w_bal[0]) > float(w_inv[0])
    # The order is preserved: still rare > medium > dominant.
    assert float(w_bal[2]) >= float(w_bal[1]) >= float(w_bal[0])
    # Largest class weight is normalised to 1.0 regardless of mode.
    assert float(w_bal.max()) == pytest.approx(1.0, abs=1e-6)


def test_class_weights_handle_empty_class(device) -> None:
    """A class with zero examples in the batch must not divide-by-zero;
    the loss factory clamps the count to ``1`` to keep gradients finite."""

    counts = np.array([100, 0, 50], dtype=np.float64)
    w = _class_weights(counts, mode="ce_balanced", cb_beta=0.999, device=device)
    assert all(math.isfinite(float(v)) for v in w.cpu().numpy().tolist())


def test_focal_loss_collapses_to_ce_when_gamma_zero(device) -> None:
    """``γ=0`` means the focal modulator is 1 for every example, so the
    loss must equal weighted cross-entropy on the same inputs."""

    logits = torch.tensor(
        [[2.0, 0.5, 0.1], [0.0, 3.0, 1.0], [1.0, 1.0, 2.5]], dtype=torch.float32
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    weight = torch.ones(3, dtype=torch.float32)

    focal = float(_focal_loss(logits, targets, weight=weight, gamma=0.0))
    ce = float(
        torch.nn.functional.cross_entropy(logits, targets, weight=weight, reduction="mean")
    )
    assert focal == pytest.approx(ce, abs=1e-6)


def test_focal_loss_downweights_confident_examples(device) -> None:
    """The whole point of focal: very confident correct predictions
    should contribute much less than ambiguous ones. Compare a perfect
    prediction (p≈1) against an ambiguous one (p≈0.4) under γ=2.0.
    """

    confident_logits = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    ambiguous_logits = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)
    weight = torch.ones(3, dtype=torch.float32)

    confident = float(
        _focal_loss(confident_logits, target, weight=weight, gamma=2.0)
    )
    ambiguous = float(
        _focal_loss(ambiguous_logits, target, weight=weight, gamma=2.0)
    )
    assert ambiguous > confident
    # And confident_loss should be near zero — the model has nothing
    # to learn from the easy example.
    assert confident < 1e-3
