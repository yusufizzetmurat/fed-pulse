"""Unit tests for the regime-classifier post-hoc calibrators.

Covers ``fit_temperature``, ``fit_platt_per_class``, the ECE / Brier /
NLL math helpers, and the end-to-end NLL-improvement property that the
calibrator is meant to deliver. The CLI / checkpoint-IO surface of the
driver script is exercised by the existing
``test_calibration_temperature`` module and by manual operator runs;
this file targets the parts the issue scopes as net-new.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from app.evaluation.calibration_temperature import (  # noqa: E402
    apply_platt_per_class,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_platt_per_class,
    fit_temperature,
    negative_log_likelihood,
)


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_overconfident_logits_yields_T_gt_one() -> None:
    """Known-overconfident fixture: half the rows predict the wrong class
    with a huge logit gap. Fitter should push T > 1 to soften."""

    torch.manual_seed(29)
    n = 400
    targets = torch.randint(0, 3, (n,))
    logits = torch.full((n, 3), -3.0)
    for i, t in enumerate(targets):
        if i % 2 == 0:
            logits[i, int(t)] = 5.0  # correct, high confidence
        else:
            wrong = (int(t) + 1) % 3
            logits[i, wrong] = 5.0  # wrong, high confidence

    T = fit_temperature(logits, targets)
    assert T > 1.0, f"overconfident logits should push T > 1; got {T}"


# ---------------------------------------------------------------------------
# fit_platt_per_class
# ---------------------------------------------------------------------------


def test_fit_platt_per_class_returns_one_param_pair_per_class() -> None:
    torch.manual_seed(11)
    logits = torch.randn(120, 3)
    targets = torch.randint(0, 3, (120,))
    params = fit_platt_per_class(logits, targets, n_classes=3)
    assert len(params) == 3
    for a, b in params:
        assert isinstance(a, float) and isinstance(b, float)


def test_fit_platt_per_class_corrects_known_per_class_shift() -> None:
    """Construct a fixture where class 0 is systematically under-scored:
    the true class-0 rows carry low softmax mass for class 0 (~0.4) but
    most are correct. Platt should recover a positive slope ``a`` for
    class 0 (or at least improve the BCE on the held-out logits)."""

    torch.manual_seed(47)
    n = 300
    targets = torch.zeros(n, dtype=torch.long)

    # Build logits where class 0 wins by a slim margin on most rows but
    # the absolute softmax score for class 0 hovers ~0.4 (underconfident).
    logits = torch.zeros(n, 3)
    for i in range(n):
        logits[i, 0] = 0.6
        logits[i, 1] = 0.3
        logits[i, 2] = 0.0

    params = fit_platt_per_class(logits, targets, n_classes=3)
    a0, b0 = params[0]

    # The class-0 BCE under raw softmax vs Platt: Platt should not be
    # worse, and on this fixture the slope+offset combo should push
    # class-0 probabilities up (a*z + b should yield p > z for z near
    # the cluster mean).
    z_cluster_mean = torch.softmax(logits[0], dim=-1)[0].item()
    raw_p = z_cluster_mean
    platt_p = 1.0 / (1.0 + math.exp(-(a0 * z_cluster_mean + b0)))
    assert platt_p > raw_p, (
        f"platt should lift the under-confident class-0 score: "
        f"raw={raw_p:.3f}, platt={platt_p:.3f}, a0={a0}, b0={b0}"
    )


def test_fit_platt_per_class_handles_empty_input() -> None:
    params = fit_platt_per_class(
        torch.empty(0, 3), torch.empty(0, dtype=torch.long), n_classes=3
    )
    assert params == [(1.0, 0.0)] * 3


def test_fit_platt_per_class_handles_absent_class() -> None:
    """If a class is never represented in val targets, identity (1, 0)
    should be returned for that class."""

    torch.manual_seed(71)
    n = 100
    targets = torch.zeros(n, dtype=torch.long)  # only class 0 seen
    logits = torch.randn(n, 3)
    params = fit_platt_per_class(logits, targets, n_classes=3)
    # Class 0 has both positive and negative examples? No -- all rows
    # are class 0, so for class 1 and 2 every target is 0 -> identity.
    assert params[1] == (1.0, 0.0)
    assert params[2] == (1.0, 0.0)


def test_apply_platt_per_class_yields_distribution() -> None:
    torch.manual_seed(97)
    logits = torch.randn(20, 3) * 2
    params = [(1.5, -0.2), (1.0, 0.0), (0.8, 0.1)]
    out = apply_platt_per_class(logits, params)
    assert out.shape == logits.shape
    row_sums = out.sum(dim=-1)
    for s in row_sums.tolist():
        assert abs(s - 1.0) < 1e-6


def test_apply_platt_per_class_rejects_param_length_mismatch() -> None:
    with pytest.raises(ValueError, match="params length"):
        apply_platt_per_class(torch.randn(5, 3), [(1.0, 0.0), (1.0, 0.0)])


# ---------------------------------------------------------------------------
# ECE hand-computed
# ---------------------------------------------------------------------------


def test_expected_calibration_error_matches_hand_computed_ten_bins() -> None:
    """Hand-built 10-row, 10-bin case.

    The fixture places one row in each of the top 5 bins (so the
    argmax-class probability lands in a distinct bin per row); the
    bottom 5 bins are empty. Half the rows are correct (alternating).
    Per-bin gap is ``|confidence - accuracy|``; mass-weighted sum is
    the ECE.

    Constraint: the max-prob row form ``[c, (1 - c) / 2, (1 - c) / 2]``
    keeps class 0 as argmax only when ``c > (1 - c) / 2`` i.e.
    ``c > 1/3``; that is why the fixture only populates the top half.
    """

    midpoints = [0.55, 0.65, 0.75, 0.85, 0.95]
    probs = [[c, (1.0 - c) / 2, (1.0 - c) / 2] for c in midpoints]
    targets = [0, 1, 0, 1, 0]  # alternating correct (class 0) / wrong (class 1)
    n_rows = len(probs)

    expected = sum(
        (1 / n_rows) * abs(c - (1.0 if t == 0 else 0.0))
        for c, t in zip(midpoints, targets)
    )
    actual = expected_calibration_error(probs, targets, n_bins=10)
    assert actual == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def test_brier_score_perfect_prediction_is_zero() -> None:
    probs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    targets = [0, 1, 2]
    assert brier_score(probs, targets, n_classes=3) == pytest.approx(0.0)


def test_brier_score_matches_manual_computation() -> None:
    # Row 0: target=0, ||p - [1,0,0]||^2 = (0.2)^2 + (0.5)^2 + (0.3)^2 = 0.38
    # Row 1: target=1, ||p - [0,1,0]||^2 = (0.6)^2 + (0.7)^2 + (0.3)^2 = 0.94
    probs = [[0.8, 0.5, 0.3], [0.6, 0.3, 0.3]]
    targets = [0, 1]
    expected = (0.38 + 0.94) / 2
    assert brier_score(probs, targets, n_classes=3) == pytest.approx(expected, abs=1e-9)


def test_brier_score_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        brier_score([[1.0, 0.0, 0.0]], [0, 1])


def test_brier_score_rejects_row_width_mismatch() -> None:
    with pytest.raises(ValueError, match="row width"):
        brier_score([[1.0, 0.0]], [0], n_classes=3)


# ---------------------------------------------------------------------------
# NLL helper
# ---------------------------------------------------------------------------


def test_negative_log_likelihood_matches_manual() -> None:
    probs = [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]
    targets = [0, 1]
    expected = (-math.log(0.8) + -math.log(0.6)) / 2
    assert negative_log_likelihood(probs, targets) == pytest.approx(expected, abs=1e-9)


def test_negative_log_likelihood_floors_zero_probability() -> None:
    probs = [[0.0, 1.0, 0.0]]
    targets = [0]
    # Floored at eps=1e-12 -> log(1e-12) ~ -27.6
    assert negative_log_likelihood(probs, targets) > 20.0


# ---------------------------------------------------------------------------
# End-to-end: synthetic overconfident logits -> NLL drops after fit
# ---------------------------------------------------------------------------


def test_temperature_fit_reduces_nll_on_overconfident_synthetic() -> None:
    """End-to-end property: on a known-overconfident synthetic fixture
    the fitted T should produce strictly lower NLL than the raw
    softmax."""

    torch.manual_seed(11)
    n = 500
    targets = torch.randint(0, 3, (n,))
    logits = torch.full((n, 3), -3.0)
    for i, t in enumerate(targets):
        if i % 2 == 0:
            logits[i, int(t)] = 5.0
        else:
            wrong = (int(t) + 1) % 3
            logits[i, wrong] = 5.0

    pre_probs = torch.softmax(logits, dim=-1).tolist()
    pre_nll = negative_log_likelihood(pre_probs, targets.tolist())

    T = fit_temperature(logits, targets)
    post_probs = apply_temperature(logits, T).tolist()
    post_nll = negative_log_likelihood(post_probs, targets.tolist())

    assert post_nll < pre_nll, (
        f"temperature fit should lower NLL; pre={pre_nll:.4f}, post={post_nll:.4f}, T={T:.3f}"
    )


def test_platt_fit_does_not_worsen_brier_on_overconfident_synthetic() -> None:
    """Companion property for Platt: on the same overconfident fixture,
    Platt should not regress the Brier score relative to raw softmax."""

    torch.manual_seed(11)
    n = 500
    targets = torch.randint(0, 3, (n,))
    logits = torch.full((n, 3), -3.0)
    for i, t in enumerate(targets):
        if i % 2 == 0:
            logits[i, int(t)] = 5.0
        else:
            wrong = (int(t) + 1) % 3
            logits[i, wrong] = 5.0

    pre_probs = torch.softmax(logits, dim=-1).tolist()
    pre_brier = brier_score(pre_probs, targets.tolist(), n_classes=3)

    params = fit_platt_per_class(logits, targets, n_classes=3)
    post_probs = apply_platt_per_class(logits, params).tolist()
    post_brier = brier_score(post_probs, targets.tolist(), n_classes=3)

    assert post_brier <= pre_brier + 1e-6, (
        f"platt fit should not regress brier; pre={pre_brier:.4f}, post={post_brier:.4f}"
    )
