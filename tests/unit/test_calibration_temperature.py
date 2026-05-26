from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from app.evaluation.calibration_temperature import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
    render_reliability_diagram_png,
)


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------


def test_calibrated_logits_recover_T_near_one() -> None:
    """When the model is already calibrated (truth-aligned logits), the
    fitted temperature should be near 1.0."""

    torch.manual_seed(11)
    # Construct 200 well-calibrated rows: large logit for the true
    # class, small logits for the others -> high confidence + high acc
    n = 200
    targets = torch.randint(0, 3, (n,))
    logits = torch.full((n, 3), -2.0)
    for i, t in enumerate(targets):
        logits[i, int(t)] = 2.5
    T = fit_temperature(logits, targets)
    assert 0.5 < T < 2.0, f"expected T near 1.0 for calibrated logits, got {T}"


def test_overconfident_logits_yield_T_above_one() -> None:
    """When logits are too sharp (model is overconfident) the fit should
    push T > 1 to soften the distribution."""

    torch.manual_seed(11)
    # Construct overconfident logits: large gap between classes but
    # only half the predictions are actually correct.
    n = 200
    targets = torch.randint(0, 3, (n,))
    logits = torch.full((n, 3), -3.0)
    for i, t in enumerate(targets):
        # 50% correct: alternate between giving high logit to the true
        # class vs giving high logit to a random wrong class.
        if i % 2 == 0:
            logits[i, int(t)] = 4.0
        else:
            wrong = (int(t) + 1) % 3
            logits[i, wrong] = 4.0
    T = fit_temperature(logits, targets)
    assert T > 1.5, f"expected T > 1.5 for overconfident logits, got {T}"


def test_underconfident_logits_yield_T_below_one() -> None:
    """When logits are too flat the fit should push T < 1 to sharpen.
    The magnitude depends on how large the available cross-entropy
    gradient is, so the test only asserts the directional move below
    1.0 -- a strict-bound assertion would be sensitive to the tiny
    gradient scale of nearly-flat logits."""

    torch.manual_seed(11)
    n = 200
    targets = torch.randint(0, 3, (n,))
    # Logits are nearly uniform but the right class consistently has
    # a small edge -- and the predictions are correct ~all the time
    # so the network is underconfident relative to its accuracy.
    logits = torch.full((n, 3), 0.0)
    for i, t in enumerate(targets):
        logits[i, int(t)] = 0.5  # modest edge for the right class
    T = fit_temperature(logits, targets)
    assert T < 1.0, f"expected T < 1.0 for underconfident logits, got {T}"


def test_fit_temperature_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_temperature(torch.randn(10, 3), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="2-D"):
        fit_temperature(torch.randn(10), torch.tensor([0] * 10))
    with pytest.raises(ValueError, match="1-D"):
        fit_temperature(torch.randn(10, 3), torch.tensor([[0]] * 10))


def test_fit_temperature_handles_empty_input() -> None:
    """No samples -> return the initial T without crashing."""

    T = fit_temperature(torch.empty(0, 3), torch.empty(0, dtype=torch.long), initial_T=2.0)
    assert T == 2.0


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------


def test_apply_temperature_preserves_argmax() -> None:
    """For any positive T, the argmax of softmax(logits / T) equals
    argmax(logits) -- temperature scaling does not change predictions."""

    torch.manual_seed(11)
    logits = torch.randn(50, 3) * 3
    pred_uncalibrated = torch.argmax(logits, dim=-1)
    for T in [0.5, 1.0, 2.0, 5.0]:
        probs = apply_temperature(logits, T)
        pred_calibrated = torch.argmax(probs, dim=-1)
        assert torch.equal(pred_uncalibrated, pred_calibrated), (
            f"argmax shifted at T={T}"
        )


def test_apply_temperature_softens_distribution_for_T_gt_one() -> None:
    """T > 1 -> the max softmax probability decreases."""

    logits = torch.tensor([[3.0, 1.0, 0.0]])
    probs_baseline = apply_temperature(logits, 1.0)
    probs_softer = apply_temperature(logits, 2.0)
    assert probs_softer.max().item() < probs_baseline.max().item()


def test_apply_temperature_rejects_zero_or_negative_T() -> None:
    logits = torch.randn(5, 3)
    with pytest.raises(ValueError, match="> 0"):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError, match="> 0"):
        apply_temperature(logits, -1.0)


# ---------------------------------------------------------------------------
# reliability_curve / ECE
# ---------------------------------------------------------------------------


def test_reliability_curve_on_perfect_predictions() -> None:
    """If every prediction is right with confidence 1.0, all rows land in
    the last bin and bin-accuracy == bin-confidence == 1.0."""

    probs = [[1.0, 0.0, 0.0]] * 20
    targets = [0] * 20
    curve = reliability_curve(probs, targets, n_bins=10)
    last_bin = curve.bins[-1]
    assert last_bin.count == 20
    assert last_bin.accuracy == 1.0
    assert last_bin.confidence_mean == 1.0
    assert curve.ece == pytest.approx(0.0)


def test_reliability_curve_detects_perfect_miscalibration() -> None:
    """All predictions wrong at confidence 0.9 -> bin accuracy = 0,
    bin confidence = 0.9, ECE picks up the 0.9 gap."""

    probs = [[0.9, 0.05, 0.05]] * 30
    targets = [1] * 30  # the model predicts 0; truth is always 1
    curve = reliability_curve(probs, targets, n_bins=10)
    # Bin 8 covers [0.8, 0.9) and bin 9 covers [0.9, 1.0]; 0.9 lands
    # in bin 9 under the closed-closed rule.
    populated = [b for b in curve.bins if b.count > 0]
    assert len(populated) == 1
    assert populated[0].accuracy == 0.0
    assert populated[0].confidence_mean == pytest.approx(0.9)
    assert curve.ece == pytest.approx(0.9, abs=0.01)


def test_reliability_curve_handles_empty_input() -> None:
    curve = reliability_curve([], [])
    assert curve.n_rows == 0
    assert curve.ece == 0.0
    assert curve.bins == ()


def test_reliability_curve_validates_input_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        reliability_curve([[0.5, 0.5]], [0, 1])
    with pytest.raises(ValueError, match="positive"):
        reliability_curve([[1.0, 0.0]], [0], n_bins=0)


def test_expected_calibration_error_matches_reliability_curve() -> None:
    """``expected_calibration_error`` is just a shortcut to
    ``reliability_curve.ece``; they must agree exactly."""

    probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.4, 0.4, 0.2]]
    targets = [0, 1, 2]
    full = reliability_curve(probs, targets, n_bins=5)
    shortcut = expected_calibration_error(probs, targets, n_bins=5)
    assert shortcut == full.ece


# ---------------------------------------------------------------------------
# Reliability diagram renderer
# ---------------------------------------------------------------------------


def test_renders_reliability_diagram_png(tmp_path) -> None:
    probs = [[0.7, 0.2, 0.1]] * 10 + [[0.4, 0.5, 0.1]] * 10
    targets = [0] * 10 + [1] * 10
    curve = reliability_curve(probs, targets, n_bins=10)
    out = tmp_path / "reliability.png"
    render_reliability_diagram_png(curve, out, title="Tier 2 test partition")
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        # Plot is fixed-size + margins; sanity check the image is the
        # right ballpark.
        assert img.size[0] > 400
        assert img.size[1] > 400


def test_renders_empty_curve_without_crash(tmp_path) -> None:
    """Edge case -- no rows -> the renderer should still produce a
    valid (empty) plot."""

    curve = reliability_curve([], [])
    out = tmp_path / "empty.png"
    render_reliability_diagram_png(curve, out)
    assert out.exists()
