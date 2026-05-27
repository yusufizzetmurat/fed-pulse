"""Class-conditional coverage + set-size distribution diagnostics (#326).

APS conformal calibration gives a marginal coverage guarantee under
exchangeability, but the per-class slice is silent. When the model's
softmax systematically excludes one class (the canonical normal-class
collapse on the 3-class vol-regime head), marginal 0.80 coverage can
hide degenerate per-class coverage like
``{calm: 0.95, normal: 0.05, high: 0.92}``.

These tests cover the helpers that surface the gap:

* ``compute_class_conditional_coverage`` -- per-class empirical
  coverage on the calibration fold.
* ``compute_set_size_distribution`` -- Pr[|S|=k] for k in {1, 2, 3}.
* ``class_conditional_gap_flag`` -- class names whose coverage falls
  >0.10 below nominal.
* ``compute_regression_band_class_coverage`` -- the dual interpretation
  on the regression-canonical surface, where buckets come from
  ``bucket_log_rv`` and the conformal object is a residual band.

Plus the manifest round-trip carrying both new fields, and an end-to-
end calibration-path test that walks
``_maybe_write_classification_conformal_manifest`` on synthetic
data and asserts the persisted sidecar carries the new fields.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path

import pytest

from app.evaluation.conformal import (
    ConformalManifest,
    calibrate_classification_conformal,
    class_conditional_gap_flag,
    compute_class_conditional_coverage,
    compute_regression_band_class_coverage,
    compute_set_size_distribution,
    load_manifest,
    predict_conformal_set,
    save_manifest,
)


# ---------------------------------------------------------------------------
# compute_class_conditional_coverage
# ---------------------------------------------------------------------------


def test_class_conditional_coverage_perfect_when_every_set_contains_truth() -> None:
    """Trivial sanity: every prediction set contains the true class →
    each class slice covers at 1.0."""

    sets = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    truths = [0, 1, 2, 0, 1]
    out = compute_class_conditional_coverage(sets, truths, ("calm", "normal", "high"))
    assert out == {"calm": 1.0, "normal": 1.0, "high": 1.0}


def test_class_conditional_coverage_detects_normal_collapse() -> None:
    """Issue #326's canonical case: APS sets systematically omit
    class 1 (``normal``). Marginal coverage stays high because the
    sets cover the other two classes; the per-class slice on
    ``normal`` collapses to ~0."""

    # 100 rows total; class 0 / 1 / 2 each get ~33 rows.
    # Sets always contain {0, 2} — never include class 1.
    sets = [[0, 2]] * 99
    truths = [i % 3 for i in range(99)]  # 33 per class
    out = compute_class_conditional_coverage(sets, truths, ("calm", "normal", "high"))
    assert out["calm"] == pytest.approx(1.0)
    assert out["normal"] == pytest.approx(0.0)
    assert out["high"] == pytest.approx(1.0)


def test_class_conditional_coverage_returns_nan_for_empty_slice() -> None:
    """A class with zero rows on the calibration partition must map
    to NaN so the gap-flag helper can skip it cleanly rather than
    flagging a 0/0 division."""

    sets = [[0]]
    truths = [0]
    out = compute_class_conditional_coverage(sets, truths, ("calm", "normal", "high"))
    assert out["calm"] == pytest.approx(1.0)
    assert math.isnan(out["normal"])
    assert math.isnan(out["high"])


def test_class_conditional_coverage_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_class_conditional_coverage([[0]], [0, 1], ("a", "b"))


def test_class_conditional_coverage_rejects_empty_class_names() -> None:
    with pytest.raises(ValueError):
        compute_class_conditional_coverage([[0]], [0], ())


# ---------------------------------------------------------------------------
# compute_set_size_distribution
# ---------------------------------------------------------------------------


def test_set_size_distribution_sums_to_one() -> None:
    sets = [[0], [0, 1], [0, 1, 2], [0], [1, 2]]
    out = compute_set_size_distribution(sets, n_classes=3)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out[1] == pytest.approx(2 / 5)
    assert out[2] == pytest.approx(2 / 5)
    assert out[3] == pytest.approx(1 / 5)


def test_set_size_distribution_handles_missing_sizes() -> None:
    """Sizes never observed must appear with mass 0.0 -- the contract
    is the full ``[1, n_classes]`` keyspace so a downstream consumer
    can render every bucket without a key-error."""

    sets = [[0], [1], [2]]
    out = compute_set_size_distribution(sets, n_classes=3)
    assert set(out.keys()) == {1, 2, 3}
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0)


def test_set_size_distribution_rejects_empty_set() -> None:
    """``predict_conformal_set`` never emits empties (argmax fallback);
    if a caller smuggles one in, the helper must surface that loudly
    rather than silently bucket into 0."""

    with pytest.raises(ValueError):
        compute_set_size_distribution([[]], n_classes=3)


def test_set_size_distribution_rejects_oversized_set() -> None:
    with pytest.raises(ValueError):
        compute_set_size_distribution([[0, 1, 2, 3]], n_classes=3)


def test_set_size_distribution_empty_input_returns_nans() -> None:
    out = compute_set_size_distribution([], n_classes=3)
    assert set(out.keys()) == {1, 2, 3}
    for v in out.values():
        assert math.isnan(v)


# ---------------------------------------------------------------------------
# class_conditional_gap_flag
# ---------------------------------------------------------------------------


def test_gap_flag_fires_on_degenerate_class() -> None:
    """The canonical issue case: ``normal`` at 0.05 must be flagged
    on the 0.80 nominal / 0.10 tolerance contract."""

    coverage = {"calm": 0.85, "normal": 0.05, "high": 0.82}
    flagged = class_conditional_gap_flag(coverage, nominal=0.80, tolerance=0.10)
    assert flagged == ["normal"]


def test_gap_flag_quiet_when_all_classes_in_band() -> None:
    coverage = {"calm": 0.78, "normal": 0.74, "high": 0.81}
    flagged = class_conditional_gap_flag(coverage, nominal=0.80, tolerance=0.10)
    assert flagged == []


def test_gap_flag_skips_nan_slices() -> None:
    """An empty class slice (NaN coverage) must not fire the flag --
    absence of evidence is not evidence of a degenerate gap."""

    coverage = {"calm": 0.85, "normal": float("nan"), "high": 0.82}
    flagged = class_conditional_gap_flag(coverage, nominal=0.80, tolerance=0.10)
    assert flagged == []


def test_gap_flag_threshold_inclusive_on_equal() -> None:
    """Coverage exactly equal to ``nominal - tolerance`` (0.70 in the
    default contract) is on the boundary -- the helper does NOT flag
    it, matching the issue's '>0.10 below' wording."""

    coverage = {"normal": 0.70}
    assert class_conditional_gap_flag(coverage, nominal=0.80, tolerance=0.10) == []


def test_gap_flag_returns_multiple_classes_in_input_order() -> None:
    coverage = {"calm": 0.10, "normal": 0.05, "high": 0.85}
    flagged = class_conditional_gap_flag(coverage, nominal=0.80, tolerance=0.10)
    assert flagged == ["calm", "normal"]


def test_gap_flag_rejects_invalid_nominal() -> None:
    with pytest.raises(ValueError):
        class_conditional_gap_flag({"a": 0.5}, nominal=1.5, tolerance=0.1)


def test_gap_flag_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        class_conditional_gap_flag({"a": 0.5}, nominal=0.8, tolerance=-0.1)


# ---------------------------------------------------------------------------
# compute_regression_band_class_coverage (regression-canonical dual)
# ---------------------------------------------------------------------------


def test_regression_band_perfect_when_band_is_wide_enough() -> None:
    """A wide residual quantile must cover every bucket at 1.0 -- a
    sanity check that the bucketing + band-membership logic does not
    accidentally exclude rows."""

    # Cutoffs in raw vol space → log cutoffs around log(0.10) and log(0.20).
    cutoffs = (0.10, 0.20)
    # Synthetic predictions perfectly equal to actuals, but bucket by truth.
    log_rv_actuals = [math.log(0.05), math.log(0.15), math.log(0.30)] * 20
    log_rv_predictions = [v + 0.01 for v in log_rv_actuals]
    out = compute_regression_band_class_coverage(
        log_rv_predictions=log_rv_predictions,
        log_rv_actuals=log_rv_actuals,
        residual_quantile=1.0,  # very wide
        raw_vol_cutoffs=cutoffs,
    )
    assert out["calm"] == pytest.approx(1.0)
    assert out["normal"] == pytest.approx(1.0)
    assert out["high"] == pytest.approx(1.0)


def test_regression_band_zero_when_band_is_too_tight() -> None:
    """A vanishingly small residual quantile + non-zero residuals →
    every class slice covers at 0.0."""

    cutoffs = (0.10, 0.20)
    log_rv_actuals = [math.log(0.05), math.log(0.15), math.log(0.30)] * 20
    log_rv_predictions = [v + 0.5 for v in log_rv_actuals]
    out = compute_regression_band_class_coverage(
        log_rv_predictions=log_rv_predictions,
        log_rv_actuals=log_rv_actuals,
        residual_quantile=1e-9,
        raw_vol_cutoffs=cutoffs,
    )
    assert out["calm"] == pytest.approx(0.0)
    assert out["normal"] == pytest.approx(0.0)
    assert out["high"] == pytest.approx(0.0)


def test_regression_band_nan_for_empty_bucket() -> None:
    cutoffs = (0.10, 0.20)
    # Only ``calm`` rows; ``normal`` and ``high`` slices empty → NaN.
    log_rv_actuals = [math.log(0.05)] * 5
    log_rv_predictions = list(log_rv_actuals)
    out = compute_regression_band_class_coverage(
        log_rv_predictions=log_rv_predictions,
        log_rv_actuals=log_rv_actuals,
        residual_quantile=0.5,
        raw_vol_cutoffs=cutoffs,
    )
    assert out["calm"] == pytest.approx(1.0)
    assert math.isnan(out["normal"])
    assert math.isnan(out["high"])


def test_regression_band_rejects_mismatched_cutoffs() -> None:
    with pytest.raises(ValueError):
        compute_regression_band_class_coverage(
            log_rv_predictions=[0.0],
            log_rv_actuals=[0.0],
            residual_quantile=0.1,
            raw_vol_cutoffs=(0.20, 0.10),  # inverted
        )


def test_regression_band_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_regression_band_class_coverage(
            log_rv_predictions=[0.0, 0.1],
            log_rv_actuals=[0.0],
            residual_quantile=0.1,
            raw_vol_cutoffs=(0.10, 0.20),
        )


# ---------------------------------------------------------------------------
# Manifest round-trip carrying the new fields
# ---------------------------------------------------------------------------


def test_manifest_round_trip_preserves_class_conditional_fields() -> None:
    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=250,
        softmax_quantile=0.42,
        class_conditional_coverage={"calm": 0.81, "normal": 0.05, "high": 0.79},
        set_size_distribution={1: 0.4, 2: 0.45, 3: 0.15},
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.conformal.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
    assert loaded.class_conditional_coverage == {
        "calm": pytest.approx(0.81),
        "normal": pytest.approx(0.05),
        "high": pytest.approx(0.79),
    }
    assert loaded.set_size_distribution is not None
    assert loaded.set_size_distribution[1] == pytest.approx(0.4)
    assert loaded.set_size_distribution[2] == pytest.approx(0.45)
    assert loaded.set_size_distribution[3] == pytest.approx(0.15)


def test_manifest_round_trip_loads_pre_326_manifests_with_none() -> None:
    """Pre-#326 manifests on disk lack both new keys. The loader must
    read them cleanly with both fields ``None`` so existing
    deployments survive the upgrade."""

    legacy_payload = {
        "alpha": 0.2,
        "nominal_coverage": 0.8,
        "residual_quantile_close": 11.0,
        "residual_quantile_volatility": 0.02,
        "calibration_n": 300,
        "softmax_quantile": 0.30,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "legacy.conformal.json"
        path.write_text(json.dumps(legacy_payload), encoding="utf-8")
        loaded = load_manifest(path)
    assert loaded.class_conditional_coverage is None
    assert loaded.set_size_distribution is None
    assert loaded.softmax_quantile == pytest.approx(0.30)


def test_manifest_round_trip_handles_nan_class_slice() -> None:
    """When a class is absent from the calibration fold,
    ``compute_class_conditional_coverage`` returns ``float('nan')`` for that
    class. The bare ``NaN`` token is not RFC-8259-compliant JSON, so
    ``save_manifest`` must serialise it as ``null`` and ``load_manifest``
    must round-trip the remaining finite entries cleanly (the all-NaN map
    collapses to ``None`` via the existing empty-after-filter guard)."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=11.0,
        residual_quantile_volatility=0.02,
        calibration_n=300,
        softmax_quantile=0.4,
        class_conditional_coverage={
            "calm": 0.83,
            "normal": float("nan"),
            "high": 0.71,
        },
        set_size_distribution={1: 0.6, 2: float("nan"), 3: 0.0},
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nan-slice.conformal.json"
        save_manifest(manifest, path)
        # The serialised file must be valid JSON (json.loads would raise
        # JSONDecodeError otherwise).
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["class_conditional_coverage"]["normal"] is None
        assert payload["set_size_distribution"]["2"] is None
        loaded = load_manifest(path)
    assert loaded.class_conditional_coverage == {"calm": 0.83, "high": 0.71}
    assert loaded.set_size_distribution == {1: 0.6, 3: 0.0}


def test_save_manifest_omits_none_class_conditional_fields() -> None:
    """A manifest without the new fields populated must serialise
    without the keys so a downstream reader of the legacy schema
    does not pick up ``null`` values it has to handle."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=11.0,
        residual_quantile_volatility=0.02,
        calibration_n=300,
        softmax_quantile=0.3,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "regression-only.conformal.json"
        save_manifest(manifest, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert "class_conditional_coverage" not in payload
    assert "set_size_distribution" not in payload


# ---------------------------------------------------------------------------
# End-to-end: calibration path populates the new fields on the sidecar
# ---------------------------------------------------------------------------


def _synthetic_collapse_softmax(n: int, *, seed: int = 0):
    """Generate softmax rows where class 1 is systematically suppressed.

    Models the issue case: class 1 is rare (~10 % of rows, matching the
    ``normal`` recall ~7 % the issue cites), and on the rare rows
    where it IS the truth, the softmax confidently picks one of the
    other two classes (class 1 stays at ~0.05). The bulk-correct
    rows (classes 0 + 2 with their own truth at ~0.85) drive the APS
    threshold low, so ``1 - softmax`` for class 1 rarely clears the
    bar → class 1 systematically excluded from prediction sets, and
    its conditional coverage collapses.
    """

    rng = random.Random(seed)
    softmax: list[list[float]] = []
    true_classes: list[int] = []
    for _ in range(n):
        # 10% class 1, ~45% class 0 / 2 each — drives a low APS
        # threshold from the confident-correct rows.
        roll = rng.random()
        if roll < 0.10:
            y = 1
        elif roll < 0.55:
            y = 0
        else:
            y = 2
        true_classes.append(y)
        if y == 1:
            # Truth is normal but model misclassifies as calm or high.
            other = rng.choice([0, 2])
            row = [0.05, 0.05, 0.05]
            row[other] = 0.90
        else:
            # Confident correct on the majority classes -- this drives
            # the calibration threshold low.
            row = [0.03, 0.03, 0.03]
            row[y] = 0.94
        s = sum(row)
        softmax.append([v / s for v in row])
    return softmax, true_classes


def test_calibration_path_persists_new_fields_and_flags_collapse() -> None:
    """End-to-end: the training-loop helper
    ``_maybe_write_classification_conformal_manifest`` consumes
    synthetic ``EvaluationMetrics`` whose class-1 softmax is
    suppressed and writes a sidecar that carries
    ``class_conditional_coverage`` + ``set_size_distribution``. The
    canonical normal-class slice must show degenerate coverage."""

    from app.evaluation.metrics import EvaluationMetrics
    from app.training.loop import _maybe_write_classification_conformal_manifest

    softmax, truths = _synthetic_collapse_softmax(450, seed=7)
    metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=softmax,
        targets=truths,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        assert sidecar.exists()
        loaded = load_manifest(sidecar)

    assert loaded.softmax_quantile is not None
    assert loaded.class_conditional_coverage is not None
    assert loaded.set_size_distribution is not None
    # The set-size distribution must sum to 1.0 within rounding.
    assert sum(loaded.set_size_distribution.values()) == pytest.approx(1.0, abs=1e-6)
    # The flag helper must fire on the synthetic collapse fixture:
    # class ``normal`` (index 1) is systematically excluded so its
    # coverage falls well below the 0.70 gap threshold.
    flagged = class_conditional_gap_flag(
        loaded.class_conditional_coverage, nominal=0.80, tolerance=0.10
    )
    assert "normal" in flagged, (
        "class_conditional_gap_flag must surface the collapsed normal "
        f"class; coverage={loaded.class_conditional_coverage}"
    )


def test_calibration_path_clean_signal_does_not_flag() -> None:
    """A well-behaved synthetic fixture (every class equally well
    served) must NOT trigger the gap flag -- guards against a
    false-positive on the rejection contract."""

    from app.evaluation.metrics import EvaluationMetrics
    from app.training.loop import _maybe_write_classification_conformal_manifest

    rng = random.Random(11)
    n = 600
    softmax: list[list[float]] = []
    truths: list[int] = []
    for _ in range(n):
        y = rng.randint(0, 2)
        truths.append(y)
        correct = rng.random() < 0.85
        row = [0.10, 0.10, 0.10]
        winner = y if correct else (y + rng.choice([1, 2])) % 3
        row[winner] = 0.80
        s = sum(row)
        softmax.append([v / s for v in row])
    metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=softmax,
        targets=truths,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        loaded = load_manifest(sidecar)

    assert loaded.class_conditional_coverage is not None
    flagged = class_conditional_gap_flag(
        loaded.class_conditional_coverage, nominal=0.80, tolerance=0.10
    )
    assert flagged == [], (
        "clean synthetic signal must not trigger the gap flag; "
        f"coverage={loaded.class_conditional_coverage}"
    )


def test_rates_calibration_preserves_class_conditional_fields() -> None:
    """The per-rates-head calibration step merges onto the sidecar
    written by ``_maybe_write_classification_conformal_manifest``;
    the merge must NOT clobber the class-conditional fields the prior
    step persisted. Regression guard against a silent drop of the
    #326 diagnostics when the dual-head path runs after
    classification."""

    from app.evaluation.metrics import EvaluationMetrics
    from app.training.loop import (
        _maybe_write_classification_conformal_manifest,
        _maybe_write_rates_conformal_manifest,
    )

    softmax, truths = _synthetic_collapse_softmax(300, seed=17)
    cls_metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=softmax,
        targets=truths,
    )
    # Build a rates metrics block with enough rows to fit the per-head
    # residual quantile but no per-head class arrays (mimicking the
    # regression-only rates surface).
    rates_metrics = {
        "2y": {
            "predictions_bps": [float(i) for i in range(20)],
            "actuals_bps": [float(i + 1) for i in range(20)],
        }
    }
    rates_eval_metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=softmax,
        targets=truths,
        rates_metrics=rates_metrics,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(cls_metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        before = load_manifest(sidecar)
        assert before.class_conditional_coverage is not None
        assert before.set_size_distribution is not None
        _maybe_write_rates_conformal_manifest(
            rates_eval_metrics, ckpt, head_names=["2y"]
        )
        after = load_manifest(sidecar)
    assert after.class_conditional_coverage == before.class_conditional_coverage
    assert after.set_size_distribution == before.set_size_distribution
    assert after.rates_residual_quantiles is not None
    assert "2y" in after.rates_residual_quantiles


def test_calibration_path_emits_diagnostic_logs(capsys) -> None:
    """The calibration helper must print one
    ``[conformal] class-conditional coverage:`` line and one
    ``[conformal] set-size distribution:`` line whenever the
    diagnostics populate, so the operator sees the per-class slice
    on the calibration breadcrumb without grepping the manifest."""

    from app.evaluation.metrics import EvaluationMetrics
    from app.training.loop import _maybe_write_classification_conformal_manifest

    softmax, truths = _synthetic_collapse_softmax(300, seed=13)
    metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=softmax,
        targets=truths,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
    captured = capsys.readouterr().out
    assert "class-conditional coverage" in captured
    assert "set-size distribution" in captured
    assert "WARNING class-conditional coverage gap" in captured
