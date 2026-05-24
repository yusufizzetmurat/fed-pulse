"""Cover the classification half of the conformal wrapper (#216).

The regression half lives in ``test_conformal.py`` and is untouched.
The new helpers (``calibrate_classification_conformal``,
``predict_conformal_set``, ``empirical_classification_coverage``,
``format_class_set_label``) get focused unit coverage here:
correctness on synthetic 3-class data, the argmax fallback contract,
input-validation, and the manifest round-trip with the new field.
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
    DEFAULT_CLASSIFICATION_ALPHA,
    calibrate_classification_conformal,
    empirical_classification_coverage,
    format_class_set_label,
    load_manifest,
    predict_conformal_set,
    save_manifest,
)


def _synthetic_softmax_set(n: int, *, seed: int = 0, accuracy: float = 0.85):
    """Generate ``n`` 3-class softmax rows where the model is right
    ``accuracy`` of the time. Used to verify the calibration recovers
    the nominal coverage on a clean signal."""

    rng = random.Random(seed)
    softmax: list[list[float]] = []
    true_classes: list[int] = []
    for _ in range(n):
        y = rng.randint(0, 2)
        true_classes.append(y)
        correct = rng.random() < accuracy
        row = [0.10, 0.10, 0.10]
        winner = y if correct else (y + rng.choice([1, 2])) % 3
        row[winner] = 0.80
        # normalise to a valid distribution
        s = sum(row)
        softmax.append([v / s for v in row])
    return softmax, true_classes


def test_calibrate_classification_recovers_nominal_coverage() -> None:
    """Synthetic 3-class softmax with known accuracy → calibrated
    threshold + APS prediction set gives empirical coverage ≥ 0.75
    at nominal 0.80 (mirrors the regression-side coverage tolerance
    in ``test_conformal.py``)."""

    softmax, true_classes = _synthetic_softmax_set(800, accuracy=0.85)
    cal_scores, cal_true = softmax[:400], true_classes[:400]
    test_scores, test_true = softmax[400:], true_classes[400:]
    threshold = calibrate_classification_conformal(
        softmax_scores=cal_scores, true_classes=cal_true, alpha=0.2
    )
    sets = [predict_conformal_set(row, threshold) for row in test_scores]
    coverage = empirical_classification_coverage(sets, test_true)
    assert coverage >= 0.75


def test_predict_conformal_set_includes_argmax_on_singleton() -> None:
    """A confident softmax row should yield a singleton set whose only
    element is the argmax class."""

    threshold = 0.5
    out = predict_conformal_set([0.9, 0.05, 0.05], threshold)
    assert out == [0]


def test_predict_conformal_set_falls_back_to_argmax_on_empty() -> None:
    """A pathological row where no class clears the threshold falls
    back to {argmax} rather than the mathematically-valid empty set.
    The empty-set case is useless as a decision-support surface and
    keeps marginal coverage asymptotically valid."""

    out = predict_conformal_set([0.34, 0.33, 0.33], threshold=0.01)
    assert out == [0]
    assert out, "predict_conformal_set must never return an empty list"


def test_calibrate_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        calibrate_classification_conformal(
            softmax_scores=[], true_classes=[], alpha=0.2
        )


def test_calibrate_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        calibrate_classification_conformal(
            softmax_scores=[[0.5, 0.3, 0.2]], true_classes=[0, 1], alpha=0.2
        )


def test_empirical_classification_coverage_rounds_to_one_when_all_included() -> None:
    sets = [[0, 1, 2]] * 50
    true_classes = [0] * 25 + [1] * 15 + [2] * 10
    assert empirical_classification_coverage(sets, true_classes) == 1.0


def test_format_class_set_label_renders_braced_string() -> None:
    labels = ("calm", "normal", "high")
    assert format_class_set_label([0, 2], labels) == "{calm, high}"
    assert format_class_set_label([], labels) == "{}"


def test_format_class_set_label_handles_unknown_index() -> None:
    """A stale manifest indexing past the label tuple should fall
    through as ``?`` so the response still serialises rather than
    raising in the JSON encoder."""

    labels = ("calm", "normal", "high")
    assert format_class_set_label([0, 7], labels) == "{calm, ?}"


def test_manifest_round_trip_preserves_softmax_quantile() -> None:
    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=10.5,
        residual_quantile_volatility=0.015,
        calibration_n=250,
        softmax_quantile=0.42,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.conformal.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
    assert loaded.softmax_quantile == pytest.approx(0.42)
    assert loaded.calibration_n == 250


def test_manifest_round_trip_loads_pre_216_manifests_with_none() -> None:
    """Pre-#216 manifests on disk have no ``softmax_quantile`` field.
    ``load_manifest`` must read them cleanly with the field at None
    so existing deployments do not break on upgrade."""

    legacy_payload = {
        "alpha": 0.2,
        "nominal_coverage": 0.8,
        "residual_quantile_close": 11.0,
        "residual_quantile_volatility": 0.02,
        "calibration_n": 300,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "legacy.conformal.json"
        path.write_text(json.dumps(legacy_payload), encoding="utf-8")
        loaded = load_manifest(path)
    assert loaded.softmax_quantile is None
    assert loaded.calibration_n == 300


def test_default_classification_alpha_matches_regression_default() -> None:
    """Both heads should default to 0.80 nominal coverage so the
    /analyze response carries a consistent confidence story across
    regression bands and classification sets."""

    assert DEFAULT_CLASSIFICATION_ALPHA == pytest.approx(0.2)


def test_save_manifest_omits_none_fields_for_back_compat() -> None:
    """Regression-only manifests written without ``softmax_quantile``
    must serialise without the key so downstream readers expecting
    the legacy schema do not key-error on the new optional field."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=11.0,
        residual_quantile_volatility=0.02,
        calibration_n=300,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "regression-only.conformal.json"
        save_manifest(manifest, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert "softmax_quantile" not in payload
    assert math.isclose(payload["nominal_coverage"], 0.8)
