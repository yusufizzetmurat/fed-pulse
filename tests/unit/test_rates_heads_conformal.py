"""Per-rates-head conformal calibration + manifest extension (#292).

The conformal manifest grows two optional fields:

- ``rates_residual_quantiles`` -- map head short name -> (1 - alpha)
  absolute-residual quantile in raw bps;
- ``rates_softmax_quantiles`` -- per-head APS threshold for the
  auxiliary 3-class classifier.

These tests pin the calibration helper, the manifest round-trip
through save/load, and the merge semantics on a pre-#292 manifest
file.
"""

from __future__ import annotations

import json

import pytest

from app.evaluation.conformal import (
    ConformalManifest,
    calibrate_classification_conformal,
    calibrate_rates_regression_conformal,
    load_manifest,
    save_manifest,
)


def test_calibrate_rates_regression_returns_finite_quantile() -> None:
    preds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    actuals = [1.1, 2.5, 2.7, 4.5, 5.0, 6.6, 7.2, 8.4, 9.1, 10.5]
    q = calibrate_rates_regression_conformal(
        predictions_bps=preds, actuals_bps=actuals, alpha=0.2
    )
    assert q > 0.0
    # The (1 - alpha) absolute-residual quantile must be <= max abs
    # residual (~0.7 here, with slack for the finite-sample correction).
    assert q <= 1.5


def test_calibrate_rates_regression_misaligned_inputs_raise() -> None:
    with pytest.raises(ValueError, match="align in length"):
        calibrate_rates_regression_conformal(
            predictions_bps=[1.0, 2.0],
            actuals_bps=[1.0, 2.0, 3.0],
            alpha=0.2,
        )


def test_calibrate_rates_regression_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        calibrate_rates_regression_conformal(
            predictions_bps=[1.0, 2.0],
            actuals_bps=[1.0, 2.0],
            alpha=1.5,
        )


def test_calibrate_rates_regression_empty_after_filter_raises() -> None:
    with pytest.raises(ValueError, match="empty after filtering"):
        calibrate_rates_regression_conformal(
            predictions_bps=[float("nan"), float("inf")],
            actuals_bps=[1.0, 2.0],
            alpha=0.2,
        )


def test_manifest_round_trips_rates_residual_quantiles(tmp_path) -> None:
    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=42,
        softmax_quantile=0.65,
        rates_residual_quantiles={"2y": 4.2, "5y": 3.1, "terminal": 5.8},
        rates_softmax_quantiles={"2y": 0.7, "5y": 0.65, "terminal": 0.55},
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(manifest, sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["rates_residual_quantiles"] == {
        "2y": 4.2,
        "5y": 3.1,
        "terminal": 5.8,
    }
    assert payload["rates_softmax_quantiles"] == {
        "2y": 0.7,
        "5y": 0.65,
        "terminal": 0.55,
    }
    loaded = load_manifest(sidecar)
    assert loaded.rates_residual_quantiles == manifest.rates_residual_quantiles
    assert loaded.rates_softmax_quantiles == manifest.rates_softmax_quantiles


def test_manifest_without_rates_fields_round_trips_cleanly(tmp_path) -> None:
    """Pre-#292 manifests load cleanly with None rates fields."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.05,
        residual_quantile_volatility=0.02,
        calibration_n=10,
        softmax_quantile=0.7,
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(manifest, sidecar)
    loaded = load_manifest(sidecar)
    assert loaded.rates_residual_quantiles is None
    assert loaded.rates_softmax_quantiles is None
    assert loaded.softmax_quantile == 0.7


def test_classification_quantile_helper_still_works() -> None:
    """The pre-#292 APS helper still produces a finite quantile."""

    scores = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
    ]
    truth = [0, 1, 2, 0, 1]
    q = calibrate_classification_conformal(
        softmax_scores=scores, true_classes=truth, alpha=0.2
    )
    assert 0.0 <= q <= 1.0


def test_rates_conformal_uses_only_val_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_maybe_write_rates_conformal_manifest`` reads only the val rates_metrics block (#317 finding #21).

    Spies on both per-head calibrators and asserts the predictions /
    softmax rows came from the val ``EvaluationMetrics`` instance.
    """

    from pathlib import Path
    from app.evaluation.metrics import EvaluationMetrics
    from app.training import loop as loop_mod

    captured: dict[str, list] = {"regression_preds": [], "softmax_rows": []}

    def _spy_regression(*, predictions_bps, actuals_bps, alpha=0.2):
        captured["regression_preds"].append(list(predictions_bps))
        return 0.42

    def _spy_classification(*, softmax_scores, true_classes, alpha=0.2):
        captured["softmax_rows"].append(list(softmax_scores))
        return 0.33

    # The helper does a local ``from app.evaluation.conformal import
    # (...)`` inside its body, so patch the conformal module directly
    # rather than the loop module's namespace.
    monkeypatch.setattr(
        "app.evaluation.conformal.calibrate_rates_regression_conformal",
        _spy_regression,
    )
    monkeypatch.setattr(
        "app.evaluation.conformal.calibrate_classification_conformal",
        _spy_classification,
    )

    val_rates_metrics = {
        "2y": {
            "predictions_bps": [10.0, 11.0, 12.0],
            "actuals_bps": [10.5, 10.8, 12.2],
            "n_rows": 3,
            "cls_softmax_scores": [
                [0.7, 0.2, 0.1],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5],
            ],
            "cls_true_classes": [0, 1, 2],
            "cls_mask": [True, True, True],
        }
    }
    val_metrics = EvaluationMetrics(
        loss=0.0,
        close_rmse=0.0,
        volatility_rmse=0.0,
        combined_rmse=0.0,
        rates_metrics=val_rates_metrics,
    )

    tmp_target = Path("/tmp/_rates_conformal_val_only.pt")
    sidecar = tmp_target.with_suffix(".conformal.json")
    if sidecar.exists():
        sidecar.unlink()
    loop_mod._maybe_write_rates_conformal_manifest(
        val_metrics, tmp_target, head_names=("2y",)
    )
    # The regression-side spy must have seen the val predictions
    # (not e.g. their negation, not e.g. a train partition).
    assert captured["regression_preds"] == [[10.0, 11.0, 12.0]]
    # The classification spy must have received the val softmax rows.
    assert captured["softmax_rows"] == [
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ]
    ]
    if sidecar.exists():
        sidecar.unlink()
