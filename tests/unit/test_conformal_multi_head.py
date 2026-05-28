"""Conformal manifest schema + per-head calibration round-trip (#292 acceptance).

The conformal manifest carries one quantile per classification head and
one interval per regression head. Per the issue contract:

- ``rates_residual_quantiles`` maps head short name to the (1 - alpha)
  absolute-residual quantile in raw bps (per-regression-head conformal
  interval);
- ``rates_softmax_quantiles`` maps the same short name to the APS
  threshold (per-classification-head quantile);
- pre-existing single-head manifests load cleanly under the extended
  schema (backwards-compatible read).

These tests pin those guarantees end-to-end against the on-disk JSON.
"""

from __future__ import annotations

import json
import math

import pytest

from app.evaluation.conformal import (
    ConformalManifest,
    calibrate_classification_conformal,
    calibrate_rates_regression_conformal,
    load_manifest,
    save_manifest,
    split_conformal_quantile,
)


# ---------------------------------------------------------------------------
# Per-head calibration math


def test_per_head_regression_quantile_is_independent_across_heads() -> None:
    """Each rates head fits its own quantile off its own residuals.

    The 2y residuals are tight (~0.1 bps) while the terminal residuals
    are wide (~2.0 bps); the calibrator must return two different
    quantiles, not a shared one.
    """

    preds_2y = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    actuals_2y = [10.0, 11.1, 12.0, 13.1, 14.0, 15.1, 16.0, 17.1, 18.0, 19.1]
    preds_terminal = [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0]
    actuals_terminal = [50.0, 54.0, 54.0, 58.0, 58.0, 62.0, 62.0, 66.0, 66.0, 70.0]
    q_2y = calibrate_rates_regression_conformal(
        predictions_bps=preds_2y, actuals_bps=actuals_2y, alpha=0.2
    )
    q_terminal = calibrate_rates_regression_conformal(
        predictions_bps=preds_terminal, actuals_bps=actuals_terminal, alpha=0.2
    )
    assert q_2y > 0.0
    assert q_terminal > 0.0
    # Terminal residuals are an order of magnitude wider; the quantile
    # must reflect that, not collapse onto the 2y level.
    assert q_terminal > 5.0 * q_2y


def test_per_head_aps_quantile_is_independent_across_heads() -> None:
    """The aux classifier APS threshold differs per head when softmax
    distributions differ — the calibrator does not share state."""

    # Confident on 2y (true class always carries ~0.8 mass).
    confident_scores = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
    ]
    # Spread on terminal (true class carries ~0.4 mass).
    spread_scores = [
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4],
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
    ]
    truth = [0, 1, 2, 0, 1]
    q_confident = calibrate_classification_conformal(
        softmax_scores=confident_scores, true_classes=truth, alpha=0.2
    )
    q_spread = calibrate_classification_conformal(
        softmax_scores=spread_scores, true_classes=truth, alpha=0.2
    )
    # 1 - 0.8 = 0.2 vs 1 - 0.4 = 0.6; the spread distribution lands a
    # larger APS threshold so the prediction set is wider, as expected.
    assert q_spread > q_confident


# ---------------------------------------------------------------------------
# Manifest schema round-trip


def test_three_head_manifest_round_trips_through_disk(tmp_path) -> None:
    """A manifest with regime + 2y + terminal entries round-trips clean."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=42,
        softmax_quantile=0.65,
        rates_residual_quantiles={"2y": 4.2, "terminal": 12.5},
        rates_softmax_quantiles={"2y": 0.7, "terminal": 0.55},
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(manifest, sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    # The two extension keys ride on disk verbatim.
    assert payload["rates_residual_quantiles"] == {"2y": 4.2, "terminal": 12.5}
    assert payload["rates_softmax_quantiles"] == {"2y": 0.7, "terminal": 0.55}
    loaded = load_manifest(sidecar)
    assert loaded.rates_residual_quantiles == manifest.rates_residual_quantiles
    assert loaded.rates_softmax_quantiles == manifest.rates_softmax_quantiles
    # The pre-#292 regime quantile (softmax_quantile for the vol-regime
    # head) survives the round-trip too.
    assert loaded.softmax_quantile == pytest.approx(0.65)


def test_regression_only_per_head_manifest_drops_softmax_block(tmp_path) -> None:
    """Aux-classifier OFF runs emit no per-head softmax quantiles."""

    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=20,
        rates_residual_quantiles={"2y": 4.2, "terminal": 12.5},
        # rates_softmax_quantiles intentionally absent.
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(manifest, sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "rates_residual_quantiles" in payload
    assert "rates_softmax_quantiles" not in payload
    loaded = load_manifest(sidecar)
    assert loaded.rates_softmax_quantiles is None


# ---------------------------------------------------------------------------
# Backwards-compatible read on a pre-#292 manifest


def test_pre_292_single_head_manifest_loads_cleanly(tmp_path) -> None:
    """A manifest written before #292 (regime quantile only, no rates
    block) must load without error and surface ``None`` rates fields."""

    pre_292_payload = {
        "alpha": 0.2,
        "nominal_coverage": 0.8,
        "calibration_n": 50,
        "softmax_quantile": 0.68,
    }
    sidecar = tmp_path / "forecaster_best.conformal.json"
    sidecar.write_text(json.dumps(pre_292_payload, indent=2), encoding="utf-8")
    loaded = load_manifest(sidecar)
    assert loaded.softmax_quantile == pytest.approx(0.68)
    assert loaded.rates_residual_quantiles is None
    assert loaded.rates_softmax_quantiles is None
    # The legacy residual fields default to 0.0 on a classification-only
    # manifest -- the inference path reads that as "no regression bands"
    # and falls back to gaussian-z, which is the documented contract.
    assert loaded.residual_quantile_close == 0.0
    assert loaded.residual_quantile_volatility == 0.0


def test_pre_292_full_manifest_loads_cleanly(tmp_path) -> None:
    """A pre-#292 manifest carrying the (close, volatility) residual
    quantiles still loads under the extended schema with no information
    loss."""

    pre_292_payload = {
        "alpha": 0.2,
        "nominal_coverage": 0.8,
        "residual_quantile_close": 0.05,
        "residual_quantile_volatility": 0.02,
        "calibration_n": 30,
        "softmax_quantile": 0.7,
    }
    sidecar = tmp_path / "forecaster_best.conformal.json"
    sidecar.write_text(json.dumps(pre_292_payload, indent=2), encoding="utf-8")
    loaded = load_manifest(sidecar)
    assert loaded.residual_quantile_close == pytest.approx(0.05)
    assert loaded.residual_quantile_volatility == pytest.approx(0.02)
    assert loaded.softmax_quantile == pytest.approx(0.7)
    assert loaded.rates_residual_quantiles is None
    assert loaded.rates_softmax_quantiles is None


def test_post_292_manifest_load_then_resave_preserves_rates_block(
    tmp_path,
) -> None:
    """Round-trip: post-#292 manifest -> load -> save -> load again."""

    original = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=42,
        softmax_quantile=0.65,
        rates_residual_quantiles={"2y": 4.2, "5y": 7.1, "terminal": 12.5},
        rates_softmax_quantiles={"2y": 0.7, "5y": 0.65, "terminal": 0.55},
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(original, sidecar)
    loaded = load_manifest(sidecar)
    save_manifest(loaded, sidecar)
    twice_loaded = load_manifest(sidecar)
    assert twice_loaded.rates_residual_quantiles == original.rates_residual_quantiles
    assert twice_loaded.rates_softmax_quantiles == original.rates_softmax_quantiles


# ---------------------------------------------------------------------------
# Manifest growth ordering (each new head is purely additive)


def test_adding_head_to_manifest_does_not_disturb_existing_heads(
    tmp_path,
) -> None:
    """A manifest extended in place adds new head keys without rewriting
    the prior ones."""

    base = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.0,
        residual_quantile_volatility=0.0,
        calibration_n=20,
        rates_residual_quantiles={"2y": 4.2},
        rates_softmax_quantiles={"2y": 0.7},
    )
    sidecar = tmp_path / "forecaster_best.conformal.json"
    save_manifest(base, sidecar)
    loaded = load_manifest(sidecar)
    # Operator adds the terminal head -- existing 2y entries persist.
    extended_residuals = dict(loaded.rates_residual_quantiles or {})
    extended_residuals["terminal"] = 12.5
    extended_softmax = dict(loaded.rates_softmax_quantiles or {})
    extended_softmax["terminal"] = 0.55
    extended = ConformalManifest(
        alpha=loaded.alpha,
        nominal_coverage=loaded.nominal_coverage,
        residual_quantile_close=loaded.residual_quantile_close,
        residual_quantile_volatility=loaded.residual_quantile_volatility,
        calibration_n=loaded.calibration_n,
        rates_residual_quantiles=extended_residuals,
        rates_softmax_quantiles=extended_softmax,
    )
    save_manifest(extended, sidecar)
    final = load_manifest(sidecar)
    assert final.rates_residual_quantiles == {"2y": 4.2, "terminal": 12.5}
    assert final.rates_softmax_quantiles == {"2y": 0.7, "terminal": 0.55}


# ---------------------------------------------------------------------------
# Sanity: the regression helper still matches a hand-computed quantile


def test_split_conformal_quantile_finite_sample_correction() -> None:
    """``ceil((1 - alpha) * (n + 1))`` rank pick, applied to the
    absolute-residual ladder. Pins the calibration math used by every
    per-head fit."""

    residuals = [-0.5, 0.3, -0.7, 0.1, 0.4]  # abs sort: 0.1, 0.3, 0.4, 0.5, 0.7
    q = split_conformal_quantile(residuals, alpha=0.2)
    # n=5; rank = ceil(0.8 * 6) = 5; the 5th sorted abs residual = 0.7.
    assert math.isclose(q, 0.7, abs_tol=1e-12)
