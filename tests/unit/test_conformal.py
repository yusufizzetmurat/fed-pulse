from __future__ import annotations

import json
import math
import random
from pathlib import Path

from app.evaluation.conformal import (
    ConformalManifest,
    apply_conformal_bands,
    bootstrap_ci_columns,
    calibrate_split_conformal,
    empirical_coverage,
    load_manifest,
    save_manifest,
    split_conformal_quantile,
)


def test_split_conformal_quantile_handles_symmetric_residuals():
    residuals = [-3, -1, 0, 0, 1, 2, 4]
    quantile = split_conformal_quantile(residuals, alpha=0.2)
    # Sorted |residuals|: 0, 0, 1, 1, 2, 3, 4. n=7 → ceil(0.8 * 8) = 7 → 4.
    assert quantile == 4.0


def test_calibrate_and_apply_recovers_nominal_coverage():
    rng = random.Random(11)
    n = 400
    # Shift data away from zero so the lower-bound price clamp does not fire.
    truth = [100.0 + rng.gauss(0.0, 1.0) for _ in range(n)]
    preds = [t + rng.gauss(0.0, 0.5) for t in truth]
    manifest = calibrate_split_conformal(
        close_predictions=preds[:n // 2],
        close_actuals=truth[:n // 2],
        volatility_predictions=[abs(p) for p in preds[:n // 2]],
        volatility_actuals=[abs(t) for t in truth[:n // 2]],
        alpha=0.2,
    )
    assert math.isclose(manifest.nominal_coverage, 0.8)
    assert manifest.calibration_n == n // 2
    lower, upper, _, _ = apply_conformal_bands(
        close_predictions=preds[n // 2:],
        volatility_predictions=[abs(p) for p in preds[n // 2:]],
        manifest=manifest,
        horizon_scale=False,
    )
    coverage = empirical_coverage(
        predictions=preds[n // 2:],
        actuals=truth[n // 2:],
        lower=lower,
        upper=upper,
    )
    # Allow a 5-point tolerance versus the nominal 80% coverage.
    assert coverage >= 0.75


def test_save_and_load_manifest_round_trip(tmp_path: Path):
    manifest = ConformalManifest(
        alpha=0.2,
        nominal_coverage=0.8,
        residual_quantile_close=0.42,
        residual_quantile_volatility=0.013,
        calibration_n=160,
    )
    out = save_manifest(manifest, tmp_path / "forecaster_best.conformal.json")
    loaded = load_manifest(out)
    assert loaded == manifest
    # Persisted JSON omits None-valued fields.
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert "notes" not in on_disk
    assert on_disk["calibration_n"] == 160


def test_bootstrap_ci_columns_emits_ci_lo_hi():
    rows = [
        {"variant": "scalar", "samples": [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]},
        {"variant": "ablation_no_text", "samples": []},
    ]
    augmented = bootstrap_ci_columns(rows, block_size=2, n_resamples=200, seed=11)
    assert augmented[0]["ci_lo"] is not None
    assert augmented[0]["ci_hi"] is not None
    assert "samples" not in augmented[0]
    assert augmented[1]["ci_lo"] is None
    assert augmented[1]["ci_hi"] is None
