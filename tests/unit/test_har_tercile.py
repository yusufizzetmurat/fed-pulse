"""HAR-tercile serving baseline contract.

Mocks the cached production spec so the test exercises the bucket / soft-prob
contract without loading torch weights or hitting HF Hub. Verifies the
wiki-section-20 macro-F1 numbers are wired through unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.services import har_tercile, rv_forecaster


def _make_spec() -> dict[str, Any]:
    """Spec with HAR coefs that are easy to reason about under test."""

    by_horizon: dict[str, Any] = {}
    for h in (1, 5, 22):
        by_horizon[f"h{h}"] = {
            # log_pred ~ daily lag (intercept 0, daily=1.0, weekly=0, monthly=0).
            # Drives the point forecast directly off the last RV value.
            "har_coef": [0.0, 1.0, 0.0, 0.0],
            "feat_mean": [0.0] * 11,
            "feat_std": [1.0] * 11,
            "resid_mean": 0.0,
            "resid_std": 0.5,
            "conformal_quantiles": {"0.20": 0.6, "0.10": 0.9},
            "seed_state_dicts": [],
            "n_oos_resid": 100,
        }
    return {
        "model": "intraday_rv_production",
        "feature_order": [
            "har_daily", "har_weekly", "har_monthly",
            "rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson", "log_rvol",
        ],
        "date_last": "2026-05-29",
        "by_horizon": by_horizon,
    }


def _make_eval() -> dict[str, Any]:
    return {
        "by_horizon": {
            "h1": {"qlike_ens": 0.197, "qlike_har": 0.223},
            "h5": {"qlike_ens": 0.197, "qlike_har": 0.219},
            "h22": {"qlike_ens": 0.327, "qlike_har": 0.360},
        },
    }


@pytest.fixture
def stub_predictor(monkeypatch: pytest.MonkeyPatch) -> rv_forecaster._RvPredictor:
    """Inject a stub _RvPredictor so har_tercile runs without HF / torch state."""

    rv_forecaster._RvPredictor.reset()
    instance = rv_forecaster._RvPredictor.__new__(rv_forecaster._RvPredictor)
    instance.model_dir = rv_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = _make_spec()  # type: ignore[attr-defined]
    instance.eval = _make_eval()  # type: ignore[attr-defined]
    instance.seed_models = {}  # type: ignore[attr-defined]
    instance.revision = "stub@2026-05-29"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        rv_forecaster._RvPredictor, "get", classmethod(lambda cls: instance)
    )
    yield instance
    rv_forecaster._RvPredictor.reset()


def test_predict_har_regime_returns_three_horizons(stub_predictor: Any) -> None:
    rng = np.random.default_rng(0)
    rv = np.abs(rng.normal(scale=1e-4, size=40)) + 1e-6
    out = har_tercile.predict_har_regime(rv.tolist())

    assert set(out.keys()) >= {"horizons", "cutoffs_q33", "cutoffs_q67", "model_revision"}
    assert out["model_revision"] == "stub@2026-05-29"
    assert out["cutoffs_q33"] <= out["cutoffs_q67"]

    horizons = out["horizons"]
    assert [row["h"] for row in horizons] == [1, 5, 22]
    for row in horizons:
        assert row["predicted_rv"] > 0.0
        assert row["tercile"] in {"low", "medium", "high"}
        probs = row["tercile_probs"]
        # Soft probabilities sum to 1.0 and are non-negative.
        assert set(probs.keys()) == {"low", "medium", "high"}
        assert all(p >= 0.0 for p in probs.values())
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_predict_har_regime_macro_f1_wired_through(stub_predictor: Any) -> None:
    """The macro-F1 triple from the pooled walk-forward eval must flow through."""

    rng = np.random.default_rng(1)
    rv = np.abs(rng.normal(scale=1e-4, size=40)) + 1e-6
    out = har_tercile.predict_har_regime(rv.tolist())
    by_h = {row["h"]: row for row in out["horizons"]}
    assert by_h[1]["macro_f1"] == pytest.approx(0.687)
    assert by_h[5]["macro_f1"] == pytest.approx(0.685)
    assert by_h[22]["macro_f1"] == pytest.approx(0.654)
    # Source attribution must describe the methodology in plain language
    # without any wiki citation; frontend renders it as the chip tooltip.
    src = by_h[1]["macro_f1_source"].lower()
    assert "walk-forward" in src and "wiki" not in src


def test_predict_har_regime_bucket_matches_manual_digitize(stub_predictor: Any) -> None:
    """Argmax bucket must match a manual q33 / q67 bucketing of the HAR point.

    The stub spec sets ``har_coef = [0, 1, 0, 0]`` so the log-RV point is
    exactly the last log-RV value; predicted_rv ≈ rv[-1]. We verify the
    returned ``tercile`` matches a manual digitize of that point against
    the q33 / q67 cutoffs the function returns.
    """

    rv = np.linspace(1e-5, 5e-4, 40).tolist()
    out = har_tercile.predict_har_regime(rv)
    q33, q67 = out["cutoffs_q33"], out["cutoffs_q67"]
    last = rv[-1]
    expected = "low" if last < q33 else ("medium" if last < q67 else "high")
    h1 = next(row for row in out["horizons"] if row["h"] == 1)
    assert h1["tercile"] == expected
    # And the highest soft-probability mass lands on the argmax bucket.
    probs = h1["tercile_probs"]
    assert max(probs, key=probs.get) == expected


def test_predict_har_regime_rejects_short_history(stub_predictor: Any) -> None:
    with pytest.raises(ValueError):
        har_tercile.predict_har_regime([1e-4] * 10)


def test_predict_har_regime_rejects_non_positive_values(stub_predictor: Any) -> None:
    rv = [1e-4] * 30
    rv[3] = 0.0
    with pytest.raises(ValueError):
        har_tercile.predict_har_regime(rv)


def test_predict_har_regime_honours_supplied_cutoffs(stub_predictor: Any) -> None:
    """When the caller pins q33 / q67, the bucketing uses those — not the series."""

    rv = np.linspace(1e-5, 5e-4, 40).tolist()
    # Pin cutoffs so the last RV value (~5e-4) sits in the high bucket.
    out = har_tercile.predict_har_regime(rv, cutoffs_q33=1e-5, cutoffs_q67=2e-5)
    h1 = next(row for row in out["horizons"] if row["h"] == 1)
    assert h1["tercile"] == "high"
    assert out["cutoffs_q33"] == pytest.approx(1e-5)
    assert out["cutoffs_q67"] == pytest.approx(2e-5)


def test_predict_har_regime_rejects_inverted_cutoffs(stub_predictor: Any) -> None:
    rv = [1e-4] * 30
    with pytest.raises(ValueError):
        har_tercile.predict_har_regime(rv, cutoffs_q33=2e-4, cutoffs_q67=1e-4)


def test_safe_float_handles_none_and_garbage() -> None:
    assert har_tercile._safe_float(None) is None
    assert har_tercile._safe_float("abc") is None
    assert har_tercile._safe_float(float("nan")) is None
    assert har_tercile._safe_float(1) == 1.0
    assert har_tercile._safe_float("3.5") == 3.5
