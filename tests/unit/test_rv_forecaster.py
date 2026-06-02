"""Serving contract for the QLIKE-DLq RV ensemble.

Mocks the singleton predictor against a tiny synthetic artifact so the test
exercises the dict-shape contract ``predict_rv`` must return without
loading 15 .pt files or hitting HF Hub.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.services import rv_forecaster


class _StubLinear:
    """Minimal callable that mimics a single-head torch module forward."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self, x: Any) -> Any:  # pragma: no cover - exercised via predict_rv
        return _StubTensor(np.array([[self.value]], dtype=np.float32))

    def eval(self) -> "_StubLinear":
        return self


class _StubTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_StubTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


def _make_spec() -> dict[str, Any]:
    n_feat = 11
    fmean = [0.0] * n_feat
    fstd = [1.0] * n_feat
    by_horizon: dict[str, Any] = {}
    for h in (1, 5, 22):
        by_horizon[f"h{h}"] = {
            "har_coef": [-0.5, 0.6, 0.2, 0.1],
            "feat_mean": fmean,
            "feat_std": fstd,
            "resid_mean": 0.0,
            "resid_std": 0.5,
            "conformal_quantiles": {"0.20": 0.6, "0.10": 0.9},
            "seed_state_dicts": [f"h{h}_seed{s}.pt" for s in (11, 22, 33, 44, 55)],
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
            "h1": {
                "qlike_ens": 0.197,
                "qlike_har": 0.223,
                "coverage": {"0.80": {"empirical": 0.74}, "0.90": {"empirical": 0.85}},
            },
            "h5": {
                "qlike_ens": 0.197,
                "qlike_har": 0.219,
                "coverage": {"0.80": {"empirical": 0.76}, "0.90": {"empirical": 0.88}},
            },
            "h22": {
                "qlike_ens": 0.327,
                "qlike_har": 0.360,
                "coverage": {"0.80": {"empirical": 0.80}, "0.90": {"empirical": 0.92}},
            },
        },
    }


@pytest.fixture
def stub_predictor(monkeypatch: pytest.MonkeyPatch) -> rv_forecaster._RvPredictor:
    """Inject a stub _RvPredictor so predict_rv runs without HF / torch state."""

    rv_forecaster._RvPredictor.reset()

    instance = rv_forecaster._RvPredictor.__new__(rv_forecaster._RvPredictor)
    instance.model_dir = rv_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = _make_spec()  # type: ignore[attr-defined]
    instance.eval = _make_eval()  # type: ignore[attr-defined]
    instance.seed_models = {  # type: ignore[attr-defined]
        f"h{h}": [_StubLinear(0.0) for _ in range(5)] for h in (1, 5, 22)
    }
    instance.revision = "stub@2026-05-29"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        rv_forecaster._RvPredictor, "get", classmethod(lambda cls: instance)
    )
    yield instance
    rv_forecaster._RvPredictor.reset()


def test_predict_rv_returns_three_horizons_with_bands(stub_predictor: Any) -> None:
    rng = np.random.default_rng(0)
    rv = np.abs(rng.normal(scale=1e-4, size=30)) + 1e-5
    out = rv_forecaster.predict_rv(rv.tolist())

    assert set(out.keys()) == {
        "horizons",
        "model_revision",
        "realized_features_source",
        "realized_features_date",
    }
    assert out["model_revision"] == "stub@2026-05-29"
    # No intraday_measures passed → forecaster must surface the
    # HAR-grade fallback flag with no measure date.
    assert out["realized_features_source"] == "training_means"
    assert out["realized_features_date"] is None
    horizons = out["horizons"]
    assert [row["h"] for row in horizons] == [1, 5, 22]

    for row in horizons:
        assert row["point"] > 0
        # 80% band sits inside the 90% band (q90 > q80 → wider).
        assert row["band_lo_90"] <= row["band_lo_80"] <= row["point"] <= row["band_hi_80"] <= row["band_hi_90"]
        # QLIKE diagnostics come back as finite floats from the eval sidecar.
        assert math.isfinite(row["qlike_model"])
        assert math.isfinite(row["qlike_har"])
        assert row["qlike_model"] < row["qlike_har"]
        assert 0.5 < row["coverage_empirical_90"] < 1.0


def test_predict_rv_surfaces_live_source_when_intraday_supplied(stub_predictor: Any) -> None:
    """Passing live intraday measures flips the source flag to ``live``.

    The forecast value itself need not differ much for the stub (which
    holds feat_mean ≈ realized values), but the flag is the load-bearing
    signal the dashboard uses to badge QLIKE-full vs HAR-fallback.
    """

    rng = np.random.default_rng(0)
    rv = np.abs(rng.normal(scale=1e-4, size=30)) + 1e-5
    intraday = {
        "rv": 1.2e-4,
        "rs_pos": 6e-5,
        "rs_neg": 6e-5,
        "bv": 1.0e-4,
        "rq": 8e-8,
        "rskew": -0.1,
        "rkurt": 4.2,
        "parkinson": 1.3e-4,
        "rvol": 1.5e9,
        "n_ret": 78.0,
    }
    out = rv_forecaster.predict_rv(rv.tolist(), intraday_measures=intraday)
    assert out["realized_features_source"] == "live"


def test_predict_rv_falls_back_when_intraday_measure_key_missing(stub_predictor: Any) -> None:
    """Malformed intraday dict (missing key) must degrade to training_means
    rather than raise — the forecast should still serve at HAR-grade.
    """

    rng = np.random.default_rng(1)
    rv = np.abs(rng.normal(scale=1e-4, size=30)) + 1e-5
    bad = {"rs_pos": 6e-5, "rs_neg": 6e-5}  # missing bv/rq/rskew/rkurt/parkinson/rvol
    out = rv_forecaster.predict_rv(rv.tolist(), intraday_measures=bad)
    assert out["realized_features_source"] == "training_means"


def test_predict_rv_rejects_short_history(stub_predictor: Any) -> None:
    with pytest.raises(ValueError):
        rv_forecaster.predict_rv([1e-4] * 10)


def test_predict_rv_rejects_non_positive_values(stub_predictor: Any) -> None:
    rv = [1e-4] * 30
    rv[5] = 0.0
    with pytest.raises(ValueError):
        rv_forecaster.predict_rv(rv)


def test_safe_float_handles_none_and_garbage() -> None:
    assert math.isnan(rv_forecaster._safe_float(None))
    assert math.isnan(rv_forecaster._safe_float("abc"))
    assert rv_forecaster._safe_float(1) == 1.0
    assert rv_forecaster._safe_float("3.5") == 3.5


def test_predict_rv_historical_bands_walks_forward(stub_predictor: Any) -> None:
    rng = np.random.default_rng(1)
    n = 60
    rv = np.abs(rng.normal(scale=1e-4, size=n)) + 1e-5
    dates = [f"2026-04-{i + 1:02d}" for i in range(n)]
    rows = rv_forecaster.predict_rv_historical_bands(rv.tolist(), dates)
    # 22 warmup days dropped → 60 - 22 rows ride.
    assert len(rows) == n - rv_forecaster._HISTORICAL_BANDS_WARMUP
    assert rows[0]["date"] == dates[rv_forecaster._HISTORICAL_BANDS_WARMUP]
    for i, row in enumerate(rows):
        # Band brackets the point (q80 > 0 in the stub) and the realized
        # RV is the actual rv at index 22+i.
        assert row["band_lo_80"] < row["band_hi_80"]
        assert row["band_lo_80"] > 0
        assert row["realized_rv"] == pytest.approx(
            float(rv[rv_forecaster._HISTORICAL_BANDS_WARMUP + i])
        )


def test_predict_rv_historical_bands_returns_empty_when_history_too_short(
    stub_predictor: Any,
) -> None:
    rv = [1e-4] * 10
    dates = [f"2026-04-{i + 1:02d}" for i in range(10)]
    assert rv_forecaster.predict_rv_historical_bands(rv, dates) == []


def test_predict_rv_historical_bands_rejects_length_mismatch(
    stub_predictor: Any,
) -> None:
    rv = [1e-4] * 30
    dates = [f"2026-04-{i + 1:02d}" for i in range(29)]
    with pytest.raises(ValueError):
        rv_forecaster.predict_rv_historical_bands(rv, dates)
