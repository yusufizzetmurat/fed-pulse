"""Serving contract for the HAR-based expected-volume forecaster.

Mocks the singleton predictor against a tiny synthetic artifact so the
test exercises the dict-shape contract ``predict_abnormal_volume``
returns without hitting HF Hub. Mirrors :mod:`test_rv_forecaster` and
:mod:`test_har_tercile`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from app.services import volume_forecaster


def _make_spec(*, with_calendar: bool = True) -> dict[str, Any]:
    """Spec with HAR coefs that are easy to reason about under test.

    ``har_coef = [0, 1, 0, 0]`` makes log_pred = lag1, so the residual
    against the 22-day mean is a direct function of the supplied series.
    """

    by_horizon: dict[str, Any] = {}
    for h in (1, 5, 22):
        row: dict[str, Any] = {
            "har_coef": [0.0, 1.0, 0.0, 0.0],
            "conformal_quantiles": {"0.20": 0.10, "0.10": 0.15},
            "r2_har": 0.82 - h * 0.01,
        }
        if with_calendar:
            row["calendar_dummy_names"] = [
                "dow_0",
                "dow_1",
                "dow_2",
                "dow_3",
                "month_end",
                "quarter_end",
            ]
            # All-zero seasonality coefficients keep the calendar layer
            # active (so the flag flips true) without disturbing the
            # point forecast — the back-transform math stays exact.
            row["calendar_dummy_coef"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        by_horizon[f"h{h}"] = row
    return {
        "model": "volume_har",
        "date_last": "2026-05-30",
        "by_horizon": by_horizon,
    }


@pytest.fixture
def stub_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> volume_forecaster._VolumePredictor:
    """Inject a stub _VolumePredictor so the service runs without HF state."""

    volume_forecaster._VolumePredictor.reset()
    instance = volume_forecaster._VolumePredictor.__new__(
        volume_forecaster._VolumePredictor
    )
    instance.model_dir = volume_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = _make_spec()  # type: ignore[attr-defined]
    instance.revision = "stub@2026-05-30"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        volume_forecaster._VolumePredictor,
        "get",
        classmethod(lambda cls: instance),
    )
    yield instance
    volume_forecaster._VolumePredictor.reset()


def test_predict_abnormal_volume_returns_three_horizons(stub_predictor: Any) -> None:
    rng = np.random.default_rng(0)
    # Generate roughly steady daily volumes around 1e9 shares.
    vol = np.exp(rng.normal(loc=21.0, scale=0.05, size=30))
    out = volume_forecaster.predict_abnormal_volume(vol.tolist())

    assert set(out.keys()) >= {
        "symbol",
        "horizons",
        "model_revision",
        "generated_at",
    }
    assert out["model_revision"] == "stub@2026-05-30"
    horizons = out["horizons"]
    assert [row["h"] for row in horizons] == [1, 5, 22]

    for row in horizons:
        # 80% band sits inside the 90% band on both sides.
        assert row["band_lo_90"] <= row["band_lo_80"] <= row["point_pct_vs_baseline"]
        assert row["point_pct_vs_baseline"] <= row["band_hi_80"] <= row["band_hi_90"]
        # R^2 wired through from the spec.
        assert math.isfinite(row["r2_har"])
        assert row["calendar_adjusted"] is True


def test_predict_abnormal_volume_back_transform_matches_log_residual(
    stub_predictor: Any,
) -> None:
    """``point_pct_vs_baseline`` must equal ``(exp(log_residual)-1)*100``."""

    rng = np.random.default_rng(1)
    vol = np.exp(rng.normal(loc=21.0, scale=0.05, size=30))
    out = volume_forecaster.predict_abnormal_volume(vol.tolist())
    for row in out["horizons"]:
        expected_pct = (math.exp(row["point_log_residual"]) - 1.0) * 100.0
        assert row["point_pct_vs_baseline"] == pytest.approx(expected_pct, rel=1e-9)


def test_predict_abnormal_volume_skips_calendar_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No seasonality block on the artifact → ``calendar_adjusted=False``."""

    volume_forecaster._VolumePredictor.reset()
    instance = volume_forecaster._VolumePredictor.__new__(
        volume_forecaster._VolumePredictor
    )
    instance.model_dir = volume_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = _make_spec(with_calendar=False)  # type: ignore[attr-defined]
    instance.revision = "stub-nocal@2026-05-30"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        volume_forecaster._VolumePredictor,
        "get",
        classmethod(lambda cls: instance),
    )
    try:
        vol = np.exp(np.linspace(20.0, 21.0, 30))
        out = volume_forecaster.predict_abnormal_volume(vol.tolist())
        for row in out["horizons"]:
            assert row["calendar_adjusted"] is False
    finally:
        volume_forecaster._VolumePredictor.reset()


def test_predict_abnormal_volume_rejects_short_history(stub_predictor: Any) -> None:
    with pytest.raises(ValueError):
        volume_forecaster.predict_abnormal_volume([1e9] * 10)


def test_predict_abnormal_volume_rejects_non_positive_values(
    stub_predictor: Any,
) -> None:
    vol = [1e9] * 30
    vol[5] = 0.0
    with pytest.raises(ValueError):
        volume_forecaster.predict_abnormal_volume(vol)


def test_predict_abnormal_volume_passes_symbol_through(stub_predictor: Any) -> None:
    vol = np.exp(np.linspace(20.0, 21.0, 30))
    out = volume_forecaster.predict_abnormal_volume(vol.tolist(), symbol="^NDX")
    assert out["symbol"] == "^NDX"


def test_safe_float_handles_none_and_garbage() -> None:
    assert volume_forecaster._safe_float(None) is None
    assert volume_forecaster._safe_float("abc") is None
    assert volume_forecaster._safe_float(float("nan")) is None
    assert volume_forecaster._safe_float(1) == 1.0
    assert volume_forecaster._safe_float("3.5") == 3.5
