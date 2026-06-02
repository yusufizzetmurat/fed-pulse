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
            # Non-zero seasonality coefficients so the calendar block
            # genuinely moves the forecast when a recognized dummy
            # fires. The ``calendar_adjusted`` flag now reflects whether
            # the dot-product against the recognized row is non-zero —
            # an all-zero coefficient vector or a non-matching date
            # would (correctly) leave the flag False.
            row["calendar_dummy_coef"] = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0]
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
    # Monday, day 25 → both ``dow_0`` and ``month_end`` fire so the
    # non-zero coefs in the stub spec produce a real adjustment and
    # ``calendar_adjusted`` flips true.
    out = volume_forecaster.predict_abnormal_volume(
        vol.tolist(), forecast_date="2026-05-25"
    )

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


def test_predict_abnormal_volume_zero_coefs_dont_flip_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-zero calendar coefficients on recognized names must NOT flip the chip.

    The flag follows the actual adjustment value, not the mere presence
    of recognized names — an artifact that ships zero coefficients does
    not move the forecast and the UI must not claim a calendar
    adjustment when none exists.
    """

    spec = _make_spec(with_calendar=True)
    for h in (1, 5, 22):
        spec["by_horizon"][f"h{h}"]["calendar_dummy_coef"] = [0.0] * 6

    volume_forecaster._VolumePredictor.reset()
    instance = volume_forecaster._VolumePredictor.__new__(
        volume_forecaster._VolumePredictor
    )
    instance.model_dir = volume_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = spec  # type: ignore[attr-defined]
    instance.revision = "stub-zero-cal@2026-05-30"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        volume_forecaster._VolumePredictor,
        "get",
        classmethod(lambda cls: instance),
    )
    try:
        vol = np.exp(np.linspace(20.0, 21.0, 30))
        out = volume_forecaster.predict_abnormal_volume(
            vol.tolist(), forecast_date="2026-05-25"
        )
        for row in out["horizons"]:
            assert row["calendar_adjusted"] is False
    finally:
        volume_forecaster._VolumePredictor.reset()


def test_predict_abnormal_volume_non_matching_date_doesnt_flip_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recognized names whose row evaluates to all-zero must NOT flip the chip.

    Friday baseline (no ``dow_0..dow_3`` fires) plus a mid-month date
    (no ``month_end`` / ``quarter_end``) makes every recognized dummy
    zero — the dot-product is zero even with non-zero coefficients, so
    the flag must report no adjustment.
    """

    volume_forecaster._VolumePredictor.reset()
    instance = volume_forecaster._VolumePredictor.__new__(
        volume_forecaster._VolumePredictor
    )
    instance.model_dir = volume_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = _make_spec(with_calendar=True)  # type: ignore[attr-defined]
    instance.revision = "stub-friday@2026-05-30"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        volume_forecaster._VolumePredictor,
        "get",
        classmethod(lambda cls: instance),
    )
    try:
        vol = np.exp(np.linspace(20.0, 21.0, 30))
        # 2026-05-15 is a Friday (weekday=4) and day 15 — Friday baseline
        # (no dow dummy fires) and mid-month (no month_end). All
        # recognized dummies evaluate to zero.
        out = volume_forecaster.predict_abnormal_volume(
            vol.tolist(), forecast_date="2026-05-15"
        )
        for row in out["horizons"]:
            assert row["calendar_adjusted"] is False
    finally:
        volume_forecaster._VolumePredictor.reset()


def test_predict_abnormal_volume_unknown_calendar_names_dont_flip_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future seasonality block this service can't decode must NOT flip the chip.

    Otherwise the UX would show "calendar-adjusted" while the math
    collapsed to the no-calendar branch (dot-product against a zero
    vector). Guard against that drift.
    """

    spec = _make_spec(with_calendar=False)
    for h in (1, 5, 22):
        spec["by_horizon"][f"h{h}"]["calendar_dummy_names"] = [
            "holiday_block_a",
            "holiday_block_b",
        ]
        spec["by_horizon"][f"h{h}"]["calendar_dummy_coef"] = [0.5, -0.5]

    volume_forecaster._VolumePredictor.reset()
    instance = volume_forecaster._VolumePredictor.__new__(
        volume_forecaster._VolumePredictor
    )
    instance.model_dir = volume_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    instance.spec = spec  # type: ignore[attr-defined]
    instance.revision = "stub-unknown-cal@2026-05-30"  # type: ignore[attr-defined]
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


def test_predict_abnormal_volume_missing_r2_returns_none(
    stub_predictor: Any,
) -> None:
    """r2_har is optional on the artifact; an absent value must serialize
    cleanly as None instead of NaN so FastAPI's stdlib JSON encoder does
    not 500 (Out of range float values are not JSON compliant)."""

    for h in (1, 5, 22):
        del stub_predictor.spec["by_horizon"][f"h{h}"]["r2_har"]
    vol = np.exp(np.linspace(20.0, 21.0, 30))
    out = volume_forecaster.predict_abnormal_volume(vol.tolist())
    for row in out["horizons"]:
        assert row["r2_har"] is None


def test_safe_float_handles_none_and_garbage() -> None:
    assert volume_forecaster._safe_float(None) is None
    assert volume_forecaster._safe_float("abc") is None
    assert volume_forecaster._safe_float(float("nan")) is None
    assert volume_forecaster._safe_float(1) == 1.0
    assert volume_forecaster._safe_float("3.5") == 3.5


def test_load_spec_falls_back_to_cold_start_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """HF download failure must trigger the in-process cold-start fit.

    Monkeypatches ``_download_artifact`` to raise
    ``VolumeForecasterUnavailable`` and ``_cold_start_fit`` to return a
    minimal valid spec, then asserts ``_load_spec`` returns it from the
    cold-start branch (no HF hit succeeded, no on-disk artifact existed).
    """

    def _raise(_target_dir: Any) -> dict[str, Any]:
        raise volume_forecaster.VolumeForecasterUnavailable("hf repo missing")

    fake_spec: dict[str, Any] = _make_spec()
    cold_start_calls: list[Any] = []

    def _stub_cold_start(target_dir: Any) -> dict[str, Any]:
        cold_start_calls.append(target_dir)
        return fake_spec

    monkeypatch.setattr(volume_forecaster, "_download_artifact", _raise)
    monkeypatch.setattr(volume_forecaster, "_cold_start_fit", _stub_cold_start)

    spec = volume_forecaster._load_spec(tmp_path)
    assert spec is fake_spec
    assert cold_start_calls == [tmp_path]
