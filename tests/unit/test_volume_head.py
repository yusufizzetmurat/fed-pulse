"""Unit tests for the volume-head serving loader (no network)."""

from __future__ import annotations

import math

import pytest

from app.services import volume_head as vh


def _fake_spec() -> dict:
    # identity standardization, coef picks lag1 only, +5 intercept -> easy to check
    h = {"feat_mean": [0.0, 0.0, 0.0], "feat_std": [1.0, 1.0, 1.0], "coef": [1.0, 0.0, 0.0], "intercept": 5.0}
    return {"model": "volume-head-HAR", "by_horizon": {"h1": h, "h5": h, "h22": h}}


def test_har_features_lag1_mean5_mean22():
    lv = [float(i) for i in range(22)]  # 0..21
    feats = vh._har_features(lv)
    assert feats[0] == 21.0  # lag1 = last
    assert feats[1] == pytest.approx(sum(range(17, 22)) / 5)  # mean last 5
    assert feats[2] == pytest.approx(sum(range(22)) / 22)  # mean last 22


def test_har_features_requires_22():
    with pytest.raises(ValueError, match=">=22"):
        vh._har_features([1.0] * 10)


def test_predict_volume_applies_coefficients(monkeypatch):
    monkeypatch.setattr(vh, "_spec_cache", _fake_spec())  # skip load
    lv = [0.0] * 21 + [10.0]  # lag1 = 10
    out = vh.predict_volume(lv, symbol="^GSPC")
    h1 = next(h for h in out.horizons if h.horizon_days == 1)
    # pred = lag1*1 + 0 + 0 + 5 = 15
    assert h1.expected_log_volume == pytest.approx(15.0)
    assert h1.expected_volume == pytest.approx(math.exp(15.0), rel=1e-6)
    assert {h.horizon_days for h in out.horizons} == {1, 5, 22}


def test_load_spec_pulls_from_hf_when_no_local(monkeypatch, tmp_path):
    import json

    import huggingface_hub

    monkeypatch.setattr(vh, "_spec_cache", None)
    monkeypatch.setattr(vh, "_LOCAL_ARTIFACT", tmp_path / "absent.json")
    art = tmp_path / "hf.json"
    art.write_text(json.dumps(_fake_spec()))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: str(art))
    spec = vh._load_spec()
    assert "by_horizon" in spec and "h1" in spec["by_horizon"]


def test_load_spec_raises_unavailable_on_hf_failure(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(vh, "_spec_cache", None)
    monkeypatch.setattr(vh, "_LOCAL_ARTIFACT", tmp_path / "absent.json")

    def _boom(**k):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
    with pytest.raises(vh.VolumeForecasterUnavailable, match="could not fetch"):
        vh._load_spec()
