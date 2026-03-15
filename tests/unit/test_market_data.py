from __future__ import annotations

import pytest
pd = pytest.importorskip("pandas")
pytest.importorskip("yfinance")

from app.services import market_data


def _series(days: list[str], values: list[float]) -> pd.Series:
    index = pd.to_datetime([f"{day} 00:00:00" for day in days])
    return pd.Series(values, index=index)


def test_fetch_market_snapshot_uses_lookback(monkeypatch):
    # requested date is Sunday, nearest valid trading day is Friday
    close_series = _series(
        ["2026-03-12", "2026-03-13"],
        [101.0, 102.0],
    )

    def fake_download(**kwargs):
        return close_series

    monkeypatch.setattr(market_data, "_download_close_series", fake_download)
    out = market_data.fetch_market_snapshot(target_date="2026-03-15", symbol="^GSPC", lookback_days=7)
    assert out["date_used"] == "2026-03-13"
    assert out["close"] == 102.0
    assert out["volatility_5d"] >= 0.0


def test_fetch_market_snapshot_invalid_date():
    with pytest.raises(ValueError):
        market_data.fetch_market_snapshot(target_date="15-03-2026")


def test_fetch_realized_forward_returns_steps(monkeypatch):
    close_series = _series(
        ["2026-03-13", "2026-03-16", "2026-03-17", "2026-03-18"],
        [100.0, 102.0, 103.0, 104.0],
    )

    def fake_window(**kwargs):
        return close_series

    monkeypatch.setattr(market_data, "_download_close_series_in_window", fake_window)
    out = market_data.fetch_realized_forward(target_date="2026-03-13", symbol="^GSPC", steps=2)
    assert len(out) == 2
    assert out[0]["date"] == "2026-03-16"
