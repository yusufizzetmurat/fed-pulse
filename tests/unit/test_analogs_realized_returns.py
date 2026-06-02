"""#299: realized 5d/20d S&P close-to-close returns on analog cards."""

from __future__ import annotations

import pytest

from app.retrieval.index import AnalogHit
from app.services import analogs as analogs_service


def _hit(event_date: str = "2022-09-21") -> AnalogHit:
    return AnalogHit(
        event_date=event_date,
        text_hash="deadbeef",
        similarity=0.9,
        axis_stance="hawkish",
        subsequent_vol_regime="high",
        excerpt="Fed signals further tightening",
    )


def test_subsequent_close_pct_attached_to_each_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """The renderer pulls realized S&P returns and stamps both 5d and 20d on every card."""

    captured: list[tuple[str, int]] = []

    def _fake(event_date: str, *, horizon: int) -> float | None:
        captured.append((event_date, horizon))
        return 2.5 if horizon == 5 else -1.0

    monkeypatch.setattr(analogs_service, "_subsequent_close_pct", _fake)

    cards = analogs_service.render_analog_cards([_hit("2022-09-21"), _hit("2020-03-15")])

    assert all("subsequent_close_pct_5d" in c for c in cards)
    assert all("subsequent_close_pct_20d" in c for c in cards)
    assert cards[0]["subsequent_close_pct_5d"] == 2.5
    assert cards[0]["subsequent_close_pct_20d"] == -1.0
    assert set(captured) == {
        ("2022-09-21", 5),
        ("2022-09-21", 20),
        ("2020-03-15", 5),
        ("2020-03-15", 20),
    }


def test_subsequent_close_pct_returns_none_when_market_fetch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yfinance failures on either fetch must not block the analogs path."""

    analogs_service._subsequent_close_pct.cache_clear()

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("yfinance offline")

    monkeypatch.setattr(
        "app.services.market_data.fetch_market_snapshot",
        _boom,
    )

    out = analogs_service._subsequent_close_pct("2022-09-21", horizon=5)
    assert out is None


def test_subsequent_close_pct_returns_none_when_short_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse forward windows produce None rather than fabricated returns."""

    analogs_service._subsequent_close_pct.cache_clear()

    monkeypatch.setattr(
        "app.services.market_data.fetch_market_snapshot",
        lambda *_args, **_kwargs: {"close": 3200.0},
    )

    def _short(*_args, **_kwargs) -> list[dict]:
        return [{"date": "2020-01-02", "close": 3257.0, "volatility_5d": 0.01}]

    monkeypatch.setattr(
        "app.services.market_data.fetch_realized_forward",
        _short,
    )

    out = analogs_service._subsequent_close_pct("2020-01-01", horizon=5)
    assert out is None


def test_subsequent_close_pct_computes_correct_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5d return = (close[event+5] / close[event] - 1) * 100 — uses the
    event-day close as the denominator (Bloomberg convention)."""

    analogs_service._subsequent_close_pct.cache_clear()

    closes = [100.0, 101.0, 102.0, 103.0, 105.0]

    monkeypatch.setattr(
        "app.services.market_data.fetch_market_snapshot",
        lambda *_args, **_kwargs: {"close": 100.0},
    )

    def _series(*_args, **kwargs) -> list[dict]:
        steps = kwargs.get("steps") or 5
        return [{"date": f"2020-01-0{i + 2}", "close": c, "volatility_5d": 0.01} for i, c in enumerate(closes[:steps])]

    monkeypatch.setattr(
        "app.services.market_data.fetch_realized_forward",
        _series,
    )

    out = analogs_service._subsequent_close_pct("2020-01-01", horizon=5)
    # (105 / 100 - 1) * 100 = 5.0
    assert out == pytest.approx(5.0, abs=1e-4)


def test_subsequent_close_pct_handles_zero_event_day_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a zero event-day close would divide-by-zero — return None."""

    analogs_service._subsequent_close_pct.cache_clear()

    monkeypatch.setattr(
        "app.services.market_data.fetch_market_snapshot",
        lambda *_args, **_kwargs: {"close": 0.0},
    )

    def _five(*_args, **_kwargs) -> list[dict]:
        return [
            {"date": "2020-01-02", "close": 99.0, "volatility_5d": 0.01},
            {"date": "2020-01-03", "close": 100.0, "volatility_5d": 0.01},
            {"date": "2020-01-06", "close": 101.0, "volatility_5d": 0.01},
            {"date": "2020-01-07", "close": 102.0, "volatility_5d": 0.01},
            {"date": "2020-01-08", "close": 103.0, "volatility_5d": 0.01},
        ]

    monkeypatch.setattr(
        "app.services.market_data.fetch_realized_forward",
        _five,
    )

    out = analogs_service._subsequent_close_pct("2020-01-01", horizon=5)
    assert out is None
