"""Unit tests for the stance-directional backtest engine (#299 PR-B)."""

from __future__ import annotations

import math

import pytest

from app.evaluation import backtest


@pytest.fixture(autouse=True)
def _stub_market(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the per-date forward % return so the engine math is deterministic."""

    pinned: dict[str, float] = {
        "2024-01-15": 1.0,
        "2024-02-15": -2.0,
        "2024-03-15": 0.5,
        "2024-04-15": -0.5,
        "2024-05-15": 2.0,
    }

    def _fake(date: str, horizon_days: int, symbol: str) -> float | None:
        return pinned.get(date)

    monkeypatch.setattr(backtest, "_lookup_forward_pct", _fake)


def _series(positions: list[tuple[str, int]]) -> list[dict]:
    return [{"date": d, "position": p} for d, p in positions]


def test_empty_realized_trades_returns_null_metrics() -> None:
    """A date with no forward data still appears in trades but metrics are null."""

    out = backtest.compute_backtest_metrics(
        _series([("2099-01-01", 1)])  # not in the stub
    )
    assert out["n_trades"] == 0
    assert out["sharpe"] is None
    assert out["hit_rate"] is None
    assert out["max_dd_pct"] is None
    assert len(out["trades"]) == 1


def test_long_only_hawkish_signal_short_sp_loses_when_market_up() -> None:
    """position=-1 with a +1% forward return → strategy return is -1%."""

    out = backtest.compute_backtest_metrics(_series([("2024-01-15", -1)]))
    assert out["trades"][0]["strategy_return_pct"] == pytest.approx(-1.0)
    assert out["n_trades"] == 1
    assert out["hit_rate"] == 0.0


def test_dovish_signal_long_sp_wins_when_market_up() -> None:
    """position=+1 with a +1% forward return → strategy return is +1%."""

    out = backtest.compute_backtest_metrics(_series([("2024-01-15", 1)]))
    assert out["trades"][0]["strategy_return_pct"] == pytest.approx(1.0)
    assert out["hit_rate"] == 1.0


def test_neutral_signal_is_a_skip() -> None:
    """position=0 → strategy_return is None (no exposure, no contribution)."""

    out = backtest.compute_backtest_metrics(
        _series([("2024-01-15", 0), ("2024-02-15", -1)])
    )
    assert out["trades"][0]["strategy_return_pct"] is None
    assert out["n_trades"] == 1  # only the hawkish trade counts


def test_sharpe_annualizes_with_holding_period_scale() -> None:
    """For a 5-day horizon: sharpe = mean/std * sqrt(252/5) = mean/std * sqrt(50.4)."""

    out = backtest.compute_backtest_metrics(
        _series([("2024-01-15", 1), ("2024-02-15", -1), ("2024-05-15", 1)]),
        horizon_days=5,
    )
    # Strategy returns: +1.0, +2.0, +2.0 → mean 1.6667, sample std 0.5774
    expected_sharpe = (5.0 / 3.0) / math.sqrt(1.0 / 3.0) * math.sqrt(252 / 5)
    assert out["sharpe"] == pytest.approx(expected_sharpe, abs=1e-2)


def test_max_drawdown_tracks_compounded_strategy_pnl() -> None:
    """A losing trade after gains produces a non-zero max DD."""

    out = backtest.compute_backtest_metrics(
        _series([("2024-01-15", 1), ("2024-02-15", 1), ("2024-04-15", 1)]),
    )
    # +1%, -2%, -0.5% → cum 1.01, 0.9898, 0.98487
    # peaks: 1.0, 1.01, 1.01, 1.01 → max DD = (0.98487 / 1.01 - 1) * 100 ≈ -2.49%
    assert out["max_dd_pct"] is not None
    assert out["max_dd_pct"] < 0
    assert out["max_dd_pct"] == pytest.approx(-2.488, abs=0.05)


def test_benchmark_and_alpha_signs() -> None:
    """Benchmark is sum of forward returns regardless of position.
    Alpha is strategy - benchmark.
    """

    out = backtest.compute_backtest_metrics(
        _series([("2024-01-15", -1), ("2024-02-15", 1)])
    )
    # forward: +1.0, -2.0 → cum buy-and-hold = 1.01 * 0.98 - 1 ≈ -0.99%
    # strategy: -1.0 ($+1%*-1) + -2.0 (-2*+1) → cum strategy = 0.99 * 0.98 - 1 ≈ -2.99%
    # alpha = -2.99 - (-0.99) ≈ -2.0
    assert out["benchmark_cum_pct"] == pytest.approx(-1.02, abs=0.02)
    assert out["cum_return_pct"] == pytest.approx(-2.99, abs=0.02)
    assert out["alpha_cum_pct"] == pytest.approx(-1.97, abs=0.05)


def test_invalid_position_raises() -> None:
    """Position must be in {-1, 0, 1}."""

    with pytest.raises(ValueError, match=r"position must be in \{-1,0,1\}"):
        backtest.compute_backtest_metrics(_series([("2024-01-15", 2)]))


def test_bool_position_rejected_as_int_lookalike() -> None:
    """``bool`` is an ``int`` subclass; reject explicitly so a True/False
    leak doesn't ride through as 1/0."""

    with pytest.raises(ValueError, match="bool"):
        backtest.compute_backtest_metrics(
            [{"date": "2024-01-15", "position": True}]
        )


def test_missing_date_raises() -> None:
    with pytest.raises(ValueError, match="missing 'date'"):
        backtest.compute_backtest_metrics([{"position": 1}])
