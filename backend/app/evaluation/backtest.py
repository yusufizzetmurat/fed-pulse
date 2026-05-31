"""Stance-directional backtest engine for the quant-facing console (#299).

Given a list of signal positions {date, position in [-1, 0, 1]} and a
forward holding period, this module looks up the market data for each
date, computes the per-trade return, and aggregates Sharpe, hit-rate,
max-drawdown, and benchmark (buy-and-hold) deltas.

This is the v1 engine: it assumes daily-resolution S&P (^GSPC) close
data via :func:`app.services.market_data.fetch_realized_forward`. The
position is treated as a simple multiplier on the forward holding
return (no slippage, no fees, no overnight gap handling) — the goal
is to surface the quant-relevant signal magnitude, not to model
execution. Real execution-quality refinements are out of scope; if
needed they belong in a follow-up.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestTrade:
    """One realized trade: signal at ``date``, held ``horizon_days``."""

    date: str
    position: int
    forward_return_pct: float | None
    strategy_return_pct: float | None


def _validate_position(position: Any) -> int:
    """Coerce + bound-check a position signal to {-1, 0, 1}.

    Bool check is the load-bearing guard for direct engine callers
    (tests, internal callers that build positions by name). The
    schema layer (``BacktestPositionEntry`` with strict=True) catches
    bool first at the endpoint boundary, but the engine accepts raw
    dicts that bypass the schema — this guard is what stops a True/
    False from riding through as 1/0 in that path.
    """

    if isinstance(position, bool):
        raise ValueError(f"position must be int in {{-1,0,1}}, got bool")
    try:
        as_int = int(position)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"position must be int in {{-1,0,1}}, got {position!r}") from exc
    if as_int not in (-1, 0, 1):
        raise ValueError(f"position must be in {{-1,0,1}}, got {as_int}")
    return as_int


def _lookup_forward_pct(date: str, horizon_days: int, symbol: str) -> float | None:
    """Return the close-to-close % return over ``horizon_days`` trading
    days from the close on ``date`` (or the nearest prior trading day).

    None when the historical window is sparse (early-history dates) or
    when yfinance / cache is unavailable. Same convention as
    :func:`app.services.analogs._subsequent_close_pct` so the
    backtest numbers line up with the historical-analog panel on the
    console.
    """

    from app.services.market_data import fetch_market_snapshot, fetch_realized_forward

    try:
        snapshot = fetch_market_snapshot(target_date=date, symbol=symbol)
        forward = fetch_realized_forward(
            target_date=date,
            symbol=symbol,
            steps=horizon_days,
            lookback_days=45,
        )
    except Exception:
        return None
    if len(forward) < horizon_days:
        return None
    try:
        start = float(snapshot["close"])
        end = float(forward[horizon_days - 1]["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def compute_backtest_metrics(
    positions: Iterable[dict[str, Any]],
    *,
    symbol: str = "^GSPC",
    horizon_days: int = 5,
) -> dict[str, Any]:
    """Run the stance-directional backtest engine.

    ``positions`` is an iterable of ``{"date": "YYYY-MM-DD", "position":
    -1|0|1}`` dicts. The engine looks up the forward % return per
    trade, multiplies by the position to get the strategy return, and
    aggregates the metrics below.

    Returns a dict with:
    - ``trades``: per-trade ``BacktestTrade`` dicts in date order
    - ``n_trades``: int, count of non-zero positions with valid forward returns
    - ``sharpe``: float | None, mean(strategy) / std(strategy) * sqrt(holding-period scale)
    - ``hit_rate``: float | None, share of trades where sign(strategy) > 0
    - ``max_dd_pct``: float | None, max % drawdown of cumulative strategy PnL
    - ``cum_return_pct``: float | None, cumulative compounded strategy return
    - ``benchmark_cum_pct``: float | None, cumulative buy-and-hold return over the same windows
    - ``alpha_cum_pct``: float | None, cum_return_pct − benchmark_cum_pct
    - ``horizon_days``: int, echo of the input horizon
    - ``symbol``: str, echo of the input symbol
    """

    trades: list[BacktestTrade] = []
    for entry in positions:
        date = str(entry.get("date", "")).strip()
        if not date:
            raise ValueError(f"position entry missing 'date': {entry!r}")
        position = _validate_position(entry.get("position"))
        forward_pct = _lookup_forward_pct(date, horizon_days, symbol)
        strategy_pct = (
            None
            if forward_pct is None or position == 0
            else round(position * forward_pct, 4)
        )
        trades.append(
            BacktestTrade(
                date=date,
                position=position,
                forward_return_pct=None if forward_pct is None else round(forward_pct, 4),
                strategy_return_pct=strategy_pct,
            )
        )

    realized = [t for t in trades if t.strategy_return_pct is not None]
    n_trades = len(realized)

    if n_trades == 0:
        return {
            "trades": [trade_to_dict(t) for t in trades],
            "n_trades": 0,
            "sharpe": None,
            "hit_rate": None,
            "max_dd_pct": None,
            "cum_return_pct": None,
            "benchmark_cum_pct": None,
            "alpha_cum_pct": None,
            "horizon_days": horizon_days,
            "symbol": symbol,
        }

    strategy_pcts = [t.strategy_return_pct for t in realized]
    # #564 review F1: benchmark compounds over EVERY trade with valid
    # forward data (incl. neutral positions). Restricting to ``realized``
    # would silently exclude neutral-position dates from the buy-and-hold
    # comparison and make alpha look favorable when neutrals span moves
    # the strategy missed. Full-window buy-and-hold is the standard
    # baseline.
    forward_pcts = [
        t.forward_return_pct for t in trades if t.forward_return_pct is not None
    ]

    mean = sum(strategy_pcts) / n_trades
    variance = (
        sum((r - mean) ** 2 for r in strategy_pcts) / (n_trades - 1)
        if n_trades > 1
        else 0.0
    )
    std = math.sqrt(variance)
    # Annualization: each trade is a discrete holding-period return.
    # The standard convention is mean / std * sqrt(periods_per_year)
    # where periods_per_year = trading_days / horizon_days.
    periods_per_year = TRADING_DAYS_PER_YEAR / max(horizon_days, 1)
    sharpe = (
        round((mean / std) * math.sqrt(periods_per_year), 4)
        if std > 0
        else None
    )

    hits = sum(1 for r in strategy_pcts if r > 0)
    hit_rate = round(hits / n_trades, 4)

    cum_strategy = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in strategy_pcts:
        cum_strategy *= 1.0 + r / 100.0
        if cum_strategy > peak:
            peak = cum_strategy
        dd = (cum_strategy - peak) / peak * 100.0
        if dd < max_dd:
            max_dd = dd

    cum_benchmark = 1.0
    for r in forward_pcts:
        cum_benchmark *= 1.0 + r / 100.0

    cum_return_pct = round((cum_strategy - 1.0) * 100.0, 4)
    benchmark_cum_pct = round((cum_benchmark - 1.0) * 100.0, 4)

    return {
        "trades": [trade_to_dict(t) for t in trades],
        "n_trades": n_trades,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_dd_pct": round(max_dd, 4),
        "cum_return_pct": cum_return_pct,
        "benchmark_cum_pct": benchmark_cum_pct,
        "alpha_cum_pct": round(cum_return_pct - benchmark_cum_pct, 4),
        "horizon_days": horizon_days,
        "symbol": symbol,
    }


def trade_to_dict(trade: BacktestTrade) -> dict[str, Any]:
    return {
        "date": trade.date,
        "position": trade.position,
        "forward_return_pct": trade.forward_return_pct,
        "strategy_return_pct": trade.strategy_return_pct,
    }


__all__ = ["compute_backtest_metrics", "trade_to_dict", "BacktestTrade", "TRADING_DAYS_PER_YEAR"]
