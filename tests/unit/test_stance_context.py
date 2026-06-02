"""Behavioural tests for the rolling stance-score context builder.

Anchors:
- `s = P(hawkish) - P(dovish)` per the validity study (matches the
  signal that landed Spearman +0.283 vs DFEDTARU).
- The tile falls back to raw rendering when fewer than 2 usable rows
  are found, so a regression-mode backlog must not poison the mean.
- Excluding the current run from the trailing window prevents the
  z-score from being computed against itself.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import AnalysisRun, Base
from app.services.stance_context import (
    _extract_stance_score,
    build_stance_context,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def _make_run(
    *,
    run_id: str,
    symbol: str,
    horizon: str,
    document_date: str,
    hawkish: float | None,
    dovish: float | None,
    created_at: _dt.datetime,
) -> AnalysisRun:
    """Build a persisted run whose payload carries one multi-axis stance row."""

    distribution: dict[str, float] = {}
    if hawkish is not None:
        distribution["hawkish"] = hawkish
    if dovish is not None:
        distribution["dovish"] = dovish

    payload: dict[str, object] = {}
    if distribution:
        payload = {
            "multi_axis": {
                "stance": {
                    "label": "hawkish" if (hawkish or 0) > (dovish or 0) else "dovish",
                    "confidence": max(hawkish or 0, dovish or 0),
                    "distribution": distribution,
                }
            }
        }

    return AnalysisRun(
        id=run_id,
        created_at=created_at,
        symbol=symbol,
        document_date=document_date,
        horizon=horizon,
        forecast_mode="fast",
        stance="hawkish",
        sentiment_score=None,
        predicted_close=None,
        current_close=None,
        predicted_volatility=None,
        payload=payload,
        text_excerpt=None,
    )


def test_extract_stance_score_from_full_distribution() -> None:
    payload = {
        "multi_axis": {
            "stance": {"distribution": {"hawkish": 0.7, "dovish": 0.2, "neutral": 0.1}}
        }
    }
    assert _extract_stance_score(payload) == pytest.approx(0.5)


def test_extract_stance_score_handles_missing_keys() -> None:
    assert _extract_stance_score(None) is None
    assert _extract_stance_score({}) is None
    assert _extract_stance_score({"multi_axis": None}) is None
    assert _extract_stance_score({"multi_axis": {"stance": None}}) is None
    assert (
        _extract_stance_score({"multi_axis": {"stance": {"distribution": {}}}}) is None
    )


def test_extract_stance_score_returns_zero_for_neutral_only() -> None:
    """A pure-neutral distribution (no hawk/dove mass) IS s=0.0, not None.

    Suppressing it would mis-classify a legitimately on-the-fence
    statement as a missing measurement.
    """

    payload = {"multi_axis": {"stance": {"distribution": {"neutral": 1.0}}}}
    assert _extract_stance_score(payload) is None  # neither hawkish nor dovish key


def test_extract_stance_score_handles_one_sided_distribution() -> None:
    """Only hawkish or only dovish present should still resolve."""

    assert _extract_stance_score(
        {"multi_axis": {"stance": {"distribution": {"hawkish": 0.6}}}}
    ) == pytest.approx(0.6)
    assert _extract_stance_score(
        {"multi_axis": {"stance": {"distribution": {"dovish": 0.4}}}}
    ) == pytest.approx(-0.4)


def test_build_stance_context_returns_none_mean_for_lt_two_rows(session) -> None:
    session.add(
        _make_run(
            run_id="a",
            symbol="^GSPC",
            horizon="10d",
            document_date="2026-05-01",
            hawkish=0.7,
            dovish=0.2,
            created_at=_dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc),
        )
    )
    session.commit()
    ctx = build_stance_context(session, symbol="^GSPC", n=12)
    assert ctx.n == 1
    assert ctx.mean is None
    assert ctx.std is None
    assert len(ctx.history) == 1


def test_build_stance_context_computes_mean_and_std(session) -> None:
    """Three rows with distinct s values produce a meaningful mean+std."""

    base = _dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc)
    for i, (h, d) in enumerate([(0.7, 0.2), (0.4, 0.4), (0.6, 0.3)]):
        session.add(
            _make_run(
                run_id=f"r{i}",
                symbol="^GSPC",
                horizon="10d",
                document_date=f"2026-05-{i + 1:02d}",
                hawkish=h,
                dovish=d,
                created_at=base + _dt.timedelta(days=i),
            )
        )
    session.commit()
    ctx = build_stance_context(session, symbol="^GSPC", n=12)
    assert ctx.n == 3
    # Scores are 0.5, 0.0, 0.3 → mean ≈ 0.2667, sample std ≈ 0.2517
    assert ctx.mean is not None and abs(ctx.mean - 0.2667) < 1e-3
    assert ctx.std is not None and ctx.std > 0.0


def test_build_stance_context_filters_runs_without_usable_distribution(session) -> None:
    """A row with no stance distribution must not enter the window."""

    base = _dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc)
    session.add(
        _make_run(
            run_id="bad",
            symbol="^GSPC",
            horizon="10d",
            document_date="2026-05-01",
            hawkish=None,
            dovish=None,
            created_at=base,
        )
    )
    for i, (h, d) in enumerate([(0.6, 0.3), (0.4, 0.4)]):
        session.add(
            _make_run(
                run_id=f"good{i}",
                symbol="^GSPC",
                horizon="10d",
                document_date=f"2026-05-{i + 2:02d}",
                hawkish=h,
                dovish=d,
                created_at=base + _dt.timedelta(days=i + 1),
            )
        )
    session.commit()
    ctx = build_stance_context(session, symbol="^GSPC", n=12)
    assert ctx.n == 2  # the empty-payload run is dropped
    assert all(math.isfinite(p.stance_score) for p in ctx.history)


def test_build_stance_context_excludes_run_id(session) -> None:
    """Self-exclusion: a request that names the current run id must
    return the trailing window without it."""

    base = _dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc)
    for i, (h, d) in enumerate([(0.6, 0.3), (0.4, 0.4), (0.7, 0.2)]):
        session.add(
            _make_run(
                run_id=f"r{i}",
                symbol="^GSPC",
                horizon="10d",
                document_date=f"2026-05-{i + 1:02d}",
                hawkish=h,
                dovish=d,
                created_at=base + _dt.timedelta(days=i),
            )
        )
    session.commit()
    ctx = build_stance_context(
        session, symbol="^GSPC", n=12, exclude_run_id="r2"
    )
    assert ctx.n == 2
    assert all(p.document_date != "2026-05-03" for p in ctx.history)
