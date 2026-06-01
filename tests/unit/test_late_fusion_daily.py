"""Unit tests for the clean-room daily companion frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.late_fusion_daily import build_daily_frame


def _daily_close(n: int, start: str = "2020-01-01") -> pd.DataFrame:
    # business-day index, deterministic rising-then-varying closes
    dates = pd.bdate_range(start, periods=n)
    closes = 100.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    return pd.DataFrame({"date": dates, "close": closes})


def test_next_day_target_and_anchor() -> None:
    daily = _daily_close(60)
    # a comm dated on the 30th trading day
    comm_date = daily["date"].iloc[30]
    corpus = pd.DataFrame(
        {
            "date": [str(comm_date.date())],
            "doc_type": ["speech"],
            "speaker": ["Powell"],
            "title": ["A speech"],
            "text": ["hawkish remarks"],
        }
    )
    out = build_daily_frame(corpus, daily)
    assert len(out) == 1
    row = out.iloc[0]
    # anchor t0 = the comm's trading day; target = log(close[t1]/close[t0])
    lc = np.log(daily["close"].to_numpy())
    assert row["ret_nextday"] == pytest.approx(lc[31] - lc[30])
    assert row["dir_nextday"] == 1  # constant positive drift
    assert row["anchor_date"] == str(daily["date"].iloc[30].date())


def test_row_hash_present_and_unique() -> None:
    daily = _daily_close(60)
    d = daily["date"].iloc[30]
    corpus = pd.DataFrame(
        {
            "date": [str(d.date()), str(d.date())],
            "doc_type": ["speech", "statement"],  # same date, different type/text
            "speaker": ["A", "B"],
            "title": ["t1", "t2"],
            "text": ["alpha", "beta"],
        }
    )
    out = build_daily_frame(corpus, daily)
    assert "row_hash" in out.columns
    assert out["row_hash"].is_unique  # distinct docs -> distinct keys


def test_comm_on_nontrading_day_anchors_forward() -> None:
    daily = _daily_close(60, start="2020-01-01")
    # pick a Saturday between two business days
    sat = pd.Timestamp("2020-02-08")  # a Saturday
    assert sat.weekday() == 5
    corpus = pd.DataFrame(
        {
            "date": [str(sat.date())],
            "doc_type": ["speech"],
            "speaker": ["X"],
            "title": ["t"],
            "text": ["txt"],
        }
    )
    out = build_daily_frame(corpus, daily)
    # anchor must be the first trading day on/after the Saturday (the Monday)
    anchor = pd.Timestamp(out.iloc[0]["anchor_date"])
    assert anchor.weekday() == 0  # Monday
    assert anchor > sat


def test_insufficient_history_skipped() -> None:
    daily = _daily_close(60)
    # comm on the 3rd trading day -> < 22 days history -> skipped
    early = daily["date"].iloc[3]
    corpus = pd.DataFrame(
        {"date": [str(early.date())], "doc_type": ["speech"], "speaker": ["x"],
         "title": ["t"], "text": ["y"]}
    )
    out = build_daily_frame(corpus, daily)
    assert out.empty


def test_features_are_as_of_anchor_not_future() -> None:
    # Make a price series with a sharp jump AFTER t0; the as-of features (ret_5d,
    # ret_22d) must not see it, but the next-day target must.
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = np.full(n, 100.0)
    closes[35:] = 200.0  # jump at index 35
    daily = pd.DataFrame({"date": dates, "close": closes})
    comm_date = dates[34]  # t0 = 34, t1 = 35 (the jump)
    corpus = pd.DataFrame(
        {"date": [str(comm_date.date())], "doc_type": ["statement"], "speaker": ["x"],
         "title": ["t"], "text": ["y"]}
    )
    row = build_daily_frame(corpus, daily).iloc[0]
    # as-of features see only flat pre-jump prices -> ~0
    assert row["ret_5d"] == pytest.approx(0.0)
    assert row["ret_22d"] == pytest.approx(0.0)
    # target captures the forward jump
    assert row["ret_nextday"] == pytest.approx(np.log(200 / 100))
    assert row["dir_nextday"] == 1
