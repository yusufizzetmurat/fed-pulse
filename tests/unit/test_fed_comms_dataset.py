"""Embargo + alignment tests for the Fed-comms fusion dataset."""

from __future__ import annotations
import numpy as np
import pandas as pd
from app.data import fed_comms_dataset as d


def _rv_df(dates, rv):
    return pd.DataFrame({"date": dates, "rv": rv})


def test_forward_windows_and_indices() -> None:
    rv = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    fwd = d._forward_log_rv_windows(rv, (1, 2))
    assert np.isclose(fwd[1][0], np.log(2.0)) and np.isnan(fwd[1][-1])
    assert np.isclose(fwd[2][0], np.log((2 + 4) / 2))
    td = ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert d._origin_after("2020-01-02", td) == 1  # strictly after
    assert d._origin_after("2020-01-04", td) == 2  # next trading day
    assert d._origin_after("2020-01-06", td) is None  # nothing after
    assert d._as_of_index("2020-01-04", td) == 1  # latest <= date
    assert d._as_of_index("2020-01-01", td) is None


def test_text_outcome_embargo_starts_next_trading_day() -> None:
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    rv = _rv_df(days, [1.0, 2.0, 4.0, 8.0])
    corpus = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "timestamp_et": "2020-01-02 14:00",
                "doc_type": "statement",
                "time_known": True,
                "speaker": "monetary",
                "text": "x" * 300,
            }
        ]
    )
    pairs = d.build_text_outcome_pairs(corpus, rv, horizons=(1,))
    assert len(pairs) == 1
    # origin is the day AFTER the statement (2020-01-03), outcome = log rv[2]=4.0
    assert pairs.iloc[0]["origin_date"] == "2020-01-03"
    assert np.isclose(pairs.iloc[0]["rv_fwd_1"], np.log(4.0))


def test_daily_fusion_has_text_and_age() -> None:
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    rv = _rv_df(days, [1.0, 2.0, 4.0, 8.0])
    corpus = pd.DataFrame(
        [
            {
                "date": "2020-01-03",
                "timestamp_et": "2020-01-03 00:00",
                "doc_type": "speech",
                "time_known": False,
                "speaker": "powell",
                "text": "y" * 300,
            }
        ]
    )
    daily = d.build_daily_fusion_frame(rv, corpus, horizons=(1,))
    by_date = {r["date"]: r for _, r in daily.iterrows()}
    assert by_date["2020-01-02"]["has_text"] is False  # before any comm
    assert by_date["2020-01-03"]["has_text"] is True  # comm available same day
    assert by_date["2020-01-03"]["doc_age_days"] == 0
    assert by_date["2020-01-06"]["doc_age_days"] == 1  # one trading day later
    assert by_date["2020-01-06"]["doc_type"] == "speech"
