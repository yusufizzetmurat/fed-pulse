"""Embargo + alignment tests for the Fed-comms fusion dataset."""

from __future__ import annotations
import numpy as np
import pandas as pd
from app.data import fed_comms_dataset as d


def _rv_df(dates, rv):
    rv = np.asarray(rv, float)
    return pd.DataFrame({
        "date": dates, "rv": rv, "rvol": rv * 1e6 + 1.0,
        "bv": rv * 0.9, "rs_neg": rv * 0.5,
    })


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


def test_trailing_vol_window_and_leak_safety() -> None:
    # First `window`-1 entries NaN until the window is full; vol[t] uses only
    # changes ending at t (backward-looking → leak-safe).
    chg = np.array([np.nan, 1.0, -1.0, 1.0, -1.0, 1.0])
    out = d._trailing_vol(chg, 3)
    # t=0,1 below window; t=2 window [nan,1,-1] still carries the leading NaN.
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert np.isfinite(out[3])  # first full all-finite window [1,-1,1]
    assert np.isclose(out[3], np.std(np.array([1.0, -1.0, 1.0])))  # ddof=0
    # A future spike at t=5 must not change vol at t=3 (no look-ahead).
    chg2 = chg.copy()
    chg2[5] = 99.0
    out2 = d._trailing_vol(chg2, 3)
    assert np.isclose(out[3], out2[3])


def test_rate_vol_measures_omitted_without_cache_columns() -> None:
    # The cache-less fixture has no rate_vol_* columns → measures cleanly omitted.
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    rv = _rv_df(days, [1.0, 2.0, 4.0, 8.0])
    for m in d._RATE_VOL_MEASURES:
        assert d._measure_present(rv, m) is False
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
    for m in d._RATE_VOL_MEASURES:
        assert f"{m}_daily" not in daily.columns
        assert f"{m}_fwd_1" not in daily.columns


def test_surprise_asof_join_is_leak_safe() -> None:
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    rv = _rv_df(days, [1.0, 2.0, 4.0, 8.0])
    corpus = pd.DataFrame(
        [
            {
                "date": "2020-01-06",
                "timestamp_et": "2020-01-06 14:00",
                "doc_type": "statement",
                "time_known": True,
                "speaker": "monetary",
                "text": "x" * 300,
            }
        ]
    )
    surprise = pd.DataFrame(
        [
            {
                "date": "2020-01-06",
                "mp_surprise_level": 12.5,
                "mp_surprise_path_factor": -3.0,
                "fed_info_factor": 4.0,
            }
        ]
    )
    daily = d.build_daily_fusion_frame(rv, corpus, horizons=(1,), surprise=surprise)
    by_date = {r["date"]: r for _, r in daily.iterrows()}
    # Strictly before the statement → neutral fill (no surprise is known yet).
    assert by_date["2020-01-03"]["surprise_level"] == 0.0
    assert by_date["2020-01-03"]["surprise_info"] == 0.0
    # On the statement date and after → the statement's surprise is attached.
    assert by_date["2020-01-06"]["surprise_level"] == 12.5
    assert by_date["2020-01-06"]["surprise_path"] == -3.0
    assert by_date["2020-01-07"]["surprise_level"] == 12.5  # carried forward
    # Omitting the surprise frame leaves the columns at the neutral fill.
    daily0 = d.build_daily_fusion_frame(rv, corpus, horizons=(1,))
    assert (daily0["surprise_level"] == 0.0).all()
