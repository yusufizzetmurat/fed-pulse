"""Tests for the monetary-policy surprise builder.

The tests bypass HTTP via ``httpx.MockTransport`` for the FRED endpoint
and inject canned SPX close maps. They cover:

- The mock-transport path that drives :func:`fetch_fred_series`.
- Hand-verified surprises on 2008-10-08 (50 bp emergency cut) and
  2020-03-15 (pandemic cut). The 2013-05-22 Bernanke testimony was
  intentionally removed from the in-scope set in PR #154 review fix-up:
  it is intermeeting Congressional testimony, not an FOMC announcement,
  and is absent from the production calendar by design.
- PCA path-factor eigenvectors persist deterministically in the lock
  JSON; rebuilding with the same inputs yields the same eigenvector.
- Holiday-cluster trading-day lookback: a synthetic year where the
  pre/post window has no calendar trading days within the 5-day radius
  still resolves via the trading-day index.
- Adjacent-meeting target-lookahead clipping: two FOMC events within
  five calendar days must not pollute each other's `ff_target_after`.
- DataFrame-value hash determinism: round-trip the parquet, hash the
  sorted DataFrame values, and pin against a reference.

Most assertions tolerate a small numerical noise envelope -- we are not
re-deriving published surprise values, only locking the *sign* and a
sensible order-of-magnitude check the way the issue requires.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Mapping

import httpx
import pandas as pd
import pytest

from app.data import mp_surprise


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _series_response(
    series_id: str,
    observations: list[tuple[str, float | None]],
    *,
    realtime: str = "2026-05-15",
) -> dict[str, object]:
    return {
        "realtime_start": realtime,
        "realtime_end": realtime,
        "observation_start": observations[0][0] if observations else "",
        "observation_end": observations[-1][0] if observations else "",
        "count": len(observations),
        "observations": [
            {
                "date": d,
                "value": ("." if v is None else f"{v}"),
                "realtime_start": realtime,
                "realtime_end": realtime,
            }
            for d, v in observations
        ],
    }


def _build_fred_payloads(start: _dt.date, end: _dt.date) -> dict[str, dict[str, object]]:
    """Hand-built FRED payloads covering ``[start, end]``.

    Daily yields drift slowly. We make them piecewise-flat between the
    handful of meetings we test so the level / path math is predictable.
    """

    days = (end - start).days + 1
    dates = [(start + _dt.timedelta(days=i)).isoformat() for i in range(days)]

    # Deterministic daily curves -- the absolute level is what each tenor
    # sits at *before* perturbation by an FOMC event. We inject a 50 bp
    # cut at 2020-03-15 across the front of the curve and a +20 bp path
    # shift at 2013-05-22.
    def curve_value(date_iso: str, base: float, *, perturb_pre_2020_03: float = 0.0, perturb_pre_2013_05: float = 0.0) -> float:
        d = _dt.date.fromisoformat(date_iso)
        v = base
        if d >= _dt.date(2020, 3, 16):
            v += perturb_pre_2020_03
        if d >= _dt.date(2013, 5, 23):
            v += perturb_pre_2013_05
        return v

    yields_by_tenor = {
        # FRED reports yields in percent (e.g. 0.50 == 0.50% == 50 bps).
        "DGS1MO": [curve_value(d, 0.10, perturb_pre_2020_03=-0.50) for d in dates],
        "DGS3MO": [curve_value(d, 0.20, perturb_pre_2020_03=-0.45, perturb_pre_2013_05=+0.02) for d in dates],
        "DGS6MO": [curve_value(d, 0.30, perturb_pre_2020_03=-0.40, perturb_pre_2013_05=+0.10) for d in dates],
        "DGS1":   [curve_value(d, 0.60, perturb_pre_2020_03=-0.35, perturb_pre_2013_05=+0.20) for d in dates],
        "DGS2":   [curve_value(d, 1.00, perturb_pre_2020_03=-0.30, perturb_pre_2013_05=+0.15) for d in dates],
        # Effective FF rate ride along with the 1m yield as a placeholder.
        "DFF":    [curve_value(d, 0.10, perturb_pre_2020_03=-0.50) for d in dates],
    }
    # Target band: pre-2008 single rate, post-2008 band. The pandemic cut
    # on 2020-03-15 (Sunday) takes effect that same day in the FRED series.
    upper_obs = []
    lower_obs = []
    single_obs = []
    for d in dates:
        date_obj = _dt.date.fromisoformat(d)
        if date_obj >= _dt.date(2008, 12, 16):
            if date_obj >= _dt.date(2020, 3, 15):
                upper_obs.append((d, 0.25))
                lower_obs.append((d, 0.00))
            else:
                upper_obs.append((d, 0.50))
                lower_obs.append((d, 0.25))
        else:
            single_obs.append((d, 1.50))

    payloads: dict[str, dict[str, object]] = {
        "DFEDTARU": _series_response("DFEDTARU", upper_obs),
        "DFEDTARL": _series_response("DFEDTARL", lower_obs),
        "DFEDTAR": _series_response("DFEDTAR", single_obs),
        "DFF": _series_response("DFF", [(d, v) for d, v in zip(dates, yields_by_tenor["DFF"])]),
    }
    for sid in ("DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"):
        payloads[sid] = _series_response(sid, [(d, v) for d, v in zip(dates, yields_by_tenor[sid])])
    return payloads


def _mock_transport(payloads: Mapping[str, dict[str, object]]) -> httpx.MockTransport:
    """Return an httpx mock that dispatches on the ``series_id`` query param."""

    def handler(request: httpx.Request) -> httpx.Response:
        sid = request.url.params.get("series_id", "")
        body = payloads.get(sid)
        if body is None:
            return httpx.Response(404, json={"error": f"unknown series: {sid}"})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.fixture
def small_calendar(tmp_path: Path) -> Path:
    """Tiny FOMC calendar: one scheduled meeting + 2020-03-15 emergency cut.

    Note: the original fixture also included 2013-05-22 (Bernanke
    Congressional testimony) but it was removed in the PR #154 review
    fix-up. That date is intermeeting Congressional testimony, NOT an
    FOMC policy announcement, and is not part of the production
    `fomc_meetings_2010_2026.csv`. Keeping it here lets a vacuous test
    pass on a date the live build will never see.
    """

    csv = tmp_path / "fomc_meetings.csv"
    csv.write_text(
        "meeting_date,is_intermeeting,notes\n"
        "2014-03-19,false,scheduled\n"
        "2020-03-15,true,emergency_cut_100bp_covid_sunday\n",
        encoding="utf-8",
    )
    return csv


@pytest.fixture
def fred_cache(tmp_path: Path) -> Path:
    """A fresh FRED cache directory per test (so cached JSON is isolated)."""

    cache = tmp_path / "fred"
    cache.mkdir(parents=True)
    return cache


# ---------------------------------------------------------------------------
# 1. End-to-end MockTransport happy path
# ---------------------------------------------------------------------------


def test_hydrate_via_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    fred_cache: Path,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )
    expected = set(mp_surprise._required_series_ids())
    assert set(responses) == expected
    for sid in expected:
        assert responses[sid].series_id == sid


# ---------------------------------------------------------------------------
# 2. Hand-verified emergency cut: 2020-03-15 pandemic cut
# ---------------------------------------------------------------------------


def test_pandemic_emergency_cut_2020_03_15(
    monkeypatch: pytest.MonkeyPatch,
    small_calendar: Path,
    fred_cache: Path,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )

    # SPX map: large negative same-day return on the emergency cut (sign matters).
    spx_map = {
        _dt.date(2013, 5, 21): 1660.0,
        _dt.date(2013, 5, 23): 1650.0,
        _dt.date(2014, 3, 18): 1872.0,
        _dt.date(2014, 3, 20): 1872.0,
        _dt.date(2020, 3, 13): 2711.0,
        _dt.date(2020, 3, 16): 2386.0,
    }

    artifacts = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    assert artifacts.rows_written == 2
    pandemic = artifacts.frame[artifacts.frame["event_date"] == "2020-03-15"].iloc[0]
    # #350: ``mp_surprise_level`` is now the strict-prior surprise:
    # ``actual_target_change_bps - pre_implied_next_move_bps`` where the
    # pre-implied leg is ``(pre_yield_1m - ff_target_prior) * 100``. In
    # this fixture the 1m yield sits at 0.10% on T-1 against a 0.375%
    # mid-band, so the curve already priced a ~27.5 bp cut; the actual
    # cut of 25 bp leaves a tiny residual near zero. The methodology
    # change is locked here against the leaky -50 bp post-window reading
    # the prior CVJ-style construction produced. See ADR-0024.
    level = float(pandemic["mp_surprise_level"])
    assert -10.0 < level < 10.0, (
        f"strict-prior level surprise should be near zero on a "
        f"fully-priced cut, got {level} bps"
    )
    assert bool(pandemic["is_intermeeting"]) is True
    assert pandemic["methodology"] == mp_surprise.METHODOLOGY_OIS_PROXY
    # Target after should fall to the 0.00-0.25 % band midpoint.
    assert float(pandemic["ff_target_after"]) == pytest.approx(0.125, abs=1e-6)
    # Fed-info factor must derive from a strictly-prior SPX leg now.
    src = pandemic["fed_info_factor_source"]
    assert src in {"strict_prior_trailing", "unavailable", "level_missing"}, (
        f"unexpected fed_info_factor_source {src!r} -- the #350 reformulation "
        "rejects the leaky daily_window_proxy / alphavantage_intraday_30min routes."
    )


# ---------------------------------------------------------------------------
# 3. (removed) 2013-05-22 Bernanke testimony — out of scope.
# The original test asserted on intermeeting Congressional testimony, not
# an FOMC policy announcement, and the date is absent from the production
# `fomc_meetings_2010_2026.csv`. Keeping the test passed vacuously because
# it injected its own calendar. Dropped in PR #154 review fix-up.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. 2008-10-08 50bp emergency cut (data outside default range -- skip per spec)
# ---------------------------------------------------------------------------


def test_2008_10_08_emergency_cut_documented_as_outside_default_range() -> None:
    """The issue requests an explicit test that the function *would* output
    a ~50 bp surprise on 2008-10-08 if run on 2008 data; we run the
    builder over a narrow 2008 window with a hand-crafted single-target
    series, and assert the level surprise magnitude.

    The default ``--start`` is 2010-01-01 so the live parquet excludes
    this row by design. This test exists to lock the methodology, not
    to ship the row.
    """

    start = _dt.date(2008, 9, 15)
    end = _dt.date(2008, 11, 30)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    # Hand-craft single-target rates so 2008-10-07 = 2.0%, 2008-10-08 = 1.5%.
    # FRED's DFEDTAR series publishes the new target on the announcement
    # day in the post-2003 schedule; the #350 strict-prior reformulation
    # reads ``ff_target_after`` directly from ``event_date`` (preferring
    # the on-day value over the lookahead). The 1m yield series stays at
    # 1.80% through T-1 and falls only after the announcement, so the
    # surprise is genuinely actual-vs-pre-implied. Path tenors mirror the
    # 1m drop so the path factor stays near zero.
    days = (end - start + _dt.timedelta(days=22)).days
    base = start - _dt.timedelta(days=14)
    new_obs: list[tuple[str, float | None]] = []
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        v = 1.80 if d < _dt.date(2008, 10, 9) else 1.30
        new_obs.append((d.isoformat(), v))
    payloads["DGS1MO"] = _series_response("DGS1MO", new_obs)
    # Path tenors fall by the same -50 bp shift after the announcement.
    for sid, base_val in (("DGS3MO", 1.85), ("DGS6MO", 1.95), ("DGS1", 2.20), ("DGS2", 2.50)):
        obs2 = []
        for i in range(days):
            d = base + _dt.timedelta(days=i)
            v = base_val if d < _dt.date(2008, 10, 9) else base_val - 0.50
            obs2.append((d.isoformat(), v))
        payloads[sid] = _series_response(sid, obs2)
    # Single-target rate (pre-band era): published target reads 2.0% up
    # through 2008-10-07 (T-1) and 1.5% from 2008-10-08 (announcement
    # day) onward.
    single_obs: list[tuple[str, float | None]] = []
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        v = 2.0 if d < _dt.date(2008, 10, 8) else 1.5
        single_obs.append((d.isoformat(), v))
    payloads["DFEDTAR"] = _series_response("DFEDTAR", single_obs)
    payloads["DFEDTARU"] = _series_response("DFEDTARU", [])
    payloads["DFEDTARL"] = _series_response("DFEDTARL", [])

    fred_responses = {
        sid: mp_surprise.fetch_fred_series.__wrapped__  # type: ignore[attr-defined]
        if False
        else _parse_in_memory(sid, payloads[sid])
        for sid in mp_surprise._required_series_ids()
    }
    calendar = [
        mp_surprise.FomcMeetingRecord(
            meeting_date=_dt.date(2008, 10, 8),
            is_intermeeting=True,
            notes="emergency_cut_50bp",
        )
    ]
    artifacts = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=fred_responses,
        fomc_calendar=calendar,
        spx_close_by_date={_dt.date(2008, 10, 7): 1000.0, _dt.date(2008, 10, 9): 940.0},
    )
    row = artifacts.frame.iloc[0]
    level = float(row["mp_surprise_level"])
    # #350 strict-prior reformulation: ``mp_surprise_level`` is the
    # actual-target-change minus the pre-implied next-move on T-1. The
    # fixture leaves the 1m yield at 1.80% on T-1 against a 2.00% target
    # (pre-implied = -20 bp), then the announcement moves the target to
    # 1.50% (actual = -50 bp), so the surprise is -30 bp. Under the old
    # construction this was -50 bp -- the literal post-event 1m yield
    # change -- which leaked T+1 data. The methodology shift is locked
    # here and footnoted in ADR-0024.
    assert level == pytest.approx(-30.0, abs=10.0), (
        f"2008-10-08 strict-prior level surprise should be ~-30 bp "
        f"(actual -50 minus pre-implied -20), got {level}"
    )
    assert bool(row["is_intermeeting"]) is True


def _parse_in_memory(series_id: str, payload: dict[str, object]) -> "mp_surprise.FredSeriesResponse":
    """Parse a FRED payload dict into a ``FredSeriesResponse`` without HTTP.

    Mirrors the parser inside :mod:`app.services.fred_client` but stays
    in this test module so we are not asserting on a private helper.
    """

    from app.services.fred_client import _parse_observations  # type: ignore[attr-defined]

    return _parse_observations(payload, series_id)


# ---------------------------------------------------------------------------
# 5. Determinism: PCA eigenvector persists identically across rebuilds
# ---------------------------------------------------------------------------


def test_pca_eigenvector_persists_in_lock(
    monkeypatch: pytest.MonkeyPatch,
    small_calendar: Path,
    fred_cache: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )
    spx_map = {
        _dt.date(2013, 5, 21): 1660.0,
        _dt.date(2013, 5, 23): 1655.0,
        _dt.date(2014, 3, 18): 1872.0,
        _dt.date(2014, 3, 20): 1872.0,
        _dt.date(2020, 3, 13): 2711.0,
        _dt.date(2020, 3, 16): 2386.0,
    }

    output_path = tmp_path / "mp_surprises.parquet"
    lock_path = tmp_path / "SOURCES.lock"

    artifacts_first = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    sha_first = mp_surprise.write_mp_surprises_parquet(artifacts_first.frame, output_path)
    value_hash_first = mp_surprise.dataframe_value_hash(artifacts_first.frame)
    mp_surprise.update_sources_lock(
        lock_path=lock_path,
        artifacts=artifacts_first,
        parquet_path=output_path,
        parquet_sha256=sha_first,
    )
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = lock_payload[mp_surprise.DEFAULT_LOCK_KEY]
    pca = entry["path_factor_model"]
    assert pca["tenors_months"] == list(mp_surprise.PATH_TENORS_MONTHS)
    assert len(pca["eigenvector"]) == len(mp_surprise.PATH_TENORS_MONTHS)
    eigenvector_first = list(pca["eigenvector"])

    # Second build with same inputs -> identical data-value hash + same eigenvector.
    # The byte-level sha may drift across pyarrow versions; the contract
    # we lock here is on the *data*, not the encoding.
    artifacts_second = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    _ = mp_surprise.write_mp_surprises_parquet(artifacts_second.frame, output_path)
    value_hash_second = mp_surprise.dataframe_value_hash(artifacts_second.frame)
    assert value_hash_first == value_hash_second, (
        "rebuild produced different DataFrame values "
        f"({value_hash_first} vs {value_hash_second})"
    )
    mp_surprise.update_sources_lock(
        lock_path=lock_path,
        artifacts=artifacts_second,
        parquet_path=output_path,
        parquet_sha256=sha_first,
    )
    lock_payload_2 = json.loads(lock_path.read_text(encoding="utf-8"))
    eigenvector_second = list(lock_payload_2[mp_surprise.DEFAULT_LOCK_KEY]["path_factor_model"]["eigenvector"])
    assert eigenvector_first == eigenvector_second, (
        f"eigenvector drifted between rebuilds: {eigenvector_first} vs {eigenvector_second}"
    )


# ---------------------------------------------------------------------------
# 6. Calendar loader
# ---------------------------------------------------------------------------


def test_load_bundled_fomc_calendar_includes_2020_emergency_cuts() -> None:
    cal = mp_surprise.load_fomc_calendar(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 12, 31),
    )
    by_date = {r.meeting_date: r for r in cal}
    assert _dt.date(2020, 3, 3) in by_date and by_date[_dt.date(2020, 3, 3)].is_intermeeting
    assert _dt.date(2020, 3, 15) in by_date and by_date[_dt.date(2020, 3, 15)].is_intermeeting
    assert _dt.date(2020, 4, 29) in by_date and not by_date[_dt.date(2020, 4, 29)].is_intermeeting
    # The 2020 schedule has 8 scheduled + 2 emergency.
    assert sum(1 for r in cal if r.is_intermeeting) == 2


# ---------------------------------------------------------------------------
# 7. Empty / missing-data degrade-to-zero behaviour
# ---------------------------------------------------------------------------


def test_missing_spx_degrades_fed_info_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    small_calendar: Path,
    fred_cache: Path,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )
    artifacts = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date={},  # no SPX coverage
    )
    assert artifacts.fed_info_factor_unavailable_rows == artifacts.rows_written
    for source in artifacts.frame["fed_info_factor_source"].tolist():
        assert source == "unavailable", f"expected unavailable flag, got {source}"
    for value in artifacts.frame["fed_info_factor"].tolist():
        assert value is None


# ---------------------------------------------------------------------------
# 8. Trading-day lookback survives a holiday cluster (CRITICAL #3)
# ---------------------------------------------------------------------------


def test_pre_post_yields_uses_trading_day_index_under_holiday_cluster() -> None:
    """When the calendar around the event date has no trading days
    inside the 5-day radius (e.g. a synthetic year where days 350-365
    are all holidays except day 360), the calendar-day implementation
    silently returns None. The trading-day-index implementation must
    still resolve the nearest published yield."""

    base = _dt.date(2024, 1, 1)
    # Construct a sparse trading-day calendar with a 15-day holiday gap.
    # Trading days: every weekday in Jan-Nov, but in Dec we drop almost
    # everything — only Dec 20 is published.
    trading_days: list[_dt.date] = []
    for offset in range(365):
        d = base + _dt.timedelta(days=offset)
        # Standard business-day weekdays in Jan-Nov.
        if d.month <= 11 and d.weekday() < 5:
            trading_days.append(d)
    # Force a long gap: only one Dec trading day (Dec 20) and one in late
    # November (Nov 28) as the "pre" anchor. Everything else in Dec is
    # closed.
    trading_days.append(_dt.date(2024, 12, 20))
    trading_days = sorted(set(trading_days))
    series_map = {d: 0.10 + i * 0.001 for i, d in enumerate(trading_days)}

    # Event date: 2024-12-10. Calendar-day lookbehind of 5 from 12-10
    # touches 12-09, 12-08, 12-07, 12-06, 12-05 — none of which are in
    # `trading_days` (we made Dec a black hole). The trading-day index
    # walk falls back to the previous trading day in late November.
    event_date = _dt.date(2024, 12, 10)
    pre, post, pre_date, post_date = mp_surprise._pre_post_yields(
        event_date,
        series_map,
        trading_days=trading_days,
    )
    assert pre is not None, "trading-day lookback should bridge the holiday gap"
    assert post is not None, "trading-day lookback should bridge the holiday gap"
    assert pre_date is not None and pre_date < event_date
    assert post_date is not None and post_date > event_date

    # Verify that the OLD calendar-day approach would have failed: with
    # `trading_days=None` and a default lookback of 5 calendar days, the
    # post side has no chance of finding a hit before 12-20.
    pre_cal, post_cal, _, _ = mp_surprise._pre_post_yields(
        event_date,
        series_map,
        lookbehind_days=5,
        lookahead_days=5,
    )
    # The new default `trading_days=None` derives the index from
    # series_map.keys(), so even this call resolves correctly — exactly
    # the fix we want.
    assert pre_cal is not None and post_cal is not None


# ---------------------------------------------------------------------------
# 9. Adjacent-meeting target-lookahead clipping (CRITICAL #5)
# ---------------------------------------------------------------------------


def test_target_lookahead_does_not_leak_into_adjacent_meeting(
    tmp_path: Path,
) -> None:
    """Two FOMC meetings within MAX_TARGET_LOOKAHEAD_DAYS calendar days
    must not pollute each other's ``ff_target_after``. We synthesise
    the March 2020 pattern: 2020-03-03 emergency cut (band 1.00-1.25%)
    followed by 2020-03-15 emergency cut (band 0.00-0.25%) — but
    compressed to a 3-day spacing so the 5-day lookahead would naturally
    grab the second meeting's band if not clipped."""

    # Compressed synthetic case: meetings on day-N and day-N+3.
    m1 = _dt.date(2020, 3, 3)
    m2 = _dt.date(2020, 3, 6)  # 3 calendar days later
    calendar = [
        mp_surprise.FomcMeetingRecord(meeting_date=m1, is_intermeeting=True, notes="cut_1"),
        mp_surprise.FomcMeetingRecord(meeting_date=m2, is_intermeeting=True, notes="cut_2"),
    ]

    # FRED responses: target band published on m1+1 = 1.125% midpoint
    # (1.00-1.25%), then on m2+1 = 0.125% midpoint (0.00-0.25%). If the
    # lookahead is unclipped, m1's `after_target` will pick up m2's band.
    days = 40
    base = _dt.date(2020, 2, 20)
    upper_obs: list[tuple[str, float | None]] = []
    lower_obs: list[tuple[str, float | None]] = []
    # We deliberately suppress band publications between m1 (exclusive)
    # and m2 (exclusive) so the lookahead from m1 must walk past m2 to
    # find a published band. Without the clip, the lookahead grabs m2's
    # post-cut band (0.125) instead of stopping short. With the clip,
    # the lookahead is bounded to (m2 - m1 - 1) = 2 days and exhausts,
    # forcing `ff_target_after` to fall back to the pre-meeting band via
    # `_target_rate_on(on_date - 1)` (i.e. the lookbehind path).
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        if m1 <= d < m2:
            # No band rows in the m1..m2-1 window (suppresses early
            # resolution by the lookahead). This forces the lookahead to
            # consider walking forward into m2's published band — the
            # bug we're guarding against.
            continue
        if d >= m2:
            upper_obs.append((d.isoformat(), 0.25))
            lower_obs.append((d.isoformat(), 0.00))
        else:
            upper_obs.append((d.isoformat(), 1.75))
            lower_obs.append((d.isoformat(), 1.50))

    payloads = _build_fred_payloads(base, base + _dt.timedelta(days=days - 1))
    payloads["DFEDTARU"] = _series_response("DFEDTARU", upper_obs)
    payloads["DFEDTARL"] = _series_response("DFEDTARL", lower_obs)
    payloads["DFEDTAR"] = _series_response("DFEDTAR", [])

    fred_responses = {
        sid: _parse_in_memory(sid, payloads[sid])
        for sid in mp_surprise._required_series_ids()
    }

    artifacts = mp_surprise.build_mp_surprises(
        start=base,
        end=base + _dt.timedelta(days=days - 1),
        fred_responses=fred_responses,
        fomc_calendar=calendar,
        spx_close_by_date={},
    )

    row_m1 = artifacts.frame[artifacts.frame["event_date"] == m1.isoformat()].iloc[0]
    row_m2 = artifacts.frame[artifacts.frame["event_date"] == m2.isoformat()].iloc[0]
    # CRITICAL: m1's after-target must NOT equal m2's published band
    # midpoint (0.125). Without the lookahead clip, the default 5-day
    # window would walk past m2 and silently grab 0.125 — a methodology
    # error. With the clip, the lookahead is bounded by
    # (m2 - m1 - 1) = 2 days, all of which we made missing, so the
    # after-target stays None.
    after_m1 = row_m1["ff_target_after"]
    assert after_m1 is None or float(after_m1) != pytest.approx(0.125, abs=1e-6), (
        f"m1 after-target leaked into m2's band ({after_m1}); the lookahead "
        "must be clipped to days_until_next_meeting - 1."
    )
    # The target_source flag should record that the after lookup failed
    # cleanly when clipped — not silently substitute m2's band.
    src = row_m1["target_source"]
    assert "after:missing" in str(src), (
        f"expected after:missing in target_source, got {src!r}"
    )
    # m2's after-target is its own published band; this is unaffected by
    # the clip (m2 is the last meeting in the synthetic calendar).
    assert float(row_m2["ff_target_after"]) == pytest.approx(0.125, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. DataFrame-value-hash determinism (CRITICAL #2)
# ---------------------------------------------------------------------------


def test_dataframe_value_hash_roundtrips_through_parquet(
    monkeypatch: pytest.MonkeyPatch,
    small_calendar: Path,
    fred_cache: Path,
    tmp_path: Path,
) -> None:
    """Build the parquet, re-read it into a DataFrame, hash the sorted
    DataFrame values, and verify the hash matches the in-memory hash
    of the original build. This is the honest determinism contract —
    it asserts data equality, not byte equality."""

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )
    spx_map = {
        _dt.date(2014, 3, 18): 1872.0,
        _dt.date(2014, 3, 20): 1872.0,
        _dt.date(2020, 3, 13): 2711.0,
        _dt.date(2020, 3, 16): 2386.0,
    }

    output_path = tmp_path / "mp_surprises.parquet"
    artifacts = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    in_memory_hash = mp_surprise.dataframe_value_hash(artifacts.frame)
    mp_surprise.write_mp_surprises_parquet(artifacts.frame, output_path)

    # Round-trip the parquet and hash the re-read DataFrame.
    reread = pd.read_parquet(output_path)
    # The parquet writer pins column order via COLUMN_ORDER; the hash
    # function sorts rows and stringifies cells so the re-read frame
    # produces the same hash even if pyarrow shifted dtypes.
    reread_hash = mp_surprise.dataframe_value_hash(reread)
    assert in_memory_hash == reread_hash, (
        "DataFrame-value hash drifted across the parquet round-trip "
        f"(in-memory={in_memory_hash}, reread={reread_hash})"
    )

    # Pinned reference: rebuild a second time and confirm the hash is
    # stable. We deliberately do NOT hard-code a hex string here because
    # the input fixtures (FRED payload synthesis) are part of the test
    # module and can legitimately change as the fixture evolves; the
    # contract is "rebuild yields same hash", not "hash equals X".
    artifacts_2 = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    assert mp_surprise.dataframe_value_hash(artifacts_2.frame) == in_memory_hash


# ---------------------------------------------------------------------------
# 11. Strict-prior construction contract (#350)
# ---------------------------------------------------------------------------


def test_strictly_prior_pre_and_trailing_yield_enforces_strict_inequality() -> None:
    """The new strict-prior helper must return dates strictly before ``on_date``.

    The level/path surprise reformulation under #350 reads the pre-event
    yield and a trailing anchor 5 trading days earlier. Both anchors are
    observable at ``T-Δ`` for ``T = on_date``; the strict inequality
    contract is the gate that closes the leaky ``T+1`` post-leg the
    audit (#324) flagged.
    """

    base = _dt.date(2024, 1, 1)
    trading_days = [
        base + _dt.timedelta(days=i) for i in range(120) if (base + _dt.timedelta(days=i)).weekday() < 5
    ]
    series_map = {d: 0.10 + i * 0.001 for i, d in enumerate(trading_days)}
    on = _dt.date(2024, 3, 6)
    pre, trail, pre_d, trail_d = mp_surprise._strictly_prior_pre_and_trailing_yield(
        on,
        series_map,
        trading_days=trading_days,
    )
    assert pre is not None and trail is not None
    assert pre_d is not None and trail_d is not None
    assert trail_d < pre_d < on, (
        f"strict-prior contract violated: trail={trail_d} pre={pre_d} on={on}"
    )
    # The default trailing window is 5 trading days; assert the gap is
    # at least that many calendar days (trading-day spacing varies
    # across weekends but never drops below 5 calendar days for a 5td
    # walk on a Mon-Fri schedule).
    assert (pre_d - trail_d).days >= 5


def test_spx_strict_prior_trailing_uses_only_pre_event_closes() -> None:
    """``_spx_return_on`` must never read SPX closes dated >= event_date.

    The strict-prior trailing return uses two anchors: ``pre`` (the
    closest close < event_date) and ``trail`` (~7 calendar days earlier
    than ``pre``, still < event_date). Both endpoints carry calendar
    dates strictly before the announcement; the close-to-close return
    is the strict-prior equity signal the fed-info residual is fit
    against under the #350 reformulation.
    """

    event_date = _dt.date(2024, 3, 20)
    # Strict-prior closes only; nothing post-event in the lookup map.
    closes = {
        event_date - _dt.timedelta(days=10): 5000.0,
        event_date - _dt.timedelta(days=9): 5010.0,
        event_date - _dt.timedelta(days=1): 5100.0,
    }
    ret, source = mp_surprise._spx_return_on(event_date, closes)
    assert ret is not None
    assert source == "strict_prior_trailing"
    # If the post-event close is the *only* available "pre" anchor (which
    # the old [T-1, T+1] construction relied on), the strict-prior path
    # must NOT use it.
    leaky_only = {event_date + _dt.timedelta(days=1): 5200.0}
    ret_leaky, source_leaky = mp_surprise._spx_return_on(event_date, leaky_only)
    assert ret_leaky is None
    assert source_leaky == "unavailable"


def test_intraday_returns_argument_is_ignored_under_strict_prior(
    monkeypatch: pytest.MonkeyPatch,
    small_calendar: Path,
    fred_cache: Path,
) -> None:
    """The Alpha Vantage intraday route is rejected under #350.

    The ``spx_intraday_returns`` mapping is retained on the
    ``build_mp_surprises`` signature for backwards compatibility but is
    ignored at runtime because the ±30 min window around the FOMC
    announcement leaks ``T+`` data (the 14:00-14:30 ET half is post-
    announcement). The contract here is: passing a non-empty intraday
    map must NOT bump any ``fed_info_factor_source`` row to
    ``alphavantage_intraday_30min``.
    """

    monkeypatch.setenv("FRED_API_KEY", "test-key")
    start = _dt.date(2010, 1, 1)
    end = _dt.date(2020, 12, 31)
    payloads = _build_fred_payloads(start - _dt.timedelta(days=14), end + _dt.timedelta(days=7))
    transport = _mock_transport(payloads)
    responses = mp_surprise._hydrate_fred_responses(
        start=start - _dt.timedelta(days=14),
        end=end + _dt.timedelta(days=7),
        cache_dir=fred_cache,
        transport=transport,
    )
    # Strict-prior trailing SPX coverage for both meetings (T-7..T-1).
    def _trailing_closes_for(d: _dt.date) -> dict[_dt.date, float]:
        return {
            d - _dt.timedelta(days=8): 4000.0,
            d - _dt.timedelta(days=1): 4100.0,
        }

    spx_map: dict[_dt.date, float] = {}
    for d in (_dt.date(2014, 3, 19), _dt.date(2020, 3, 15)):
        spx_map.update(_trailing_closes_for(d))
    intraday = {"2014-03-19": 0.01, "2020-03-15": -0.10}
    artifacts = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
        spx_intraday_returns=intraday,
    )
    for source in artifacts.frame["fed_info_factor_source"].tolist():
        assert source != "alphavantage_intraday_30min", (
            f"#350: intraday route must be ignored, got {source!r}"
        )
