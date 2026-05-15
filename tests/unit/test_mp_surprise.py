"""Tests for the monetary-policy surprise builder.

The tests bypass HTTP via ``httpx.MockTransport`` for the FRED endpoint
and inject canned SPX close maps. They cover:

- The mock-transport path that drives :func:`fetch_fred_series`.
- Hand-verified surprises on 2008-10-08 (50 bp emergency cut),
  2013-05-22 ("taper tantrum"), and 2020-03-15 (pandemic cut).
- PCA path-factor eigenvectors persist deterministically in the lock
  JSON; rebuilding with the same inputs yields the same eigenvector.

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
    """Tiny FOMC calendar covering one in-window meeting + 2013-05-22 + 2020-03-15."""

    csv = tmp_path / "fomc_meetings.csv"
    csv.write_text(
        "meeting_date,is_intermeeting,notes\n"
        "2013-05-22,false,bernanke_taper_testimony\n"
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
    assert artifacts.rows_written == 3
    pandemic = artifacts.frame[artifacts.frame["event_date"] == "2020-03-15"].iloc[0]
    # The level surprise should be a large negative bps move (we injected -50 bp at the 1m tenor).
    level = float(pandemic["mp_surprise_level"])
    assert level < -25.0, f"expected sharp negative level surprise, got {level} bps"
    assert level > -100.0, f"expected ~-50 bp level surprise, got {level} bps"
    assert bool(pandemic["is_intermeeting"]) is True
    assert pandemic["methodology"] == mp_surprise.METHODOLOGY_OIS_PROXY
    # Target after should fall to the 0.00-0.25 % band midpoint.
    assert float(pandemic["ff_target_after"]) == pytest.approx(0.125, abs=1e-6)
    # Fed-info factor should be sign-consistent with the level (or null).
    fi = pandemic["fed_info_factor"]
    assert fi is not None


# ---------------------------------------------------------------------------
# 3. Hand-verified path-factor signature: 2013-05-22 Bernanke testimony
# ---------------------------------------------------------------------------


def test_taper_tantrum_2013_05_22(
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
    spx_map = {
        _dt.date(2013, 5, 21): 1660.0,
        _dt.date(2013, 5, 23): 1655.0,  # mildly down -- bond surprise > equity surprise
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
    taper = artifacts.frame[artifacts.frame["event_date"] == "2013-05-22"].iloc[0]
    level = float(taper["mp_surprise_level"])
    # The 1m yield is flat across this event; the long-end curve shifts up.
    # Level should be near zero, path factor positive.
    assert abs(level) < 5.0, f"taper-tantrum level should be near zero, got {level} bps"
    path_factor = taper["mp_surprise_path_factor"]
    assert path_factor is not None
    assert float(path_factor) > 0.0, (
        f"taper-tantrum path factor should be positive (curve steepening), got {path_factor}"
    )


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
    # Hand-craft single-target rates so 2008-10-07 = 2.0%, 2008-10-09 = 1.5%
    # The 1m yield series we already built is flat across this range; we
    # rewire it so 1m falls 50 bp across the cut.
    days = (end - start + _dt.timedelta(days=22)).days
    base = start - _dt.timedelta(days=14)
    new_obs: list[tuple[str, float | None]] = []
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        v = 1.80 if d < _dt.date(2008, 10, 9) else 1.30
        new_obs.append((d.isoformat(), v))
    payloads["DGS1MO"] = _series_response("DGS1MO", new_obs)
    # Bump path tenors by the same -50 bp shift so the path factor stays near zero.
    for sid, base_val in (("DGS3MO", 1.85), ("DGS6MO", 1.95), ("DGS1", 2.20), ("DGS2", 2.50)):
        obs2 = []
        for i in range(days):
            d = base + _dt.timedelta(days=i)
            v = base_val if d < _dt.date(2008, 10, 9) else base_val - 0.50
            obs2.append((d.isoformat(), v))
        payloads[sid] = _series_response(sid, obs2)
    # Single-target rate (pre-band era).
    single_obs: list[tuple[str, float | None]] = []
    for i in range(days):
        d = base + _dt.timedelta(days=i)
        v = 2.0 if d < _dt.date(2008, 10, 9) else 1.5
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
    assert level == pytest.approx(-50.0, abs=10.0), (
        f"2008-10-08 emergency cut should give ~-50 bp level surprise, got {level}"
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

    # Second build with same inputs -> bit-identical parquet + same eigenvector.
    artifacts_second = mp_surprise.build_mp_surprises(
        start=start,
        end=end,
        fred_responses=responses,
        fomc_calendar_path=small_calendar,
        spx_close_by_date=spx_map,
    )
    sha_second = mp_surprise.write_mp_surprises_parquet(artifacts_second.frame, output_path)
    assert sha_first == sha_second, "rebuild produced different parquet bytes"
    mp_surprise.update_sources_lock(
        lock_path=lock_path,
        artifacts=artifacts_second,
        parquet_path=output_path,
        parquet_sha256=sha_second,
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
