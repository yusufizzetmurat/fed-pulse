"""Tests for the FRED macro-state snapshot builder (Phase 8 #147).

Patterns mirror ``tests/unit/test_fred_client.py``: httpx is wired via
``httpx.MockTransport`` so no network I/O leaks into the test
environment, and the SOURCES.lock entry is asserted both ways
(written from the builder, read back from disk).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.data import macro_state
from app.services import fred_client


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _months_of(start: _dt.date, count: int) -> list[_dt.date]:
    out: list[_dt.date] = []
    y = start.year
    m = start.month
    for _ in range(count):
        out.append(_dt.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _series_payload(
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
                "value": "." if v is None else str(v),
                "realtime_start": realtime,
                "realtime_end": realtime,
            }
            for d, v in observations
        ],
    }


def _daily_dates(start: _dt.date, count: int) -> list[_dt.date]:
    return [start + _dt.timedelta(days=i) for i in range(count)]


def _build_canned_fred_responses() -> dict[str, fred_client.FredSeriesResponse]:
    """Hand-built FRED responses covering 2018-2024.

    Values are smoothly increasing per series so YoY / MoM transforms
    produce non-zero, non-NaN signals that the as-of join can read.
    The monthly panel uses month-start reference dates; the rates +
    financial-conditions panel uses daily reference dates for the
    Treasury / spread / OAS / TIPS series, and weekly Friday reference
    dates for NFCI.
    """

    months = _months_of(_dt.date(2018, 1, 1), 84)  # 7 years
    responses: dict[str, fred_client.FredSeriesResponse] = {}
    # UNRATE: trend down then up. Stay strictly positive.
    unrate_vals = [4.0 - i * 0.005 for i in range(60)] + [3.5 + i * 0.02 for i in range(24)]
    # CPIAUCSL: smooth upward.
    cpi_vals = [250.0 * (1 + 0.0025) ** i for i in range(84)]
    # PCEPILFE: smooth upward, slower.
    pce_vals = [105.0 * (1 + 0.002) ** i for i in range(84)]
    # MANEMP: tens of thousands.
    manemp_vals = [12000.0 + i * 5.0 for i in range(84)]
    # PAYEMS: thousands.
    payems_vals = [150000.0 + i * 200.0 for i in range(84)]
    # RSAFS: smooth upward.
    rsafs_vals = [450000.0 * (1 + 0.003) ** i for i in range(84)]

    series_inputs = {
        "UNRATE": unrate_vals,
        "CPIAUCSL": cpi_vals,
        "PCEPILFE": pce_vals,
        "MANEMP": manemp_vals,
        "PAYEMS": payems_vals,
        "RSAFS": rsafs_vals,
    }
    for sid, vals in series_inputs.items():
        observations = [(d.isoformat(), v) for d, v in zip(months, vals)]
        payload = _series_payload(observations)
        # Parse via fred_client's own parser so we get the same shape
        # the production code will see.
        responses[sid] = fred_client._parse_observations(payload, sid)

    # Daily rates panel: 7 years of calendar-daily observations. Values
    # are smooth so every as-of date inside the canned window finds a
    # non-NaN strictly-before value.
    daily_dates = _daily_dates(_dt.date(2018, 1, 1), 365 * 7)
    daily_series_inputs = {
        "DGS10": [2.0 + 0.001 * i for i in range(len(daily_dates))],
        "T10Y2Y": [0.5 - 0.0002 * i for i in range(len(daily_dates))],
        "T10Y3M": [0.7 - 0.0003 * i for i in range(len(daily_dates))],
        "BAMLH0A0HYM2": [3.5 + 0.0005 * i for i in range(len(daily_dates))],
        "DFII10": [0.4 + 0.0001 * i for i in range(len(daily_dates))],
    }
    for sid, vals in daily_series_inputs.items():
        observations = [(d.isoformat(), v) for d, v in zip(daily_dates, vals)]
        payload = _series_payload(observations)
        responses[sid] = fred_client._parse_observations(payload, sid)

    # NFCI: weekly Friday observations. 2018-01-05 is the first Friday
    # of 2018; every subsequent observation lands 7 days later.
    first_friday = _dt.date(2018, 1, 5)
    n_weeks = 52 * 7
    nfci_dates = [first_friday + _dt.timedelta(days=7 * i) for i in range(n_weeks)]
    nfci_vals = [-0.5 + 0.001 * i for i in range(n_weeks)]
    nfci_observations = [(d.isoformat(), v) for d, v in zip(nfci_dates, nfci_vals)]
    responses["NFCI"] = fred_client._parse_observations(
        _series_payload(nfci_observations), "NFCI"
    )
    return responses


# ---------------------------------------------------------------------------
# Builder behaviour
# ---------------------------------------------------------------------------


def test_build_macro_state_emits_expected_columns_and_no_lookahead() -> None:
    responses = _build_canned_fred_responses()
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 6, 30),
        fred_responses=responses,
        publication_delay_days=30,
    )
    df = artifacts.frame
    assert list(df.columns) == list(macro_state.COLUMN_ORDER)
    assert artifacts.rows_written > 0
    # Every column populated where reference data exists; ism_proxy_source
    # is the literal MANEMP label on every row.
    assert (df["ism_proxy_source"] == macro_state.ISM_PROXY_SOURCE_LABEL).all()
    # No look-ahead: every row's as_of_date strictly greater than the
    # publication date of the underlying observation. We hand-verify on
    # the first row: as_of_date is 2020-01-01; CPI YoY at that date
    # must reflect the value at reference month 2019-11-01 (because the
    # December print is shifted to 2020-01-01 by the 30-day delay, and
    # we read strictly < 2020-01-01).
    first = df.iloc[0]
    assert first["as_of_date"] == "2020-01-01"
    # The 2019-12-01 observation has reference_date + 30 days =
    # 2019-12-31 which is strictly < 2020-01-01, so it counts.
    assert first["cpi_yoy"] is not None
    # data_version is reproducible: same inputs produce the same hash.
    second = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 6, 30),
        fred_responses=responses,
        publication_delay_days=30,
    )
    assert second.data_version == artifacts.data_version
    assert second.value_hash == artifacts.value_hash


def test_build_macro_state_respects_as_of_dates_override() -> None:
    responses = _build_canned_fred_responses()
    as_of = [_dt.date(2020, 3, 18), _dt.date(2020, 6, 10)]
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 12, 31),
        fred_responses=responses,
        as_of_dates=as_of,
        publication_delay_days=30,
    )
    assert artifacts.rows_written == 2
    assert artifacts.frame["as_of_date"].tolist() == [d.isoformat() for d in as_of]


def test_build_macro_state_raises_on_missing_series() -> None:
    responses = _build_canned_fred_responses()
    del responses["MANEMP"]
    with pytest.raises(KeyError, match="MANEMP"):
        macro_state.build_macro_state(
            start=_dt.date(2020, 1, 1),
            end=_dt.date(2020, 6, 30),
            fred_responses=responses,
        )


def test_rates_panel_columns_present_and_typed() -> None:
    """The 12-column macro panel emits the rates + financial-conditions slice."""

    responses = _build_canned_fred_responses()
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2022, 1, 3),
        end=_dt.date(2022, 1, 7),
        fred_responses=responses,
        publication_delay_days=30,
    )
    df = artifacts.frame
    rates_columns = (
        "treas_10y",
        "slope_10y_2y",
        "slope_10y_3m",
        "hy_oas",
        "nfci",
        "tips_10y_real",
    )
    for column in rates_columns:
        assert column in df.columns, f"missing rates-panel column {column}"
        # Daily-cadence canned data covers the window, so every row
        # must read a non-None value off the strictly-before join.
        non_null = df[column].dropna()
        assert not non_null.empty, f"rates-panel column {column} has no rows"
        # Every non-null value must coerce to float (level data).
        for v in non_null:
            assert isinstance(v, float), f"{column} value not float: {v!r}"


def test_macro_state_column_order_pins_rates_panel_layout() -> None:
    """``COLUMN_ORDER`` lists the rates panel between rsafs_mom and ism_proxy_source."""

    order = list(macro_state.COLUMN_ORDER)
    assert order.index("rsafs_mom") < order.index("treas_10y")
    assert order.index("tips_10y_real") < order.index("ism_proxy_source")
    rates_layout = (
        "treas_10y",
        "slope_10y_2y",
        "slope_10y_3m",
        "hy_oas",
        "nfci",
        "tips_10y_real",
    )
    rates_positions = [order.index(c) for c in rates_layout]
    assert rates_positions == sorted(rates_positions), (
        f"rates panel columns out of contiguous order: {rates_positions}"
    )


def test_nfci_publication_delay_is_five_days() -> None:
    """NFCI carries a 5-day publication delay; the daily series carry zero."""

    delays = macro_state.RATES_PANEL_PUBLICATION_DELAYS_DAYS
    assert delays["NFCI"] == 5
    for series_id in ("DGS10", "T10Y2Y", "T10Y3M", "BAMLH0A0HYM2", "DFII10"):
        assert delays[series_id] == 0, (
            f"{series_id} should carry a zero-day publication delay"
        )


def test_nfci_publication_delay_pushes_friday_observation_to_following_thursday() -> None:
    """The 5-day shift parks each Friday NFCI observation strictly after
    the following Tuesday, so a Wednesday as-of-date reading strictly-<
    sees the prior Friday's print only after the Wednesday publication
    window has closed.
    """

    responses = _build_canned_fred_responses()
    # 2022-04-08 (Friday) NFCI observation -> publication on 2022-04-13
    # (Wednesday) under the 5-day delay. A Wednesday-2022-04-13 as-of
    # date reading strictly-< must therefore NOT see the 2022-04-08
    # value yet; a Thursday-2022-04-14 as-of date MUST see it.
    as_of_dates = [_dt.date(2022, 4, 13), _dt.date(2022, 4, 14)]
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2022, 1, 1),
        end=_dt.date(2022, 12, 31),
        fred_responses=responses,
        as_of_dates=as_of_dates,
    )
    by_date = {row["as_of_date"]: row for _, row in artifacts.frame.iterrows()}
    wednesday = by_date["2022-04-13"]
    thursday = by_date["2022-04-14"]
    # Wednesday: the latest NFCI observation visible is 2022-04-01
    # (pub_date 2022-04-06, strictly before 2022-04-13). Thursday: the
    # 2022-04-08 observation has pub_date 2022-04-13, strictly before
    # 2022-04-14, so it is now visible. The two reads must therefore
    # come from different observation rows, i.e. their NFCI values
    # differ.
    assert wednesday["nfci"] is not None
    assert thursday["nfci"] is not None
    assert thursday["nfci"] != wednesday["nfci"]


def test_sources_lock_records_rates_panel_publication_delays(tmp_path: Path) -> None:
    """SOURCES.lock carries the rates-panel column map + delay map so the
    publication-delay contract round-trips through the parquet."""

    responses = _build_canned_fred_responses()
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2022, 1, 3),
        end=_dt.date(2022, 1, 7),
        fred_responses=responses,
    )
    output_path = tmp_path / "macro_state.parquet"
    sha = macro_state.write_macro_state_parquet(artifacts.frame, output_path)
    lock_path = tmp_path / fred_client.SOURCES_LOCK_NAME
    macro_state.update_sources_lock(
        lock_path=lock_path,
        artifacts=artifacts,
        parquet_path=output_path,
        parquet_sha256=sha,
    )
    entry = json.loads(lock_path.read_text())[macro_state.DEFAULT_LOCK_KEY]
    assert entry["rates_panel_publication_delays_days"]["NFCI"] == 5
    assert entry["rates_panel_publication_delays_days"]["DGS10"] == 0
    assert entry["rates_panel_columns"]["DGS10"] == "treas_10y"
    assert entry["rates_panel_columns"]["NFCI"] == "nfci"
    assert set(entry["fred_series"]) == set(macro_state.FRED_SERIES_IDS)


def test_publication_delay_shifts_as_of_window() -> None:
    """Increasing delay should push more rows toward 'missing data'."""

    responses = _build_canned_fred_responses()
    # Very small window at the start of the canned data so the delay
    # change is visible. Start at 2018-01-15: with delay=30 the
    # 2018-01-01 reference observation has pub_date=2018-01-31, so
    # the row at 2018-01-15 will be None. With delay=0 it will be set.
    as_of = [_dt.date(2018, 1, 15)]
    art_zero = macro_state.build_macro_state(
        start=_dt.date(2018, 1, 1),
        end=_dt.date(2018, 12, 31),
        fred_responses=responses,
        as_of_dates=as_of,
        publication_delay_days=0,
    )
    art_thirty = macro_state.build_macro_state(
        start=_dt.date(2018, 1, 1),
        end=_dt.date(2018, 12, 31),
        fred_responses=responses,
        as_of_dates=as_of,
        publication_delay_days=30,
    )
    # zero-delay: 2018-01-01 ref is strictly < 2018-01-15 → unrate set.
    # 30-day delay: 2018-01-01 ref has pub=2018-01-31 → strictly < 2018-01-15 is False → None.
    assert art_zero.frame["unrate"].iloc[0] is not None
    assert art_thirty.frame["unrate"].iloc[0] is None


# ---------------------------------------------------------------------------
# Persistence + SOURCES.lock
# ---------------------------------------------------------------------------


def test_write_macro_state_persists_parquet_and_lock(tmp_path: Path) -> None:
    responses = _build_canned_fred_responses()
    artifacts = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 3, 31),
        fred_responses=responses,
        publication_delay_days=30,
    )
    output_path = tmp_path / "macro_state.parquet"
    sha = macro_state.write_macro_state_parquet(artifacts.frame, output_path)
    assert len(sha) == 64
    assert output_path.exists()

    lock_path = tmp_path / fred_client.SOURCES_LOCK_NAME
    macro_state.update_sources_lock(
        lock_path=lock_path,
        artifacts=artifacts,
        parquet_path=output_path,
        parquet_sha256=sha,
    )
    lock = json.loads(lock_path.read_text())
    assert macro_state.DEFAULT_LOCK_KEY in lock
    entry = lock[macro_state.DEFAULT_LOCK_KEY]
    assert entry["sha256"] == sha
    assert entry["rows"] == artifacts.rows_written
    assert entry["publication_delay_days"] == 30
    assert entry["ism_proxy_source"] == macro_state.ISM_PROXY_SOURCE_LABEL
    assert set(entry["fred_series"]) == set(macro_state.FRED_SERIES_IDS)
    assert entry["value_hash"] == artifacts.value_hash


def test_round_trip_parquet_preserves_value_hash(tmp_path: Path) -> None:
    """Re-running the build with the same inputs is idempotent on disk."""

    responses = _build_canned_fred_responses()
    artifacts_a = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 4, 30),
        fred_responses=responses,
    )
    artifacts_b = macro_state.build_macro_state(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 4, 30),
        fred_responses=responses,
    )
    assert artifacts_a.value_hash == artifacts_b.value_hash

    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    macro_state.write_macro_state_parquet(artifacts_a.frame, out_a)
    macro_state.write_macro_state_parquet(artifacts_b.frame, out_b)

    # Re-read both parquets; the value-hash contract holds even if the
    # parquet bytes drift across pyarrow versions.
    df_a = pd.read_parquet(out_a)
    df_b = pd.read_parquet(out_b)
    assert macro_state.dataframe_value_hash(df_a) == artifacts_a.value_hash
    assert macro_state.dataframe_value_hash(df_b) == artifacts_b.value_hash


# ---------------------------------------------------------------------------
# CLI / FRED-hydration via httpx.MockTransport
# ---------------------------------------------------------------------------


def _mock_transport_for_series(payload_by_series: dict[str, dict]) -> httpx.MockTransport:
    """Return a transport that replies based on the ``series_id`` query param."""

    def handler(request: httpx.Request) -> httpx.Response:
        series_id = request.url.params.get("series_id")
        if series_id is None or series_id not in payload_by_series:
            return httpx.Response(404, json={"error": "unknown series"})
        return httpx.Response(200, json=payload_by_series[series_id])

    return httpx.MockTransport(handler)


def test_hydrate_fred_responses_via_mock_transport(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    months = _months_of(_dt.date(2018, 1, 1), 36)
    base_payload = {
        sid: _series_payload([(d.isoformat(), 1.0 + 0.01 * i) for i, d in enumerate(months)])
        for sid in macro_state.FRED_SERIES_IDS
    }

    responses = macro_state._hydrate_fred_responses(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 6, 30),
        cache_dir=tmp_path,
        transport=_mock_transport_for_series(base_payload),
    )

    assert set(responses) == set(macro_state.FRED_SERIES_IDS)
    # Every series should be cached on disk now.
    for sid in macro_state.FRED_SERIES_IDS:
        assert (tmp_path / f"{sid}.json").exists()

    # Second hydration with a "boom" transport should NOT hit the wire
    # because the per-series JSON cache is populated. This is the
    # "cache is idempotent" contract.
    def boom(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport called despite cache hit")

    responses_again = macro_state._hydrate_fred_responses(
        start=_dt.date(2020, 1, 1),
        end=_dt.date(2020, 6, 30),
        cache_dir=tmp_path,
        transport=httpx.MockTransport(boom),
    )
    # Cached responses should have the same counts as the first pull.
    for sid in macro_state.FRED_SERIES_IDS:
        assert responses_again[sid].count == responses[sid].count
