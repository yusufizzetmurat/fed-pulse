"""Strict-backward semantics + byte stability for the rates panel builder (#291)."""

from __future__ import annotations

import datetime as _dt

import pytest

from app.data.rates_panel import (
    COLUMN_BY_SERIES,
    COLUMN_ORDER,
    PUBLICATION_DELAY_DAYS,
    RATES_SERIES_IDS,
    RatesPanelLookup,
    build_rates_panel,
    dataframe_value_hash,
    load_rates_panel_lookup,
    write_rates_panel_parquet,
)
from app.services.fred_client import FredObservation, FredSeriesResponse


def _make_response(series_id: str, rows: list[tuple[str, float]]) -> FredSeriesResponse:
    observations = [
        FredObservation(
            date=d,
            value=v,
            realtime_start="2026-05-26",
            realtime_end="2026-05-26",
        )
        for d, v in rows
    ]
    return FredSeriesResponse(
        series_id=series_id,
        realtime_start="2026-05-26",
        realtime_end="2026-05-26",
        observation_start=rows[0][0] if rows else "",
        observation_end=rows[-1][0] if rows else "",
        count=len(rows),
        observations=observations,
    )


def _stub_fred_responses() -> dict[str, FredSeriesResponse]:
    """Pin a small synthetic set of rates observations for the test bench.

    Every series carries the same three reference dates so the lookup
    semantics can be inspected without juggling differing calendars per
    series. The values are picked so each series has a recognisable
    fingerprint (treas_2y at 4.50 / 4.60 / 4.70, treas_5y at 4.10 / ...
    etc.) and the strict-backward lookups have unambiguous answers.
    """

    base_dates = ("2026-05-19", "2026-05-20", "2026-05-21")
    series_values = {
        "DGS1": [4.90, 4.95, 5.00],
        "DGS2": [4.50, 4.60, 4.70],
        "DGS5": [4.10, 4.15, 4.20],
        "DGS10": [4.30, 4.35, 4.40],
        "T10Y2Y": [-0.20, -0.25, -0.30],
        "T10Y3M": [-0.40, -0.45, -0.50],
        "DFEDTARU": [5.50, 5.50, 5.50],
        "DFEDTARL": [5.25, 5.25, 5.25],
    }
    return {
        sid: _make_response(sid, list(zip(base_dates, values)))
        for sid, values in series_values.items()
    }


def test_build_rates_panel_pins_column_order_and_publication_delay() -> None:
    """The schema and per-series delay contract are stable."""

    assert PUBLICATION_DELAY_DAYS == 0, (
        "Rates panel publishes same-day on FRED; the delay must stay 0."
    )
    assert set(COLUMN_BY_SERIES) == set(RATES_SERIES_IDS), (
        "Every FRED series must map to a parquet column."
    )
    for col in COLUMN_BY_SERIES.values():
        assert col in COLUMN_ORDER, f"missing column {col!r} in COLUMN_ORDER"


def test_value_strictly_before_excludes_event_day() -> None:
    """A value published on event_date is NOT visible to a strict-backward lookup."""

    responses = _stub_fred_responses()
    artifacts = build_rates_panel(
        start=_dt.date(2026, 5, 19),
        end=_dt.date(2026, 5, 22),
        fred_responses=responses,
    )
    frame = artifacts.frame
    # Row dated 2026-05-21 must reflect the 2026-05-20 observation
    # (strictly before): DGS2 = 4.60.
    row = frame.loc[frame["as_of_date"] == "2026-05-21"].iloc[0]
    assert row["treas_2y"] == pytest.approx(4.60), (
        "Strict-backward lookup at 2026-05-21 must see 2026-05-20's "
        "DGS2 (4.60) rather than the same-day 4.70."
    )


def test_lookup_yield_strictly_before_excludes_target_date() -> None:
    """RatesPanelLookup.yield_strictly_before drops same-day observations."""

    responses = _stub_fred_responses()
    artifacts = build_rates_panel(
        start=_dt.date(2026, 5, 19),
        end=_dt.date(2026, 5, 22),
        fred_responses=responses,
    )
    # Construct the lookup in-memory from the build artifacts. Avoids a
    # disk round-trip while exercising the same RatesPanelLookup shape
    # the production loader returns.
    dates_by_column = {}
    values_by_column = {}
    for sid, column in COLUMN_BY_SERIES.items():
        series_pairs = sorted(
            (
                _dt.date.fromisoformat(obs.date),
                float(obs.value),
            )
            for obs in responses[sid].observations
            if obs.value is not None
        )
        dates_by_column[column] = tuple(p[0] for p in series_pairs)
        values_by_column[column] = tuple(p[1] for p in series_pairs)
    lookup = RatesPanelLookup(
        dates_by_column=dates_by_column, values_by_column=values_by_column
    )

    # Strictly before 2026-05-21 should see 2026-05-20's value (4.60).
    assert lookup.yield_strictly_before("treas_2y", _dt.date(2026, 5, 21)) == pytest.approx(
        4.60
    )
    # ``on_or_before`` includes the target date itself (4.70).
    assert lookup.yield_on_or_before("treas_2y", _dt.date(2026, 5, 21)) == pytest.approx(
        4.70
    )
    # A target before every observation returns None.
    assert lookup.yield_strictly_before("treas_2y", _dt.date(2026, 5, 19)) is None


def test_value_hash_is_deterministic_across_runs() -> None:
    """Two builds on the same inputs produce the same value hash."""

    responses = _stub_fred_responses()
    a = build_rates_panel(
        start=_dt.date(2026, 5, 19),
        end=_dt.date(2026, 5, 22),
        fred_responses=responses,
    )
    b = build_rates_panel(
        start=_dt.date(2026, 5, 19),
        end=_dt.date(2026, 5, 22),
        fred_responses=responses,
    )
    assert a.value_hash == b.value_hash
    assert dataframe_value_hash(a.frame) == dataframe_value_hash(b.frame)


def test_round_trip_via_parquet_preserves_lookup(tmp_path) -> None:
    """Writing + reloading the parquet returns the same strict-backward answers."""

    responses = _stub_fred_responses()
    artifacts = build_rates_panel(
        start=_dt.date(2026, 5, 19),
        end=_dt.date(2026, 5, 22),
        fred_responses=responses,
    )
    parquet_path = tmp_path / "rates_panel.parquet"
    write_rates_panel_parquet(artifacts.frame, parquet_path)

    lookup = load_rates_panel_lookup(parquet_path)
    # The parquet snapshots the per-business-day frame, so the
    # lookup's strict-backward semantics now operate against the
    # *as-of dates* embedded in the parquet rows. The 2026-05-21
    # row's treas_2y is 4.60 (the strict-backward value computed on
    # build) and ``yield_on_or_before(2026-05-21)`` returns it.
    assert lookup.yield_on_or_before("treas_2y", _dt.date(2026, 5, 21)) == pytest.approx(
        4.60
    )


def test_missing_parquet_degrades_to_empty_lookup(tmp_path) -> None:
    """A missing rates panel falls back to an empty lookup."""

    missing = tmp_path / "absent_rates_panel.parquet"
    lookup = load_rates_panel_lookup(missing)
    assert lookup.yield_strictly_before("treas_2y", _dt.date(2026, 5, 21)) is None
    assert lookup.yield_on_or_before("treas_2y", _dt.date(2026, 5, 21)) is None
