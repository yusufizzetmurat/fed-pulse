"""Unit tests for ``app.data.event_dataset_builder``.

Covers the methodological contract:

- α/β regression on a hand-built 252-day window
- Multi-horizon targets exist with the right sign for an engineered move
- Look-ahead guard: a prior window that includes ``as_of_ts`` raises
- No survivorship filter: zero-move events still emit a row
- Determinism: building twice yields byte-identical parquet
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.data import event_dataset_builder as edb


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_registry(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _make_trading_dates(start: _dt.date, n: int) -> list[_dt.date]:
    out: list[_dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


def _series_from_closes(dates: list[_dt.date], closes: list[float], volume: float = 1_000_000.0) -> edb._CloseSeries:
    return edb._CloseSeries(
        dates=list(dates),
        close=[float(c) for c in closes],
        volume=[float(volume)] * len(closes),
    )


def _registry_entry_for_statement(event_date: str, text: str = "Sample statement text.") -> dict:
    return {
        "record_id": hashlib.sha256(f"{event_date}|{text}".encode()).hexdigest()[:16],
        "source": "scraped_fed",
        "source_record_id": f"fomc_statements.json:{event_date}",
        "document_type": "Statement",
        "source_type": "fomc_statement",
        "event_date": event_date,
        "text": text,
        "mapped_label": "neutral",
        "axes": {"stance": "neutral", "time": None, "certainty": None, "factor": None, "topic": None},
        "multi_axis_extras": {},
        "sample_weight": 1.0,
    }


# ---------------------------------------------------------------------------
# α/β regression — hand-checked
# ---------------------------------------------------------------------------


def test_fit_market_model_recovers_known_alpha_beta() -> None:
    """When asset_return = 0.001 + 1.3 * bench_return (exact, no noise),
    OLS must recover (alpha, beta) ≈ (0.001, 1.3) on the 252-day window."""

    import math

    bench_returns = [math.sin(i / 7.0) * 0.005 for i in range(300)]
    asset_returns = [0.001 + 1.3 * r for r in bench_returns]

    alpha, beta = edb._fit_market_model(asset_returns[-252:], bench_returns[-252:])
    assert abs(beta - 1.3) < 1e-9, f"beta drifted: {beta}"
    assert abs(alpha - 0.001) < 1e-9, f"alpha drifted: {alpha}"


def test_market_model_for_event_uses_252d_window_ending_strictly_before_as_of() -> None:
    """The window must end on the last bar strictly before ``as_of``; the
    bar at ``as_of`` must not appear in the regression."""

    import math

    dates = _make_trading_dates(_dt.date(2020, 1, 1), 400)
    # Build closes so that returns match a known (alpha, beta).
    bench_closes = [100.0]
    asset_closes = [100.0]
    for i in range(1, 400):
        b_ret = math.sin(i / 11.0) * 0.004
        bench_closes.append(bench_closes[-1] * (1 + b_ret))
        # Inject a wildly different return at the event date so we can
        # detect leakage if it ever appears in the regression.
        if i == 350:
            asset_closes.append(asset_closes[-1] * 1.05)  # +5% on the event day
        else:
            a_ret = 0.0005 + 0.8 * b_ret
            asset_closes.append(asset_closes[-1] * (1 + a_ret))

    asset_series = _series_from_closes(dates, asset_closes)
    bench_series = _series_from_closes(dates, bench_closes)
    as_of = dates[350]  # event day

    fit = edb._market_model_for_event(asset_series, bench_series, as_of)
    assert fit is not None
    alpha, beta = fit
    # The event-day +5% spike should NOT contaminate the regression. So
    # we expect beta ≈ 0.8 and alpha ≈ 0.0005.
    assert abs(beta - 0.8) < 1e-2, f"beta = {beta}; window included the event"
    assert abs(alpha - 0.0005) < 1e-3, f"alpha = {alpha}; window included the event"


# ---------------------------------------------------------------------------
# Multi-horizon targets
# ---------------------------------------------------------------------------


def test_multi_horizon_targets_for_engineered_next_day_pop(tmp_path: Path) -> None:
    """An engineered +1% next-trading-day return must surface as
    positive realized_return at h=1, and abnormal_return == realized_return
    when asset == benchmark (alpha=0, beta=1)."""

    package = tmp_path / "package"
    package.mkdir()
    event_date = "2023-06-14"
    _write_registry(package / "registry_normalized.jsonl", [_registry_entry_for_statement(event_date)])

    # 1000-day series so we have plenty of pre-event history AND >30 days post.
    dates = _make_trading_dates(_dt.date(2020, 1, 2), 1000)
    closes = [100.0]
    # Flat at 100 through the trading day strictly before event_date, then
    # +1% on event_date (which is the first trading-day on-or-after the
    # event and equals h=1 by our convention: base = close_{t-1}, target =
    # close_t for h=1), then flat post-pop. h=5/10/30 still see the same
    # +1% since the series is flat after the pop.
    event_dt = _dt.date.fromisoformat(event_date)
    pop_idx = None
    for i in range(1, 1000):
        if dates[i] >= event_dt and pop_idx is None:
            pop_idx = i
            closes.append(101.0)
        elif pop_idx is not None and i > pop_idx:
            closes.append(closes[-1])  # flat post-pop
        else:
            closes.append(100.0)
    series = _series_from_closes(dates, closes)

    df = edb.build_event_rows(
        package_dir=package,
        asset="^GSPC",
        benchmark="^GSPC",
        asset_series=series,
        bench_series=series,
    )
    assert not df.empty
    assert set(df["horizon"].tolist()) == {1, 5, 10, 30}

    by_h = {int(h): row for h, row in zip(df["horizon"], df.to_dict(orient="records"))}
    # h=1 should be ~+1%
    assert by_h[1]["realized_return"] == pytest.approx(0.01, abs=1e-9)
    assert by_h[1]["abnormal_return"] == pytest.approx(0.01, abs=1e-9)
    assert by_h[1]["direction_t1d"] == 1
    # All horizons after the pop also see the same +1% (since post-pop flat)
    for h in (5, 10, 30):
        assert by_h[h]["realized_return"] == pytest.approx(0.01, abs=1e-9)
    # alpha=0, beta=1 when asset == benchmark
    for row in df.to_dict(orient="records"):
        assert row["alpha"] == 0.0
        assert row["beta"] == 1.0


# ---------------------------------------------------------------------------
# Look-ahead guard
# ---------------------------------------------------------------------------


def test_lookahead_guard_raises_when_prior_bar_overlaps_as_of() -> None:
    """A prior-window assertion must fire if the last bar's date isn't
    strictly less than ``as_of.date()``."""

    dates = _make_trading_dates(_dt.date(2023, 1, 2), 40)
    as_of = dates[-1]  # last bar's date == as_of -> contract violation
    bars = [
        edb._PriorBar(date=d, close=100.0, volume=0.0, vol_5d=0.0, cum_return_20d=0.0)
        for d in dates[-20:]
    ]
    # Forge a contract violation: last bar.date == as_of
    bars[-1] = edb._PriorBar(
        date=as_of, close=100.0, volume=0.0, vol_5d=0.0, cum_return_20d=0.0
    )
    with pytest.raises(ValueError, match="prior-window contract violated"):
        edb._assert_no_lookahead(as_of, bars)


# ---------------------------------------------------------------------------
# No survivorship: zero-move events still emit a row
# ---------------------------------------------------------------------------


def test_no_survivorship_filter_zero_move_event(tmp_path: Path) -> None:
    """A perfectly flat market still produces an event row -- we never
    drop events on the basis of how the market moved."""

    package = tmp_path / "package"
    package.mkdir()
    _write_registry(
        package / "registry_normalized.jsonl",
        [_registry_entry_for_statement("2024-03-20")],
    )
    dates = _make_trading_dates(_dt.date(2022, 6, 1), 600)
    flat = [100.0] * 600
    series = _series_from_closes(dates, flat)

    df = edb.build_event_rows(
        package_dir=package,
        asset="^GSPC",
        benchmark="^GSPC",
        asset_series=series,
        bench_series=series,
    )
    assert not df.empty
    assert (df["realized_return"] == 0.0).all()
    assert (df["direction_t1d"] == 0).all()


# ---------------------------------------------------------------------------
# Determinism: byte-identical parquet across builds
# ---------------------------------------------------------------------------


def test_build_is_deterministic_byte_identical(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_registry(
        package / "registry_normalized.jsonl",
        [
            _registry_entry_for_statement("2023-06-14"),
            _registry_entry_for_statement("2023-07-26", text="Second statement."),
            _registry_entry_for_statement("2023-09-20", text="Third statement."),
        ],
    )
    dates = _make_trading_dates(_dt.date(2021, 1, 4), 800)
    # Mildly varying closes so the prior-window hashes differ across rows.
    import math

    closes = [100.0 + math.sin(i / 13.0) * 2.0 for i in range(800)]
    series = _series_from_closes(dates, closes)

    df1 = edb.build_event_rows(
        package_dir=package,
        asset="^GSPC",
        benchmark="^GSPC",
        asset_series=series,
        bench_series=series,
    )
    df2 = edb.build_event_rows(
        package_dir=package,
        asset="^GSPC",
        benchmark="^GSPC",
        asset_series=series,
        bench_series=series,
    )
    p1 = tmp_path / "out1.parquet"
    p2 = tmp_path / "out2.parquet"
    edb.write_events_parquet(df1, p1)
    edb.write_events_parquet(df2, p2)
    sha1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    sha2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    assert sha1 == sha2, f"Parquet bytes differ across builds: {sha1} != {sha2}"
    # Frames also identical row-wise (catches column-order drift):
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Source preference + multi-axis label lift
# ---------------------------------------------------------------------------


def test_source_preference_picks_full_document_over_sentence_shards(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    # Two sources cover the same event_date+kind. scraped_fed should win.
    _write_registry(
        package / "registry_normalized.jsonl",
        [
            {
                **_registry_entry_for_statement("2023-06-14", text="Sentence one."),
                "source": "hf_fomc_communication",
                "source_record_id": "train:1",
                "document_type": "statement",
                "mapped_label": "hawkish",
            },
            {
                **_registry_entry_for_statement("2023-06-14", text="Full doc text."),
                "source": "scraped_fed",
                "source_record_id": "fomc_statements.json:5",
                "document_type": "Statement",
                "mapped_label": "neutral",
            },
        ],
    )
    dates = _make_trading_dates(_dt.date(2021, 1, 4), 800)
    series = _series_from_closes(dates, [100.0] * 800)
    df = edb.build_event_rows(
        package_dir=package,
        asset_series=series,
        bench_series=series,
    )
    assert (df["source"] == "scraped_fed").all()
    assert (df["axis_stance"] == "neutral").all()


def test_concurrent_macro_release_flag_is_boolean(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_registry(
        package / "registry_normalized.jsonl",
        [_registry_entry_for_statement("2023-06-14")],
    )
    dates = _make_trading_dates(_dt.date(2021, 1, 4), 800)
    series = _series_from_closes(dates, [100.0] * 800)
    df = edb.build_event_rows(
        package_dir=package,
        asset_series=series,
        bench_series=series,
    )
    assert df["concurrent_macro_release"].dtype == bool
    # The flag must exist for every row; we never drop on this.
    assert df["concurrent_macro_release"].notna().all()


# ---------------------------------------------------------------------------
# events_full.parquet -- keep_all_sources keeps every source's rows
# ---------------------------------------------------------------------------


def test_events_full_has_more_rows_than_collapsed_on_multi_source_fixture(
    tmp_path: Path,
) -> None:
    """When the same event_date+kind is covered by three sources, the
    full view emits 3 rows-per-horizon while the collapsed view emits 1.
    Both share the same schema; the full view never drops a source."""

    package = tmp_path / "package"
    package.mkdir()
    event_date = "2023-06-14"
    _write_registry(
        package / "registry_normalized.jsonl",
        [
            {
                **_registry_entry_for_statement(event_date, text="hf sentence."),
                "source": "hf_fomc_communication",
                "source_record_id": "train:1",
                "document_type": "statement",
                "mapped_label": "hawkish",
            },
            {
                **_registry_entry_for_statement(event_date, text="kaggle row."),
                "source": "kaggle_fed_statements_minutes",
                "source_record_id": "kaggle:1",
                "document_type": "statement",
                "mapped_label": "dovish",
            },
            {
                **_registry_entry_for_statement(event_date, text="Full doc text."),
                "source": "scraped_fed",
                "source_record_id": "fomc_statements.json:5",
                "document_type": "Statement",
                "mapped_label": "neutral",
            },
        ],
    )
    dates = _make_trading_dates(_dt.date(2021, 1, 4), 800)
    series = _series_from_closes(dates, [100.0] * 800)
    df_collapsed = edb.build_event_rows(
        package_dir=package,
        asset_series=series,
        bench_series=series,
        keep_all_sources=False,
    )
    df_full = edb.build_event_rows(
        package_dir=package,
        asset_series=series,
        bench_series=series,
        keep_all_sources=True,
    )
    # collapsed should be exactly one chosen source x 4 horizons
    assert len(df_collapsed) == 4
    assert set(df_collapsed["source"]) == {"scraped_fed"}
    # full should preserve all three sources -- 3 x 4 = 12 rows
    assert len(df_full) == 12
    assert set(df_full["source"]) == {
        "hf_fomc_communication",
        "kaggle_fed_statements_minutes",
        "scraped_fed",
    }
    # Schema parity: same columns
    assert list(df_collapsed.columns) == list(df_full.columns)
    # source_record_id is populated in both views
    assert (df_full["source_record_id"].str.len() > 0).all()
    assert (df_collapsed["source_record_id"].str.len() > 0).all()
    # Determinism: building the full view twice yields identical bytes
    p1 = tmp_path / "full1.parquet"
    p2 = tmp_path / "full2.parquet"
    edb.write_events_parquet(df_full, p1)
    df_full2 = edb.build_event_rows(
        package_dir=package,
        asset_series=series,
        bench_series=series,
        keep_all_sources=True,
    )
    edb.write_events_parquet(df_full2, p2)
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(p2.read_bytes()).hexdigest()

