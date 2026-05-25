"""Daily Treasury yield + Fed Funds target panel for the rates-complex heads.

Mirrors :mod:`app.data.macro_state` but covers the daily rates surface
the FOMC forecaster needs both *before* the meeting (strict-backward
pre-meeting expectation features at ``t-1``) and *after* the meeting
(strict-forward ``t+1..t+5`` change targets stored in raw basis points).

The panel emits one row per business day in ``[start, end]`` with the
last-published value strictly before that date for every series. Two
lookup helpers expose the same shape to event-time consumers:

- :func:`yield_strictly_before` reads the last value published with
  ``pub_date < event_date`` -- the convention for pre-meeting features.
- :func:`yield_on_or_before` reads the last value with
  ``pub_date <= event_date`` -- the convention for the close-of-day
  baseline used by forward change targets.

Series
------

==================== ============================ =================================
column               FRED series                  units
==================== ============================ =================================
``treas_1y``         ``DGS1``                     % (1-year constant maturity)
``treas_2y``         ``DGS2``                     % (2-year constant maturity)
``treas_5y``         ``DGS5``                     % (5-year constant maturity)
``treas_10y``        ``DGS10``                    % (10-year constant maturity)
``slope_10y_2y``     ``T10Y2Y``                   pct points (10y minus 2y)
``slope_10y_3m``     ``T10Y3M``                   pct points (10y minus 3m)
``ff_target_upper``  ``DFEDTARU``                 % (Fed Funds upper bound)
``ff_target_lower``  ``DFEDTARL``                 % (Fed Funds lower bound)
==================== ============================ =================================

All eight series publish daily on FRED with a zero-day delay (observation
date == publication date), so the strict-backward / on-or-before lookups
operate directly on the reference date without a publication-delay shift.

Pre-2008 the FOMC used a single ``DFEDTAR`` (target rate) series rather
than the upper/lower band. The bundle here covers 2008-present cleanly;
earlier history surfaces as nulls and downstream consumers fall back to
``treas_1y`` as the terminal-rate proxy.

Determinism
-----------

Same FRED inputs imply the same parquet rows. Output is sorted by
``as_of_date`` (mergesort, stable) and re-runs produce byte-identical
parquet under zstd-level-3 + statistics-off + dictionary-off, matching
the pattern from :mod:`app.data.macro_state`.

CLI
---

::

    python -m app.data.rates_panel \\
        --start 2008-01-01 --end today \\
        --output data/external/fred/rates_panel.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.services.fred_client import (
    DEFAULT_CACHE_DIR as FRED_CACHE_DIR,
    FredSeriesResponse,
    SOURCES_LOCK_NAME,
    fetch_fred_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_START = "2008-01-01"
DEFAULT_OUTPUT_NAME = "rates_panel.parquet"
DEFAULT_LOCK_KEY = "rates_panel"

# FRED series feeding the panel. Order is pinned so the SOURCES.lock
# entry and tests can iterate the tuple without surprises.
RATES_SERIES_IDS: tuple[str, ...] = (
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS10",
    "T10Y2Y",
    "T10Y3M",
    "DFEDTARU",
    "DFEDTARL",
)

# All series are daily market-data with a zero publication delay (FRED
# observation date == publication date for the rates surface).
PUBLICATION_DELAY_DAYS: int = 0

# Mapping from FRED series id to the parquet column name.
COLUMN_BY_SERIES: dict[str, str] = {
    "DGS1": "treas_1y",
    "DGS2": "treas_2y",
    "DGS5": "treas_5y",
    "DGS10": "treas_10y",
    "T10Y2Y": "slope_10y_2y",
    "T10Y3M": "slope_10y_3m",
    "DFEDTARU": "ff_target_upper",
    "DFEDTARL": "ff_target_lower",
}

# Output column order. Pinned for determinism + tests.
COLUMN_ORDER: tuple[str, ...] = (
    "as_of_date",
    "treas_1y",
    "treas_2y",
    "treas_5y",
    "treas_10y",
    "slope_10y_2y",
    "slope_10y_3m",
    "ff_target_upper",
    "ff_target_lower",
    "publication_delay_days",
    "data_version",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DailyObservation:
    reference_date: _dt.date
    value: float


@dataclass
class RatesPanelArtifacts:
    """Returned by :func:`build_rates_panel`."""

    frame: pd.DataFrame
    fred_series_used: tuple[str, ...]
    publication_delay_days: int
    rows_written: int
    data_version: str
    value_hash: str


@dataclass(frozen=True)
class RatesPanelLookup:
    """In-memory view of the rates panel keyed by publication date.

    Constructed once per pipeline run and queried per FOMC event.
    ``yield_strictly_before`` returns the last value with
    ``pub_date < target`` (pre-meeting convention). ``yield_on_or_before``
    returns the last value with ``pub_date <= target`` (close-of-day
    baseline for forward-change targets).
    """

    dates_by_column: Mapping[str, tuple[_dt.date, ...]]
    values_by_column: Mapping[str, tuple[float, ...]]

    def yield_strictly_before(self, column: str, target: _dt.date) -> float | None:
        dates = self.dates_by_column.get(column)
        values = self.values_by_column.get(column)
        if not dates or not values:
            return None
        import bisect as _bisect

        idx = _bisect.bisect_left(dates, target)
        if idx == 0:
            return None
        return values[idx - 1]

    def yield_on_or_before(self, column: str, target: _dt.date) -> float | None:
        dates = self.dates_by_column.get(column)
        values = self.values_by_column.get(column)
        if not dates or not values:
            return None
        import bisect as _bisect

        idx = _bisect.bisect_right(dates, target)
        if idx == 0:
            return None
        return values[idx - 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if fv != fv:  # NaN
        return None
    return fv


def _daily_observations(series: FredSeriesResponse) -> list[_DailyObservation]:
    out: list[_DailyObservation] = []
    for obs in series.observations:
        if obs.value is None:
            continue
        try:
            d = _dt.date.fromisoformat(obs.date)
        except ValueError:
            continue
        out.append(_DailyObservation(reference_date=d, value=float(obs.value)))
    out.sort(key=lambda r: r.reference_date)
    return out


def _value_on_or_before(
    pub_index: Sequence[tuple[_dt.date, float]], as_of: _dt.date
) -> tuple[_dt.date | None, float | None]:
    """Return the latest ``(pub_date, value)`` with ``pub_date <= as_of``.

    The panel stores this raw "close as of date" value at each parquet
    row. :class:`RatesPanelLookup` then layers strict-backward or
    on-or-before semantics on top via its own bisect calls, so the
    panel build itself must NOT pre-apply a shift — doing so would
    double-shift every value returned by :meth:`yield_on_or_before`
    and :meth:`yield_strictly_before` (an earlier draft of this
    module used strict-before semantics in the build and silently
    emitted values one trading day stale).
    """

    import bisect as _bisect

    if not pub_index:
        return (None, None)
    dates = [d for d, _ in pub_index]
    idx = _bisect.bisect_right(dates, as_of)
    if idx == 0:
        return (None, None)
    pub_date, val = pub_index[idx - 1]
    return (pub_date, val)


def _shifted_publication_index(
    observations: Sequence[_DailyObservation], *, delay_days: int
) -> list[tuple[_dt.date, float]]:
    if delay_days < 0:
        raise ValueError(f"delay_days must be >= 0, got {delay_days}")
    return [
        (obs.reference_date + _dt.timedelta(days=delay_days), float(obs.value))
        for obs in observations
    ]


def _business_days(start: _dt.date, end: _dt.date) -> list[_dt.date]:
    out: list[_dt.date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_rates_panel(
    *,
    start: _dt.date,
    end: _dt.date,
    fred_responses: Mapping[str, FredSeriesResponse],
    as_of_dates: Sequence[_dt.date] | None = None,
    publication_delay_days: int = PUBLICATION_DELAY_DAYS,
) -> RatesPanelArtifacts:
    """Assemble the rates panel frame.

    Parameters
    ----------
    start, end:
        Inclusive bounds on ``as_of_date``. When ``as_of_dates`` is None,
        emits one row per business day in ``[start, end]``.
    fred_responses:
        Pre-loaded FRED responses keyed by series id. Must cover every id
        in :data:`RATES_SERIES_IDS`.
    as_of_dates:
        Optional explicit list of as-of dates. When supplied, the frame
        has one row per supplied date (sorted, deduped).
    publication_delay_days:
        Days added to each daily observation's reference date before the
        as-of join. Defaults to 0 (the rates surface publishes same-day).
    """

    missing = [sid for sid in RATES_SERIES_IDS if sid not in fred_responses]
    if missing:
        raise KeyError(f"Missing FRED series for rates_panel: {missing}")

    panel_indices: dict[str, list[tuple[_dt.date, float]]] = {}
    for series_id in RATES_SERIES_IDS:
        observations = _daily_observations(fred_responses[series_id])
        panel_indices[series_id] = _shifted_publication_index(
            observations, delay_days=publication_delay_days
        )

    if as_of_dates is None:
        target_dates = _business_days(start, end)
    else:
        target_dates = sorted({d for d in as_of_dates if start <= d <= end})

    rows: list[dict[str, Any]] = []
    for d in target_dates:
        payload: dict[str, Any] = {"as_of_date": d.isoformat()}
        for series_id, column_name in COLUMN_BY_SERIES.items():
            # Store the close-of-day value as of `d`. Strict-backward
            # semantics are applied by RatesPanelLookup at query time,
            # so pre-shifting here would double-shift every consumer.
            _, val = _value_on_or_before(panel_indices[series_id], d)
            payload[column_name] = _round(_clean_float(val))
        payload["publication_delay_days"] = int(publication_delay_days)
        rows.append(payload)

    data_version = _data_version_hash(
        fred_responses,
        publication_delay_days=publication_delay_days,
        target_dates=target_dates,
    )
    for r in rows:
        r["data_version"] = data_version

    frame = pd.DataFrame(rows) if rows else _empty_frame()
    if not frame.empty:
        frame = frame.sort_values("as_of_date", kind="mergesort").reset_index(drop=True)
        frame = frame[list(COLUMN_ORDER)]

    return RatesPanelArtifacts(
        frame=frame,
        fred_series_used=tuple(sorted(fred_responses)),
        publication_delay_days=int(publication_delay_days),
        rows_written=len(rows),
        data_version=data_version,
        value_hash=dataframe_value_hash(frame),
    )


def _data_version_hash(
    series_responses: Mapping[str, FredSeriesResponse],
    *,
    publication_delay_days: int,
    target_dates: Sequence[_dt.date],
) -> str:
    parts: list[str] = [f"delay={publication_delay_days}"]
    for series_id in sorted(series_responses):
        resp = series_responses[series_id]
        parts.append(f"{series_id}|{resp.observation_end}|{resp.count}")
    parts.append(f"target_n={len(target_dates)}")
    if target_dates:
        parts.append(f"target_first={target_dates[0].isoformat()}")
        parts.append(f"target_last={target_dates[-1].isoformat()}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def dataframe_value_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    cols = [c for c in COLUMN_ORDER if c in frame.columns]
    if not cols:
        cols = list(frame.columns)
    ordered = frame[cols].copy()
    str_rows = ordered.astype(object).map(
        lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v)
    )
    serialised = ["|".join(str(v) for v in row) for row in str_rows.to_numpy().tolist()]
    serialised.sort()
    payload = "\n".join(serialised) + "\n" + "|".join(cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parquet writer + SOURCES.lock update
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_rates_panel_parquet(frame: pd.DataFrame, output_path: Path) -> str:
    """Write deterministically and return the sha256 of the parquet bytes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
        use_dictionary=False,
    )
    return _sha256_of_file(output_path)


def update_sources_lock(
    *,
    lock_path: Path,
    artifacts: RatesPanelArtifacts,
    parquet_path: Path,
    parquet_sha256: str,
    lock_key: str = DEFAULT_LOCK_KEY,
) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    entry = {
        "parquet_path": str(parquet_path.name),
        "sha256": parquet_sha256,
        "fred_series": list(artifacts.fred_series_used),
        "rows": int(artifacts.rows_written),
        "publication_delay_days": int(artifacts.publication_delay_days),
        "columns": dict(COLUMN_BY_SERIES),
        "data_version": artifacts.data_version,
        "value_hash": artifacts.value_hash,
        "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    existing[lock_key] = entry
    lock_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Lookup loader (read-side helper for event_dataset_builder)
# ---------------------------------------------------------------------------


def load_rates_panel_lookup(parquet_path: Path) -> RatesPanelLookup:
    """Read ``rates_panel.parquet`` and build the in-memory lookup.

    Falls back to an empty lookup when the file is missing -- callers
    that depend on the rates panel pass through the strict-backward /
    forward queries which return ``None`` for every column, and the
    event-row builder records the missing values without aborting the
    whole pipeline. This mirrors how the linguistic and credibility
    loaders degrade when their inputs are absent.
    """

    if not parquet_path.exists():
        return RatesPanelLookup(dates_by_column={}, values_by_column={})

    frame = pd.read_parquet(parquet_path)
    if frame.empty or "as_of_date" not in frame.columns:
        return RatesPanelLookup(dates_by_column={}, values_by_column={})

    frame = frame.sort_values("as_of_date", kind="mergesort").reset_index(drop=True)
    parsed_dates = [_dt.date.fromisoformat(str(d)[:10]) for d in frame["as_of_date"]]

    dates_by_column: dict[str, tuple[_dt.date, ...]] = {}
    values_by_column: dict[str, tuple[float, ...]] = {}
    for column in COLUMN_BY_SERIES.values():
        if column not in frame.columns:
            continue
        pairs = [
            (d, float(v))
            for d, v in zip(parsed_dates, frame[column].tolist())
            if v is not None and v == v  # filter NaN
        ]
        dates_by_column[column] = tuple(p[0] for p in pairs)
        values_by_column[column] = tuple(p[1] for p in pairs)

    return RatesPanelLookup(
        dates_by_column=dates_by_column,
        values_by_column=values_by_column,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _hydrate_fred_responses(
    *,
    start: _dt.date,
    end: _dt.date,
    cache_dir: Path,
    transport: httpx.BaseTransport | None = None,
    force_refresh: bool = False,
) -> dict[str, FredSeriesResponse]:
    """Fetch every required series via :func:`fetch_fred_series`."""

    out: dict[str, FredSeriesResponse] = {}
    for sid in RATES_SERIES_IDS:
        out[sid] = fetch_fred_series(
            sid,
            start=start.isoformat(),
            end=end.isoformat(),
            cache_dir=cache_dir,
            transport=transport,
            force_refresh=force_refresh,
        )
    return out


def _parse_end(value: str) -> _dt.date:
    if value.lower() == "today":
        return _dt.date.today()
    return _dt.date.fromisoformat(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the daily rates panel parquet at "
            "data/external/fred/rates_panel.parquet for the #291 rates-complex heads."
        ),
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default="today")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--cache-dir", default=str(FRED_CACHE_DIR))
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    start = _dt.date.fromisoformat(args.start)
    end = _parse_end(args.end)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    responses = _hydrate_fred_responses(
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=args.force_refresh,
    )

    artifacts = build_rates_panel(
        start=start,
        end=end,
        fred_responses=responses,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cache_dir / output_path
    parquet_sha = write_rates_panel_parquet(artifacts.frame, output_path)
    update_sources_lock(
        lock_path=cache_dir / SOURCES_LOCK_NAME,
        artifacts=artifacts,
        parquet_path=output_path,
        parquet_sha256=parquet_sha,
    )

    print(f"[rates-panel] rows: {artifacts.rows_written}")
    print(f"[rates-panel] data_version: {artifacts.data_version}")
    print(f"[rates-panel] parquet sha256: {parquet_sha}")
    print(f"[rates-panel] series: {', '.join(artifacts.fred_series_used)}")
    if not artifacts.frame.empty:
        tail = artifacts.frame.tail(3)
        print("[rates-panel] 3 most recent rows:")
        for _, row in tail.iterrows():
            print(
                "  {d} 1y={y1} 2y={y2} 5y={y5} 10y={y10} 10y-2y={s2} 10y-3m={s3} ffu={u} ffl={l}".format(
                    d=row["as_of_date"],
                    y1=_fmt(row["treas_1y"]),
                    y2=_fmt(row["treas_2y"]),
                    y5=_fmt(row["treas_5y"]),
                    y10=_fmt(row["treas_10y"]),
                    s2=_fmt(row["slope_10y_2y"]),
                    s3=_fmt(row["slope_10y_3m"]),
                    u=_fmt(row["ff_target_upper"]),
                    l=_fmt(row["ff_target_lower"]),
                )
            )
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    sys.exit(main())
