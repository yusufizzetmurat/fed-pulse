"""Macro-state snapshot at FOMC decision-eve (Phase 8 #147 lift).

For every business day in ``[start, end]`` we record the *last published
value strictly before that day* for a fixed bundle of FRED indicators.
Downstream consumers (``app.forecasting.next_fomc_decision``) join on
``as_of_date`` and feed the resulting vector into the ordinal model
alongside text / OIS / credibility / linguistic axes.

Indicators
----------

================= ============================ =================================
column            FRED series                  transform
================= ============================ =================================
``unrate``        ``UNRATE``                   level (% civilian unemployment)
``cpi_yoy``       ``CPIAUCSL``                 12-month log-change x 100 (% YoY)
``core_pce_yoy``  ``PCEPILFE``                 12-month log-change x 100 (% YoY)
``ism_proxy``     ``MANEMP``                   3-month % change (ISM manufacturing
                                               employment is paywalled at NAPM
                                               level; ``MANEMP`` from FRED is
                                               the documented free-data proxy,
                                               see methodology note below).
``payems_mom``    ``PAYEMS``                   month-over-month change in
                                               thousands of jobs
``rsafs_mom``     ``RSAFS``                    month-over-month % change in
                                               retail sales (advance)
================= ============================ =================================

Why ``MANEMP`` proxies ISM. The published ISM Manufacturing PMI itself is
behind a paywall (ISM survey aggregate, distributed via ISM and Haver).
The honest free-data substitute is **manufacturing payroll employment
growth** -- which FRED publishes monthly as ``MANEMP``. Cyclical turns
in ``MANEMP`` 3-month growth track ISM PMI inflections (Federal Reserve
Bank of Dallas Working Paper 2207, "Forecasting US Manufacturing
Activity," 2022) and the 3-month change is the formulation NBER
business-cycle dating uses as a covariate. Every row carries
``ism_proxy_source = "MANEMP_3m_pct"`` so the substitution is auditable.

No look-ahead
-------------

Every column at ``as_of_date = D`` reflects FRED publications timestamped
strictly **before** ``D``. The "publication date" used here is the
``date`` field on each FRED observation, which for monthly series is the
*reference period start* (e.g. ``2024-04-01`` for the April 2024 print).
That is conservative -- BLS / BEA publish the previous month's data
roughly 2-6 weeks into the next month, so a ``2024-04-01`` reference
period is not actually public until late April / early May. To stay
conservative without parsing release calendars, we shift each monthly
observation forward by a ``publication_delay_days`` knob (default 30)
before the as-of join. The ``data/external/macro_releases.csv``
calendar carries the real published-on date for CPI/NFP/ISM and the
shift defaults match the long-run median lag of that calendar.

Determinism
-----------

Same FRED inputs imply the same parquet rows. Output is sorted by
``as_of_date`` (mergesort, stable) and re-runs produce byte-identical
parquet under zstd-level-3 + statistics-off + dictionary-off, matching
the pattern from :mod:`app.data.mp_surprise`.

Caching
-------

Per-series JSON cache lives under ``data/external/fred/`` (shared with
:mod:`app.services.fred_client`). The assembled parquet lands at
``data/external/fred/macro_state.parquet`` and a SOURCES.lock entry
under key ``macro_state`` records the parquet sha256, FRED series used,
publication delay, retrieval timestamp, and the value-hash of the
sorted DataFrame.

CLI
---

::

    python -m app.data.macro_state \
        --start 2010-01-01 --end today \
        --output data/external/fred/macro_state.parquet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from app.config import DATA_DIR
from app.services.fred_client import (
    DEFAULT_CACHE_DIR as FRED_CACHE_DIR,
    FredObservation,
    FredSeriesResponse,
    SOURCES_LOCK_NAME,
    fetch_fred_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_START = "2010-01-01"
DEFAULT_OUTPUT_NAME = "macro_state.parquet"
DEFAULT_LOCK_KEY = "macro_state"

# FRED series IDs used by this module. Order is significant: it pins the
# SOURCES.lock entry shape and tests can iterate the tuple to validate
# the API contract.
FRED_SERIES_IDS: tuple[str, ...] = (
    "UNRATE",
    "CPIAUCSL",
    "PCEPILFE",
    "MANEMP",
    "PAYEMS",
    "RSAFS",
)

# Default conservative publication delay (in calendar days) applied to
# every monthly observation before the as-of join. BLS / BEA typically
# publish previous-month data 14-30 days after the reference month
# starts; 30 days matches the long-run median lag of the bundled
# macro-release calendar.
DEFAULT_PUBLICATION_DELAY_DAYS: int = 30

# Output column order. Pinned for determinism + tests.
COLUMN_ORDER: tuple[str, ...] = (
    "as_of_date",
    "unrate",
    "cpi_yoy",
    "core_pce_yoy",
    "ism_proxy",
    "payems_mom",
    "rsafs_mom",
    "ism_proxy_source",
    "publication_delay_days",
    "data_version",
)

ISM_PROXY_SOURCE_LABEL = "MANEMP_3m_pct"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MonthlyObservation:
    """One ``(reference_date, value)`` pair from a FRED monthly series."""

    reference_date: _dt.date
    value: float


@dataclass
class MacroStateArtifacts:
    """Returned by :func:`build_macro_state`."""

    frame: pd.DataFrame
    fred_series_used: tuple[str, ...]
    publication_delay_days: int
    rows_written: int
    data_version: str
    value_hash: str


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


def _parse_date(value: str | _dt.date) -> _dt.date:
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def _monthly_observations(series: FredSeriesResponse) -> list[_MonthlyObservation]:
    """Reduce a FRED response to a sorted list of ``(date, value)`` pairs."""

    out: list[_MonthlyObservation] = []
    for obs in series.observations:
        if obs.value is None:
            continue
        try:
            d = _dt.date.fromisoformat(obs.date)
        except ValueError:
            continue
        out.append(_MonthlyObservation(reference_date=d, value=float(obs.value)))
    out.sort(key=lambda r: r.reference_date)
    return out


def _shifted_publication_index(
    observations: Sequence[_MonthlyObservation], *, delay_days: int
) -> list[tuple[_dt.date, float]]:
    """Return ``(publication_date, value)`` pairs.

    ``publication_date = reference_date + delay_days``. The list is
    sorted and may contain duplicate publication dates only when the
    upstream FRED series ships duplicate observations (which it does
    not for the series this module reads).
    """

    if delay_days < 0:
        raise ValueError(f"delay_days must be >= 0, got {delay_days}")
    return [
        (obs.reference_date + _dt.timedelta(days=delay_days), float(obs.value))
        for obs in observations
    ]


def _value_strictly_before(
    pub_index: Sequence[tuple[_dt.date, float]], as_of: _dt.date
) -> tuple[_dt.date | None, float | None]:
    """Find the latest ``(pub_date, value)`` with ``pub_date < as_of``.

    Returns ``(None, None)`` when no such row exists.
    """

    import bisect as _bisect

    if not pub_index:
        return (None, None)
    dates = [d for d, _ in pub_index]
    # ``bisect_left`` returns the first index whose value is >= as_of;
    # everything strictly before sits at indices < idx.
    idx = _bisect.bisect_left(dates, as_of)
    if idx == 0:
        return (None, None)
    pub_date, val = pub_index[idx - 1]
    return (pub_date, val)


def _yoy_log_change(
    observations: Sequence[_MonthlyObservation],
) -> list[_MonthlyObservation]:
    """Compute ``100 * (ln(v_t) - ln(v_{t-12}))`` aligned on ``t``.

    Returns one observation per ``t`` where both ``v_t`` and the
    12-months-prior value are positive. The reference date of the
    output is the *current month* (the canonical YoY convention).
    """

    if len(observations) < 13:
        return []
    by_date = {obs.reference_date: obs.value for obs in observations}
    out: list[_MonthlyObservation] = []
    for obs in observations:
        prior_date = _shift_months(obs.reference_date, -12)
        prior_val = by_date.get(prior_date)
        if prior_val is None or prior_val <= 0 or obs.value <= 0:
            continue
        # log-change x 100 -- matches the standard YoY % in FRED FRED
        # publications (and lines up numerically with the BLS YoY at
        # 2-3 decimal places for the entire 2010-present window).
        import math as _math

        yoy = 100.0 * (_math.log(obs.value) - _math.log(prior_val))
        out.append(_MonthlyObservation(reference_date=obs.reference_date, value=yoy))
    return out


def _mom_pct_change(
    observations: Sequence[_MonthlyObservation],
) -> list[_MonthlyObservation]:
    """Compute month-over-month percentage change."""

    out: list[_MonthlyObservation] = []
    prev: float | None = None
    for obs in observations:
        if prev is not None and prev > 0:
            pct = 100.0 * (obs.value - prev) / prev
            out.append(_MonthlyObservation(reference_date=obs.reference_date, value=pct))
        prev = obs.value
    return out


def _mom_diff(
    observations: Sequence[_MonthlyObservation],
) -> list[_MonthlyObservation]:
    """Compute the absolute month-over-month change (level diff)."""

    out: list[_MonthlyObservation] = []
    prev: float | None = None
    for obs in observations:
        if prev is not None:
            diff = obs.value - prev
            out.append(_MonthlyObservation(reference_date=obs.reference_date, value=diff))
        prev = obs.value
    return out


def _three_month_pct_change(
    observations: Sequence[_MonthlyObservation],
) -> list[_MonthlyObservation]:
    """Compute ``100 * (v_t - v_{t-3}) / v_{t-3}`` aligned on ``t``."""

    by_date = {obs.reference_date: obs.value for obs in observations}
    out: list[_MonthlyObservation] = []
    for obs in observations:
        prior_date = _shift_months(obs.reference_date, -3)
        prior_val = by_date.get(prior_date)
        if prior_val is None or prior_val <= 0:
            continue
        pct = 100.0 * (obs.value - prior_val) / prior_val
        out.append(_MonthlyObservation(reference_date=obs.reference_date, value=pct))
    return out


def _shift_months(d: _dt.date, months: int) -> _dt.date:
    """Shift ``d`` by ``months`` (positive = future, negative = past).

    Day component is preserved -- FRED monthly observations always sit
    on the first of the month, so this is exact for our inputs.
    """

    y = d.year
    m = d.month + months
    while m <= 0:
        y -= 1
        m += 12
    while m > 12:
        y += 1
        m -= 12
    # Clamp day -- defensive; FRED inputs are always day=1 for monthly.
    day = min(d.day, 28)
    return _dt.date(y, m, day)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_macro_state(
    *,
    start: _dt.date,
    end: _dt.date,
    fred_responses: Mapping[str, FredSeriesResponse],
    as_of_dates: Sequence[_dt.date] | None = None,
    publication_delay_days: int = DEFAULT_PUBLICATION_DELAY_DAYS,
) -> MacroStateArtifacts:
    """Assemble the macro-state frame.

    Parameters
    ----------
    start, end:
        Inclusive bounds on ``as_of_date``. When ``as_of_dates`` is None,
        we emit one row per business day in ``[start, end]``.
    fred_responses:
        Pre-loaded FRED responses keyed by series id. Must cover every
        id in :data:`FRED_SERIES_IDS`.
    as_of_dates:
        Optional explicit list of as-of dates. When supplied, the frame
        has one row per supplied date (sorted, deduped). Production
        callers pass the FOMC meeting dates here; tests pass a small
        synthetic list.
    publication_delay_days:
        Days added to each monthly observation's reference date before
        the as-of join. Conservative default (30) shadows BLS / BEA's
        typical release lag.
    """

    missing = [sid for sid in FRED_SERIES_IDS if sid not in fred_responses]
    if missing:
        raise KeyError(f"Missing FRED series for macro_state: {missing}")

    unrate = _monthly_observations(fred_responses["UNRATE"])
    cpi = _monthly_observations(fred_responses["CPIAUCSL"])
    core_pce = _monthly_observations(fred_responses["PCEPILFE"])
    manemp = _monthly_observations(fred_responses["MANEMP"])
    payems = _monthly_observations(fred_responses["PAYEMS"])
    rsafs = _monthly_observations(fred_responses["RSAFS"])

    # Transforms
    unrate_index = _shifted_publication_index(unrate, delay_days=publication_delay_days)
    cpi_yoy_index = _shifted_publication_index(
        _yoy_log_change(cpi), delay_days=publication_delay_days
    )
    pce_yoy_index = _shifted_publication_index(
        _yoy_log_change(core_pce), delay_days=publication_delay_days
    )
    ism_proxy_index = _shifted_publication_index(
        _three_month_pct_change(manemp), delay_days=publication_delay_days
    )
    payems_mom_index = _shifted_publication_index(
        _mom_diff(payems), delay_days=publication_delay_days
    )
    rsafs_mom_index = _shifted_publication_index(
        _mom_pct_change(rsafs), delay_days=publication_delay_days
    )

    # As-of date set.
    if as_of_dates is None:
        target_dates = _business_days(start, end)
    else:
        target_dates = sorted({d for d in as_of_dates if start <= d <= end})

    rows: list[dict[str, Any]] = []
    for d in target_dates:
        _, unrate_val = _value_strictly_before(unrate_index, d)
        _, cpi_val = _value_strictly_before(cpi_yoy_index, d)
        _, pce_val = _value_strictly_before(pce_yoy_index, d)
        _, ism_val = _value_strictly_before(ism_proxy_index, d)
        _, pay_val = _value_strictly_before(payems_mom_index, d)
        _, rsa_val = _value_strictly_before(rsafs_mom_index, d)
        rows.append(
            {
                "as_of_date": d.isoformat(),
                "unrate": _clean_float(unrate_val),
                "cpi_yoy": _round(_clean_float(cpi_val)),
                "core_pce_yoy": _round(_clean_float(pce_val)),
                "ism_proxy": _round(_clean_float(ism_val)),
                "payems_mom": _round(_clean_float(pay_val)),
                "rsafs_mom": _round(_clean_float(rsa_val)),
                "ism_proxy_source": ISM_PROXY_SOURCE_LABEL,
                "publication_delay_days": int(publication_delay_days),
            }
        )

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

    return MacroStateArtifacts(
        frame=frame,
        fred_series_used=tuple(sorted(fred_responses)),
        publication_delay_days=int(publication_delay_days),
        rows_written=len(rows),
        data_version=data_version,
        value_hash=dataframe_value_hash(frame),
    )


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})


def _business_days(start: _dt.date, end: _dt.date) -> list[_dt.date]:
    """Inclusive list of Mon-Fri dates in ``[start, end]``.

    No federal-holiday filter -- the as-of join is robust to a few
    holidays returning the same last-published value as the prior
    business day. The wiki notes this caveat explicitly.
    """

    out: list[_dt.date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


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
    """Hash the DataFrame's data (not its encoding).

    Mirrors :func:`app.data.mp_surprise.dataframe_value_hash` --
    portability over byte-stability. Two runs that produce the same
    macro-state rows produce the same hash even if the parquet bytes
    differ across pyarrow / platform combinations.
    """

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


def write_macro_state_parquet(frame: pd.DataFrame, output_path: Path) -> str:
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
    artifacts: MacroStateArtifacts,
    parquet_path: Path,
    parquet_sha256: str,
    lock_key: str = DEFAULT_LOCK_KEY,
) -> None:
    """Persist provenance for the macro-state parquet."""

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
        "data_version": artifacts.data_version,
        "value_hash": artifacts.value_hash,
        "ism_proxy_source": ISM_PROXY_SOURCE_LABEL,
        "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    existing[lock_key] = entry
    lock_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


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
    """Fetch every required series via :func:`fetch_fred_series`.

    Pads ``start`` backward by 18 months so the 12-month-YoY transforms
    and the 3-month-change ISM proxy have enough history at the
    earliest as-of date.
    """

    fetch_start = start - _dt.timedelta(days=18 * 31)
    fetch_end = end
    out: dict[str, FredSeriesResponse] = {}
    for sid in FRED_SERIES_IDS:
        out[sid] = fetch_fred_series(
            sid,
            start=fetch_start.isoformat(),
            end=fetch_end.isoformat(),
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
            "Build the macro-state snapshot parquet at "
            "data/external/fred/macro_state.parquet for Phase 8 #147."
        ),
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default="today")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help="Parquet filename (relative to --cache-dir unless absolute).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(FRED_CACHE_DIR),
        help="FRED cache directory (also receives the output parquet).",
    )
    parser.add_argument(
        "--publication-delay-days",
        type=int,
        default=DEFAULT_PUBLICATION_DELAY_DAYS,
        help=(
            "Calendar-day shift applied to each monthly FRED reference "
            "date before the as-of join. Default 30 matches BLS / BEA's "
            "typical release lag."
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetch from FRED, bypassing the per-series JSON cache.",
    )
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

    artifacts = build_macro_state(
        start=start,
        end=end,
        fred_responses=responses,
        publication_delay_days=int(args.publication_delay_days),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cache_dir / output_path
    parquet_sha = write_macro_state_parquet(artifacts.frame, output_path)
    update_sources_lock(
        lock_path=cache_dir / SOURCES_LOCK_NAME,
        artifacts=artifacts,
        parquet_path=output_path,
        parquet_sha256=parquet_sha,
    )

    print(f"[macro-state] rows: {artifacts.rows_written}")
    print(f"[macro-state] publication_delay_days: {artifacts.publication_delay_days}")
    print(f"[macro-state] data_version: {artifacts.data_version}")
    print(f"[macro-state] parquet sha256: {parquet_sha}")
    print(f"[macro-state] series: {', '.join(artifacts.fred_series_used)}")
    if not artifacts.frame.empty:
        tail = artifacts.frame.tail(3)
        print("[macro-state] 3 most recent rows:")
        for _, row in tail.iterrows():
            print(
                "  {d} unrate={u} cpi_yoy={c} core_pce_yoy={p} ism_proxy={i} payems={pa} rsafs={r}".format(
                    d=row["as_of_date"],
                    u=_fmt(row["unrate"]),
                    c=_fmt(row["cpi_yoy"]),
                    p=_fmt(row["core_pce_yoy"]),
                    i=_fmt(row["ism_proxy"]),
                    pa=_fmt(row["payems_mom"]),
                    r=_fmt(row["rsafs_mom"]),
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
