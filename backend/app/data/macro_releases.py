"""Macro release calendar for the Phase 8 ``concurrent_macro_release`` flag.

The original heuristic in :mod:`app.data.event_dataset_builder` flagged
~43 % of FOMC events because it stamped three rule-based dates per month
(first Friday for NFP, second Wednesday for CPI, first business day for
ISM). BLS / ISM actually publish on slightly different days -- holiday
shifts, scheduled Tuesdays/Thursdays for CPI, occasional reschedules.

This module loads a real release calendar built from one of:

- BLS / ISM published schedules, hand-curated and shipped at
  ``data/external/macro_releases.csv``. This is the default and keeps the
  build deterministic (same input -> same output).
- FRED's ``release/dates`` endpoint, fetched on demand via
  :func:`refresh_from_fred`. Useful for refreshing the CSV; not called
  inside :func:`build_event_rows`.

Schema of the CSV / parquet:

    release_kind,release_date
    NFP,2008-01-04
    CPI,2008-01-16
    ISM,2008-01-02
    ...

``release_kind`` is one of ``{NFP, CPI, ISM}``. Anything else is ignored
(future-compat: add PCE, GDP without changing the consumer).

Public API:

- :class:`MacroReleaseCalendar`
- :func:`load_macro_release_calendar`
- :func:`build_heuristic_calendar` -- legacy fallback when the CSV is
  absent. Used by tests that assert the new calendar fires *less* often
  than the heuristic on the same input.
- :func:`refresh_from_fred` -- optional one-shot refresh utility.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import DATA_DIR as _DATA_DIR

DEFAULT_MACRO_RELEASES_CSV = _DATA_DIR / "external" / "macro_releases.csv"

# Map release_kind -> FRED release_id. Used by refresh_from_fred. Verified
# against https://api.stlouisfed.org/fred/releases?api_key=<key>&file_type=json
_FRED_RELEASE_IDS: dict[str, int] = {
    "NFP": 50,  # "Employment Situation" (BLS) -- includes Nonfarm Payrolls
    "CPI": 10,  # "Consumer Price Index" (BLS)
    # ISM Manufacturing PMI is a private (non-FRED) release. We keep the
    # ISM column in the CSV but refresh_from_fred leaves it untouched.
}

_KNOWN_KINDS = frozenset({"NFP", "CPI", "ISM"})


@dataclass(frozen=True)
class MacroReleaseCalendar:
    """Immutable lookup of major US macro release dates by kind.

    ``dates`` is a flat set of release dates (any kind) used for the
    ±2-trading-day overlap test. ``by_kind`` is the kind-stratified view
    used for tests / diagnostics.
    """

    dates: frozenset[_dt.date]
    by_kind: dict[str, frozenset[_dt.date]] = field(default_factory=dict)
    source_label: str = "unknown"

    def dates_in_year(self, year: int) -> set[_dt.date]:
        return {d for d in self.dates if d.year == year}

    def count_by_kind(self) -> dict[str, int]:
        return {kind: len(values) for kind, values in self.by_kind.items()}


def load_macro_release_calendar(path: Path) -> MacroReleaseCalendar:
    """Load a release calendar from a CSV or parquet file at ``path``.

    The file must have ``release_kind`` and ``release_date`` columns; rows
    with unknown kinds or unparseable dates are dropped silently (and the
    dropped count is logged when the loader is run via the CLI helper).
    """

    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        records = _read_parquet(path)
    else:
        raise ValueError(f"Unsupported macro release calendar format: {path} ({suffix})")
    by_kind: dict[str, set[_dt.date]] = {kind: set() for kind in _KNOWN_KINDS}
    for kind, raw_date in records:
        if kind not in _KNOWN_KINDS:
            continue
        try:
            d = _dt.date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        by_kind[kind].add(d)
    flat: set[_dt.date] = set()
    for s in by_kind.values():
        flat |= s
    return MacroReleaseCalendar(
        dates=frozenset(flat),
        by_kind={kind: frozenset(values) for kind, values in by_kind.items()},
        source_label=str(path),
    )


def _read_csv(path: Path) -> list[tuple[str, str]]:
    import csv

    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kind = (row.get("release_kind") or "").strip().upper()
            date_str = (row.get("release_date") or "").strip()
            if kind and date_str:
                rows.append((kind, date_str))
    return rows


def _read_parquet(path: Path) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(path)
    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        out.append((str(row.get("release_kind", "")).upper(), str(row.get("release_date", ""))))
    return out


def build_heuristic_calendar(
    *,
    start_year: int = 1970,
    end_year: int = 2030,
) -> MacroReleaseCalendar:
    """Legacy rule-based calendar.

    Identical semantics to the pre-PR-153-follow-up heuristic:

    - NFP : first Friday of each month
    - CPI : second Wednesday of each month
    - ISM : first business day of each month

    Exposed so tests can compare the real calendar's hit rate against the
    heuristic and so the builder still works on a fresh checkout when the
    CSV is absent.
    """

    by_kind: dict[str, set[_dt.date]] = {"NFP": set(), "CPI": set(), "ISM": set()}
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            by_kind["ISM"].add(_first_business_day(year, month))
            by_kind["NFP"].add(_first_friday(year, month))
            by_kind["CPI"].add(_second_wednesday(year, month))
    flat: set[_dt.date] = set()
    for s in by_kind.values():
        flat |= s
    return MacroReleaseCalendar(
        dates=frozenset(flat),
        by_kind={kind: frozenset(values) for kind, values in by_kind.items()},
        source_label="heuristic",
    )


def _first_business_day(year: int, month: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return d


def _first_friday(year: int, month: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    while d.weekday() != 4:
        d += _dt.timedelta(days=1)
    return d


def _second_wednesday(year: int, month: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    seen = 0
    while True:
        if d.weekday() == 2:
            seen += 1
            if seen == 2:
                return d
        d += _dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# Optional FRED refresher
# ---------------------------------------------------------------------------


def refresh_from_fred(
    *,
    api_key: str | None = None,
    start: str = "2008-01-01",
    end: str | None = None,
    output_csv: Path | None = None,
    existing_ism_dates: Iterable[_dt.date] | None = None,
) -> Path:
    """Fetch real release dates from FRED and write a CSV at ``output_csv``.

    ISM Manufacturing PMI is a private release (not on FRED), so this
    helper passes through ``existing_ism_dates`` -- typically the ISM
    dates already in the curated CSV.

    Returns the path of the freshly-written CSV. Not called inside the
    builder; run manually when refreshing the bundled calendar.
    """

    import csv

    import httpx

    key = api_key or os.environ.get("FRED_API_KEY") or os.environ.get("FRED_TOKEN")
    if not key:
        raise RuntimeError("FRED_API_KEY not set; cannot refresh macro release calendar")

    base_url = "https://api.stlouisfed.org/fred/release/dates"
    end_str = end or _dt.date.today().isoformat()
    output = output_csv or DEFAULT_MACRO_RELEASES_CSV
    output.parent.mkdir(parents=True, exist_ok=True)

    by_kind: dict[str, set[_dt.date]] = {kind: set() for kind in _KNOWN_KINDS}
    if existing_ism_dates is not None:
        for d in existing_ism_dates:
            by_kind["ISM"].add(d)

    with httpx.Client(timeout=30.0) as client:
        for kind, release_id in _FRED_RELEASE_IDS.items():
            params: dict[str, Any] = {
                "release_id": release_id,
                "api_key": key,
                "file_type": "json",
                "include_release_dates_with_no_data": "false",
                "limit": 10000,
                "realtime_start": start,
                "realtime_end": end_str,
            }
            resp = client.get(base_url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            for row in payload.get("release_dates", []) or []:
                raw = str(row.get("date") or "")
                try:
                    d = _dt.date.fromisoformat(raw[:10])
                except ValueError:
                    continue
                if d >= _dt.date.fromisoformat(start):
                    by_kind[kind].add(d)

    rows: list[tuple[str, _dt.date]] = []
    for kind, dates in by_kind.items():
        for d in sorted(dates):
            rows.append((kind, d))
    rows.sort(key=lambda t: (t[1], t[0]))
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["release_kind", "release_date"])
        for kind, d in rows:
            writer.writerow([kind, d.isoformat()])
    # Sidecar lockfile so the bundled CSV's provenance is auditable.
    lock_path = output.with_suffix(output.suffix + ".lock.json")
    lock_path.write_text(
        json.dumps(
            {
                "source": "fred_release_dates",
                "fred_release_ids": _FRED_RELEASE_IDS,
                "refreshed_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
                "count_by_kind": {k: len(v) for k, v in by_kind.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output
