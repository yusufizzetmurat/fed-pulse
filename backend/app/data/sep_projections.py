"""Summary of Economic Projections (SEP) loader (#215).

Builds ``data/external/fred/sep_projections.parquet`` -- one row per
quarterly SEP release (March / June / September / December FOMC
meetings) carrying the FOMC's median fed-funds-rate projections plus
the current-year central-tendency band.

Source policy
-------------

The SEP is published as a structured table alongside the FOMC statement
at the meeting where it is released. Two ingestion paths:

- (a) **FRED median series.** FRED republishes the SEP medians under
  fed-funds-rate projection series IDs. The current-year median
  (``FEDTARMD``) and longer-run median (``FEDTARMDLR``) sit on stable
  multi-vintage series and are pulled once per build. The next-year
  median is published as a separate year-specific series per vintage
  (``FEDTARMD<YYYY>``); the loader pivots at each release date,
  pulling ``FEDTARMD<year(release)+1>`` and reading the value stamped
  on the release. When the operator has a FRED API key the loader
  hydrates every required series at the meeting dates listed in the
  FOMC calendar and writes the parquet directly.

- (b) **CSV fixture.** When FRED is unreachable or the operator wants
  to pin a manually-vetted source, the loader reads
  ``data/external/sep_projections.csv`` (one row per release, columns
  matching the parquet schema below) and emits the parquet from that
  fixture. This is the path the test suite drives so the unit tests
  carry no network dependency.

The parquet schema is the single contract the training-package loader
joins against on ``meeting_date``. Columns are documented inline.

Provenance
----------

SEP rows are ``T (snapshot)``: the projections are observable from the
SEP release document on the meeting date itself, same band as the
``stance_*`` text features. On non-SEP meetings the training loader
forward-fills from the most recent prior SEP release (strict-prior by
construction; see ``app.training.sep_features.compute_sep_features_for_event``
and ``docs/feature-provenance-audit.md``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

DEFAULT_OUTPUT_NAME = "sep_projections.parquet"
DEFAULT_LOCK_KEY = "sep_projections"
DEFAULT_FIXTURE_CSV = DATA_DIR / "external" / "sep_projections.csv"

# Live FRED series IDs the SEP loader pulls. The longer-run median
# uses ``FEDTARMDLR`` (the published longer-run median across SEP
# vintages); the range high / low bounds are the full all-participants
# spread (range, not central tendency). The next-year median is NOT a
# single rolling series on FRED -- FRED publishes one year-specific
# ``FEDTARMD<YYYY>`` series per calendar year (e.g. ``FEDTARMD2025``).
# The loader therefore pivots per release: at each SEP meeting it
# pulls ``FEDTARMD<year(release)+1>`` and reads the median value
# stamped on that release. See ``DEFAULT_FRED_NEXT_YEAR_PREFIX`` and
# ``_resolve_next_year_series_id`` for the pivot. Names below match
# what FRED actually publishes.
DEFAULT_FRED_SERIES_IDS: dict[str, str] = {
    "ffr_median_current_year": "FEDTARMD",
    "ffr_median_longer_run": "FEDTARMDLR",
    "ffr_range_upper_current": "FEDTARRH",
    "ffr_range_lower_current": "FEDTARRL",
}

# Per-release next-year median pivot. FRED's year-specific median
# series follow the ``FEDTARMD<YYYY>`` naming convention; the loader
# resolves the series ID at each release date by appending
# ``year(release_date) + 1`` to the prefix below. Earliest year-
# specific series on FRED begins with the 2014 vintage; pre-2014 SEPs
# pre-date the next-year-median publication cadence and the loader
# leaves the slot ``None`` for those rows (graceful fallback — the
# composer collapses the missing scalar to ``0.0`` and the per-event
# missing flag carries the signal).
DEFAULT_FRED_NEXT_YEAR_PREFIX: str = "FEDTARMD"
EARLIEST_NEXT_YEAR_VINTAGE: int = 2014

# Parquet column order. The training-package loader reads these
# columns by name so the order is cosmetic for the join, but pinning
# it here keeps the parquet metadata stable across re-builds.
COLUMN_ORDER: tuple[str, ...] = (
    "meeting_date",
    "ffr_median_current_year",
    "ffr_median_next_year",
    "ffr_median_longer_run",
    "ffr_range_upper_current",
    "ffr_range_lower_current",
    "source",
    "data_version",
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SepReleaseRow:
    """One SEP release, as the parquet writer serialises it."""

    meeting_date: _dt.date
    ffr_median_current_year: float | None
    ffr_median_next_year: float | None
    ffr_median_longer_run: float | None
    ffr_range_upper_current: float | None
    ffr_range_lower_current: float | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "meeting_date": self.meeting_date.isoformat(),
            "ffr_median_current_year": _clean_float(self.ffr_median_current_year),
            "ffr_median_next_year": _clean_float(self.ffr_median_next_year),
            "ffr_median_longer_run": _clean_float(self.ffr_median_longer_run),
            "ffr_range_upper_current": _clean_float(
                self.ffr_range_upper_current
            ),
            "ffr_range_lower_current": _clean_float(
                self.ffr_range_lower_current
            ),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN guard
        return None
    return out


def _parse_iso_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _series_to_map(series: FredSeriesResponse) -> dict[_dt.date, float]:
    out: dict[_dt.date, float] = {}
    for obs in series.observations:
        if obs.value is None:
            continue
        d = _parse_iso_date(obs.date)
        if d is None:
            continue
        out[d] = float(obs.value)
    return out


def _value_on_or_before(
    series_map: Mapping[_dt.date, float], target: _dt.date
) -> float | None:
    """Return the most recent series value with date ``<= target``.

    FRED publishes SEP series at quarter-end dates that occasionally
    drift from the FOMC meeting day by a day or two. The on-or-before
    lookup matches the SEP value to the meeting whose release it
    accompanied without requiring exact date equality.
    """

    eligible = [d for d in series_map.keys() if d <= target]
    if not eligible:
        return None
    return float(series_map[max(eligible)])


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


def load_fixture_csv(
    path: Path | str | None = None,
) -> list[SepReleaseRow]:
    """Read SEP releases from a CSV fixture.

    Columns: ``meeting_date``, ``ffr_median_current_year``,
    ``ffr_median_next_year``, ``ffr_median_longer_run``,
    ``ffr_range_upper_current``, ``ffr_range_lower_current``. Header
    row required. The ``ffr_median_next_year`` column may be empty on
    pre-2014 SEP rows where FRED does not publish a year-specific
    next-year series; the loader treats the missing value as
    ``None`` so the composer can collapse it to a zero with the
    per-event missing flag carrying the signal.

    Returns an empty list when the fixture does not exist. The test
    suite injects a synthetic fixture; production builds use the FRED
    path when an API key is available and fall back to the fixture
    otherwise.
    """

    resolved = Path(path) if path is not None else DEFAULT_FIXTURE_CSV
    if not resolved.exists():
        return []
    rows: list[SepReleaseRow] = []
    with resolved.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            md = _parse_iso_date(row.get("meeting_date"))
            if md is None:
                continue
            rows.append(
                SepReleaseRow(
                    meeting_date=md,
                    ffr_median_current_year=_clean_float(
                        row.get("ffr_median_current_year")
                    ),
                    ffr_median_next_year=_clean_float(
                        row.get("ffr_median_next_year")
                    ),
                    ffr_median_longer_run=_clean_float(row.get("ffr_median_longer_run")),
                    ffr_range_upper_current=_clean_float(
                        row.get("ffr_range_upper_current")
                    ),
                    ffr_range_lower_current=_clean_float(
                        row.get("ffr_range_lower_current")
                    ),
                    source="fixture_csv",
                )
            )
    rows.sort(key=lambda r: r.meeting_date)
    return rows


# ---------------------------------------------------------------------------
# FRED path
# ---------------------------------------------------------------------------


def _resolve_next_year_series_id(release_date: _dt.date) -> str:
    """Return the ``FEDTARMD<YYYY>`` series ID for the release's next-year median.

    The pivot is per-release: a SEP published in 2024 reports its
    next-year median against the 2025 vintage, so the loader pulls
    ``FEDTARMD2025``. The earliest vintage FRED publishes for this
    series family is :data:`EARLIEST_NEXT_YEAR_VINTAGE`; callers are
    expected to skip the lookup for pre-2014 releases where no
    matching series exists.
    """

    return f"{DEFAULT_FRED_NEXT_YEAR_PREFIX}{release_date.year + 1}"


def build_from_fred_responses(
    *,
    fred_responses: Mapping[str, FredSeriesResponse],
    sep_meeting_dates: Sequence[_dt.date],
    series_ids: Mapping[str, str] = DEFAULT_FRED_SERIES_IDS,
    next_year_responses: Mapping[str, FredSeriesResponse] | None = None,
) -> list[SepReleaseRow]:
    """Project FRED median + central-tendency series onto the SEP calendar.

    ``sep_meeting_dates`` lists the quarterly meetings at which the
    SEP was released. For each such meeting we read the on-or-before
    value from every required series and emit one row. Meetings
    without a matching FRED observation drop out -- the operator can
    backfill via the CSV fixture path.

    ``next_year_responses`` maps the per-release year-specific median
    series ID (``FEDTARMD<YYYY>``) to its FRED payload. The loader
    pivots on the release date: a SEP published in 2024 reads its
    next-year median off ``FEDTARMD2025``. Releases for which the
    year-specific series is missing (pre-2014 vintages or a transient
    FRED outage) emit ``None`` in the slot; the row is otherwise
    written normally so the rest of the block remains populated.
    """

    maps: dict[str, dict[_dt.date, float]] = {}
    for field_name, series_id in series_ids.items():
        resp = fred_responses.get(series_id)
        if resp is None:
            maps[field_name] = {}
            continue
        maps[field_name] = _series_to_map(resp)

    next_year_maps: dict[str, dict[_dt.date, float]] = {}
    if next_year_responses:
        for sid, resp in next_year_responses.items():
            if resp is None:
                continue
            next_year_maps[sid] = _series_to_map(resp)

    rows: list[SepReleaseRow] = []
    for md in sorted(set(sep_meeting_dates)):
        next_year_value: float | None = None
        if md.year + 1 >= EARLIEST_NEXT_YEAR_VINTAGE:
            sid = _resolve_next_year_series_id(md)
            next_year_value = _value_on_or_before(
                next_year_maps.get(sid, {}), md
            )
        row = SepReleaseRow(
            meeting_date=md,
            ffr_median_current_year=_value_on_or_before(
                maps.get("ffr_median_current_year", {}), md
            ),
            ffr_median_next_year=next_year_value,
            ffr_median_longer_run=_value_on_or_before(
                maps.get("ffr_median_longer_run", {}), md
            ),
            ffr_range_upper_current=_value_on_or_before(
                maps.get("ffr_range_upper_current", {}), md
            ),
            ffr_range_lower_current=_value_on_or_before(
                maps.get("ffr_range_lower_current", {}), md
            ),
            source="fred",
        )
        # Drop a row if both medians are missing; one of them at
        # minimum is required for the row to be useful.
        if (
            row.ffr_median_current_year is None
            and row.ffr_median_longer_run is None
        ):
            continue
        rows.append(row)
    return rows


def _hydrate_fred_responses(
    *,
    start: _dt.date,
    end: _dt.date,
    cache_dir: Path,
    series_ids: Mapping[str, str],
    force_refresh: bool = False,
) -> dict[str, FredSeriesResponse]:
    responses: dict[str, FredSeriesResponse] = {}
    for sid in sorted(set(series_ids.values())):
        responses[sid] = fetch_fred_series(
            sid,
            start=start.isoformat(),
            end=end.isoformat(),
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
    return responses


def _hydrate_next_year_responses(
    *,
    release_dates: Sequence[_dt.date],
    cache_dir: Path,
    force_refresh: bool = False,
) -> dict[str, FredSeriesResponse]:
    """Pull ``FEDTARMD<YYYY>`` per SEP release for the next-year median.

    Iterates the release calendar and fetches the year-specific
    series for ``year(release) + 1``. Each series ID is fetched at
    most once (multiple releases inside the same vintage year share
    the same target series). Pre-2014 releases are skipped because
    FRED's year-specific median series did not begin publication
    until the 2014 vintage. A series whose lookup raises is swallowed
    so the rest of the calendar can still resolve; the affected
    release emits ``None`` in the next-year slot.
    """

    responses: dict[str, FredSeriesResponse] = {}
    for md in sorted(set(release_dates)):
        if md.year + 1 < EARLIEST_NEXT_YEAR_VINTAGE:
            continue
        sid = _resolve_next_year_series_id(md)
        if sid in responses:
            continue
        try:
            responses[sid] = fetch_fred_series(
                sid,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
        except Exception as exc:  # noqa: BLE001 - any failure on a single
            # vintage (404 for a vintage not yet published, 400 for a
            # series id FRED has renamed, network timeout, etc.) must not
            # block the build; the next-year slot for that release
            # collapses to ``None`` and the composer's missing flag
            # carries the signal.
            print(
                f"[sep-projections] next-year series {sid} unavailable ({exc}); skipping"
            )
    return responses


# ---------------------------------------------------------------------------
# SEP meeting calendar
# ---------------------------------------------------------------------------


def filter_sep_meeting_dates(
    fomc_meeting_dates: Iterable[_dt.date],
) -> list[_dt.date]:
    """Keep only the FOMC meetings that release an SEP.

    The SEP is published quarterly at the March, June, September, and
    December meetings. We filter the bundled FOMC calendar to those
    months; the result is the calendar the FRED path projects values
    against.
    """

    sep_months = {3, 6, 9, 12}
    return sorted({d for d in fomc_meeting_dates if d.month in sep_months})


# ---------------------------------------------------------------------------
# Frame assembly + parquet writer
# ---------------------------------------------------------------------------


def _data_version_hash(rows: Sequence[SepReleaseRow]) -> str:
    """Short sha over the row payload.

    Stable across re-builds with the same data; used in the parquet
    column for downstream auditing and in the SOURCES.lock entry.
    """

    parts: list[str] = []
    for row in rows:
        d = row.to_dict()
        parts.append(
            "|".join(
                str(d.get(col, "")) for col in COLUMN_ORDER if col != "data_version"
            )
        )
    payload = "\n".join(sorted(parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def to_frame(rows: Sequence[SepReleaseRow]) -> pd.DataFrame:
    """Materialise ``SepReleaseRow`` records into the parquet schema."""

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMN_ORDER})
    data_version = _data_version_hash(rows)
    payload = [{**r.to_dict(), "data_version": data_version} for r in rows]
    frame = pd.DataFrame(payload)
    frame = frame.sort_values("meeting_date", kind="mergesort").reset_index(drop=True)
    return frame[list(COLUMN_ORDER)]


def write_parquet(frame: pd.DataFrame, output_path: Path) -> str:
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
    digest = hashlib.sha256()
    digest.update(output_path.read_bytes())
    return digest.hexdigest()


def update_sources_lock(
    *,
    lock_path: Path,
    rows: Sequence[SepReleaseRow],
    parquet_path: Path,
    parquet_sha256: str,
    series_ids: Mapping[str, str],
    next_year_series_ids: Sequence[str] | None = None,
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
        "rows": int(len(rows)),
        "fred_series": sorted(set(series_ids.values())),
        "fred_next_year_series": sorted(set(next_year_series_ids or [])),
        "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    existing[DEFAULT_LOCK_KEY] = entry
    lock_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_end(value: str) -> _dt.date:
    if value.lower() == "today":
        return _dt.date.today()
    return _dt.date.fromisoformat(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Summary of Economic Projections parquet at "
            "data/external/fred/sep_projections.parquet (#215). Reads "
            "FRED median series when an API key is set; falls back to "
            "data/external/sep_projections.csv otherwise."
        ),
    )
    parser.add_argument("--start", default="2012-01-01")
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
        "--fomc-calendar-csv",
        default=None,
        help=(
            "Override the bundled FOMC calendar; the loader filters this "
            "to the March / June / September / December meetings."
        ),
    )
    parser.add_argument(
        "--fixture-csv",
        default=None,
        help=(
            "Read SEP releases from this CSV fixture instead of FRED. "
            "When omitted and FRED is reachable, the FRED path runs; when "
            "omitted and FRED fails, the loader falls back to the bundled "
            "fixture at data/external/sep_projections.csv."
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

    rows: list[SepReleaseRow] = []
    source_used = "fixture_csv"
    next_year_series_used: list[str] = []
    if args.fixture_csv:
        rows = load_fixture_csv(args.fixture_csv)
    else:
        from app.data.mp_surprise import load_fomc_calendar

        calendar_path = (
            Path(args.fomc_calendar_csv) if args.fomc_calendar_csv else None
        )
        calendar = load_fomc_calendar(path=calendar_path, start=start, end=end)
        sep_dates = filter_sep_meeting_dates(m.meeting_date for m in calendar)
        try:
            responses = _hydrate_fred_responses(
                start=start,
                end=end,
                cache_dir=cache_dir,
                series_ids=DEFAULT_FRED_SERIES_IDS,
                force_refresh=args.force_refresh,
            )
            next_year_responses = _hydrate_next_year_responses(
                release_dates=sep_dates,
                cache_dir=cache_dir,
                force_refresh=args.force_refresh,
            )
            next_year_series_used = sorted(next_year_responses.keys())
            rows = build_from_fred_responses(
                fred_responses=responses,
                sep_meeting_dates=sep_dates,
                next_year_responses=next_year_responses,
            )
            source_used = "fred"
            if not rows:
                rows = load_fixture_csv()
                source_used = "fixture_csv"
        except (RuntimeError, OSError) as exc:
            # FRED unavailable (missing API key, network error). Degrade
            # to the bundled fixture and stamp the source on every row.
            print(f"[sep-projections] FRED path failed ({exc}); using fixture")
            rows = load_fixture_csv()
            source_used = "fixture_csv"

    frame = to_frame(rows)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cache_dir / output_path
    parquet_sha = write_parquet(frame, output_path)
    update_sources_lock(
        lock_path=cache_dir / SOURCES_LOCK_NAME,
        rows=rows,
        parquet_path=output_path,
        parquet_sha256=parquet_sha,
        series_ids=DEFAULT_FRED_SERIES_IDS,
        next_year_series_ids=next_year_series_used,
    )
    print(f"[sep-projections] source: {source_used}")
    print(f"[sep-projections] rows: {len(rows)}")
    print(f"[sep-projections] parquet sha256: {parquet_sha}")
    print(f"[sep-projections] output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
