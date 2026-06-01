"""Ingest the FOMC Summary of Economic Projections (SEP).

The Federal Reserve publishes, at the four quarterly projection meetings, an
"accessible version" projection-materials page at a stable URL:

    https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{YYYYMMDD}.htm

where ``YYYYMMDD`` is the meeting decision day. The standalone page exists from
2012 onward; before 2012 the projections shipped with the minutes (lagged) and
are out of scope here.

Two table formats appear:

* 2015 onward: a combined table with ``Median`` / ``Central Tendency`` / ``Range``
  column groups and one row per variable.
* 2012-2014: ``Central tendency`` / ``Range`` only (no median column); cells are
  string ranges such as ``"2.8 to 3.0"``.

This module fetches each meeting's page, parses the main projection table into a
tidy long frame, and writes ``sep.parquet`` keyed by meeting date.

Leak note: the projection tables / dot plot are released at 2pm ET with the
statement, so they are a valid event-frame feature for that meeting date. The
narrative SEP document (released ~3 weeks later with the minutes) is NOT ingested
here and must not be used as an event-frame feature.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

# The Fed is inconsistent on the filename: most meetings use "fomcprojtabl",
# but at least one (2022-03-16) uses the "fomcprojtable" spelling. Try both.
# Pre-2013 meetings use an older per-variable table format (no "Variable" column,
# variable identified by an <h4> heading) and are out of scope here; they parse to
# empty and are skipped with a warning. The 2013+ event frame is fully covered.
PROJTABL_URL_TEMPLATES = (
    "https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{date}.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcprojtable{date}.htm",
)
_USER_AGENT = "fed-pulse-research/1.0"
_REQUEST_TIMEOUT = 30

# Canonical variable keys, matched against the leading "Variable" cell.
_VARIABLE_PATTERNS: dict[str, str] = {
    "gdp": r"real gdp",
    "unemployment": r"unemployment",
    "core_pce": r"core pce",
    "pce": r"pce inflation",
    "ffr": r"federal funds rate",
}
# Rows that repeat the prior meeting's projections for comparison; dropped.
_PRIOR_ROW = re.compile(r"projection", re.IGNORECASE)


def _canon_variable(label: object) -> str | None:
    """Map a raw row label to a canonical variable key, or None to skip it."""
    text = str(label).strip().lower()
    if not text or _PRIOR_ROW.search(text):
        return None
    # core_pce must be tested before pce so "core pce" does not match "pce".
    for key in ("gdp", "unemployment", "core_pce", "pce", "ffr"):
        if re.search(_VARIABLE_PATTERNS[key], text):
            return key
    return None


def _group_of(name: object) -> str | None:
    """Classify a column's top-level group label."""
    text = str(name).strip().lower()
    if text.startswith("median"):
        return "median"
    if text.startswith("central"):
        return "central"
    if text.startswith("range"):
        return "range"
    return None


def _norm_horizon(year: object) -> str | None:
    """Normalise a column year label to a horizon key ('2024', 'LR'), or None."""
    text = str(year).strip()
    if "longer" in text.lower():
        return "LR"
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else None


def _scalar(cell: object) -> float | None:
    """Parse a single projection value; '-'/blank/NaN -> None."""
    text = str(cell).strip()
    if not text or text in {"-", "nan", "NaN", "—"}:
        return None
    match = re.search(r"-?\d+\.?\d*", text)
    return float(match.group(0)) if match else None


def _range(cell: object) -> tuple[float | None, float | None]:
    """Parse a 'low to high' / 'low–high' range cell into (low, high)."""
    text = str(cell).strip()
    if not text or text in {"-", "nan", "NaN", "—"}:
        return (None, None)
    # Split on the word "to", en/em dashes, a space-flanked hyphen, or a
    # digit-flanked hyphen. A leading hyphen (unary minus, e.g. "-0.1") is NOT a
    # separator, so negative boundaries survive.
    parts = re.split(r"\s+to\s+|[–—]|\s+-\s+|(?<=\d)-(?=\d)", text)
    nums = [float(p) for p in parts if re.fullmatch(r"-?\d+\.?\d*", p.strip())]
    if not nums:
        return (None, None)
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (min(nums), max(nums))


def _select_main_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the projection table: a 'Variable'-led table with median/CT columns."""
    for table in tables:
        # Flatten MultiIndex columns to "level0 level1" rather than relying on the
        # tuple repr, so detection does not depend on pandas' __repr__ formatting.
        flat = [
            (" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c)).lower()
            for c in table.columns
        ]
        has_variable = any("variable" in c for c in flat)
        has_group = any(("median" in c) or ("central" in c) for c in flat)
        if has_variable and has_group:
            return table
    return None


def parse_projection_tables(html: str, meeting_date: str) -> pd.DataFrame:
    """Parse the main SEP projection table into a tidy long frame.

    Columns: meeting_date, variable, horizon, median, central_low, central_high,
    range_low, range_high. One row per (variable, horizon).
    """
    tables = pd.read_html(io.StringIO(html))
    main = _select_main_table(tables)
    if main is None:
        logger.warning("no projection table found for %s", meeting_date)
        return pd.DataFrame()

    columns = list(main.columns)
    var_col = next(
        (c for c in columns if "variable" in str(c).lower()),
        columns[0],
    )

    records: dict[tuple[str, str], dict[str, float | None]] = {}
    seen_variables: set[str] = set()
    for _, row in main.iterrows():
        variable = _canon_variable(row[var_col])
        if variable is None:
            continue
        if variable in seen_variables:
            # A second current-projection row for the same variable should not
            # happen; if a prior-meeting comparison row slips past _canon_variable
            # it would silently overwrite. Warn rather than corrupt silently.
            logger.warning(
                "duplicate '%s' row at %s; later row overwrites earlier",
                variable,
                meeting_date,
            )
        seen_variables.add(variable)
        for col in columns:
            if col == var_col:
                continue
            group = _group_of(col[0] if isinstance(col, tuple) else col)
            horizon = _norm_horizon(col[1] if isinstance(col, tuple) else col)
            if group is None or horizon is None:
                continue
            entry = records.setdefault((variable, horizon), {})
            if group == "median":
                entry["median"] = _scalar(row[col])
            elif group == "central":
                low, high = _range(row[col])
                entry["central_low"], entry["central_high"] = low, high
            else:
                low, high = _range(row[col])
                entry["range_low"], entry["range_high"] = low, high

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(
        [
            {"meeting_date": meeting_date, "variable": var, "horizon": hor, **vals}
            for (var, hor), vals in records.items()
        ]
    )
    for field in ("median", "central_low", "central_high", "range_low", "range_high"):
        if field not in out.columns:
            out[field] = pd.NA
    return out[
        [
            "meeting_date",
            "variable",
            "horizon",
            "median",
            "central_low",
            "central_high",
            "range_low",
            "range_high",
        ]
    ]


def fetch_projection_page(date: str, session: requests.Session | None = None) -> str | None:
    """GET the projection page for a YYYYMMDD date; None if no variant exists.

    Tries each filename spelling in turn; returns the first 200, None if all 404.
    """
    getter = session or requests
    last_error_status: int | None = None
    for template in PROJTABL_URL_TEMPLATES:
        url = template.format(date=date)
        response = getter.get(
            url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        )
        if response.status_code == 200:
            response.encoding = "utf-8-sig"
            return response.text
        if response.status_code == 404:
            continue
        # Transient/other error: record it and try the next spelling rather than
        # aborting before the fallback URL has been attempted.
        last_error_status = response.status_code
        logger.debug("HTTP %d for %s; trying next variant", response.status_code, url)
    if last_error_status is not None:
        # All variants failed and at least one was a non-404 error: do not silently
        # treat this meeting as "no SEP" — surface it.
        raise RuntimeError(
            f"SEP fetch failed for {date}: last HTTP status {last_error_status}"
        )
    return None


def ingest_sep(meeting_dates: list[str], out_path: Path, pause: float = 0.5) -> pd.DataFrame:
    """Fetch + parse SEP for each meeting date; write a tidy long parquet.

    meeting_dates are 'YYYY-MM-DD'; only meetings with a projection page are kept.
    """
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    found = 0
    for iso_date in meeting_dates:
        compact = iso_date.replace("-", "")
        html = fetch_projection_page(compact, session=session)
        if html is None:
            continue
        parsed = parse_projection_tables(html, iso_date)
        if parsed.empty:
            logger.warning("page present but no rows parsed for %s", iso_date)
            continue
        frames.append(parsed)
        found += 1
        logger.info("SEP %s: %d rows", iso_date, len(parsed))
        time.sleep(pause)

    if not frames:
        raise RuntimeError("no SEP meetings ingested; check URL pattern / dates")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["meeting_date", "variable", "horizon"]).reset_index(
        drop=True
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(
        "wrote %d rows across %d SEP meetings -> %s", len(combined), found, out_path
    )
    return combined


def _read_meeting_dates(csv_path: Path) -> list[str]:
    meetings = pd.read_csv(csv_path)
    return [str(d) for d in meetings["meeting_date"].tolist()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest FOMC SEP projection tables.")
    parser.add_argument(
        "--meetings-csv",
        type=Path,
        default=DATA_DIR / "external" / "fomc_meetings_2010_2026.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "external" / "fed_comms" / "sep.parquet",
    )
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    dates = _read_meeting_dates(args.meetings_csv)
    combined = ingest_sep(dates, args.out, pause=args.pause)
    meetings = combined["meeting_date"].nunique()
    variables = sorted(combined["variable"].unique())
    print(f"SEP ingested: {meetings} meetings, variables={variables}")
    print(combined.groupby("variable").size().to_string())


if __name__ == "__main__":
    main()
