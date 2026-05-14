"""Extract the GSS 2005 (Gürkaynak-Sack-Swanson, IJCB) per-FOMC factor data
appendix from the published PDF into CSV files the ingestion pipeline can
read.

Source PDF: `data/external/gss/gss_ijcb_2005.pdf` (downloaded manually from
IJCB). Output: `gss_factors.csv` (Table 3: target/path factors per FOMC date)
and `gss_surprises.csv` (Table 2: 30-min / 1-hour / 1-day monetary-policy
surprise windows per FOMC date).

Run via the backend container so pdfplumber is available:

    docker compose run --rm --no-deps backend python -m scripts.extract_gss_factors
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PDF = Path("/data/external/gss/gss_ijcb_2005.pdf")
DEFAULT_OUT = Path("/data/external/gss")

# Page-7 layout: rows of three (Date, Target, Path, [T]) tuples laid out
# left-to-right by meeting date. Triples may end with " T" if the FOMC
# released an explanatory statement that day.
_FACTOR_TRIPLE = re.compile(
    r"(\d{1,2}-[A-Za-z]{3}-\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)(\s+T)?"
)

# Pages 4-6 layout: one row per FOMC date with five numeric columns plus
# zero, one, or two trailing T flags. "omitted" appears for 17-Sep-01.
_SURPRISE_ROW = re.compile(
    r"^(\d{1,2}-[A-Za-z]{3}-\d{2})"
    r"\s+(-?\d+\.\d+|omitted)\s+(-?\d+\.\d+|omitted)\s+(-?\d+\.\d+|omitted)"
    r"\s+(-?\d+\.\d+|omitted)\s+(-?\d+\.\d+|omitted)(.*)$"
)


def _parse_two_digit_year_date(raw: str) -> str:
    dt = datetime.strptime(raw, "%d-%b-%y")
    if dt.year > 2030:
        # strptime defaults pre-2070 two-digit years to 20xx; pull back
        # those that fall after 2030 (the GSS appendix tops out at 2004).
        dt = dt.replace(year=dt.year - 100)
    return dt.strftime("%Y-%m-%d")


def _maybe_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw == "omitted":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def extract_factors(page_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in page_text.splitlines():
        for date_raw, target_raw, path_raw, t_flag in _FACTOR_TRIPLE.findall(line):
            rows.append(
                {
                    "meeting_date": _parse_two_digit_year_date(date_raw),
                    "target_factor": float(target_raw),
                    "path_factor": float(path_raw),
                    "fomc_statement": "T" if t_flag.strip() else "",
                }
            )
    rows.sort(key=lambda r: r["meeting_date"])
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["meeting_date"] in seen:
            continue
        seen.add(row["meeting_date"])
        out.append(row)
    return out


def extract_surprises(page_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in page_text.splitlines():
        match = _SURPRISE_ROW.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "meeting_date": _parse_two_digit_year_date(match.group(1)),
                "surprise_30min_bp": _maybe_float(match.group(2)),
                "surprise_1hour_bp": _maybe_float(match.group(3)),
                "surprise_1day_bp": _maybe_float(match.group(4)),
                "diff_wide_minus_tight": _maybe_float(match.group(5)),
                "diff_daily_minus_tight": _maybe_float(match.group(6)),
                "flags_raw": match.group(7).strip(),
            }
        )
    rows.sort(key=lambda r: r["meeting_date"])
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["meeting_date"] in seen:
            continue
        seen.add(row["meeting_date"])
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    import pdfplumber  # noqa: WPS433 — lazy import keeps the module importable in tests

    with pdfplumber.open(args.pdf) as pdf:
        factor_text = pdf.pages[6].extract_text() or ""
        surprise_text = "\n".join(
            (pdf.pages[i].extract_text() or "") for i in (3, 4, 5)
        )

    factors = extract_factors(factor_text)
    surprises = extract_surprises(surprise_text)

    out_dir = Path(args.output_dir)
    _write_csv(
        out_dir / "gss_factors.csv",
        factors,
        ["meeting_date", "target_factor", "path_factor", "fomc_statement"],
    )
    _write_csv(
        out_dir / "gss_surprises.csv",
        surprises,
        [
            "meeting_date",
            "surprise_30min_bp",
            "surprise_1hour_bp",
            "surprise_1day_bp",
            "diff_wide_minus_tight",
            "diff_daily_minus_tight",
            "flags_raw",
        ],
    )
    print(f"Wrote {len(factors)} factor rows and {len(surprises)} surprise rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
