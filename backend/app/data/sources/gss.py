"""GSS factor-decomposition external-corpus adapter.

Wraps the existing `_iter_gss_factors_records` read path under the
`BaseSourceScraper` Protocol so the Gurkaynak-Sack-Swanson (2005 IJCB) per-FOMC
target / path factor decomposition advertises the same metadata surface as the
in-house scrapers. The on-disk format is two CSVs (factors + optional
surprise-window side-table) so `fetch_listing` reads the factors CSV text and
yields one parsed-row dict per meeting; `parse_entry` accepts a JSON-encoded
row and emits the registry record shape `_iter_gss_factors_records` builds.

The rows are stance-unlabelled — the factor decomposition is continuous, not
categorical — so they populate the factor axis only. The cross-source transfer
matrix's stance classifier head skips this source_type.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register


def _coerce_event_date(raw: str) -> str:
    """Light-weight date coercion matching ingest_sources._coerce_event_date.

    Kept local so the adapter is loadable without importing ingest_sources
    (avoids a hard import cycle if ingest_sources is later split).
    """

    raw = (raw or "").strip()
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    if len(digits) == 4:
        return f"{digits}-01-01"
    return raw


def _parse_gss_factors_row(
    row: dict[str, str],
    surprises_by_date: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Parse one GSS factors CSV row into the registry record shape."""

    event_date = ""
    for key in ("meeting_date", "date", "event_date"):
        event_date = _coerce_event_date(row.get(key, ""))
        if event_date:
            break
    if not event_date:
        return None

    target_raw = (row.get("target_factor") or "").strip()
    path_raw = (row.get("path_factor") or "").strip()
    if not target_raw and not path_raw:
        return None

    extras: dict[str, Any] = {}
    try:
        extras["gss_target_factor"] = float(target_raw) if target_raw else None
    except ValueError:
        extras["gss_target_factor"] = None
    try:
        extras["gss_path_factor"] = float(path_raw) if path_raw else None
    except ValueError:
        extras["gss_path_factor"] = None
    extras["gss_fomc_statement"] = (row.get("fomc_statement") or "").strip().upper() == "T"
    extras.update(surprises_by_date.get(event_date, {}))

    target_repr = (
        f"{extras['gss_target_factor']:+.2f}"
        if extras.get("gss_target_factor") is not None
        else "n/a"
    )
    path_repr = (
        f"{extras['gss_path_factor']:+.2f}"
        if extras.get("gss_path_factor") is not None
        else "n/a"
    )
    text = (
        f"GSS factor decomposition for {event_date}: "
        f"target={target_repr} bp, path={path_repr} bp"
    )

    return {
        "source_record_id": f"gss_{event_date}",
        "event_date_hint": event_date,
        "document_type": "statement",
        "title": f"GSS target/path factors {event_date}",
        "text": text,
        "label": "",  # factor axis is continuous; no categorical stance
        "license_scope": "research_only",
        "citation_ref": "gurkaynak_sack_swanson_2005_ijcb",
        "multi_axis_extras": extras,
    }


def _parse_surprises_csv(text: str) -> dict[str, dict[str, Any]]:
    """Parse the surprises side-table CSV into a per-meeting-date dict."""

    by_date: dict[str, dict[str, Any]] = {}
    if not text:
        return by_date
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        event_date = ""
        for key in ("meeting_date", "date", "event_date"):
            event_date = _coerce_event_date(row.get(key, ""))
            if event_date:
                break
        if not event_date:
            continue
        extras: dict[str, Any] = {}
        for key in (
            "surprise_30min_bp",
            "surprise_1hour_bp",
            "surprise_1day_bp",
            "diff_wide_minus_tight",
            "diff_daily_minus_tight",
        ):
            raw = (row.get(key) or "").strip()
            if not raw:
                continue
            try:
                extras[key] = float(raw)
            except ValueError:
                continue
        if extras:
            by_date[event_date] = extras
    return by_date


class GssFactorsScraper:
    """File-backed `BaseSourceScraper` adapter for the GSS factor-decomposition release.

    The factors CSV is the primary listing; the optional surprises CSV is a
    per-meeting side-table merged onto `multi_axis_extras`. Pass surprise CSV
    text through the constructor when available — `fetch_listing` then enriches
    matching rows at parse time.
    """

    metadata = SourceMetadata(
        name="GSS factor decomposition (Gurkaynak-Sack-Swanson 2005)",
        source_type="gss_factor_decomposition",
        provenance=Provenance.PEER_REVIEWED,
        citation="Gurkaynak, R. S., Sack, B., Swanson, E. T. (2005). IJCB 1(1).",
    )

    def __init__(self, surprises_csv_text: str | None = None) -> None:
        self._surprises_by_date = _parse_surprises_csv(surprises_csv_text or "")

    def fetch_listing(self, html: str) -> list[dict[str, str]]:
        """Read raw factors-CSV text and return the list of row dicts.

        `html` is the entire file contents — keeping the Protocol name so the
        adapter sits in the same registry as the HTML-scraped sources.
        """

        if not html:
            return []
        reader = csv.DictReader(io.StringIO(html))
        return [dict(row) for row in reader]

    def parse_entry(self, raw_html: str, *, source_url: str) -> dict[str, Any] | None:
        """Parse a single GSS factors CSV row.

        `raw_html` is a JSON-encoded row dict (so the Protocol signature stays
        a plain string). `source_url` records provenance on the parsed entry.
        """

        try:
            row = json.loads(raw_html)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        parsed = _parse_gss_factors_row(
            {str(k): str(v) if v is not None else "" for k, v in row.items()},
            self._surprises_by_date,
        )
        if parsed is None:
            return None
        parsed["source_url"] = source_url
        return parsed

    def write(self, parsed: Iterable[dict[str, Any]], output_path: Path) -> int:
        """Serialise parsed GSS entries to JSONL. Returns the row count."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in parsed:
                if entry is None:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count


register(GssFactorsScraper())
