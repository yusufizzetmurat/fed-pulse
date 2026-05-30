"""Op-Fed external-corpus adapter (Keith et al. 2025).

Wraps the existing `_iter_op_fed_records` read path under the
`BaseSourceScraper` Protocol so the replication-package corpus advertises the
same metadata / contract surface as the in-house scrapers. The on-disk format
is CSV, not HTML, so `fetch_listing` reads the CSV path as text and yields one
parsed-row dict per data line. `parse_entry` accepts a single CSV row (as a
JSON-encoded string for parity with the HTML-on-the-wire scrapers) and emits
the same registry record `_iter_op_fed_records` builds. `write` serialises
parsed entries to a JSONL file under the registry's text-source contract.

The adapter is the first external replication-package corpus to ship a
`BaseSourceScraper` wrapper. Earlier adapters (Op-Fed, GSS, vtasca FOMC
archive, gtfintechlab cross-bank) reached the source registry via direct
`_iter_*` loaders in `ingest_sources.py` only.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register

# Public mirror of the Op-Fed v1 CSV release published alongside Keith et
# al. (2025), arXiv:2509.13539. Pinned to the ``main`` branch's
# ``data/opfed_v1.csv`` per the repo README. The on-disk filename matches
# ``OP_FED_DEFAULT_RELATIVE`` in ``app.data.ingest_sources`` so the
# downstream ``--include-op-fed`` ingest path picks the pull up without
# extra wiring.
OP_FED_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/kakeith/op-fed/main/data/opfed_v1.csv"
)

# Stance mapping kept local so the adapter is loadable without importing
# ingest_sources (avoids a hard import cycle if ingest_sources is later split).
_OP_FED_STANCE_MAP: dict[str, str] = {
    "entailment": "hawkish",
    "contradiction": "dovish",
    "neutral": "neutral",
}


def _parse_op_fed_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Parse one Op-Fed CSV row into the dict shape downstream consumers expect.

    Returns None when the row is missing the unique id, the sentence text, or
    a parseable date. The parsed dict carries the same keys the registry
    writer in `ingest_sources._iter_op_fed_records` populates so the two paths
    stay byte-compatible.
    """

    unique_id = (row.get("unique_id") or "").strip()
    if not unique_id:
        return None
    date_part = unique_id.split("_", 1)[0]
    # Loose validation; ingest pipeline applies `_coerce_event_date` itself.
    if not date_part or len(date_part) < 8:
        return None
    sentence = (row.get("sentence") or "").strip().strip('"')
    if not sentence:
        return None
    stance_raw = (row.get("4_stance_nli") or "").strip().lower()
    label = _OP_FED_STANCE_MAP.get(stance_raw, "")
    speaker = (row.get("speaker") or "").strip()
    title = f"FOMC meeting transcript {date_part} — {speaker}".strip(" —")
    extras = {
        "op_fed_opinion": (row.get("1_opinion") or "").strip(),
        "op_fed_mp": (row.get("2_mp") or "").strip(),
        "op_fed_mp_context": (row.get("3_mp_context") or "").strip(),
        "op_fed_stance_nli": stance_raw,
        "op_fed_stance_nli_context": (row.get("5_stance_nli_context") or "").strip(),
    }
    extras = {k: v for k, v in extras.items() if v}
    return {
        "source_record_id": unique_id,
        "event_date_hint": date_part,
        "document_type": "meeting_transcript",
        "title": title,
        "text": sentence,
        "label": label,
        "license_scope": "mit",
        "citation_ref": "keith_etal_2025_op_fed",
        "stance_raw": stance_raw,
        "multi_axis_extras": extras,
    }


class OpFedScraper:
    """File-backed `BaseSourceScraper` adapter for the Op-Fed CSV release.

    Op-Fed publishes one CSV per release; the adapter reads the entire file
    in `fetch_listing` rather than walking a paginated HTML index. The
    Protocol's `html` parameter therefore carries the raw CSV text.
    """

    metadata = SourceMetadata(
        name="Op-Fed (Keith et al. 2025)",
        source_type="fomc_meeting_transcript",
        provenance=Provenance.PEER_REVIEWED,
        citation="Keith, K. A. et al. (2025). Op-Fed. arXiv:2509.13539",
    )

    def fetch_listing(self, html: str) -> list[dict[str, str]]:
        """Read raw CSV text and return the list of row dicts.

        `html` is the entire file contents — keeping the Protocol name so the
        adapter sits in the same registry as the HTML-scraped sources.
        """

        if not html:
            return []
        reader = csv.DictReader(io.StringIO(html))
        return [dict(row) for row in reader]

    def parse_entry(self, raw_html: str, *, source_url: str) -> dict[str, Any] | None:
        """Parse a single Op-Fed CSV row.

        `raw_html` is a JSON-encoded row dict (so the Protocol signature stays
        a plain string). `source_url` records provenance on the parsed entry.
        """

        try:
            row = json.loads(raw_html)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        parsed = _parse_op_fed_row({str(k): str(v) if v is not None else "" for k, v in row.items()})
        if parsed is None:
            return None
        parsed["source_url"] = source_url
        return parsed

    def write(self, parsed: Iterable[dict[str, Any]], output_path: Path) -> int:
        """Serialise parsed Op-Fed entries to JSONL. Returns the row count."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in parsed:
                if entry is None:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count


# Only register when imported as a module. When the file runs under
# `python -m app.data.sources.op_fed`, Python first imports the
# package (which imports this file as a module and registers the
# scraper), then re-executes this file as __main__. Without the guard
# the second pass tries to register again and the registry raises a
# duplicate-source_type error.
if __name__ != "__main__":
    register(OpFedScraper())


def pull_op_fed_csv(
    target_path: Path,
    *,
    force: bool = False,
    url: str = OP_FED_UPSTREAM_URL,
    timeout: float = 60.0,
) -> int:
    """Download the Op-Fed CSV to ``target_path``. Returns the parsed row count.

    Idempotent: when ``target_path`` already exists and ``force`` is False
    the existing file is re-counted and returned. If the cache parses to
    zero rows it is treated as corrupt and a re-pull is forced.

    Best-effort atomic on POSIX: the body is written to a sibling ``.tmp``
    file, parsed as CSV, and only renamed into place once it parses to a
    non-empty row set. ``Path.replace`` is a single ``rename(2)`` on
    Linux/macOS; on Windows the rename is not atomic. Any HTTP error or
    zero-row parse raises ``RuntimeError`` and removes the tmp file.
    """

    if target_path.exists() and not force:
        with target_path.open("r", encoding="utf-8", newline="") as handle:
            cached_rows = sum(1 for _ in csv.DictReader(handle))
        if cached_rows > 0:
            return cached_rows
        # Cache exists but is empty — treat as corrupt and re-pull.

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        try:
            response = urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Op-Fed upstream returned HTTP {exc.code} from {url}"
            ) from exc
        with response:
            body = response.read()
        tmp_path.write_bytes(body)
        with tmp_path.open("r", encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        if row_count == 0:
            raise RuntimeError(
                f"Op-Fed download from {url} produced zero rows"
            )
        tmp_path.replace(target_path)
        return row_count
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


if __name__ == "__main__":
    import argparse

    from app.config import DATA_DIR as _DEFAULT_DATA_DIR
    from app.data.ingest_sources import OP_FED_DEFAULT_RELATIVE

    parser = argparse.ArgumentParser(
        description=(
            "Pull the Op-Fed CSV release into the local data cache so "
            "`python -m app.data.ingest_sources --include-op-fed` can "
            "materialise rows into the source registry."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        help="Base data directory (default: app.config.DATA_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the cache file already exists.",
    )
    parser.add_argument(
        "--url",
        default=OP_FED_UPSTREAM_URL,
        help="Override the upstream URL (default: kakeith/op-fed main).",
    )
    ns = parser.parse_args()
    target = Path(ns.data_dir) / OP_FED_DEFAULT_RELATIVE
    rows = pull_op_fed_csv(target, force=ns.force, url=ns.url)
    print(f"Op-Fed cache at {target} (rows: {rows})")
