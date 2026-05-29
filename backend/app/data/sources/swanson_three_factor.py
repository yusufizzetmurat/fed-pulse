"""Swanson 2021 three-factor monetary-policy adapter.

Extends the GSS 2005 (Gürkaynak-Sack-Swanson) factor decomposition into the
ZLB era. Swanson (2021, JME) re-estimates the principal-component
decomposition over the pre- and post-ZLB samples and exposes a third axis
(LSAP / asset-purchase factor) on top of GSS's target-rate / forward-
guidance pair.

The canonical xlsx mirror is on Swanson's UC Irvine homepage:

    https://sites.socsci.uci.edu/~swanson2/papers/pre-and-post-ZLB-factors-extended.xlsx

One sheet ("Data") with one row per FOMC meeting, columns ``Federal Funds
Rate factor``, ``Forward Guidance factor``, ``LSAP factor``, and the
sign-flipped ``- LSAP factor``. Coverage 1991-07-05 through 2019-06-19 at
last verification (241 meetings). The xlsx is updated in place by the
author, so the adapter records the upstream URL on every row's
``multi_axis_extras`` so downstream consumers can audit which release
they are training against.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register

# Public xlsx mirror — verified live 2026-05-29. Pinned to the UCI
# homepage because Swanson's prior author domain (ericswanson.org) is
# dead per #420. The on-disk filename matches
# ``SWANSON_THREE_FACTOR_DEFAULT_RELATIVE`` in
# ``app.data.ingest_sources`` so the downstream
# ``--include-swanson-three-factor`` ingest path picks the pull up
# without extra wiring.
SWANSON_THREE_FACTOR_UPSTREAM_URL = (
    "https://sites.socsci.uci.edu/~swanson2/papers/"
    "pre-and-post-ZLB-factors-extended.xlsx"
)

_USER_AGENT = (
    "fed-pulse-data-ingester/1.0 "
    "(+https://github.com/yusufizzetmurat/fed-pulse)"
)


def pull_swanson_three_factor_xlsx(
    target_path: Path,
    *,
    force: bool = False,
    url: str = SWANSON_THREE_FACTOR_UPSTREAM_URL,
    timeout: float = 60.0,
) -> int:
    """Download the Swanson three-factor xlsx to ``target_path``.

    Returns the row count after parsing. Idempotent: when the cache
    exists and parses to a non-empty row set, the existing row count
    is returned without HTTP traffic. Atomic-on-POSIX via
    ``Path.replace`` after a sibling ``.tmp`` write.
    """

    # pandas import is deferred so module import does not pull the
    # heavy dependency for callers that only want the metadata.
    import pandas as pd

    if target_path.exists() and not force:
        try:
            cached = pd.read_excel(
                target_path, engine="openpyxl", sheet_name="Data", header=1
            )
            if len(cached) > 0:
                return int(len(cached))
        except Exception:
            pass  # fall through to re-pull on corrupt cache

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": _USER_AGENT}
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Swanson three-factor upstream returned HTTP {exc.code} "
                f"from {url}"
            ) from exc
        with response:
            body: bytes = response.read()
        tmp_path.write_bytes(body)
        parsed = pd.read_excel(
            tmp_path, engine="openpyxl", sheet_name="Data", header=1
        )
        row_count = int(len(parsed))
        if row_count == 0:
            raise RuntimeError(
                f"Swanson three-factor download from {url} produced zero rows"
            )
        tmp_path.replace(target_path)
        return row_count
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class SwansonThreeFactorScraper:
    """``BaseSourceScraper``-shaped wrapper around the xlsx release.

    The adapter does NOT walk an HTML index — the xlsx is the entire
    corpus in a single file — so ``fetch_listing`` returns a list of
    one dict per spreadsheet row and ``parse_entry`` accepts each one
    as a JSON-encoded string for protocol parity with the HTML-on-the-
    wire scrapers.
    """

    metadata = SourceMetadata(
        name="Swanson three-factor (2021 JME, UCI mirror)",
        source_type="swanson_three_factor",
        provenance=Provenance.PEER_REVIEWED,
        citation=(
            "Swanson, E. T. (2021). Measuring the Effects of Federal "
            "Reserve Forward Guidance and Asset Purchases on Financial "
            "Markets. J. Monetary Economics 118."
        ),
    )

    def fetch_listing(self, html: str) -> list[dict[str, Any]]:
        """Parse the xlsx-on-disk path passed via ``html`` and return the
        row dicts. ``html`` here carries the path string to keep the
        Protocol signature uniform across CSV/HTML/xlsx sources.
        """

        if not html:
            return []
        import pandas as pd

        df = pd.read_excel(
            html, engine="openpyxl", sheet_name="Data", header=1
        )
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "meeting_date": row.iloc[1],
                    "target_factor": row.get("Federal Funds Rate factor"),
                    "forward_guidance_factor": row.get(
                        "Forward Guidance factor"
                    ),
                    "lsap_factor": row.get("LSAP factor"),
                }
            )
        return rows

    def parse_entry(
        self, raw_html: str, *, source_url: str
    ) -> dict[str, Any] | None:
        try:
            row = json.loads(raw_html)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        parsed = _parse_swanson_row(row)
        if parsed is None:
            return None
        parsed["source_url"] = source_url
        return parsed

    def write(
        self, parsed: Iterable[dict[str, Any]], output_path: Path
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in parsed:
                if entry is None:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count


def _parse_swanson_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build one registry record from a Swanson row dict.

    Returns None when the meeting date or all three factor values are
    missing. Factor values are kept as floats; downstream consumers read
    them off ``multi_axis_extras``.
    """

    raw_date = row.get("meeting_date")
    if raw_date is None:
        return None
    iso_date: str = ""
    try:
        # pandas Timestamp / datetime
        iso_date = raw_date.strftime("%Y-%m-%d")
    except AttributeError:
        # xlsx text cells (e.g. "12/16/2015") don't have strftime. Coerce
        # to a Timestamp via pandas so the registry sees ISO format.
        import pandas as pd

        parsed_dt = pd.to_datetime(str(raw_date), errors="coerce")
        if pd.isna(parsed_dt):
            return None
        iso_date = parsed_dt.strftime("%Y-%m-%d")
    if not iso_date or len(iso_date) < 10:
        return None

    def _float_or_none(v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        # pandas reads blanks as NaN; treat NaN as missing
        if f != f:  # noqa: PLR0124
            return None
        return f

    target = _float_or_none(row.get("target_factor"))
    fg = _float_or_none(row.get("forward_guidance_factor"))
    lsap = _float_or_none(row.get("lsap_factor"))
    if target is None and fg is None and lsap is None:
        return None

    extras = {
        "swanson_target_factor": target,
        "swanson_forward_guidance_factor": fg,
        "swanson_lsap_factor": lsap,
        "swanson_upstream_url": SWANSON_THREE_FACTOR_UPSTREAM_URL,
    }
    title = f"Swanson three-factor decomposition {iso_date}"
    text = (
        f"Swanson three-factor decomposition for {iso_date}: "
        f"target={target if target is not None else 'n/a':+.4f}, "
        f"forward_guidance={fg if fg is not None else 'n/a':+.4f}, "
        f"lsap={lsap if lsap is not None else 'n/a':+.4f}"
    )
    return {
        "source_record_id": f"swanson_{iso_date}",
        "event_date_hint": iso_date,
        "document_type": "statement",
        "title": title,
        "text": text,
        "label": "",
        "license_scope": "research_only",
        "citation_ref": "swanson_2021_jme",
        "multi_axis_extras": extras,
    }


# Only register when imported as a module; see op_fed.py for the same
# guard's rationale (python -m re-executes this file as __main__ after
# the package init already imported and registered it).
if __name__ != "__main__":
    register(SwansonThreeFactorScraper())


if __name__ == "__main__":
    import argparse

    from app.config import DATA_DIR as _DEFAULT_DATA_DIR
    from app.data.ingest_sources import SWANSON_THREE_FACTOR_DEFAULT_RELATIVE

    parser = argparse.ArgumentParser(
        description=(
            "Pull the Swanson 2021 three-factor xlsx into the local data "
            "cache so `python -m app.data.ingest_sources "
            "--include-swanson-three-factor` can materialise rows into "
            "the source registry."
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
        default=SWANSON_THREE_FACTOR_UPSTREAM_URL,
        help="Override the upstream URL (default: UCI mirror).",
    )
    ns = parser.parse_args()
    target = Path(ns.data_dir) / SWANSON_THREE_FACTOR_DEFAULT_RELATIVE
    rows = pull_swanson_three_factor_xlsx(
        target, force=ns.force, url=ns.url
    )
    print(f"Swanson three-factor cache at {target} (rows: {rows})")
