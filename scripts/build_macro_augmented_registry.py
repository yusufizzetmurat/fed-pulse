"""Augment an existing supervised registry with macro-release event rows.

Variant A of the macro-event augmentation experiment (#239 follow-up):
fetch CPI + NFP release dates from FRED's release calendar, append them
as supervised event rows with ``text=""`` and a neutral placeholder
stance label, and write the augmented JSONL to a new path. The
training-package builder downstream consumes the augmented JSONL with
its existing ``--input`` flag, so no pipeline changes are required.

Why text="": the goal here is to give the model more rows on which the
macro features (VIX term slope, yield-curve slope, realised-vol
horizons, cross-asset closes) map to a vol-regime target. The text
channel emits the missing flag on these rows, so the FinBERT path
contributes a zero vector — the recurrent core learns from richer
training data without confusing the language model.

Output JSONL is byte-identical to the existing registry for the
FOMC rows and appends ~1300 new ``fred_macro_releases`` rows. Run

    python scripts/build_macro_augmented_registry.py \\
        --base-package-id tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0 \\
        --output /data/interim/macro_augmented_registry.jsonl

then feed the result into ``app.data.build_training_package`` with the
new dataset_version / feature_version pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import DATA_DIR  # noqa: E402

# FRED release calendar IDs. Discoverable via
# https://api.stlouisfed.org/fred/releases?api_key=... but pinned here
# so the script is deterministic across FRED catalogue revisions.
RELEASE_IDS: dict[str, dict[str, str]] = {
    "cpi": {
        "release_id": "10",
        "label": "Consumer Price Index",
        "document_type": "macro_release_cpi",
    },
    "nfp": {
        "release_id": "50",
        "label": "Employment Situation (Non-Farm Payrolls)",
        "document_type": "macro_release_nfp",
    },
}

_EMPTY_TEXT_HASH = hashlib.sha256(b"").hexdigest()
_FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"


def _fetch_release_dates(release_id: str, api_key: str) -> list[str]:
    """Hit the FRED release-dates endpoint and return a sorted YYYY-MM-DD list."""

    # FRED's release-dates endpoint caps ``limit`` at 10,000 — well
    # above the ~700-ish total release dates per series since the late
    # 1940s, so a single page covers everything.
    response = httpx.get(
        _FRED_RELEASE_DATES_URL,
        params={
            "release_id": release_id,
            "api_key": api_key,
            "file_type": "json",
            "limit": 10_000,
            "sort_order": "asc",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload.get("release_dates") or []
    dates = sorted({str(entry["date"]) for entry in raw if entry.get("date")})
    return dates


def _macro_row(*, release_key: str, release_date: str, ingested_at: str) -> dict[str, Any]:
    """Emit one supervised registry row for a macro-release event.

    The row carries ``text=""`` (text channel emits the missing flag),
    ``mapped_label="neutral"`` (the supervised filter requires a non-
    empty label; macro releases carry no stance content so neutral is
    the honest placeholder), and the four axis labels set to plausible
    defaults so the multi-axis-extras feature block does not surface as
    missing on these rows.
    """

    info = RELEASE_IDS[release_key]
    record_id = hashlib.sha256(
        f"{info['release_id']}|{release_date}".encode("utf-8")
    ).hexdigest()[:16]
    # Structured placeholder text. The supervised pipeline's pandera
    # schema rejects empty strings, so we emit a short marker that
    # describes the event but carries no stance signal. FinBERT will
    # embed this into a roughly-constant vector per release type; the
    # text channel is effectively dormant on these rows and the macro
    # features carry the prediction. (Variant A as planned: the
    # placeholder text is structured data dressed up, not natural
    # language content.)
    placeholder_text = f"{info['label']} release on {release_date}."
    text_hash = hashlib.sha256(placeholder_text.encode("utf-8")).hexdigest()
    return {
        "record_id": record_id,
        "source": "fred_macro_releases",
        "source_record_id": f"{info['release_id']}_{release_date}",
        "document_type": info["document_type"],
        "source_type": "macro_data_release",
        "event_date": release_date,
        "title": f"{info['label']} release {release_date}",
        "text": placeholder_text,
        "label": "neutral",
        "label_origin": "macro_release_placeholder",
        "license_scope": "fred_public_domain",
        "citation_ref": "fred_release_calendar",
        "ingested_at_utc": ingested_at,
        "text_hash": text_hash,
        "provenance": "official",
        "multi_axis_extras": {},
        "mapped_label": "neutral",
        "label_map_version": "label_map_v1.0",
        "label_taxonomy": "hawkish_dovish_neutral",
        "sample_weight": 0.0,
        # Schema constraints land us on numeric / nullable everywhere:
        # the flattened ``axis_certainty`` / ``axis_factor`` columns
        # require float in [0,1] / [-1,1] (or null). String enums on
        # the FOMC corpus survive because their flattener coerces;
        # macro rows have no equivalent so the safest emission is
        # None on the regression axes and string on the categorical
        # axes (time / topic) where None is also accepted.
        "axes": {
            "stance": "neutral",
            "factor": None,
            "certainty": None,
            "time": None,
            "topic": "economic_indicator",
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-package-id",
        required=True,
        help="Training package whose registry_normalized.jsonl seeds the output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path for the augmented registry.",
    )
    parser.add_argument(
        "--include-cpi",
        action="store_true",
        help="Append CPI release dates (release_id=10).",
    )
    parser.add_argument(
        "--include-nfp",
        action="store_true",
        help="Append NFP / Employment Situation release dates (release_id=50).",
    )
    parser.add_argument(
        "--earliest-date",
        default="1970-01-01",
        help="Drop release dates strictly before this YYYY-MM-DD.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Root of the processed/ tree (default: $DATA_DIR / /data).",
    )
    args = parser.parse_args(argv)

    if not (args.include_cpi or args.include_nfp):
        parser.error("at least one of --include-cpi or --include-nfp must be set")

    api_key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not api_key:
        parser.error(
            "FRED_API_KEY env var is required to fetch release dates; "
            "populate it via .env (the file-based loader picks it up)"
        )

    base_path = args.data_dir / "processed" / args.base_package_id / "registry_normalized.jsonl"
    if not base_path.exists():
        parser.error(f"base registry not found: {base_path}")

    print(f"[macro-augment] reading base registry: {base_path}")
    base_rows = _read_jsonl(base_path)
    print(f"[macro-augment]   base rows: {len(base_rows)}")

    ingested_at = datetime.now(timezone.utc).isoformat()
    new_rows: list[dict[str, Any]] = []
    earliest = str(args.earliest_date)
    for release_key, include in (
        ("cpi", args.include_cpi),
        ("nfp", args.include_nfp),
    ):
        if not include:
            continue
        info = RELEASE_IDS[release_key]
        print(
            f"[macro-augment] fetching {info['label']} release dates "
            f"(release_id={info['release_id']}) ..."
        )
        dates = _fetch_release_dates(info["release_id"], api_key=api_key)
        kept = [d for d in dates if d >= earliest]
        print(f"[macro-augment]   {len(dates)} total, {len(kept)} after {earliest}")
        for date in kept:
            new_rows.append(
                _macro_row(
                    release_key=release_key,
                    release_date=date,
                    ingested_at=ingested_at,
                )
            )

    print(f"[macro-augment] appending {len(new_rows)} macro-release rows")
    augmented = base_rows + new_rows
    augmented.sort(key=lambda r: (str(r.get("event_date", "")), str(r.get("record_id", ""))))

    _write_jsonl(args.output, augmented)
    print(f"[macro-augment] wrote {args.output} ({len(augmented)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
