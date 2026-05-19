"""B1 (#212) runner: extract LLM catalogue features for every event row.

Walks the events.parquet for the configured training package, resolves
each event's document text from the source registry, and calls the
extractor for any row not already in the cache. Cost-aware: a re-run
against an existing cache is free (skips every cached text_hash).

Usage::

    python scripts/extract_llm_features.py \\
        --training-package-id <pkg> \\
        [--limit 5]        # cap the number of API calls for dry runs
        [--dry-run]         # print what would be sent but make no API call

Cost on the full corpus (4 100 events, ~10K-50K tokens each):
~$7-10 with Claude Sonnet 4.7 at $3/1M input tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable


_DEFAULT_CACHE_DIR = Path("/data/raw/llm_features")
_LOG = logging.getLogger("extract_llm_features")


def _resolve_package_dir(training_package_id: str) -> Path:
    for c in (
        Path(f"/data/processed/{training_package_id}"),
        Path(f"data/processed/{training_package_id}"),
        Path(f"backend/data/processed/{training_package_id}"),
    ):
        if (c / "events.parquet").exists():
            return c
    raise FileNotFoundError(f"events.parquet missing for {training_package_id}")


def _iter_documents(
    training_package_id: str,
    *,
    limit: int | None = None,
    min_chars: int = 500,
) -> Iterable[tuple[str, str]]:
    """Yield (text_hash, document_text) for every event row, deduped on
    text_hash. Order is event_date asc so the cache fills temporally.

    Reads ``text`` directly from ``events.parquet`` -- the column is
    already populated by the event-dataset builder and carries the
    canonical document text per event row. Rows below ``min_chars``
    are filtered out so we don't spend an API call on a 30-char
    document fragment.
    """

    import pandas as pd

    pkg_dir = _resolve_package_dir(training_package_id)
    frame = pd.read_parquet(pkg_dir / "events.parquet")
    if "text" not in frame.columns:
        raise RuntimeError(
            f"events.parquet at {pkg_dir} is missing the 'text' column "
            "needed for LLM feature extraction"
        )
    frame = (
        frame[["text_hash", "event_date", "text"]]
        .drop_duplicates(subset="text_hash")
        .sort_values("event_date")
    )
    yielded = 0
    for _, row in frame.iterrows():
        text_hash = str(row["text_hash"])
        document = row.get("text") or ""
        if not isinstance(document, str) or len(document) < min_chars:
            continue
        yield text_hash, document
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of API calls. Useful for a dry-budget run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the (text_hash, document length) pairs that would be sent; no API call.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Root directory for the LLM-features cache parquet.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-extract every row currently in the cache, including "
            "rows marked ok. Use after editing the catalogue prompt "
            "wording (without bumping CATALOG_VERSION). Default: "
            "skip ok / document_too_short rows; retry api_error / "
            "invalid_json / out_of_vocab rows automatically."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help=(
            "Flush the cache parquet to disk after every N successful "
            "extractions. Default 1 (write per row) keeps SIGKILL "
            "recovery within a single document; raise to batch I/O."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    documents = list(_iter_documents(args.training_package_id, limit=args.limit))
    _LOG.info("Resolved %d (text_hash, document) pairs.", len(documents))

    if args.dry_run:
        for text_hash, text in documents[:5]:
            _LOG.info("dry-run: %s (%d chars)", text_hash[:12], len(text))
        if len(documents) > 5:
            _LOG.info("... and %d more (truncated for dry-run preview).", len(documents) - 5)
        return 0

    from app.data.llm_feature_extractor import extract_for_package

    cache_path = extract_for_package(
        training_package_id=args.training_package_id,
        documents=documents,
        cache_dir=args.cache_dir,
        retry_failed=args.retry_failed,
        progress_every=max(1, args.flush_every),
    )
    _LOG.info("Cache parquet: %s", cache_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
