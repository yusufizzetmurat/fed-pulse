"""Cache voyage-finance-2 sentence embeddings for the FOMC corpus.

The Voyage AI ``voyage-finance-2`` model is a 1024-dim finance-tuned
sentence-embedding encoder. It is delivered through Voyage's hosted
REST API rather than the Hugging Face hub, so it cannot be cached
through :mod:`app.data.embedding_cache` (which loads a HF tokenizer +
model). This script mirrors the existing cache's parquet schema so the
downstream chunk-embedding consumer reads voyage rows with no shape
change.

The output parquet lives at
``data/raw/embeddings/voyage_finance_2_<slug>.parquet`` with columns
``record_id, doc_id, event_date, chunk_index, chunk_preview,
embedding``. A SOURCES.lock entry under
``data/raw/embeddings/SOURCES.lock`` carries the encoder alias, model
name, parquet sha256, registry sha256, row count, and retrieval
timestamp -- one JSON line per encoder, matching the format used by
the embedding_cache writer.

The Voyage API expects ``input`` as a list of strings (or one string)
and returns embeddings sized ``embed_dim`` per input. We batch up to
:data:`DEFAULT_BATCH_SIZE` inputs per request and retry transient
``429`` / ``5xx`` responses with exponential backoff.

CLI
---

::

    python scripts/cache_voyage_embeddings.py --allow-network \\
        --training-package-id tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0

The ``--allow-network`` flag must be passed explicitly. Without it the
script hard-fails before contacting Voyage so accidental network calls
on a training-time path raise a clear error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import DATA_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = DATA_DIR / "raw" / "embeddings"
DEFAULT_MODEL = "voyage-finance-2"
DEFAULT_ENCODER_ALIAS = "voyage_finance_2"
DEFAULT_TRAINING_PACKAGE_ID = (
    "tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0"
)
DEFAULT_MIN_TEXT_CHARS = 64
DEFAULT_PREVIEW_CHARS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_DOCS = 0  # 0 = no cap
DEFAULT_INPUT_TYPE = "document"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_API_BASE = "https://api.voyageai.com/v1/embeddings"
SOURCES_LOCK_NAME = "SOURCES.lock"

# Retry policy for transient failures (429 + 5xx).
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_slug(value: str, length: int = 12) -> str:
    """Return a short, filesystem-safe slug for a model name."""

    return value.replace("/", "_").replace("-", "_")[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _doc_id(record: dict[str, Any]) -> str:
    raw = (
        record.get("record_id")
        or record.get("source_record_id")
        or record.get("text_hash")
    )
    if isinstance(raw, str) and raw:
        return raw
    text = str(record.get("text", "")).strip()
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _iter_registry(path: Path, *, min_text_chars: int) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(record.get("text", "")).strip()
            event_date = str(record.get("event_date", "")).strip()
            if not text or not event_date:
                continue
            if len(text) < min_text_chars:
                continue
            yield record


# ---------------------------------------------------------------------------
# Voyage REST client
# ---------------------------------------------------------------------------


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key.strip()
    env_value = os.environ.get("VOYAGE_API_KEY")
    if not env_value:
        raise RuntimeError(
            "VOYAGE_API_KEY not set. Add it to .env or pass --api-key."
        )
    return env_value.strip()


def _post_voyage_embeddings(
    client: httpx.Client,
    *,
    api_url: str,
    api_key: str,
    inputs: list[str],
    model: str,
    input_type: str,
) -> list[list[float]]:
    """POST one batch to the Voyage embeddings endpoint with retries.

    Treats 429 + 5xx as transient and applies exponential backoff. Any
    other status raises ``httpx.HTTPStatusError``.
    """

    payload = {
        "model": model,
        "input": inputs,
        "input_type": input_type,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    backoff = INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None
    for _attempt in range(MAX_RETRIES):
        try:
            response = client.post(api_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:  # noqa: PERF203 -- network retries
            last_exc = exc
            time.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff *= 2
            continue
        if response.status_code == 429 or 500 <= response.status_code < 600:
            last_exc = httpx.HTTPStatusError(
                f"voyage transient {response.status_code}",
                request=response.request,
                response=response,
            )
            time.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff *= 2
            continue
        response.raise_for_status()
        body = response.json()
        rows = body.get("data") or []
        # Voyage returns rows in input order, but each carries an
        # ``index`` field. Sort by index to keep ordering robust.
        rows_sorted = sorted(rows, key=lambda r: int(r.get("index", 0)))
        return [list(r["embedding"]) for r in rows_sorted]
    raise RuntimeError(
        f"voyage embedding call failed after {MAX_RETRIES} attempts"
    ) from last_exc


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def cache_voyage_embeddings(
    *,
    training_package_id: str,
    model: str,
    encoder_alias: str,
    data_dir: Path,
    cache_dir: Path,
    batch_size: int,
    max_docs: int,
    min_text_chars: int,
    preview_chars: int,
    input_type: str,
    allow_network: bool,
    api_key: str | None,
    api_url: str,
    force: bool,
) -> dict[str, Any]:
    """Compute + persist the voyage-finance-2 embedding parquet.

    Returns a dict carrying the parquet path, row count, sha256, and
    SOURCES.lock entry. The dict is suitable for printing or for a
    downstream test harness that wants to assert the contract without
    re-reading the parquet from disk.
    """

    registry_path = (
        data_dir / "processed" / training_package_id / "registry_normalized.jsonl"
    )
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Missing training-package registry at {registry_path}. "
            "Run pipeline_data_prep before caching voyage embeddings."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = _short_slug(model)
    parquet_path = cache_dir / f"{encoder_alias}_{slug}.parquet"
    sources_lock_path = cache_dir / SOURCES_LOCK_NAME

    if parquet_path.exists() and not force:
        existing = pd.read_parquet(parquet_path)
        return {
            "parquet_path": parquet_path,
            "rows": int(len(existing)),
            "reused_existing_cache": True,
            "sha256": _file_sha256(parquet_path),
            "sources_lock_path": sources_lock_path,
        }

    if not allow_network:
        raise RuntimeError(
            f"Voyage embedding cache missing at {parquet_path}. "
            "Re-run with --allow-network to call the Voyage REST API."
        )

    api_key_resolved = _resolve_api_key(api_key)
    rows: list[dict[str, Any]] = []
    pending_inputs: list[str] = []
    pending_meta: list[tuple[str, str, str, int, str]] = []
    docs_processed = 0
    started_at = time.time()

    def _flush(client: httpx.Client) -> None:
        if not pending_inputs:
            return
        embeddings = _post_voyage_embeddings(
            client,
            api_url=api_url,
            api_key=api_key_resolved,
            inputs=pending_inputs,
            model=model,
            input_type=input_type,
        )
        for (record_id, doc_id, event_date, chunk_index, preview), embedding in zip(
            pending_meta, embeddings, strict=False
        ):
            rows.append(
                {
                    "record_id": record_id,
                    "doc_id": doc_id,
                    "event_date": event_date,
                    "chunk_index": chunk_index,
                    "chunk_preview": preview,
                    "embedding": embedding,
                }
            )
        pending_inputs.clear()
        pending_meta.clear()

    progress_step = 200
    last_progress_at = started_at
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        for record in _iter_registry(registry_path, min_text_chars=min_text_chars):
            if max_docs and docs_processed >= max_docs:
                break
            text = str(record["text"]).strip()
            doc_id = _doc_id(record)
            record_id = str(record.get("record_id") or doc_id)
            event_date = str(record["event_date"]).strip()
            preview = text[:preview_chars]
            # voyage-finance-2 is a sentence-embedding encoder; one
            # row per document with chunk_index=0 matches the cache
            # shape that BGE / Nomic already write.
            pending_inputs.append(text)
            pending_meta.append((record_id, doc_id, event_date, 0, preview))
            docs_processed += 1
            if len(pending_inputs) >= batch_size:
                _flush(client)
            if docs_processed % progress_step == 0:
                now = time.time()
                docs_per_sec = (
                    progress_step / max(now - last_progress_at, 1e-6)
                )
                last_progress_at = now
                print(
                    f"[voyage] progress docs={docs_processed} "
                    f"rate={docs_per_sec:.1f}/s elapsed={now - started_at:.0f}s",
                    flush=True,
                )
        _flush(client)

    elapsed = time.time() - started_at
    print(
        f"[voyage] encoder={encoder_alias} model={model} docs={docs_processed} "
        f"rows={len(rows)} elapsed={elapsed:.1f}s"
    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "Voyage cache build produced zero rows -- "
            "check the registry path and min_text_chars filter."
        )
    df["event_date"] = df["event_date"].astype(str)
    df.to_parquet(parquet_path, index=False)

    parquet_sha = _file_sha256(parquet_path)
    registry_sha = _file_sha256(registry_path)
    entry = {
        "encoder_alias": encoder_alias,
        "encoder_repo": f"voyageai/{model}",
        "encoder_revision": model,
        "encoder_task": "sentence_embedding",
        "training_package_id": training_package_id,
        "registry_sha256": registry_sha,
        "parquet_relpath": parquet_path.name,
        "parquet_sha256": parquet_sha,
        "row_count": int(len(df)),
        "embedding_dim": int(len(df["embedding"].iloc[0])),
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with sources_lock_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return {
        "parquet_path": parquet_path,
        "rows": int(len(df)),
        "reused_existing_cache": False,
        "sha256": parquet_sha,
        "sources_lock_path": sources_lock_path,
        "embedding_dim": int(len(df["embedding"].iloc[0])),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache voyage-finance-2 sentence embeddings for a training "
            "package. The default training-package-id targets the v2 "
            "sprint-1 sentiment+market-core package."
        )
    )
    parser.add_argument(
        "--training-package-id",
        default=DEFAULT_TRAINING_PACKAGE_ID,
        help="Training-package id under data/processed/.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Voyage model name (default: voyage-finance-2).",
    )
    parser.add_argument(
        "--encoder-alias",
        default=DEFAULT_ENCODER_ALIAS,
        help="Encoder alias slug used in the parquet filename and the lock entry.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FED_PULSE_DATA_DIR") or str(DATA_DIR),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument(
        "--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS
    )
    parser.add_argument(
        "--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS
    )
    parser.add_argument(
        "--input-type",
        default=DEFAULT_INPUT_TYPE,
        choices=("document", "query"),
        help="Voyage input-type marker (default: document).",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--api-key",
        default=None,
        help="VOYAGE_API_KEY override. Reads VOYAGE_API_KEY from env if absent.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Required to call the Voyage REST API. Defensive default off.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-build even if the parquet already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    result = cache_voyage_embeddings(
        training_package_id=args.training_package_id,
        model=args.model,
        encoder_alias=args.encoder_alias,
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        min_text_chars=args.min_text_chars,
        preview_chars=args.preview_chars,
        input_type=args.input_type,
        allow_network=args.allow_network,
        api_key=args.api_key,
        api_url=args.api_url,
        force=args.force,
    )
    if result.get("reused_existing_cache"):
        print(
            f"[voyage] reused existing cache at {result['parquet_path']} "
            f"(rows={result['rows']}, sha256={result['sha256'][:12]})"
        )
    else:
        print(
            f"[voyage] parquet={result['parquet_path']} "
            f"rows={result['rows']} embedding_dim={result['embedding_dim']} "
            f"sha256={result['sha256'][:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
