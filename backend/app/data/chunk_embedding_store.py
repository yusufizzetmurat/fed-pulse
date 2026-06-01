"""Persist chunk-level CLS embeddings for every FOMC document in the registry.

Output: ``data/processed/<package_id>/chunk_embeddings.parquet`` with columns
``doc_id, event_date, chunk_index, chunk_preview, embedding`` (embedding stored
as a list of floats — pyarrow widens it to a fixed-size list at write time).

The Phase 4 attention ablation reads this store at training time to retrieve
chunk embeddings from FOMC documents within a lookback window. We embed once,
ablate cheaply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch

from app.services.text_encoder import (
    DEFAULT_CLASSIFIER_MAX_LENGTH,
    get_classifier,
    split_into_chunks,
)

DEFAULT_PREVIEW_CHARS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TEXT_CHARS = 64
DEFAULT_MAX_DOCS = 0  # 0 = no cap


def _doc_id(record: dict[str, Any]) -> str:
    raw = record.get("record_id") or record.get("source_record_id") or record.get("text_hash")
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


def _embed_batch(
    chunks: list[str],
    *,
    classifier,
    device: torch.device,
) -> list[list[float]]:
    if not chunks:
        return []
    enc = classifier.tokenizer(
        chunks,
        truncation=True,
        max_length=DEFAULT_CLASSIFIER_MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = classifier.model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, T, H)
        cls = hidden[:, 0, :]  # CLS token
    return cls.detach().cpu().tolist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunk embedding store for a training package."
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--output-name", default="chunk_embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="0 = no cap.")
    parser.add_argument("--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS)
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS)
    parser.add_argument("--force", action="store_true", help="Overwrite existing store.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    package_dir = Path(args.data_dir) / "processed" / args.training_package_id
    registry_path = package_dir / "registry_normalized.jsonl"
    if not registry_path.exists():
        raise SystemExit(f"Missing registry: {registry_path}")

    output_path = package_dir / args.output_name
    if output_path.exists() and not args.force:
        existing = pd.read_parquet(output_path)
        print(f"[chunk_store] reusing existing store: {output_path} (rows={len(existing)})")
        return 0

    classifier = get_classifier()
    device = next(classifier.model.parameters()).device
    print(f"[chunk_store] device={device} embedding_dim={classifier.model.config.hidden_size}")

    pending_chunks: list[str] = []
    pending_meta: list[tuple[str, str, int, str]] = []  # (doc_id, event_date, chunk_index, preview)
    rows: list[dict[str, Any]] = []
    embedded = 0
    docs_processed = 0
    started_at = time.time()

    def _flush() -> None:
        nonlocal embedded
        if not pending_chunks:
            return
        embeddings = _embed_batch(pending_chunks, classifier=classifier, device=device)
        for (doc_id, event_date, chunk_index, preview), embedding in zip(pending_meta, embeddings):
            rows.append(
                {
                    "doc_id": doc_id,
                    "event_date": event_date,
                    "chunk_index": chunk_index,
                    "chunk_preview": preview,
                    "embedding": embedding,
                }
            )
        embedded += len(pending_chunks)
        pending_chunks.clear()
        pending_meta.clear()

    for record in _iter_registry(registry_path, min_text_chars=args.min_text_chars):
        if args.max_docs and docs_processed >= args.max_docs:
            break
        text = str(record["text"]).strip()
        chunks = split_into_chunks(text, classifier=classifier)
        if not chunks:
            continue
        doc_id = _doc_id(record)
        event_date = str(record["event_date"]).strip()
        for chunk_index, chunk_text in enumerate(chunks):
            preview = chunk_text[: args.preview_chars]
            pending_chunks.append(chunk_text)
            pending_meta.append((doc_id, event_date, chunk_index, preview))
            if len(pending_chunks) >= args.batch_size:
                _flush()
        docs_processed += 1
        if docs_processed % 200 == 0:
            elapsed = time.time() - started_at
            rate = docs_processed / max(elapsed, 1e-6)
            print(
                f"[chunk_store] docs={docs_processed} chunks={embedded + len(pending_chunks)} rate={rate:.1f} doc/s"
            )

    _flush()
    elapsed = time.time() - started_at
    print(
        f"[chunk_store] done: docs={docs_processed} chunks={embedded} "
        f"elapsed={elapsed:.1f}s rate={docs_processed / max(elapsed, 1e-6):.1f} doc/s"
    )

    df = pd.DataFrame(rows)
    df["event_date"] = df["event_date"].astype(str)
    df.to_parquet(output_path, index=False)
    print(f"[chunk_store] wrote {len(df)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
