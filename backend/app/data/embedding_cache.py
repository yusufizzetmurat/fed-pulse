"""Encoder-keyed embedding cache for the bake-off and credibility paths.

Writes one parquet per ``(encoder_alias, training_package_id)`` pair under
``data/raw/embeddings/`` together with a ``SOURCES.lock`` line recording the
encoder revision, the source-registry SHA, and the artefact SHA-256. The
parquet schema is the same as the legacy chunk-embedding store
(``doc_id, event_date, chunk_index, chunk_preview, embedding``) so the
existing ``chunk_embedding_retrieval`` consumer can read it with no change.

Network-dependent encoders are only loaded when the caller passes
``allow_network=True``. Without that flag the builder hard-fails before
touching HF — that is the policy training-time call sites rely on.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.config import DATA_DIR
from app.models.registry import EncoderRef, encoder_ref

DEFAULT_CACHE_DIR = DATA_DIR / "raw" / "embeddings"
DEFAULT_PREVIEW_CHARS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TEXT_CHARS = 64
DEFAULT_MAX_DOCS = 0  # 0 = no cap
DEFAULT_MAX_LENGTH = 512

# Encoders whose recipe is "one slot per document" rather than "many chunks per
# document". Sentence-embedding models typically want mean-pool over the whole
# input; classifier/MLM encoders use CLS pooling over chunks.
SENTENCE_EMBEDDING_TASKS = {"sentence_embedding"}


@dataclass(frozen=True)
class CachePaths:
    parquet: Path
    sources_lock: Path

    @property
    def directory(self) -> Path:
        return self.parquet.parent


def _short_revision(revision: str, length: int = 12) -> str:
    if not revision:
        return "unpinned"
    return revision[:length]


def resolve_cache_paths(
    encoder_alias: str,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
) -> CachePaths:
    """Return parquet + SOURCES.lock paths for a given encoder alias.

    The parquet name embeds a short revision slug so two different pinned
    revisions of the same encoder don't collide.
    """

    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    rev_slug = _short_revision(revision or "")
    parquet = base / f"{encoder_alias}_{rev_slug}.parquet"
    return CachePaths(parquet=parquet, sources_lock=base / "SOURCES.lock")


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


def _registry_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _load_encoder(ref: EncoderRef):
    """Lazy-import AutoModel/AutoTokenizer so test code can monkeypatch."""

    from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

    revision = ref.revision or None
    tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=revision)
    model = AutoModel.from_pretrained(ref.repo, revision=revision)
    return tokenizer, model


def _split_text_for_encoder(
    text: str,
    *,
    tokenizer,
    max_length: int,
    sentence_embedding: bool,
) -> list[str]:
    """Produce the input strings to be embedded for one document.

    Sentence-embedding encoders see the whole document (truncated by the
    tokenizer). Chunk-pool encoders see token-window chunks so the model
    output is per-chunk CLS, matching the legacy chunk store.
    """

    if sentence_embedding:
        return [text]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    stride = max(1, max_length - 2)  # leave room for [CLS] / [SEP]
    chunks: list[str] = []
    for start in range(0, len(token_ids), stride):
        slice_ids = token_ids[start : start + stride]
        if not slice_ids:
            continue
        chunks.append(tokenizer.decode(slice_ids, skip_special_tokens=True))
    return chunks


def _embed_batch(
    inputs: list[str],
    *,
    tokenizer,
    model,
    max_length: int,
    sentence_embedding: bool,
) -> list[list[float]]:
    import torch  # local import keeps test-time monkeypatching cheap

    if not inputs:
        return []
    enc = tokenizer(
        inputs,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, T, H)
        if sentence_embedding:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        else:
            pooled = hidden[:, 0, :]
    return pooled.detach().cpu().tolist()


@dataclass(frozen=True)
class BuildResult:
    parquet_path: Path
    row_count: int
    encoder_alias: str
    encoder_revision: str
    sources_lock_path: Path


def build_cache(
    *,
    encoder_alias: str,
    training_package_id: str,
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_docs: int = DEFAULT_MAX_DOCS,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    max_length: int = DEFAULT_MAX_LENGTH,
    allow_network: bool = False,
    force: bool = False,
) -> BuildResult:
    """Build (or reuse) the per-encoder embedding parquet for a package.

    ``allow_network`` MUST be set explicitly when the encoder is not already
    cached locally — the builder will instantiate the HF model otherwise and
    hit the network. Training-time call sites pass ``allow_network=False``
    and rely on this hard-fail to prevent silent downloads.
    """

    ref = encoder_ref(encoder_alias)
    if ref is None:
        raise ValueError(
            f"Unknown encoder alias {encoder_alias!r}. Add it to backend/app/models/registry.yaml."
        )
    if not ref.revision:
        raise ValueError(
            f"Encoder {encoder_alias!r} has no pinned revision in registry.yaml. "
            "Train the checkpoint or paste a revision SHA before caching."
        )

    package_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    registry_path = package_dir / "processed" / training_package_id / "registry_normalized.jsonl"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Missing training-package registry at {registry_path}. "
            "Run pipeline_data_prep before caching embeddings."
        )

    paths = resolve_cache_paths(encoder_alias, revision=ref.revision, cache_dir=cache_dir)
    paths.directory.mkdir(parents=True, exist_ok=True)

    if paths.parquet.exists() and not force:
        existing = pd.read_parquet(paths.parquet)
        return BuildResult(
            parquet_path=paths.parquet,
            row_count=len(existing),
            encoder_alias=encoder_alias,
            encoder_revision=ref.revision,
            sources_lock_path=paths.sources_lock,
        )

    if not allow_network:
        raise RuntimeError(
            f"Embedding cache missing for encoder={encoder_alias!r} at {paths.parquet}. "
            "Re-run with allow_network=True (or `make cache-embeddings ENCODER=… ALLOW_NETWORK=1`)."
        )

    sentence_embedding = ref.task in SENTENCE_EMBEDDING_TASKS
    tokenizer, model = _load_encoder(ref)
    model.eval()

    pending_inputs: list[str] = []
    pending_meta: list[tuple[str, str, str, int, str]] = []  # (record_id, doc_id, event_date, chunk_index, preview)
    rows: list[dict[str, Any]] = []
    docs_processed = 0
    started_at = time.time()

    def _flush() -> None:
        if not pending_inputs:
            return
        embeddings = _embed_batch(
            pending_inputs,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            sentence_embedding=sentence_embedding,
        )
        for (record_id, doc_id, event_date, chunk_index, preview), embedding in zip(pending_meta, embeddings):
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

    for record in _iter_registry(registry_path, min_text_chars=min_text_chars):
        if max_docs and docs_processed >= max_docs:
            break
        text = str(record["text"]).strip()
        chunks = _split_text_for_encoder(
            text,
            tokenizer=tokenizer,
            max_length=max_length,
            sentence_embedding=sentence_embedding,
        )
        if not chunks:
            continue
        doc_id = _doc_id(record)
        record_id = str(record.get("record_id") or doc_id)
        event_date = str(record["event_date"]).strip()
        for chunk_index, chunk_text in enumerate(chunks):
            preview = chunk_text[:preview_chars]
            pending_inputs.append(chunk_text)
            pending_meta.append((record_id, doc_id, event_date, chunk_index, preview))
            if len(pending_inputs) >= batch_size:
                _flush()
        docs_processed += 1

    _flush()
    elapsed = time.time() - started_at
    print(
        f"[embedding_cache] encoder={encoder_alias} docs={docs_processed} "
        f"rows={len(rows)} elapsed={elapsed:.1f}s"
    )

    df = pd.DataFrame(rows)
    df["event_date"] = df["event_date"].astype(str)
    df.to_parquet(paths.parquet, index=False)
    artefact_sha = _file_sha256(paths.parquet)
    registry_sha = _registry_sha256(registry_path)

    sources_entry = {
        "encoder_alias": encoder_alias,
        "encoder_repo": ref.repo,
        "encoder_revision": ref.revision,
        "encoder_task": ref.task,
        "training_package_id": training_package_id,
        "registry_sha256": registry_sha,
        "parquet_relpath": str(paths.parquet.relative_to(paths.directory)),
        "parquet_sha256": artefact_sha,
        "row_count": len(rows),
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with paths.sources_lock.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sources_entry, sort_keys=True) + "\n")

    return BuildResult(
        parquet_path=paths.parquet,
        row_count=len(rows),
        encoder_alias=encoder_alias,
        encoder_revision=ref.revision,
        sources_lock_path=paths.sources_lock,
    )


def require_cache_exists(
    encoder_alias: str,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
) -> Path:
    """Return the parquet path or raise with a fixable error message."""

    paths = resolve_cache_paths(encoder_alias, revision=revision, cache_dir=cache_dir)
    if not paths.parquet.exists():
        cmd = f"make cache-embeddings ENCODER={encoder_alias} ALLOW_NETWORK=1"
        raise FileNotFoundError(
            f"Embedding cache missing for encoder={encoder_alias!r} at {paths.parquet}. "
            f"Run: {cmd}"
        )
    return paths.parquet


def _parse_args() -> "object":
    import argparse

    parser = argparse.ArgumentParser(description="Build per-encoder embedding cache for a training package.")
    parser.add_argument("--encoder", required=True, help="Encoder alias from models/registry.yaml.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--data-dir", default=os.environ.get("FED_PULSE_DATA_DIR") or str(DATA_DIR))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_cache(
        encoder_alias=args.encoder,
        training_package_id=args.training_package_id,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        min_text_chars=args.min_text_chars,
        max_length=args.max_length,
        allow_network=args.allow_network,
        force=args.force,
    )
    print(
        f"[embedding_cache] {result.encoder_alias} rev={result.encoder_revision[:12]} "
        f"rows={result.row_count} parquet={result.parquet_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
