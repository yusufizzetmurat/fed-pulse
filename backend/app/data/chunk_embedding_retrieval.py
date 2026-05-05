"""Read helpers for the chunk embedding store.

Loads the parquet produced by :mod:`app.data.chunk_embedding_store` and exposes
a ``build_lookback_tensors`` function that returns padded ``(embeddings,
elapsed_days, mask)`` tensors suitable for :class:`ChunkAttentionPooler`.

Variant C (``embedding_source="llm"``) reads from the LLM embeddings parquet
produced by :mod:`app.data.llm_embedding_store`.  Each document contributes
exactly one row (one "chunk-of-one"), so the output tensor shape
``(max_chunks, embedding_dim)`` is identical to the chunk path; the pooler
sees the same interface regardless of which source is active.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_LLM_EMBEDDINGS_PARQUET = DEFAULT_DATA_DIR / "interim" / "phase2" / "llm_embeddings.parquet"


@dataclass(frozen=True)
class ChunkRetrievalResult:
    embeddings: torch.Tensor  # (max_chunks, D)
    elapsed_days: torch.Tensor  # (max_chunks,)
    mask: torch.Tensor  # (max_chunks,) — 1 where valid, 0 where padded
    doc_ids: list[str]
    event_dates: list[str]
    chunk_previews: list[str]
    actual_count: int


@lru_cache(maxsize=8)
def load_chunk_store(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["event_date"] = df["event_date"].astype(str)
    df = df.sort_values(["event_date", "doc_id", "chunk_index"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=4)
def _load_llm_store(path: str) -> pd.DataFrame:
    """Load the LLM embeddings parquet (or jsonl fallback) into a DataFrame.

    Expected columns: ``document_id``, ``event_date``, ``embedding``.
    """
    p = Path(path)
    if p.exists() and p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        # Fallback: check for .jsonl alongside the parquet path.
        jsonl_path = p.with_suffix(".jsonl")
        if jsonl_path.exists():
            rows: list[dict] = []
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            df = pd.DataFrame(rows)
        else:
            raise FileNotFoundError(
                f"LLM embeddings not found at {p} or {p.with_suffix('.jsonl')}. "
                "Run app.data.llm_embedding_store first."
            )
    df["event_date"] = df["event_date"].astype(str)
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def _parse_iso(date_str: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return None


def build_lookback_tensors(
    store: pd.DataFrame,
    *,
    anchor_date: str,
    lookback_days: int,
    max_chunks: int,
    embedding_size: int | None = None,
    include_anchor_day: bool = True,
    embedding_source: str = "chunk",
) -> ChunkRetrievalResult:
    """Return padded tensors for docs within ``[anchor - lookback, anchor]``.

    Parameters
    ----------
    store:
        For ``embedding_source="chunk"`` this is the chunk-level parquet
        DataFrame (columns: ``doc_id``, ``event_date``, ``chunk_index``,
        ``embedding``, ``chunk_preview``).  For ``embedding_source="llm"``
        this argument is *ignored*; the LLM embeddings parquet is loaded from
        ``DEFAULT_LLM_EMBEDDINGS_PARQUET`` automatically.
    embedding_source:
        ``"chunk"`` (default, Variant B) or ``"llm"`` (Variant C).
        When ``"llm"`` each document contributes exactly one slot so the
        output shape ``(max_chunks, embedding_dim)`` is identical to the
        chunk path, making the pooler interface source-agnostic.

    The most recent docs win when ``max_chunks`` cap is hit.
    """
    if embedding_source == "llm":
        return _build_lookback_tensors_llm(
            anchor_date=anchor_date,
            lookback_days=lookback_days,
            max_chunks=max_chunks,
            embedding_size=embedding_size,
            include_anchor_day=include_anchor_day,
        )
    return _build_lookback_tensors_chunk(
        store,
        anchor_date=anchor_date,
        lookback_days=lookback_days,
        max_chunks=max_chunks,
        embedding_size=embedding_size,
        include_anchor_day=include_anchor_day,
    )


def _build_lookback_tensors_chunk(
    store: pd.DataFrame,
    *,
    anchor_date: str,
    lookback_days: int,
    max_chunks: int,
    embedding_size: int | None = None,
    include_anchor_day: bool = True,
) -> ChunkRetrievalResult:
    """Chunk-path implementation (Variant B). One row = one chunk."""
    anchor = _parse_iso(anchor_date)
    if anchor is None:
        raise ValueError(f"anchor_date is not ISO date: {anchor_date!r}")
    lower = anchor - datetime.timedelta(days=int(lookback_days))

    parsed_dates: list[datetime.date | None] = [_parse_iso(d) for d in store["event_date"].tolist()]
    rows: list[tuple[datetime.date, int]] = []
    for idx, parsed in enumerate(parsed_dates):
        if parsed is None:
            continue
        if parsed < lower:
            continue
        if parsed > anchor:
            continue
        if not include_anchor_day and parsed == anchor:
            continue
        rows.append((parsed, idx))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    rows = rows[:max_chunks]
    rows.sort(key=lambda item: (item[0], item[1]))

    if embedding_size is None:
        if rows:
            sample = store.iloc[rows[0][1]]["embedding"]
            embedding_size = int(len(sample))
        else:
            embedding_size = 768

    embeddings = np.zeros((max_chunks, embedding_size), dtype=np.float32)
    elapsed = np.zeros((max_chunks,), dtype=np.float32)
    mask = np.zeros((max_chunks,), dtype=np.float32)
    doc_ids: list[str] = []
    event_dates: list[str] = []
    chunk_previews: list[str] = []

    for slot, (parsed, idx) in enumerate(rows):
        record = store.iloc[idx]
        vec = np.asarray(record["embedding"], dtype=np.float32)
        if vec.shape[0] != embedding_size:
            continue
        embeddings[slot] = vec
        elapsed[slot] = float((anchor - parsed).days)
        mask[slot] = 1.0
        doc_ids.append(str(record["doc_id"]))
        event_dates.append(str(record["event_date"]))
        chunk_previews.append(str(record.get("chunk_preview", "")))

    while len(doc_ids) < max_chunks:
        doc_ids.append("")
        event_dates.append("")
        chunk_previews.append("")

    return ChunkRetrievalResult(
        embeddings=torch.from_numpy(embeddings),
        elapsed_days=torch.from_numpy(elapsed),
        mask=torch.from_numpy(mask),
        doc_ids=doc_ids,
        event_dates=event_dates,
        chunk_previews=chunk_previews,
        actual_count=int(mask.sum()),
    )


def _build_lookback_tensors_llm(
    *,
    anchor_date: str,
    lookback_days: int,
    max_chunks: int,
    embedding_size: int | None = None,
    include_anchor_day: bool = True,
    llm_store_path: str | Path | None = None,
) -> ChunkRetrievalResult:
    """LLM-path implementation (Variant C).

    One document = one slot in the output tensor.  The shape contract
    ``(max_chunks, embedding_dim)`` is identical to the chunk path so the
    ``ChunkAttentionPooler`` receives the same interface regardless of source.
    """
    store_path = str(llm_store_path) if llm_store_path is not None else str(DEFAULT_LLM_EMBEDDINGS_PARQUET)
    store = _load_llm_store(store_path)

    anchor = _parse_iso(anchor_date)
    if anchor is None:
        raise ValueError(f"anchor_date is not ISO date: {anchor_date!r}")
    lower = anchor - datetime.timedelta(days=int(lookback_days))

    parsed_dates: list[datetime.date | None] = [_parse_iso(d) for d in store["event_date"].tolist()]
    rows: list[tuple[datetime.date, int]] = []
    for idx, parsed in enumerate(parsed_dates):
        if parsed is None:
            continue
        if parsed < lower:
            continue
        if parsed > anchor:
            continue
        if not include_anchor_day and parsed == anchor:
            continue
        rows.append((parsed, idx))

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    rows = rows[:max_chunks]
    rows.sort(key=lambda item: (item[0], item[1]))

    if embedding_size is None:
        if rows:
            sample = store.iloc[rows[0][1]]["embedding"]
            embedding_size = int(len(sample))
        else:
            embedding_size = 768

    embeddings = np.zeros((max_chunks, embedding_size), dtype=np.float32)
    elapsed = np.zeros((max_chunks,), dtype=np.float32)
    mask = np.zeros((max_chunks,), dtype=np.float32)
    doc_ids: list[str] = []
    event_dates: list[str] = []
    chunk_previews: list[str] = []

    for slot, (parsed, idx) in enumerate(rows):
        record = store.iloc[idx]
        vec = np.asarray(record["embedding"], dtype=np.float32)
        if vec.shape[0] != embedding_size:
            continue
        embeddings[slot] = vec
        elapsed[slot] = float((anchor - parsed).days)
        mask[slot] = 1.0
        doc_ids.append(str(record.get("document_id", "")))
        event_dates.append(str(record["event_date"]))
        chunk_previews.append("")  # LLM path has no chunk_preview column

    while len(doc_ids) < max_chunks:
        doc_ids.append("")
        event_dates.append("")
        chunk_previews.append("")

    return ChunkRetrievalResult(
        embeddings=torch.from_numpy(embeddings),
        elapsed_days=torch.from_numpy(elapsed),
        mask=torch.from_numpy(mask),
        doc_ids=doc_ids,
        event_dates=event_dates,
        chunk_previews=chunk_previews,
        actual_count=int(mask.sum()),
    )


def resolve_store_path(data_dir: str | Path, training_package_id: str, *, name: str = "chunk_embeddings.parquet") -> Path:
    return Path(data_dir) / "processed" / training_package_id / name
