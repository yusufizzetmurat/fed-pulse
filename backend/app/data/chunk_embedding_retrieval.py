"""Read helpers for the chunk embedding store.

Loads the parquet produced by :mod:`app.data.chunk_embedding_store` and exposes
a ``build_lookback_tensors`` function that returns padded ``(embeddings,
elapsed_days, mask)`` tensors suitable for :class:`ChunkAttentionPooler`.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
) -> ChunkRetrievalResult:
    """Return padded chunk tensors for docs within ``[anchor - lookback, anchor]``.

    The most recent docs win when ``max_chunks`` cap is hit.
    """
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


def resolve_store_path(data_dir: str | Path, training_package_id: str, *, name: str = "chunk_embeddings.parquet") -> Path:
    return Path(data_dir) / "processed" / training_package_id / name
