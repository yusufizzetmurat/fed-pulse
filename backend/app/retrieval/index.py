"""On-disk retrieval index for historical FOMC statements (#294).

The retrieval index is a small numpy embedding matrix paired with a
metadata parquet. Each row in the metadata represents one past FOMC
statement (rows from ``events.parquet`` filtered to
``event_kind == "statement"``); each row in the matrix is the
mean-pooled sentence embedding produced by the fine-tuned encoder
registered at ``finbert_fed_adjacent_xbank_dapt_retrieval``.

The total population is ~250 historical statements, so a plain
``(N, d)`` matrix + dot-product top-k beats a FAISS dependency on
operational simplicity. ``query`` normalises both index and query
vectors so the dot product is the cosine similarity directly.

Layout on disk under ``DATA_DIR / "artifacts" / "retrieval" / <run_id>``::

    index.parquet     metadata rows (event_date, text_hash,
                      axis_stance, forward_realized_vol_10d, excerpt)
    embeddings.npy    float32 (N, d) embedding matrix; row order
                      matches the parquet
    manifest.json     encoder alias + revision + source training
                      package id + row count + build timestamp

The runtime singleton at ``app.services.analogs`` loads these three
files at first /analyze/analogs hit and reuses them for the lifetime
of the worker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Truncate stored excerpts so the parquet stays cheap and the API response
# never echoes a multi-page statement back to the client. ~280 chars matches
# the AnalogCard schema documented in app/schemas.py.
EXCERPT_CHARS = 280

# Defensive cap on per-statement input length. The encoder tokenizer
# truncates internally, but capping the string upstream keeps the
# embedding helper from spending time on tail tokens that get dropped
# anyway. 4096 chars ≈ 1024 tokens is comfortably above the FOMC
# statement length distribution.
MAX_TEXT_CHARS = 4096

INDEX_PARQUET_NAME = "index.parquet"
EMBEDDINGS_NPY_NAME = "embeddings.npy"
MANIFEST_NAME = "manifest.json"

METADATA_COLUMNS = (
    "event_date",
    "text_hash",
    "axis_stance",
    "forward_realized_vol_10d",
    "excerpt",
)


@dataclass(frozen=True)
class IndexPaths:
    directory: Path

    @property
    def parquet(self) -> Path:
        return self.directory / INDEX_PARQUET_NAME

    @property
    def embeddings(self) -> Path:
        return self.directory / EMBEDDINGS_NPY_NAME

    @property
    def manifest(self) -> Path:
        return self.directory / MANIFEST_NAME


@dataclass(frozen=True)
class LoadedIndex:
    """In-memory retrieval index ready to serve top-k queries.

    ``embeddings`` is row-normalised at load time so the dot product
    against an L2-normalised query is the cosine similarity directly.
    """

    embeddings: np.ndarray  # (N, d), float32, L2-normalised rows
    metadata: pd.DataFrame  # N rows, columns = METADATA_COLUMNS
    encoder_alias: str
    encoder_revision: str
    training_package_id: str | None
    built_at_utc: str

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.size == 0:
            return 0
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class AnalogHit:
    """One retrieved analog row."""

    event_date: str
    text_hash: str
    axis_stance: str | None
    forward_realized_vol_10d: float | None
    excerpt: str
    similarity: float


def _l2_normalise(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation so dot products become cosine similarities.

    Zero-length rows (degenerate empty inputs) keep their zero vector;
    division-by-zero is guarded by ``eps``.
    """

    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    arr = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def _ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce / project a DataFrame to the canonical METADATA_COLUMNS shape.

    Missing optional columns (axis_stance, forward_realized_vol_10d) are
    filled with ``None``; the required event_date and text_hash columns
    raise if absent so a malformed parquet fails loudly at load time.
    """

    for required in ("event_date", "text_hash"):
        if required not in df.columns:
            raise KeyError(
                f"retrieval metadata is missing required column {required!r}; "
                f"got columns {list(df.columns)!r}"
            )
    out = pd.DataFrame()
    out["event_date"] = df["event_date"].astype(str)
    out["text_hash"] = df["text_hash"].astype(str)
    out["axis_stance"] = df.get("axis_stance", pd.Series([None] * len(df)))
    out["forward_realized_vol_10d"] = df.get(
        "forward_realized_vol_10d", pd.Series([None] * len(df))
    )
    excerpts = df.get("excerpt", pd.Series([""] * len(df))).astype(str)
    out["excerpt"] = excerpts.str.slice(0, EXCERPT_CHARS)
    return out


def _filter_statement_events(events: pd.DataFrame) -> pd.DataFrame:
    """Project ``events.parquet`` to one row per historical FOMC statement.

    The events parquet expands every event by horizon (1d / 5d / 10d
    rows duplicate the text), so we group by ``text_hash`` and keep the
    smallest-horizon row — the 1-day-horizon row whose
    ``forward_realized_vol_10d`` is the post-event measure documented
    in the data schema. Row order is preserved otherwise: the first
    appearance of each ``text_hash`` in the source frame defines the
    output order, which lets the test fixtures assert a deterministic
    metadata sequence without depending on hash-sort.
    """

    if "event_kind" not in events.columns:
        raise KeyError(
            "events.parquet must carry an 'event_kind' column; "
            f"got {list(events.columns)!r}"
        )
    mask = events["event_kind"].astype(str).str.lower() == "statement"
    df = events.loc[mask].copy()
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    df["_original_order"] = np.arange(len(df))
    sort_cols = ["text_hash"]
    if "horizon" in df.columns:
        sort_cols.append("horizon")
    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(subset=["text_hash"], keep="first")
    df = df.sort_values("_original_order").drop(columns=["_original_order"])
    return df.reset_index(drop=True)


def _statement_text(row: pd.Series) -> str:
    text = str(row.get("text") or "").strip()
    if not text:
        return ""
    return text[:MAX_TEXT_CHARS]


def _empty_embedding_matrix(dim: int) -> np.ndarray:
    return np.zeros((0, max(dim, 1)), dtype=np.float32)


def build_index_from_events(  # noqa: PLR0913 — keyword-only builder args mirror the CLI; grouping would obscure the call site.
    events_parquet: Path,
    *,
    encoder_alias: str,
    encoder_revision: str,
    embed_fn,
    training_package_id: str | None = None,
    out_dir: Path,
) -> LoadedIndex:
    """Build a retrieval index by embedding every statement row.

    ``embed_fn`` is a callable taking ``list[str]`` and returning a
    ``(len, d)`` ndarray — the test suite passes a dummy projection
    here, while the production caller in ``app.retrieval.train`` passes
    a closure that runs the fine-tuned encoder.

    Persists the parquet + embedding matrix + manifest under ``out_dir``
    and returns the in-memory ``LoadedIndex`` so the caller can validate
    the build without a re-load round-trip.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(events_parquet)
    statements = _filter_statement_events(events)
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for _, raw_row in statements.iterrows():
        text = _statement_text(raw_row)
        if not text:
            continue
        rows.append(
            {
                "event_date": str(raw_row.get("event_date", "")),
                "text_hash": str(raw_row.get("text_hash", "")),
                "axis_stance": raw_row.get("axis_stance"),
                "forward_realized_vol_10d": raw_row.get("forward_realized_vol_10d"),
                "excerpt": text[:EXCERPT_CHARS],
            }
        )
        texts.append(text)

    if not rows:
        _logger.warning("retrieval_index_empty events_parquet=%s", events_parquet)
        embeddings = _empty_embedding_matrix(1)
        metadata = pd.DataFrame(columns=list(METADATA_COLUMNS))
    else:
        raw_embeddings = embed_fn(texts)
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
            raise ValueError(
                "embed_fn must return a 2-D array with one row per input; "
                f"got shape {embeddings.shape!r} for {len(rows)} inputs"
            )
        metadata = _ensure_metadata_columns(pd.DataFrame(rows))
        embeddings = _l2_normalise(embeddings)

    paths = IndexPaths(directory=out_dir)
    metadata.to_parquet(paths.parquet, index=False)
    np.save(paths.embeddings, embeddings)
    manifest = {
        "encoder_alias": encoder_alias,
        "encoder_revision": encoder_revision,
        "training_package_id": training_package_id,
        "row_count": int(len(metadata)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "events_parquet": str(events_parquet),
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return LoadedIndex(
        embeddings=embeddings,
        metadata=metadata,
        encoder_alias=encoder_alias,
        encoder_revision=encoder_revision,
        training_package_id=training_package_id,
        built_at_utc=manifest["built_at_utc"],
    )


def load_index(directory: Path) -> LoadedIndex:
    """Load a previously persisted retrieval index from disk.

    Raises ``FileNotFoundError`` when the directory is missing the
    embeddings / parquet / manifest triple — the runtime singleton
    catches this and degrades the /analyze/analogs endpoint to
    ``available=False`` instead of crashing the worker.
    """

    paths = IndexPaths(directory=Path(directory))
    for required in (paths.parquet, paths.embeddings, paths.manifest):
        if not required.exists():
            raise FileNotFoundError(
                f"retrieval index incomplete at {paths.directory}: "
                f"missing {required.name}"
            )
    metadata = _ensure_metadata_columns(pd.read_parquet(paths.parquet))
    raw_embeddings = np.load(paths.embeddings)
    embeddings = _l2_normalise(np.asarray(raw_embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "retrieval index row mismatch: "
            f"{embeddings.shape[0]} embeddings vs {len(metadata)} metadata rows"
        )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return LoadedIndex(
        embeddings=embeddings,
        metadata=metadata,
        encoder_alias=str(manifest.get("encoder_alias", "")),
        encoder_revision=str(manifest.get("encoder_revision", "")),
        training_package_id=manifest.get("training_package_id"),
        built_at_utc=str(manifest.get("built_at_utc", "")),
    )


def query(index: LoadedIndex, query_embedding: np.ndarray, *, k: int = 5) -> list[AnalogHit]:
    """Return the top-``k`` analogs for a pre-computed query embedding.

    The query vector is L2-normalised before the dot product so the
    similarity scores live in ``[-1, 1]`` regardless of the encoder's
    output scale. ``k`` is clipped to the available index size; an
    empty index returns ``[]``.
    """

    if index.size == 0:
        return []
    if k <= 0:
        return []
    arr = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if arr.shape[0] != index.embedding_dim:
        raise ValueError(
            f"query embedding dim {arr.shape[0]} does not match index dim {index.embedding_dim}"
        )
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return []
    arr = arr / norm
    scores = index.embeddings @ arr  # (N,)
    k_effective = min(int(k), index.size)
    # ``argpartition`` is cheaper than a full sort for the typical
    # 250-row index, but we still need the partition slice sorted so
    # the API output is descending by similarity.
    top_idx = np.argpartition(-scores, k_effective - 1)[:k_effective]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    hits: list[AnalogHit] = []
    for row_idx in top_idx:
        row = index.metadata.iloc[int(row_idx)]
        raw_vol = row.get("forward_realized_vol_10d")
        try:
            forward_vol: float | None = float(raw_vol) if raw_vol is not None and not pd.isna(raw_vol) else None
        except (TypeError, ValueError):
            forward_vol = None
        raw_stance = row.get("axis_stance")
        stance: str | None
        if raw_stance is None or (isinstance(raw_stance, float) and pd.isna(raw_stance)):
            stance = None
        else:
            stance = str(raw_stance)
        hits.append(
            AnalogHit(
                event_date=str(row["event_date"]),
                text_hash=str(row["text_hash"]),
                axis_stance=stance,
                forward_realized_vol_10d=forward_vol,
                excerpt=str(row["excerpt"]),
                similarity=float(scores[int(row_idx)]),
            )
        )
    return hits


__all__ = [
    "AnalogHit",
    "IndexPaths",
    "LoadedIndex",
    "build_index_from_events",
    "load_index",
    "query",
]
