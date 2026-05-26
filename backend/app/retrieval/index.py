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
                      axis_stance, subsequent_vol_regime, excerpt)
    embeddings.npy    float32 (N, d) embedding matrix; row order
                      matches the parquet
    manifest.json     encoder alias + revision + source training
                      package id + row count + build timestamp +
                      train_end walk-forward boundary

The runtime singleton at ``app.services.analogs`` loads these three
files at first /analyze/analogs hit and reuses them for the lifetime
of the worker.

The metadata DOES NOT carry the raw supervised target
``forward_realized_vol_10d`` — surfacing it as an analog field
would tempt downstream consumers to feed API output back into a
trained model and silently leak labels. Instead the build path
buckets the value into ``calm`` / ``normal`` / ``high`` using the
``VOL_REGIME_BUCKET_EDGES`` constant below (pinned from a held-out
2000-2015 reference distribution) so the UI gets a coarse, stable
flag without exposing the exact target value.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
# statement length distribution. Also applied to /analyze/analogs query
# payloads in app.services.analogs.encode_query so a 10MB payload never
# reaches the tokenizer.
MAX_TEXT_CHARS = 4096

# Bucket edges for ``forward_realized_vol_10d`` -> ``subsequent_vol_regime``.
# Pinned as a module constant rather than computed at build time so the
# bucket assignment is reproducible across rebuilds. The two cut points
# correspond to the ~33rd and ~66th percentile of the 10-day realised
# vol distribution on the ^GSPC train slice from 2000-01-01 through
# 2015-12-31 (held-out from the production walk-forward folds, which
# start at 2016-09-21). Values come from the same forward-realized-vol
# computation that backs ``forward_realized_vol_10d`` in events.parquet
# so the bucket boundaries are on the same scale as the underlying
# series. Centring them on a pre-2016 slice means downstream rebuilds
# do not move the bucket edges with new market data.
VOL_REGIME_BUCKET_EDGES: tuple[float, float] = (0.012, 0.020)
VolRegime = Literal["calm", "normal", "high"]

INDEX_PARQUET_NAME = "index.parquet"
EMBEDDINGS_NPY_NAME = "embeddings.npy"
MANIFEST_NAME = "manifest.json"

METADATA_COLUMNS = (
    "event_date",
    "text_hash",
    "axis_stance",
    "subsequent_vol_regime",
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
    train_end: str | None = None

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
    subsequent_vol_regime: VolRegime | None
    excerpt: str
    similarity: float


def _bucket_vol(value: Any) -> VolRegime | None:
    """Bucket a raw 10d-forward vol scalar into ``calm`` / ``normal`` / ``high``.

    Returns ``None`` for missing / non-finite inputs so the column stays
    nullable in the parquet without polluting the bucket distribution.
    """

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    low, high = VOL_REGIME_BUCKET_EDGES
    if scalar < low:
        return "calm"
    if scalar < high:
        return "normal"
    return "high"


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

    Missing optional columns (axis_stance, subsequent_vol_regime) are
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
    out["subsequent_vol_regime"] = df.get(
        "subsequent_vol_regime", pd.Series([None] * len(df))
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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically via a sibling .tmp file."""

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, payload.encode(encoding))


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_save_npy(array: np.ndarray, path: Path) -> None:
    # np.save appends ``.npy`` when the destination does not already end
    # in that suffix, which would silently rename our staged file. Write
    # bytes via a BytesIO buffer instead so the on-disk tmp path is exact.
    import io

    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    _atomic_write_bytes(path, buf.getvalue())


def build_index_from_events(  # noqa: PLR0913 — keyword-only builder args mirror the CLI; grouping would obscure the call site.
    events_parquet: Path,
    *,
    encoder_alias: str,
    encoder_revision: str,
    embed_fn,
    training_package_id: str | None = None,
    out_dir: Path,
    train_end: str | None = None,
) -> LoadedIndex:
    """Build a retrieval index by embedding every statement row.

    ``embed_fn`` is a callable taking ``list[str]`` and returning a
    ``(len, d)`` ndarray — the test suite passes a dummy projection
    here, while the production caller in ``app.retrieval.train`` passes
    a closure that runs the fine-tuned encoder.

    Persists the parquet + embedding matrix + manifest under ``out_dir``
    and returns the in-memory ``LoadedIndex`` so the caller can validate
    the build without a re-load round-trip. Each file is written via a
    sibling .tmp + atomic rename so a mid-write crash never leaves a
    half-built bundle on disk; the manifest is written last so the
    bundle is either complete or absent.

    ``train_end`` is the resolved walk-forward boundary (ISO date or
    ``None``) — persisted into the manifest so downstream consumers can
    enforce the same boundary at query time.
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
                "subsequent_vol_regime": _bucket_vol(
                    raw_row.get("forward_realized_vol_10d")
                ),
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
    # Parquet + .npy first; manifest last so a mid-write crash never
    # leaves a half-built bundle that ``load_index`` would happily
    # consume.
    _atomic_write_parquet(metadata, paths.parquet)
    _atomic_save_npy(embeddings, paths.embeddings)
    built_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "encoder_alias": encoder_alias,
        "encoder_revision": encoder_revision,
        "training_package_id": training_package_id,
        "row_count": int(len(metadata)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "events_parquet": str(events_parquet),
        "built_at_utc": built_at_utc,
        "train_end": train_end,
    }
    _atomic_write_text(
        paths.manifest, json.dumps(manifest, indent=2, sort_keys=True)
    )

    return LoadedIndex(
        embeddings=embeddings,
        metadata=metadata,
        encoder_alias=encoder_alias,
        encoder_revision=encoder_revision,
        training_package_id=training_package_id,
        built_at_utc=built_at_utc,
        train_end=train_end,
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
    # allow_pickle=False guards against a tampered .npy attempting to
    # smuggle arbitrary pickled objects through np.load — defence in
    # depth since the embeddings on disk are always float32 arrays.
    raw_embeddings = np.load(paths.embeddings, allow_pickle=False)
    embeddings = _l2_normalise(np.asarray(raw_embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "retrieval index row mismatch: "
            f"{embeddings.shape[0]} embeddings vs {len(metadata)} metadata rows"
        )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    train_end_raw = manifest.get("train_end")
    train_end = str(train_end_raw) if train_end_raw else None
    return LoadedIndex(
        embeddings=embeddings,
        metadata=metadata,
        encoder_alias=str(manifest.get("encoder_alias", "")),
        encoder_revision=str(manifest.get("encoder_revision", "")),
        training_package_id=manifest.get("training_package_id"),
        built_at_utc=str(manifest.get("built_at_utc", "")),
        train_end=train_end,
    )


def _as_of_to_iso(as_of_date: date_type | str | None) -> str | None:
    """Normalise an as-of-date argument to an ISO YYYY-MM-DD string.

    Accepts ``None`` (no filter), ``datetime.date`` (or ``datetime``),
    or a string already in ISO form. Anything else raises ``ValueError``
    so the caller sees a clean signal rather than a silent no-op.
    """

    if as_of_date is None:
        return None
    if isinstance(as_of_date, datetime):
        return as_of_date.date().isoformat()
    if isinstance(as_of_date, date_type):
        return as_of_date.isoformat()
    if isinstance(as_of_date, str):
        # Validate that the string parses as an ISO date.
        try:
            return date_type.fromisoformat(as_of_date).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"as_of_date {as_of_date!r} is not a valid ISO date (YYYY-MM-DD)"
            ) from exc
    raise ValueError(f"as_of_date must be date | str | None, got {type(as_of_date)!r}")


def query(  # noqa: PLR0912, C901 — guard clauses on the optional filters keep the happy path readable.
    index: LoadedIndex,
    query_embedding: np.ndarray,
    *,
    k: int = 5,
    as_of_date: date_type | str | None = None,
    exclude_text_hash: str | None = None,
) -> list[AnalogHit]:
    """Return the top-``k`` analogs for a pre-computed query embedding.

    The query vector is L2-normalised before the dot product so the
    similarity scores live in ``[-1, 1]`` regardless of the encoder's
    output scale. ``k`` is clipped to the available index size; an
    empty index returns ``[]``.

    ``as_of_date`` enforces a strict-backward walk-forward boundary —
    only rows with ``event_date < as_of_date`` are eligible. When the
    filtered pool has fewer than ``k`` rows we return what we have
    rather than padding with future rows.

    ``exclude_text_hash`` drops any candidate whose stored ``text_hash``
    matches the supplied value. The runtime singleton passes the
    sha256 of the cleaned query text so a caller that submits an
    indexed statement does not get the trivial self-match back.
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

    # Build the eligibility mask over the full index, then derive
    # the candidate row indices in one pass — keeps the scoring step
    # contiguous against a 1-D similarity vector.
    eligible = np.ones(index.size, dtype=bool)
    cutoff_iso = _as_of_to_iso(as_of_date)
    if cutoff_iso is not None:
        event_dates = index.metadata["event_date"].astype(str).to_numpy()
        eligible &= event_dates < cutoff_iso
    if exclude_text_hash:
        text_hashes = index.metadata["text_hash"].astype(str).to_numpy()
        eligible &= text_hashes != exclude_text_hash
    candidate_idx = np.nonzero(eligible)[0]
    if candidate_idx.size == 0:
        return []

    scores_all = index.embeddings @ arr  # (N,)
    scores = scores_all[candidate_idx]
    k_effective = min(int(k), candidate_idx.size)
    # ``argpartition`` is cheaper than a full sort for the typical
    # 250-row index, but we still need the partition slice sorted so
    # the API output is descending by similarity.
    if k_effective < scores.size:
        partition = np.argpartition(-scores, k_effective - 1)[:k_effective]
    else:
        partition = np.arange(scores.size)
    partition = partition[np.argsort(-scores[partition])]
    top_idx = candidate_idx[partition]

    hits: list[AnalogHit] = []
    for row_idx in top_idx:
        row = index.metadata.iloc[int(row_idx)]
        raw_regime = row.get("subsequent_vol_regime")
        regime: VolRegime | None
        if raw_regime is None or (isinstance(raw_regime, float) and pd.isna(raw_regime)):
            regime = None
        else:
            regime_str = str(raw_regime)
            regime = regime_str if regime_str in ("calm", "normal", "high") else None  # type: ignore[assignment]
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
                subsequent_vol_regime=regime,
                excerpt=str(row["excerpt"]),
                similarity=float(scores_all[int(row_idx)]),
            )
        )
    return hits


def text_hash_for_query(text: str) -> str:
    """Return the sha256 hex digest of ``text`` after the cleaning the
    runtime applies before tokenisation.

    Centralised so the index's text_hash convention and the runtime
    self-match filter share one definition.
    """

    cleaned = (text or "").strip()[:MAX_TEXT_CHARS]
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


__all__ = [
    "AnalogHit",
    "IndexPaths",
    "LoadedIndex",
    "MAX_TEXT_CHARS",
    "VOL_REGIME_BUCKET_EDGES",
    "VolRegime",
    "build_index_from_events",
    "load_index",
    "query",
    "text_hash_for_query",
]
