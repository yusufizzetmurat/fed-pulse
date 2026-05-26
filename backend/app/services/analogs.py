"""Runtime singleton for the historical-analog retrieval endpoint (#294).

Backs ``POST /analyze/analogs`` in :mod:`app.main`. Loads the
fine-tuned encoder + persisted retrieval index produced by
:mod:`app.retrieval.train` on the first request, caches the bundle on
the worker, and serves subsequent queries from the in-memory state.

The encoder is loaded via plain ``AutoTokenizer`` / ``AutoModel`` so
the inference path does not depend on the ``sentence-transformers``
package — the SBERT save format is HF-compatible at the model level
and the SBERT mean-pooling is reproduced here with a one-liner masked
mean. Keeping the inference path SBERT-free lets the FastAPI worker
import without paying the SBERT cold-start cost when the analogs
feature is not used.

Mirrors the singleton patterns used by
:mod:`app.services.multi_axis_classifier` and
:mod:`app.services.sentiment` — thread-safe lazy init, a ``reset``
hook for the test suite, and a graceful "not available" state so a
missing checkpoint never crashes the worker.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Any

import numpy as np

from app.config import DATA_DIR
from app.retrieval.index import (
    MAX_TEXT_CHARS,
    AnalogHit,
    LoadedIndex,
    load_index,
    query,
    text_hash_for_query,
)

_logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_DIR = DATA_DIR / "artifacts" / "retrieval" / "finbert_fed_adjacent_xbank_dapt_retrieval"
DEFAULT_CHECKPOINT_SUBDIR = "checkpoint"
DEFAULT_MAX_LENGTH = 256


@dataclass(frozen=True)
class _AnalogsState:
    """Loaded encoder + retrieval index ready to serve top-k queries."""

    tokenizer: Any
    model: Any
    device: Any
    max_length: int
    index: LoadedIndex
    bundle_dir: Path

    @property
    def encoder_alias(self) -> str:
        return self.index.encoder_alias or "finbert_fed_adjacent_xbank_dapt_retrieval"


_state: _AnalogsState | None = None
_state_lock = threading.Lock()


def _resolve_bundle_dir() -> Path:
    """Resolve the persisted retrieval-bundle directory.

    The environment knob lets the test suite drop the singleton onto a
    tmp_path-built fixture without touching the production default.
    """

    override = (os.environ.get("FED_PULSE_RETRIEVAL_DIR") or "").strip()
    if override:
        return Path(override)
    return DEFAULT_RETRIEVAL_DIR


def _resolve_checkpoint_dir(bundle_dir: Path) -> Path:
    """Resolve the encoder checkpoint sub-directory inside the bundle."""

    candidate = bundle_dir / DEFAULT_CHECKPOINT_SUBDIR
    if candidate.exists():
        return candidate
    return bundle_dir


def bundle_available() -> bool:
    """Lightweight check used by /analyze/analogs to short-circuit absent bundles.

    Called as the endpoint pre-flight in :mod:`app.main` so a permanently
    absent bundle does not pay the full ``_load_state`` round-trip on
    every request — the handler short-circuits to a clean 503 the moment
    this returns ``False``.
    """

    bundle = _resolve_bundle_dir()
    if not bundle.exists():
        return False
    # The bundle must carry the index triple; the checkpoint dir is a
    # secondary concern caught at load time.
    return all(
        (bundle / name).exists()
        for name in ("index.parquet", "embeddings.npy", "manifest.json")
    )


def _load_state() -> _AnalogsState | None:
    """Build the singleton from the persisted bundle.

    Returns ``None`` when the bundle is missing (so /analyze/analogs
    can degrade to ``available=False``) or when the encoder fails to
    load. Logs the failure either way so a broken bundle is visible in
    uvicorn output without taking the worker down.
    """

    bundle_dir = _resolve_bundle_dir()
    try:
        loaded_index = load_index(bundle_dir)
    except FileNotFoundError:
        _logger.info("analogs_bundle_missing path=%s", bundle_dir)
        return None
    except Exception:  # pragma: no cover — guarded so a malformed parquet does not 500 the worker
        _logger.warning("analogs_index_load_failed path=%s", bundle_dir, exc_info=True)
        return None

    checkpoint_dir = _resolve_checkpoint_dir(bundle_dir)
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found,unused-ignore]
        import torch  # type: ignore[import-not-found,unused-ignore]

        # local_files_only + trust_remote_code=False keep the load
        # strictly offline: a malformed or missing directory raises here
        # instead of silently triggering a HF Hub lookup that may hang
        # in air-gapped environments or fetch an unrelated repo.
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        model = AutoModel.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception:
        _logger.warning(
            "analogs_encoder_load_failed checkpoint=%s", checkpoint_dir, exc_info=True
        )
        return None

    return _AnalogsState(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=DEFAULT_MAX_LENGTH,
        index=loaded_index,
        bundle_dir=bundle_dir,
    )


def get_state() -> _AnalogsState | None:
    """Return the cached analogs state, building it on first call."""

    global _state
    if _state is not None:
        return _state
    with _state_lock:
        if _state is None:
            _state = _load_state()
        return _state


def reset_state() -> None:
    """Drop the singleton so the next call rebuilds (test hook + refresh)."""

    global _state
    with _state_lock:
        _state = None


def install_state(state: _AnalogsState) -> None:
    """Install a pre-built state directly. Used by tests to bypass disk I/O."""

    global _state
    with _state_lock:
        _state = state


def build_state_from_index(
    index: LoadedIndex,
    *,
    embed_fn,
    encoder_alias: str | None = None,
) -> _AnalogsState:
    """Convenience constructor for tests + smoke harnesses.

    Wraps a callable ``embed_fn(list[str]) -> ndarray`` as a stand-in
    tokenizer/model pair so the endpoint can run without loading real
    HF weights. The endpoint path always calls ``encode_query`` to
    produce the query vector, and ``encode_query`` accepts any state
    object that exposes ``tokenizer``, ``model``, and ``device`` — we
    cover the contract here with tiny adapter classes.
    """

    class _CallableModel:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, texts: list[str]) -> np.ndarray:
            return np.asarray(self._fn(texts), dtype=np.float32)

    bound_index = (
        dataclasses.replace(index, encoder_alias=encoder_alias)
        if encoder_alias
        else index
    )
    return _AnalogsState(
        tokenizer=None,
        model=_CallableModel(embed_fn),
        device=None,
        max_length=DEFAULT_MAX_LENGTH,
        index=bound_index,
        bundle_dir=Path("/dev/null"),
    )


def _mean_pool_last_hidden(
    last_hidden_state: Any,
    attention_mask: Any,
) -> Any:
    """Reproduce SBERT mean-pooling without depending on sentence-transformers.

    ``last_hidden_state`` is ``(1, T, H)`` and ``attention_mask`` is
    ``(1, T)``. We zero out padding positions, sum, then divide by the
    valid-token count (clamped to ≥1 to avoid division-by-zero on a
    degenerate empty input).
    """

    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (summed / counts).detach().cpu().numpy()


def encode_query(state: _AnalogsState, text: str) -> np.ndarray:
    """Embed a single query text using the loaded encoder.

    Supports the two call-shapes the singleton accepts:

    * The production path holds a real HF tokenizer + model and runs a
      forward pass + mean-pool.
    * Test fixtures install a ``_CallableModel`` (see
      :func:`build_state_from_index`) that ignores tokenization and
      returns the embedding directly from a Python callable.

    Both paths return a 1-D ``(d,)`` ndarray. The query string is
    truncated to ``MAX_TEXT_CHARS`` before tokenisation so a 10MB
    payload never reaches the tokenizer.
    """

    cleaned = (text or "").strip()[:MAX_TEXT_CHARS]
    if not cleaned:
        return np.zeros(state.index.embedding_dim or 1, dtype=np.float32)

    if state.tokenizer is None or state.device is None:
        # Test path: model is a plain callable returning the embedding(s).
        out = state.model([cleaned])
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr.reshape(-1)

    import torch

    encoded = state.tokenizer(
        cleaned,
        max_length=state.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(state.device)
    attention_mask = encoded["attention_mask"].to(state.device)
    with torch.no_grad():
        outputs = state.model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = _mean_pool_last_hidden(outputs.last_hidden_state, attention_mask)
    return pooled.reshape(-1).astype(np.float32, copy=False)


def find_analogs(
    text: str,
    *,
    k: int = 5,
    as_of_date: date_type | None = None,
) -> dict[str, Any] | None:
    """Top-k retrieval entry point for ``POST /analyze/analogs``.

    Returns ``None`` when no bundle is loaded — the endpoint then
    responds with ``available=False``-equivalent shape (empty
    ``analogs`` list, ``index_size=0``).

    ``as_of_date`` enforces a strict-backward walk-forward boundary at
    query time (only rows with ``event_date < as_of_date`` are
    eligible). Self-match suppression always runs: the sha256 of the
    cleaned query text is computed and any candidate carrying that
    text_hash is dropped, so a caller that submits the literal text of
    an indexed statement never sees the trivial similarity ≈ 1.0 hit.
    """

    state = get_state()
    if state is None:
        return None
    cleaned = (text or "").strip()[:MAX_TEXT_CHARS]
    query_vec = encode_query(state, cleaned)
    exclude_hash = text_hash_for_query(cleaned) if cleaned else None
    hits = query(
        state.index,
        query_vec,
        k=k,
        as_of_date=as_of_date,
        exclude_text_hash=exclude_hash,
    )
    return {
        "encoder_alias": state.encoder_alias,
        "index_size": state.index.size,
        "hits": hits,
    }


def render_analog_cards(hits: list[AnalogHit]) -> list[dict[str, Any]]:
    """Adapt :class:`AnalogHit` rows to the ``AnalogCard`` schema shape."""

    return [
        {
            "event_date": hit.event_date,
            "similarity": hit.similarity,
            "axis_stance": hit.axis_stance,
            "subsequent_vol_regime": hit.subsequent_vol_regime,
            "excerpt": hit.excerpt,
        }
        for hit in hits
    ]


__all__ = [
    "build_state_from_index",
    "bundle_available",
    "encode_query",
    "find_analogs",
    "get_state",
    "install_state",
    "render_analog_cards",
    "reset_state",
]
