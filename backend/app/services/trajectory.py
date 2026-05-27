"""Runtime singleton for the trajectory model endpoint (#296).

Backs ``POST /analyze/trajectory`` in :mod:`app.main`. Loads the
trained trajectory bundle produced by :mod:`app.trajectory.train` on
the first request, caches the bundle on the worker, and serves
subsequent queries from the in-memory state.

Mirrors :mod:`app.services.analogs` — thread-safe lazy init, a
``reset`` hook for the test suite, and a graceful "not available"
state so a missing bundle never crashes the worker.

The handler can be exercised in tests without a real torch checkpoint
via :func:`install_state`, which lets a fixture inject a pre-built
state directly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.trajectory.model import (
    DEFAULT_HISTORY_LENGTH,
    MARKET_FEATURE_DIM,
    STANCE_CLASSES,
    TrajectoryConfig,
    load_model,
    market_feature_vector,
    pad_sequence,
)

_logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_DIR = DATA_DIR / "artifacts" / "trajectory" / "trajectory_transformer"
PARQUET_NAME = "embedding_index.parquet"
NPZ_NAME = "embedding_index.npz"
MODEL_NAME = "model.pt"
MANIFEST_NAME = "manifest.json"
CONFORMAL_NAME = "conformal.json"
METRICS_NAME = "metrics.json"

MAX_HISTORY_LENGTH = 60
MAX_TEXT_CHARS = 4096


@dataclass(frozen=True)
class _TrajectoryState:
    """Loaded trajectory bundle ready to serve projections."""

    model: Any
    config: TrajectoryConfig
    embeddings: np.ndarray  # (N, embedding_dim)
    feature_mean: np.ndarray  # (embedding_dim,)
    feature_std: np.ndarray  # (embedding_dim,)
    metadata: pd.DataFrame
    encoder_alias: str
    encoder_revision: str
    train_end: str | None
    architecture: str
    bundle_dir: Path
    conformal_quantile: float | None
    conformal_alpha: float | None
    market_provider: Callable[[str], dict[str, float | None]] | None = None
    # Lift-vs-baseline verdict (#332). Persisted to ``metrics.json``
    # by the trainer; surfaced on the API envelope so the UI can render
    # the "lift / no-lift" badge. All three default to None so a
    # bundle trained before #332 reads through cleanly.
    lift_vs_baseline: bool = False
    delta_dir_acc: float | None = None
    baseline_used: str | None = None

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])


_state: _TrajectoryState | None = None
_state_lock = threading.Lock()


def _resolve_bundle_dir() -> Path:
    override = (os.environ.get("FED_PULSE_TRAJECTORY_DIR") or "").strip()
    if override:
        return Path(override)
    return DEFAULT_BUNDLE_DIR


def bundle_available() -> bool:
    """Lightweight check used by /analyze/trajectory to short-circuit absent bundles."""

    bundle = _resolve_bundle_dir()
    if not bundle.exists():
        return False
    return all(
        (bundle / name).exists()
        for name in (PARQUET_NAME, NPZ_NAME, MODEL_NAME, MANIFEST_NAME)
    )


def _load_npz_arrays(
    npz_path: Path, bundle_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the embedding matrix + standardisation stats from the npz.

    Falls back to zero-mean / unit-std with a logged warning when the
    bundle predates the standardisation contract. ``NpzFile`` does not
    support ``.get`` so the probe uses an explicit ``in .files`` check.
    """

    try:
        npz = np.load(npz_path, allow_pickle=False)
        embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
    except Exception:  # pragma: no cover — guarded so a malformed npz never 500s the worker
        _logger.warning(
            "trajectory_bundle_load_failed path=%s", bundle_dir, exc_info=True
        )
        return None
    if "feature_mean" in npz.files:
        feature_mean = np.asarray(npz["feature_mean"], dtype=np.float32)
    else:
        _logger.warning(
            "trajectory_bundle_missing_feature_mean path=%s — falling back to zeros",
            bundle_dir,
        )
        feature_mean = np.zeros(embeddings.shape[1], dtype=np.float32)
    if "feature_std" in npz.files:
        feature_std = np.asarray(npz["feature_std"], dtype=np.float32)
    else:
        _logger.warning(
            "trajectory_bundle_missing_feature_std path=%s — falling back to ones",
            bundle_dir,
        )
        feature_std = np.ones(embeddings.shape[1], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    return embeddings, feature_mean, feature_std


def _load_lift_check(bundle_dir: Path) -> tuple[bool, float | None, str | None]:
    """Read the lift / no-lift verdict from the bundle's metrics.json (#332).

    Returns ``(lift_vs_baseline, delta_dir_acc, baseline_used)``. A
    missing / unreadable file degrades to ``(False, None, None)`` so
    bundles trained before #332 surface a no-lift badge by default
    instead of crashing the worker.
    """

    metrics_path = bundle_dir / METRICS_NAME
    if not metrics_path.exists():
        return False, None, None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return False, None, None
    lift_check = payload.get("lift_check") if isinstance(payload, dict) else None
    if not isinstance(lift_check, dict):
        return False, None, None
    raw_lift = lift_check.get("lift_vs_baseline")
    raw_delta = lift_check.get("delta_dir_acc")
    raw_baseline = lift_check.get("baseline_used")
    lift = bool(raw_lift) if raw_lift is not None else False
    delta: float | None
    if raw_delta is None:
        delta = None
    else:
        try:
            delta = float(raw_delta)
        except (TypeError, ValueError):
            delta = None
    baseline_used = str(raw_baseline) if raw_baseline is not None else None
    return lift, delta, baseline_used


def _load_conformal(bundle_dir: Path) -> tuple[float | None, float | None]:
    conformal_path = bundle_dir / CONFORMAL_NAME
    if not conformal_path.exists():
        return None, None
    try:
        payload = json.loads(conformal_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return None, None
    raw_q = payload.get("softmax_quantile")
    raw_a = payload.get("alpha")
    q = float(raw_q) if raw_q is not None else None
    a = float(raw_a) if raw_a is not None else None
    return q, a


def _load_state() -> _TrajectoryState | None:
    bundle_dir = _resolve_bundle_dir()
    if not bundle_available():
        _logger.info("trajectory_bundle_missing path=%s", bundle_dir)
        return None
    try:
        metadata = pd.read_parquet(bundle_dir / PARQUET_NAME)
    except Exception:  # pragma: no cover
        _logger.warning(
            "trajectory_bundle_load_failed path=%s", bundle_dir, exc_info=True
        )
        return None
    loaded = _load_npz_arrays(bundle_dir / NPZ_NAME, bundle_dir)
    if loaded is None:
        return None
    embeddings, feature_mean, feature_std = loaded

    try:
        model, config = load_model(bundle_dir / MODEL_NAME)
    except Exception:
        _logger.warning(
            "trajectory_model_load_failed path=%s", bundle_dir, exc_info=True
        )
        return None

    try:
        manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        manifest = {}

    conformal_quantile, conformal_alpha = _load_conformal(bundle_dir)
    lift_vs_baseline, delta_dir_acc, baseline_used = _load_lift_check(bundle_dir)

    return _TrajectoryState(
        model=model,
        config=config,
        embeddings=embeddings,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata=metadata,
        encoder_alias=str(manifest.get("encoder_alias", "")),
        encoder_revision=str(manifest.get("encoder_revision", "")),
        train_end=manifest.get("train_end"),
        architecture=str(manifest.get("architecture", config.architecture)),
        bundle_dir=bundle_dir,
        conformal_quantile=conformal_quantile,
        conformal_alpha=conformal_alpha,
        lift_vs_baseline=lift_vs_baseline,
        delta_dir_acc=delta_dir_acc,
        baseline_used=baseline_used,
    )


def get_state() -> _TrajectoryState | None:
    """Return the cached state, building it on first call.

    Double-checked locking: the lock-free pre-check serves the steady
    state, the lock is only contested on cold start, and the actual
    bundle load (HF + torch + npz) runs OUTSIDE the lock so concurrent
    first-hit requests do not serialize on the slow path. We accept
    that two simultaneous first hits may both call ``_load_state``;
    the second result is discarded by the compare-and-assign under the
    lock so the worker still ends up with a single shared state.
    """

    global _state
    if _state is not None:
        return _state
    # Build outside the lock so a concurrent first-hit caller does not
    # block on a multi-second torch.load. The worst case is duplicate
    # work; correctness is preserved by the compare-and-assign below.
    candidate = _load_state()
    with _state_lock:
        if _state is None:
            _state = candidate
        return _state


def reset_state() -> None:
    global _state
    with _state_lock:
        _state = None


def install_state(state: _TrajectoryState) -> None:
    global _state
    with _state_lock:
        _state = state


def build_state_for_tests(  # noqa: PLR0913 — keyword-only fixture knobs.
    *,
    model: Any,
    config: TrajectoryConfig,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    encoder_alias: str = "test_trajectory_encoder",
    encoder_revision: str = "",
    train_end: str | None = None,
    architecture: str = "lstm",
    bundle_dir: Path = Path("/dev/null"),
    conformal_quantile: float | None = None,
    conformal_alpha: float | None = None,
    market_provider: Callable[[str], dict[str, float | None]] | None = None,
    lift_vs_baseline: bool = False,
    delta_dir_acc: float | None = None,
    baseline_used: str | None = None,
) -> _TrajectoryState:
    """Convenience constructor for tests / smoke harnesses."""

    dim = int(embeddings.shape[1]) if embeddings.size else int(config.embedding_dim)
    if feature_mean is None:
        feature_mean = np.zeros(dim, dtype=np.float32)
    if feature_std is None:
        feature_std = np.ones(dim, dtype=np.float32)
    return _TrajectoryState(
        model=model,
        config=config,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
        metadata=metadata,
        encoder_alias=encoder_alias,
        encoder_revision=encoder_revision,
        train_end=train_end,
        architecture=architecture,
        bundle_dir=Path(bundle_dir),
        conformal_quantile=conformal_quantile,
        conformal_alpha=conformal_alpha,
        market_provider=market_provider,
        lift_vs_baseline=lift_vs_baseline,
        delta_dir_acc=delta_dir_acc,
        baseline_used=baseline_used,
    )


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def _slice_history(
    metadata: pd.DataFrame,
    *,
    as_of_iso: str,
    history_length: int,
) -> pd.DataFrame:
    """Strict-backward window: rows with ``event_date < as_of`` (most recent ``N``).

    Matches the strict-backward convention enforced by
    :func:`app.retrieval.index.query` — a meeting whose ``event_date``
    equals the query date is the meeting being projected, never an
    eligible history marker.
    """

    if metadata.empty:
        return metadata
    eligible = metadata[metadata["event_date"].astype(str) < as_of_iso]
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values("event_date").reset_index(drop=True)
    return eligible.tail(int(history_length)).reset_index(drop=True)


def _market_for(
    state: _TrajectoryState,
    event_date: str,
    *,
    text_hash: str | None = None,
) -> tuple[float | None, float | None]:
    """Return the per-meeting market block inputs for ``event_date``.

    Production path: the metadata frame already carries
    ``pre_meeting_trailing_2y_yield_change_5d_bps`` / ``vix_close``
    columns when the trainer included them; fall through to the
    optional ``market_provider`` callable when those columns are
    absent (tests / hot-reload of an older bundle).

    ``text_hash`` disambiguates duplicate ``event_date`` rows (e.g.
    intermeeting release stamped same-day as a statement). When set,
    we key the lookup on the hash so the row returned matches the
    history marker the embedding window was built from.
    """

    if state.metadata is not None and not state.metadata.empty:
        if text_hash and "text_hash" in state.metadata.columns:
            row = state.metadata[
                state.metadata["text_hash"].astype(str) == str(text_hash)
            ]
        else:
            row = state.metadata[
                state.metadata["event_date"].astype(str) == event_date
            ]
        if not row.empty:
            r0 = row.iloc[0]
            y = r0.get("pre_meeting_trailing_2y_yield_change_5d_bps")
            v = r0.get("vix_close")
            return _to_float_or_none(y), _to_float_or_none(v)
    if state.market_provider is not None:
        payload = state.market_provider(event_date)
        return (
            _to_float_or_none(payload.get("trailing_2y_yield_change_5d_bps")),
            _to_float_or_none(payload.get("vix_close")),
        )
    return None, None


def _to_float_or_none(value: Any) -> float | None:
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
    import math

    if not math.isfinite(scalar):
        return None
    return scalar


def _standardise_embedding(
    embedding: np.ndarray, state: _TrajectoryState
) -> np.ndarray:
    if state.feature_mean.size == 0:
        return embedding
    return np.asarray(
        (embedding - state.feature_mean) / state.feature_std, dtype=np.float32
    )


def build_trajectory_inputs(
    state: _TrajectoryState,
    *,
    as_of_date: date_type,
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> tuple[list[dict[str, Any]], "Any | None", "Any | None", str, str | None]:
    """Slice + pad the trajectory history into the model's forward inputs.

    Extracted from :func:`project_trajectory` so callers that only need
    the inputs tensor (e.g. the XAI panel-attribution dispatcher in
    :mod:`app.services.forecaster`) do not have to re-implement the
    history-slice + scaler + pad pipeline.

    Returns ``(history_markers, inputs_tensor, mask_tensor, as_of_iso,
    warning)``. ``inputs_tensor`` and ``mask_tensor`` are ``None`` when
    the strict-backward window contains no eligible meetings — the
    caller then surfaces a "no history" payload.
    """

    import torch

    history_length = max(1, min(int(history_length), MAX_HISTORY_LENGTH))
    as_of_iso = as_of_date.isoformat()
    history_df = _slice_history(
        state.metadata, as_of_iso=as_of_iso, history_length=history_length
    )

    history_markers: list[dict[str, Any]] = []
    embeddings_window: list[np.ndarray] = []
    market_blocks: list[np.ndarray] = []
    metadata_has_hash = (
        state.metadata is not None and "text_hash" in state.metadata.columns
    )
    for _, row in history_df.iterrows():
        event_date = str(row["event_date"])
        text_hash_raw = row.get("text_hash") if metadata_has_hash else None
        text_hash = (
            str(text_hash_raw) if text_hash_raw is not None and str(text_hash_raw) != "" else None
        )
        history_markers.append(
            {
                "event_date": event_date,
                "axis_stance": _str_or_none(row.get("axis_stance")),
                "embedding_2d": (
                    float(row.get("embedding_2d_x", 0.0) or 0.0),
                    float(row.get("embedding_2d_y", 0.0) or 0.0),
                ),
            }
        )
        # Resolve the row's embedding from the npz matrix. With
        # duplicate ``event_date`` rows (e.g. intermeeting release
        # same-day as a statement) the date is not unique; prefer
        # ``text_hash`` when the bundle carries it so the embedding,
        # market block, and history marker all point at the same row.
        if text_hash and metadata_has_hash:
            matches = state.metadata.index[
                state.metadata["text_hash"].astype(str) == text_hash
            ]
        else:
            matches = state.metadata.index[
                state.metadata["event_date"].astype(str) == event_date
            ]
        parquet_idx = int(matches[0])
        raw_emb = state.embeddings[parquet_idx]
        embeddings_window.append(_standardise_embedding(raw_emb, state))
        y, v = _market_for(state, event_date, text_hash=text_hash)
        market_blocks.append(
            market_feature_vector(
                trailing_2y_yield_change_5d_bps=y, vix_close=v
            )
        )

    warning: str | None = None
    if state.train_end and as_of_iso > str(state.train_end):
        # The bundle's model never trained on meetings past ``train_end``;
        # honour the request but flag the extrapolation so the caller
        # can decide whether to surface a banner in the UI.
        warning = (
            "as_of_date is beyond train_end; projection extrapolates "
            "beyond the walk-forward fold boundary"
        )

    if not embeddings_window:
        return history_markers, None, None, as_of_iso, warning

    padded_inputs, mask = pad_sequence(
        embeddings_window,
        market_blocks,
        history_length=history_length,
    )
    inputs_tensor = torch.tensor(
        padded_inputs[np.newaxis, ...], dtype=torch.float32
    )
    mask_tensor = torch.tensor(mask[np.newaxis, ...], dtype=torch.bool)
    return history_markers, inputs_tensor, mask_tensor, as_of_iso, warning


def project_trajectory(
    state: _TrajectoryState,
    *,
    as_of_date: date_type,
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> dict[str, Any]:
    """Run inference + assemble the response payload.

    Returns a dict matching the API shape — the FastAPI handler wraps
    it with the typed response model. ``history`` is the most-recent
    ``history_length`` real meetings with their 2D anchors and stance
    labels; ``projected_next`` carries the predicted class
    distribution + a confidence band derived from the conformal
    softmax quantile (when the bundle ships one).
    """

    import torch

    history_length = max(1, min(int(history_length), MAX_HISTORY_LENGTH))
    history_markers, inputs_tensor, mask_tensor, as_of_iso, warning = (
        build_trajectory_inputs(
            state, as_of_date=as_of_date, history_length=history_length
        )
    )

    if inputs_tensor is None or mask_tensor is None:
        return {
            "history": history_markers,
            "projected_next": None,
            "architecture": state.architecture,
            "encoder_alias": state.encoder_alias,
            "history_length": int(history_length),
            "train_end": state.train_end,
            "as_of_date": as_of_iso,
            "available": False,
            "warning": warning,
            "lift_vs_baseline": bool(state.lift_vs_baseline),
            "delta_dir_acc": state.delta_dir_acc,
            "baseline_used": state.baseline_used,
        }

    state.model.eval()
    with torch.no_grad():
        logits, _ = state.model(inputs_tensor, mask_tensor)
        softmax = torch.softmax(logits, dim=-1)
        probs = softmax[0].detach().cpu().numpy()

    class_probs = {STANCE_CLASSES[i]: float(probs[i]) for i in range(len(STANCE_CLASSES))}
    # Defensive normalisation so the dict is always a valid probability
    # distribution even if float drift accumulated through softmax.
    total = float(sum(class_probs.values()))
    if total > 0:
        class_probs = {k: v / total for k, v in class_probs.items()}
    argmax = max(class_probs, key=class_probs.get)  # type: ignore[arg-type]

    confidence_band: list[str] | None = None
    if state.conformal_quantile is not None:
        keep = 1.0 - float(state.conformal_quantile)
        included = [k for k, v in class_probs.items() if v >= keep]
        confidence_band = included if included else [argmax]

    projection = {
        "predicted_stance": argmax,
        "class_probs": class_probs,
        "confidence_band": confidence_band,
        "conformal_alpha": state.conformal_alpha,
    }
    return {
        "history": history_markers,
        "projected_next": projection,
        "architecture": state.architecture,
        "encoder_alias": state.encoder_alias,
        "history_length": int(history_length),
        "train_end": state.train_end,
        "as_of_date": as_of_iso,
        "available": True,
        "warning": warning,
        "lift_vs_baseline": bool(state.lift_vs_baseline),
        "delta_dir_acc": state.delta_dir_acc,
        "baseline_used": state.baseline_used,
    }


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_HISTORY_LENGTH",
    "MARKET_FEATURE_DIM",
    "MAX_HISTORY_LENGTH",
    "STANCE_CLASSES",
    "build_state_for_tests",
    "build_trajectory_inputs",
    "bundle_available",
    "get_state",
    "install_state",
    "project_trajectory",
    "reset_state",
]
