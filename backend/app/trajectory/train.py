"""Walk-forward trainer for the trajectory model (#296).

Builds the per-meeting input panel from ``events.parquet``, applies
the same strict-forward walk-forward cut as the retrieval encoder
(:mod:`app.retrieval.train`), trains the architecture selected by
``--architecture {lstm,transformer}``, and persists the bundle under
``data/artifacts/trajectory/<run_name>/``.

Bundle layout (matches the §13 acceptance for #296)::

    model.pt           torch state_dict + config
    embedding_index.npz
                       pooled encoder embeddings per meeting,
                       float32 (N, d). Row order tracks the
                       sibling ``embedding_index.parquet``.
    embedding_index.parquet
                       per-meeting metadata (event_date, text_hash,
                       axis_stance, embedding_2d). Used by the
                       runtime singleton to assemble the response.
    manifest.json      encoder alias + revision + train_end + fold_id
                       + architecture + history_length.
    metrics.json       per-architecture macro-F1 + directional
                       accuracy with bootstrap CIs (when the
                       holdout slice is non-empty).
    conformal.json     APS softmax quantile for the calibrated
                       prediction set (when the calibration slice
                       has rows).

Every file lands via a sibling ``.tmp`` + ``os.replace`` so a mid-write
crash never leaves a half-built bundle behind.

The encoder used to materialise per-meeting embeddings is the
cross-bank DAPT checkpoint pinned at
``finbert_fed_adjacent_xbank_dapt``. The forward pass mirrors the
mean-pool the analog retrieval singleton uses (so the two surfaces
read the same statement the same way). For test / smoke runs the
``embed_fn`` argument lets callers inject a Python projection without
loading the real encoder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.evaluation.conformal import calibrate_classification_conformal
from app.evaluation.regression_metrics import with_block_bootstrap_ci
from app.trajectory.model import (
    DEFAULT_HISTORY_LENGTH,
    MARKET_FEATURE_DIM,
    N_STANCE_CLASSES,
    STANCE_CLASSES,
    Architecture,
    TrajectoryConfig,
    build_model,
    market_feature_vector,
    pad_sequence,
    save_model,
)

_logger = logging.getLogger(__name__)


DEFAULT_BASE_ENCODER_ALIAS = "finbert_fed_adjacent_xbank_dapt"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "artifacts" / "trajectory"
DEFAULT_RUN_NAME_LSTM = "trajectory_lstm"
DEFAULT_RUN_NAME_TRANSFORMER = "trajectory_transformer"
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_SEED = 11
DEFAULT_HOLDOUT_SHARE = 0.2
DEFAULT_CALIBRATION_SHARE = 0.15
DEFAULT_BOOTSTRAP_RESAMPLES = 500
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 4

FOLD_MANIFEST_FILENAME = "fold_manifest_expanding_walk_forward.json"

STANCE_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(STANCE_CLASSES)}


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSequence:
    """One training row — a left-padded meeting window plus the next-meeting label."""

    target_event_date: str
    history_event_dates: tuple[str, ...]
    inputs: np.ndarray  # (history_length, embedding_dim + market_feature_dim)
    mask: np.ndarray  # (history_length,) bool, True = real meeting
    label_index: int  # 0..n_classes-1


@dataclass(frozen=True)
class MeetingRow:
    """One distilled meeting row — keeps the columns the trainer cares about."""

    event_date: str
    text_hash: str
    text: str
    axis_stance: str | None
    trailing_2y_yield_change_5d_bps: float | None
    vix_close: float | None


def _validate_train_end(train_end: str | None) -> str | None:
    if train_end is None or str(train_end).strip() == "":
        return None
    text = str(train_end).strip()
    try:
        return date_type.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"train_end {train_end!r} is not a valid ISO date (YYYY-MM-DD)"
        ) from exc


def resolve_train_end_from_fold(
    *,
    events_parquet: Path,
    fold_id: str,
) -> str:
    """Mirror :func:`app.retrieval.train.resolve_train_end_from_fold` so the
    trajectory training surface respects the same fold definitions used by
    every other walk-forward training script.
    """

    manifest_path = events_parquet.parent / FOLD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"fold manifest not found at {manifest_path}; pass --train-end explicitly"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = payload.get("folds") or []
    for fold in folds:
        if fold.get("fold_id") == fold_id:
            train_end = fold.get("train_end")
            if not train_end:
                raise ValueError(
                    f"fold {fold_id!r} in {manifest_path} carries no train_end"
                )
            return str(train_end)
    available = sorted(str(f.get("fold_id", "")) for f in folds)
    raise ValueError(
        f"fold_id {fold_id!r} not found in {manifest_path}; available: {available}"
    )


def distill_meeting_rows(events: pd.DataFrame) -> list[MeetingRow]:
    """Project ``events.parquet`` to one ``MeetingRow`` per FOMC statement.

    The events parquet duplicates statements across horizons (1d / 5d
    / 10d / 30d), so we keep the smallest-horizon row per ``text_hash``
    — that row carries the same text and the same pre-meeting features
    (which do not depend on horizon). Output rows are sorted by
    ``event_date`` ascending so the sequence builder consumes them in
    chronological order.
    """

    if "event_kind" not in events.columns:
        raise KeyError("events parquet missing 'event_kind' column")
    required_columns = ("event_date", "text", "text_hash", "axis_stance")
    for column in required_columns:
        if column not in events.columns:
            raise KeyError(f"events parquet missing {column!r} column")

    mask = events["event_kind"].astype(str).str.lower() == "statement"
    df = events.loc[mask].copy()
    if df.empty:
        return []
    df["event_date"] = df["event_date"].astype(str)
    sort_cols = ["event_date", "text_hash"]
    if "horizon" in df.columns:
        sort_cols.append("horizon")
    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(subset=["text_hash"], keep="first")
    df = df.sort_values("event_date").reset_index(drop=True)

    rows: list[MeetingRow] = []
    for _, raw in df.iterrows():
        rows.append(
            MeetingRow(
                event_date=str(raw["event_date"]),
                text_hash=str(raw["text_hash"]),
                text=str(raw.get("text") or ""),
                axis_stance=_normalise_stance(raw.get("axis_stance")),
                trailing_2y_yield_change_5d_bps=_safe_float(
                    raw.get("pre_meeting_trailing_2y_yield_change_5d_bps")
                ),
                vix_close=_safe_float(raw.get("vix_close")),
            )
        )
    return rows


def _normalise_stance(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in STANCE_TO_INDEX:
        return text
    return None


def _safe_float(value: Any) -> float | None:
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


def build_training_sequences(
    meetings: Sequence[MeetingRow],
    *,
    embeddings: Sequence[np.ndarray] | np.ndarray,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    train_end: str | None = None,
) -> list[TrainingSequence]:
    """Build supervised next-meeting training rows over a meeting panel.

    Each row uses ``meetings[i - history_length : i]`` as the history
    window and ``meetings[i].axis_stance`` as the target. Rows where
    the target stance is unknown are skipped. ``train_end`` (ISO date)
    drops every row whose TARGET ``event_date >= train_end`` so the
    model never trains on a meeting from the holdout slice — the same
    strict-forward cut the retrieval encoder applies.
    """

    if history_length <= 0:
        raise ValueError(f"history_length must be positive; got {history_length}")
    embeddings_list = list(embeddings)
    if len(embeddings_list) != len(meetings):
        raise ValueError(
            "embeddings and meetings must align in length; "
            f"got {len(embeddings_list)} vs {len(meetings)}"
        )
    cutoff = _validate_train_end(train_end)

    sequences: list[TrainingSequence] = []
    for target_idx in range(1, len(meetings)):
        target = meetings[target_idx]
        if cutoff is not None and target.event_date >= cutoff:
            # Strict-forward: a target on or after train_end leaks the
            # holdout slice into training.
            continue
        if target.axis_stance is None:
            continue
        history_start = max(0, target_idx - history_length)
        window = meetings[history_start:target_idx]
        if not window:
            continue
        window_embeddings = [embeddings_list[i] for i in range(history_start, target_idx)]
        window_markets = [
            market_feature_vector(
                trailing_2y_yield_change_5d_bps=row.trailing_2y_yield_change_5d_bps,
                vix_close=row.vix_close,
            )
            for row in window
        ]
        inputs, mask = pad_sequence(
            window_embeddings,
            window_markets,
            history_length=history_length,
        )
        sequences.append(
            TrainingSequence(
                target_event_date=target.event_date,
                history_event_dates=tuple(row.event_date for row in window),
                inputs=inputs,
                mask=mask,
                label_index=STANCE_TO_INDEX[target.axis_stance],
            )
        )
    return sequences


def standardise_inputs(
    sequences: list[TrainingSequence],
    *,
    embedding_dim: int,
) -> tuple[list[TrainingSequence], np.ndarray, np.ndarray]:
    """Z-score the per-meeting input slabs using train-slice statistics.

    Fits the mean / std on the concatenation of every REAL (non-pad)
    timestep in ``sequences``, then applies the same transform to all
    sequences in place. Returns the standardised sequences plus the
    ``(D,)`` mean and std arrays so the inference path can reapply the
    same transform without re-fitting.

    Statistics are computed across the embedding axis only — the market
    block enters the model already in interpretable units (bps, z-vol)
    and z-scoring it a second time would over-shrink the signal.
    """

    if not sequences:
        mean = np.zeros(embedding_dim, dtype=np.float32)
        std = np.ones(embedding_dim, dtype=np.float32)
        return sequences, mean, std
    stacked = np.concatenate(
        [seq.inputs[seq.mask, :embedding_dim] for seq in sequences if seq.mask.any()],
        axis=0,
    )
    if stacked.size == 0:
        mean = np.zeros(embedding_dim, dtype=np.float32)
        std = np.ones(embedding_dim, dtype=np.float32)
        return sequences, mean, std
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # guard against constant features.
    rescaled: list[TrainingSequence] = []
    for seq in sequences:
        new_inputs = seq.inputs.copy()
        for t in range(seq.inputs.shape[0]):
            if not seq.mask[t]:
                continue
            new_inputs[t, :embedding_dim] = (
                new_inputs[t, :embedding_dim] - mean
            ) / std
        rescaled.append(
            TrainingSequence(
                target_event_date=seq.target_event_date,
                history_event_dates=seq.history_event_dates,
                inputs=new_inputs,
                mask=seq.mask,
                label_index=seq.label_index,
            )
        )
    return rescaled, mean, std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def _to_tensor_batch(sequences: Sequence[TrainingSequence], torch_mod: Any) -> tuple[Any, Any, Any]:
    inputs = torch_mod.tensor(
        np.stack([seq.inputs for seq in sequences], axis=0), dtype=torch_mod.float32
    )
    mask = torch_mod.tensor(
        np.stack([seq.mask for seq in sequences], axis=0), dtype=torch_mod.bool
    )
    labels = torch_mod.tensor(
        [seq.label_index for seq in sequences], dtype=torch_mod.long
    )
    return inputs, mask, labels


def train_model(  # noqa: PLR0913 — keyword-only knobs mirror the train_and_persist CLI surface.
    sequences: list[TrainingSequence],
    config: TrajectoryConfig,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
) -> Any:
    """Fit a trajectory model on the supplied sequence list.

    Single-process CPU/GPU loop — small dataset (~250 meetings) means
    even the full-fold pass takes seconds and a multi-worker DataLoader
    would only add overhead. Returns the trained module in eval mode.
    """

    import torch

    _set_all_seeds(seed)
    model = build_model(config)
    if not sequences:
        return model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    indices = list(range(len(sequences)))
    rng = random.Random(seed)
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = [sequences[i] for i in batch_idx]
            inputs, mask, labels = _to_tensor_batch(batch, torch)
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def evaluate_model(
    model: Any,
    sequences: list[TrainingSequence],
) -> dict[str, Any]:
    """Run inference on a sequence list and return logits + labels + softmax."""

    import torch

    if not sequences:
        return {
            "logits": np.zeros((0, N_STANCE_CLASSES), dtype=np.float32),
            "softmax": np.zeros((0, N_STANCE_CLASSES), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "predictions": np.zeros(0, dtype=np.int64),
        }
    model.eval()
    with torch.no_grad():
        inputs, mask, labels = _to_tensor_batch(sequences, torch)
        logits, _ = model(inputs, mask)
        softmax = torch.softmax(logits, dim=-1)
        predictions = softmax.argmax(dim=-1)
    return {
        "logits": logits.detach().cpu().numpy(),
        "softmax": softmax.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
        "predictions": predictions.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def macro_f1(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    if len(true_labels) == 0:
        return float("nan")
    per_class: list[float] = []
    for cls in range(N_STANCE_CLASSES):
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == cls and p != cls)
        if tp + fp == 0 or tp + fn == 0:
            per_class.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            per_class.append(0.0)
            continue
        per_class.append(2 * precision * recall / (precision + recall))
    return float(sum(per_class) / len(per_class))


def directional_accuracy(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> float:
    """Share of rows where the predicted class matches the truth.

    Direct accuracy here — the 3-class stance set has no natural
    ordering for a "signed delta" metric, so the classification
    accuracy is the cleanest analogue of the regression-side directional
    accuracy. Returns ``nan`` on empty input.
    """

    if len(true_labels) == 0:
        return float("nan")
    matches = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    return matches / len(true_labels)


def bootstrap_macro_f1(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    coverage: float = 0.95,
    seed: int = 11,
) -> tuple[float, float, float]:
    """Naive (non-block) bootstrap CI for macro-F1.

    Macro-F1 is not in :func:`with_block_bootstrap_ci`'s statistic
    table (it is classification-specific), so we implement the
    resample loop here. The block-bootstrap form fits the rates
    regression metrics; for the classification head the rows are
    already independent at the meeting level (one row per FOMC
    meeting) and a naive bootstrap is the standard CI choice.
    """

    if len(true_labels) == 0:
        return float("nan"), float("nan"), float("nan")
    point = macro_f1(true_labels, predicted_labels)
    if len(true_labels) < 2:
        return point, point, point
    rng = random.Random(seed)
    n = len(true_labels)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sampled_true = [true_labels[i] for i in idx]
        sampled_pred = [predicted_labels[i] for i in idx]
        samples.append(macro_f1(sampled_true, sampled_pred))
    samples.sort()
    alpha = (1.0 - coverage) / 2.0
    lo_idx = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    hi_idx = max(0, min(len(samples) - 1, int((1.0 - alpha) * len(samples)) - 1))
    return point, samples[lo_idx], samples[hi_idx]


def evaluate_metrics(  # noqa: PLR0913 — keyword-only bootstrap knobs forwarded to two CI helpers.
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE,
    coverage: float = 0.95,
    seed: int = 11,
) -> dict[str, Any]:
    """Compose macro-F1 + directional-accuracy with bootstrap CIs."""

    f1_point, f1_lo, f1_hi = bootstrap_macro_f1(
        true_labels,
        predicted_labels,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )
    # The block-bootstrap directional accuracy from the regression
    # metrics module accepts numeric pairs; encoding classes 0/1/2 as
    # floats keeps the metric well-defined.
    dir_ci = with_block_bootstrap_ci(
        name="directional_accuracy",
        predicted=[float(p) for p in predicted_labels],
        observed=[float(t) for t in true_labels],
        statistic="directional_accuracy",
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )
    return {
        "macro_f1": {"point": f1_point, "lo": f1_lo, "hi": f1_hi, "coverage": coverage},
        "directional_accuracy": {
            "point": directional_accuracy(true_labels, predicted_labels),
            "lo": dir_ci.lo,
            "hi": dir_ci.hi,
            "coverage": coverage,
            "block_size": block_size,
        },
        "n": int(len(true_labels)),
    }


# ---------------------------------------------------------------------------
# 2D projection
# ---------------------------------------------------------------------------


def project_2d(matrix: np.ndarray) -> np.ndarray:
    """Project an ``(N, d)`` embedding matrix to ``(N, 2)`` via PCA.

    Uses ``np.linalg.svd`` directly so we do not pull a scikit-learn
    dependency. Centres the matrix on the column mean first; with
    fewer than 2 rows / 2 columns the function returns zeros so the
    panel still renders a graceful empty state.
    """

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((max(arr.shape[0], 0), 2), dtype=np.float32)
    centred = arr - arr.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((arr.shape[0], 2), dtype=np.float32)
    components = vt[:2]
    projected = centred @ components.T
    return np.asarray(projected, dtype=np.float32)


# ---------------------------------------------------------------------------
# Bundle persistence
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    import io

    buf = io.BytesIO()
    # mypy's numpy stubs flag the **kwargs splat as targeting savez's
    # legacy ``allow_pickle`` positional. Cast through Any so the
    # idiomatic call shape stays readable.
    np.savez(buf, **arrays)  # type: ignore[arg-type]
    _atomic_write_bytes(path, buf.getvalue())


def persist_bundle(  # noqa: PLR0913 — keyword-only persistence args mirror the trainer CLI.
    *,
    out_dir: Path,
    model: Any,
    config: TrajectoryConfig,
    meetings: Sequence[MeetingRow],
    raw_embeddings: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    encoder_alias: str,
    encoder_revision: str,
    train_end: str | None,
    fold_id: str | None,
    metrics: dict[str, Any] | None,
    conformal_quantile: float | None,
    conformal_alpha: float | None,
    training_package_id: str | None = None,
) -> Path:
    """Persist the full trajectory bundle atomically.

    Writes happen in this order so a half-built bundle cannot pass the
    runtime singleton's pre-flight check:

    1. parquet (meeting metadata + 2D coords)
    2. .npz (raw embeddings + standardisation statistics)
    3. model.pt (state dict)
    4. metrics.json, conformal.json (optional)
    5. manifest.json (written LAST so its presence implies the bundle
       is complete; the runtime singleton keys availability on this file.)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projected = project_2d(raw_embeddings)
    if projected.shape[0] != len(meetings):
        # PCA degraded to empty — fall back to zeros so the panel
        # renders something rather than crashing the writer.
        projected = np.zeros((len(meetings), 2), dtype=np.float32)
    parquet_path = out_dir / "embedding_index.parquet"
    rows = []
    for idx, row in enumerate(meetings):
        rows.append(
            {
                "event_date": row.event_date,
                "text_hash": row.text_hash,
                "axis_stance": row.axis_stance,
                "embedding_2d_x": float(projected[idx, 0]) if idx < projected.shape[0] else 0.0,
                "embedding_2d_y": float(projected[idx, 1]) if idx < projected.shape[0] else 0.0,
            }
        )
    metadata_df = pd.DataFrame(rows)
    _atomic_write_parquet(metadata_df, parquet_path)

    npz_path = out_dir / "embedding_index.npz"
    _atomic_save_npz(
        npz_path,
        embeddings=raw_embeddings.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )

    save_model(model, config, out_dir / "model.pt")

    if metrics is not None:
        _atomic_write_text(
            out_dir / "metrics.json",
            json.dumps(metrics, indent=2, sort_keys=True),
        )

    if conformal_quantile is not None:
        _atomic_write_text(
            out_dir / "conformal.json",
            json.dumps(
                {
                    "softmax_quantile": float(conformal_quantile),
                    "alpha": float(conformal_alpha or 0.2),
                },
                indent=2,
                sort_keys=True,
            ),
        )

    manifest = {
        "architecture": config.architecture,
        "encoder_alias": encoder_alias,
        "encoder_revision": encoder_revision,
        "train_end": train_end,
        "fold_id": fold_id,
        "history_length": config.history_length,
        "embedding_dim": config.embedding_dim,
        "n_classes": config.n_classes,
        "row_count": int(len(meetings)),
        "training_package_id": training_package_id,
        "stance_classes": list(STANCE_CLASSES),
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config.to_dict(),
    }
    _atomic_write_text(
        out_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return out_dir


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def _default_embed_fn() -> Callable[[list[str]], np.ndarray]:
    """Build the production embedder lazily (loads the DAPT encoder)."""

    def _embed(texts: list[str]) -> np.ndarray:
        from app.models.registry import encoder_ref
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found,unused-ignore]
        import torch

        ref = encoder_ref(DEFAULT_BASE_ENCODER_ALIAS)
        if ref is None:
            raise ValueError(
                f"Encoder alias {DEFAULT_BASE_ENCODER_ALIAS!r} not in registry"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            ref.repo, local_files_only=True, trust_remote_code=False
        )
        model = AutoModel.from_pretrained(
            ref.repo, local_files_only=True, trust_remote_code=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        outs: list[np.ndarray] = []
        for text in texts:
            encoded = tokenizer(
                text or "",
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"].to(device)
            attn = encoded["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn)
            mask = attn.unsqueeze(-1).to(out.last_hidden_state.dtype)
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = (summed / counts).detach().cpu().numpy()
            outs.append(np.asarray(pooled, dtype=np.float32).reshape(-1))
        return np.stack(outs, axis=0).astype(np.float32)

    return _embed


def train_and_persist(  # noqa: PLR0913, C901, PLR0912, PLR0915 — keyword-only knobs mirror the CLI surface.
    *,
    events_parquet: Path,
    architecture: Architecture,
    base_encoder_alias: str = DEFAULT_BASE_ENCODER_ALIAS,
    encoder_revision: str = "",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str | None = None,
    history_length: int = DEFAULT_HISTORY_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    train_end: str | None = None,
    fold_id: str | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    holdout_share: float = DEFAULT_HOLDOUT_SHARE,
    calibration_share: float = DEFAULT_CALIBRATION_SHARE,
    conformal_alpha: float = 0.2,
    training_package_id: str | None = None,
) -> Path:
    """End-to-end: read events, train, evaluate, persist the bundle.

    Returns the bundle directory. The function is intentionally
    rebuild-safe: every output file is written atomically and the
    manifest lands last so a partial run cannot fool the runtime
    pre-flight check.
    """

    if architecture not in ("lstm", "transformer"):
        raise ValueError(
            f"architecture must be 'lstm' or 'transformer'; got {architecture!r}"
        )
    if train_end is not None and fold_id is not None:
        raise ValueError(
            "--train-end and --fold-id are mutually exclusive; pass only one"
        )
    resolved_train_end: str | None
    if fold_id is not None:
        resolved_train_end = resolve_train_end_from_fold(
            events_parquet=Path(events_parquet), fold_id=fold_id
        )
    else:
        resolved_train_end = _validate_train_end(train_end)

    events = pd.read_parquet(events_parquet)
    meetings = distill_meeting_rows(events)
    if not meetings:
        raise RuntimeError(
            f"events_parquet {events_parquet} yielded zero statement rows"
        )

    embedder = embed_fn if embed_fn is not None else _default_embed_fn()
    raw_embeddings = np.asarray(
        embedder([row.text for row in meetings]), dtype=np.float32
    )
    if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] != len(meetings):
        raise ValueError(
            "embed_fn must return a 2-D array with one row per meeting; "
            f"got shape {raw_embeddings.shape!r}"
        )
    embedding_dim = int(raw_embeddings.shape[1])
    config = TrajectoryConfig(
        architecture=architecture,
        embedding_dim=embedding_dim,
        history_length=history_length,
    )

    sequences = build_training_sequences(
        meetings,
        embeddings=raw_embeddings,
        history_length=history_length,
        train_end=resolved_train_end,
    )
    if not sequences:
        raise RuntimeError(
            "training sequence build yielded zero rows — verify the parquet "
            "carries statement rows with non-null axis_stance and that "
            "train_end is not stripping every target."
        )

    sequences, feature_mean, feature_std = standardise_inputs(
        sequences, embedding_dim=embedding_dim
    )

    # Holdout / calibration carve from the END of the training pool so
    # the slices are temporally contiguous (matches the walk-forward
    # spirit of the rest of the project).
    n_total = len(sequences)
    n_calibration = max(0, int(n_total * calibration_share))
    n_holdout = max(0, int(n_total * holdout_share))
    n_train = max(1, n_total - n_calibration - n_holdout)
    train_sequences = sequences[:n_train]
    calibration_sequences = sequences[n_train : n_train + n_calibration]
    holdout_sequences = sequences[n_train + n_calibration :]

    model = train_model(
        train_sequences,
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    metrics_payload: dict[str, Any] | None = None
    conformal_quantile: float | None = None
    if holdout_sequences:
        evaluation = evaluate_model(model, holdout_sequences)
        metrics_payload = {
            "architecture": architecture,
            "holdout": evaluate_metrics(
                evaluation["labels"].tolist(),
                evaluation["predictions"].tolist(),
                seed=seed,
            ),
        }
        # Naive-prior baseline: predict the modal class of the train slice.
        if train_sequences:
            train_labels = [seq.label_index for seq in train_sequences]
            modal = max(set(train_labels), key=train_labels.count)
            baseline_pred = [modal] * len(holdout_sequences)
            metrics_payload["baseline_modal"] = evaluate_metrics(
                evaluation["labels"].tolist(),
                baseline_pred,
                seed=seed,
            )
    if calibration_sequences:
        cal_eval = evaluate_model(model, calibration_sequences)
        try:
            conformal_quantile = float(
                calibrate_classification_conformal(
                    softmax_scores=cal_eval["softmax"].tolist(),
                    true_classes=cal_eval["labels"].tolist(),
                    alpha=conformal_alpha,
                )
            )
        except ValueError:
            conformal_quantile = None

    resolved_run_name = run_name or (
        DEFAULT_RUN_NAME_LSTM
        if architecture == "lstm"
        else DEFAULT_RUN_NAME_TRANSFORMER
    )
    out_dir = Path(output_root) / resolved_run_name
    persist_bundle(
        out_dir=out_dir,
        model=model,
        config=config,
        meetings=meetings,
        raw_embeddings=raw_embeddings,
        feature_mean=feature_mean,
        feature_std=feature_std,
        encoder_alias=base_encoder_alias,
        encoder_revision=encoder_revision,
        train_end=resolved_train_end,
        fold_id=fold_id,
        metrics=metrics_payload,
        conformal_quantile=conformal_quantile,
        conformal_alpha=conformal_alpha if conformal_quantile is not None else None,
        training_package_id=training_package_id,
    )
    _logger.info(
        "trajectory_train_done architecture=%s rows=%d holdout=%d out_dir=%s",
        architecture,
        n_train,
        len(holdout_sequences),
        out_dir,
    )
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a trajectory model (LSTM or Transformer) (#296)."
    )
    parser.add_argument("--events-parquet", required=True, type=Path)
    parser.add_argument(
        "--architecture", required=True, choices=("lstm", "transformer")
    )
    parser.add_argument("--base-encoder-alias", default=DEFAULT_BASE_ENCODER_ALIAS)
    parser.add_argument("--encoder-revision", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--fold-id", default=None)
    parser.add_argument("--holdout-share", type=float, default=DEFAULT_HOLDOUT_SHARE)
    parser.add_argument(
        "--calibration-share", type=float, default=DEFAULT_CALIBRATION_SHARE
    )
    parser.add_argument("--conformal-alpha", type=float, default=0.2)
    parser.add_argument("--training-package-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s %(message)s"
    )
    out_dir = train_and_persist(
        events_parquet=args.events_parquet,
        architecture=args.architecture,
        base_encoder_alias=args.base_encoder_alias,
        encoder_revision=args.encoder_revision,
        output_root=args.output_root,
        run_name=args.run_name,
        history_length=args.history_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_end=args.train_end,
        fold_id=args.fold_id,
        holdout_share=args.holdout_share,
        calibration_share=args.calibration_share,
        conformal_alpha=args.conformal_alpha,
        training_package_id=args.training_package_id,
    )
    print(f"[trajectory.train] saved bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
