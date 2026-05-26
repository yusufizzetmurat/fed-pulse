"""Walk-forward trainer for the trajectory model (#296, #332).

Builds the per-meeting input panel from ``events.parquet``, applies
the same strict-forward walk-forward cut as the retrieval encoder
(:mod:`app.retrieval.train`), trains the architecture selected by
``--architecture {lstm,transformer}``, and persists the bundle under
``data/artifacts/trajectory/<run_name>/``.

Parameter-count cap (#332). With ~250 historical statements, a 4-layer
x 64-d_model x 4-head Transformer (the #296 default at ~250k parameters)
sits at a poor data:parameter ratio. Per the lift-or-no-lift convention
spelled out in the issue, the Transformer arm caps at 2 layers x 32
d_model unless it beats the strongest naive baseline (previous_stance /
rolling_majority(3) / small-LSTM) by >= 5pp directional accuracy on the
canonical fold protocol. The cap is enforced via
:func:`assert_parameter_count_within_cap` before the training loop
starts; ``--no-param-cap`` opts out only when the lift threshold has
already been demonstrated on a downstream sweep. The cap default
(75_000) accommodates a 2x32 Transformer at the canonical 768-d DAPT
embedding (~50k parameters) with comfortable headroom and rejects the
historical 4x64 default (~250k parameters).

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
from app.trajectory.baselines import (
    DEFAULT_ROLLING_WINDOW,
    compare_against_transformer,
    evaluate_previous_stance,
    evaluate_rolling_majority,
    evaluate_small_lstm,
)
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

# Default parameter-count cap (#332). Sized to comfortably admit the
# 2 layers x 32 d_model Transformer arm at the canonical 768-d DAPT
# embedding (~50.6k params) while rejecting the original 4 layers x 64
# d_model default (~250k params). Lift threshold for an override is
# >= 5pp directional-accuracy beat over the strongest naive baseline.
DEFAULT_PARAMETER_COUNT_CAP: int = 75_000

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


def resolve_test_end_from_fold(
    *,
    events_parquet: Path,
    fold_id: str,
) -> str | None:
    """Return the fold's ``test_end`` (or ``None`` when the manifest omits it).

    The walk-forward holdout for fold ``F`` is every meeting with
    ``event_date >= F.train_end AND event_date < F.test_end``. When the
    manifest omits a ``test_end`` (older training packages) we treat
    the slice as open-ended (every meeting past ``train_end`` becomes
    holdout) so the trajectory metrics still measure something
    walk-forward-correct.
    """

    manifest_path = events_parquet.parent / FOLD_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for fold in payload.get("folds") or []:
        if fold.get("fold_id") == fold_id:
            raw = fold.get("test_end")
            if not raw:
                return None
            return str(raw)
    return None


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


def fit_standardisation_stats(
    sequences: list[TrainingSequence],
    *,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-embedding-feature mean / std over REAL timesteps.

    Returns ``(mean, std)`` shaped ``(embedding_dim,)`` each. ``std`` is
    floored at ``1e-6`` so a constant feature does not divide-by-zero
    in the transform step. Empty input yields ``(zeros, ones)``.
    """

    if not sequences:
        return (
            np.zeros(embedding_dim, dtype=np.float32),
            np.ones(embedding_dim, dtype=np.float32),
        )
    stacked = np.concatenate(
        [seq.inputs[seq.mask, :embedding_dim] for seq in sequences if seq.mask.any()],
        axis=0,
    )
    if stacked.size == 0:
        return (
            np.zeros(embedding_dim, dtype=np.float32),
            np.ones(embedding_dim, dtype=np.float32),
        )
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # guard against constant features.
    return mean, std


def apply_standardisation(
    sequences: list[TrainingSequence],
    *,
    embedding_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[TrainingSequence]:
    """Apply pre-fit ``(mean, std)`` to the embedding slab of every sequence."""

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
    return rescaled


def standardise_inputs(
    sequences: list[TrainingSequence],
    *,
    embedding_dim: int,
    train_sequences: list[TrainingSequence] | None = None,
) -> tuple[list[TrainingSequence], np.ndarray, np.ndarray]:
    """Z-score the per-meeting input slabs using TRAIN-slice statistics.

    The mean / std are fit on ``train_sequences`` when supplied — that
    is the path the trainer takes after the temporal carve into
    train / calibration / holdout, so the calibration and holdout
    rows never contribute to the standardisation statistics. When
    ``train_sequences`` is ``None`` we fall back to fitting on the
    full ``sequences`` list (kept for backwards compatibility with
    standalone callers, e.g. ad-hoc unit tests).

    Statistics are computed across the embedding axis only — the market
    block enters the model already in interpretable units (bps, z-vol)
    and z-scoring it a second time would over-shrink the signal.
    """

    if not sequences:
        return (
            sequences,
            np.zeros(embedding_dim, dtype=np.float32),
            np.ones(embedding_dim, dtype=np.float32),
        )
    fit_source = train_sequences if train_sequences is not None else sequences
    mean, std = fit_standardisation_stats(fit_source, embedding_dim=embedding_dim)
    rescaled = apply_standardisation(
        sequences, embedding_dim=embedding_dim, mean=mean, std=std
    )
    return rescaled, mean, std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _history_label_lists(
    sequences: Sequence[TrainingSequence],
    meetings: Sequence[MeetingRow],
) -> list[list[int]]:
    """Build per-row stance-index history lists from the meetings panel.

    The naive baselines (#332) consume stance-only history — the same
    sequence of past meeting labels the Transformer arm sees, projected
    down to the label space. Returns one history list per row in
    ``sequences``, with unrecognised stances dropped. Used by the
    trainer to feed ``evaluate_previous_stance`` /
    ``evaluate_rolling_majority`` / ``evaluate_small_lstm``.
    """

    label_by_date: dict[str, int] = {}
    for row in meetings:
        if row.axis_stance is not None and row.axis_stance in STANCE_TO_INDEX:
            label_by_date[row.event_date] = STANCE_TO_INDEX[row.axis_stance]
    histories: list[list[int]] = []
    for seq in sequences:
        history: list[int] = []
        for event_date in seq.history_event_dates:
            idx = label_by_date.get(str(event_date))
            if idx is not None:
                history.append(idx)
        histories.append(history)
    return histories


def assert_parameter_count_within_cap(
    model: Any, *, cap: int = DEFAULT_PARAMETER_COUNT_CAP
) -> int:
    """Block oversized trajectory architectures from entering the train loop (#332).

    With ~250 historical statements, even the 2x32 Transformer (~50k
    params at the 768-d DAPT embedding) is data-starved; the 4x64
    default (~250k) is the configuration this cap exists to reject.
    Lifting the cap requires the lift-or-no-lift verdict on the
    canonical fold protocol to already favour the larger architecture
    by >= 5pp directional accuracy over the strongest naive baseline.

    Returns the actual parameter count so the caller can record it in
    the metrics payload.
    """

    actual = int(sum(p.numel() for p in model.parameters()))
    if actual > int(cap):
        raise AssertionError(
            f"trajectory model parameter count {actual} exceeds cap {cap}; "
            "shrink the architecture (e.g. transformer_layers=2, "
            "transformer_d_model=32) or pass --no-param-cap when the "
            ">=5pp lift threshold over the strongest naive baseline has "
            "already been demonstrated on the canonical fold protocol."
        )
    return actual


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


def fit_pca_axes(
    train_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the centred-mean + top-2 principal axes on ``train_matrix``.

    Returns ``(mean, components)`` where ``mean`` is ``(d,)`` and
    ``components`` is ``(2, d)`` — the same shape ``vt[:2]`` returns
    from :func:`np.linalg.svd`. Callers re-use the pair via
    :func:`project_with_axes` so train / cal / holdout all land on the
    same train-fit basis. Degraded inputs (fewer than 2 rows or
    fewer than 2 columns) return all-zero axes; downstream projection
    then falls back to zero coords so the panel still renders.
    """

    arr = np.asarray(train_matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        mean = (
            np.zeros(arr.shape[1] if arr.ndim == 2 else 0, dtype=np.float32)
        )
        components = np.zeros((2, mean.shape[0]), dtype=np.float32)
        return mean, components
    mean = arr.mean(axis=0).astype(np.float32)
    centred = arr - mean
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return mean, np.zeros((2, arr.shape[1]), dtype=np.float32)
    components = vt[:2].astype(np.float32)
    return mean, components


def project_with_axes(
    matrix: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project an ``(N, d)`` matrix onto pre-fit ``mean`` + ``components``."""

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((max(arr.shape[0], 0), 2), dtype=np.float32)
    if components.shape[0] < 2 or (components == 0).all():
        return np.zeros((arr.shape[0], 2), dtype=np.float32)
    centred = arr - mean.reshape(1, -1)
    projected = centred @ components.T
    return np.asarray(projected, dtype=np.float32)


def project_2d(matrix: np.ndarray) -> np.ndarray:
    """Fit + project in one call — convenience for callers without a holdout.

    Prefer :func:`fit_pca_axes` + :func:`project_with_axes` in the
    walk-forward trainer so the holdout slice's embedding geometry
    never leaks into the principal-axes fit.
    """

    mean, components = fit_pca_axes(matrix)
    return project_with_axes(matrix, mean=mean, components=components)


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


def persist_bundle(  # noqa: PLR0913, C901, PLR0912, PLR0915 — keyword-only persistence args mirror the trainer CLI.
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
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    model_parameter_count: int | None = None,
) -> Path:
    """Persist the full trajectory bundle atomically.

    Writes happen in this order so a half-built bundle cannot pass the
    runtime singleton's pre-flight check:

    1. parquet (meeting metadata + market columns + 2D coords)
    2. .npz (raw embeddings + standardisation statistics + PCA axes)
    3. model.pt (state dict)
    4. metrics.json, conformal.json (optional)
    5. manifest.json (written LAST so its presence implies the bundle
       is complete; the runtime singleton keys availability on this file.)

    Bundle hygiene rules enforced here:

    * **train_end filter** — the persisted parquet / npz only carry
      meetings with ``event_date < train_end``. The model never trained
      on a meeting from the holdout slice, so projecting from one would
      be a walk-forward leak at inference time. ``train_end=None``
      degrades to "persist everything" for ad-hoc / smoke runs.
    * **PCA axes from train slice** — when ``pca_mean`` and
      ``pca_components`` are supplied, the 2D coords land on the
      caller-fit basis (train-only). When omitted, we fit + project on
      the persisted slice as a fallback for legacy callers.
    * **market columns** — ``pre_meeting_trailing_2y_yield_change_5d_bps``
      and ``vix_close`` ride along on the parquet so the runtime path
      reads the same numbers the trainer saw. Missing inputs persist
      as NaN; the runtime ``_market_for`` honours that with the
      explicit-missing bit in the market feature vector.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_iso = _validate_train_end(train_end)
    if cutoff_iso is not None:
        keep_mask = np.array(
            [row.event_date < cutoff_iso for row in meetings], dtype=bool
        )
    else:
        keep_mask = np.ones(len(meetings), dtype=bool)
    kept_meetings = [m for m, keep in zip(meetings, keep_mask) if keep]
    kept_embeddings = (
        raw_embeddings[keep_mask] if raw_embeddings.size else raw_embeddings
    )

    # 2D projection: project the persisted slice onto the caller's
    # pre-fit axes when available (walk-forward path); otherwise fit
    # on the persisted slice itself (legacy / smoke path).
    if pca_mean is not None and pca_components is not None and kept_embeddings.size:
        projected = project_with_axes(
            kept_embeddings, mean=pca_mean, components=pca_components
        )
    else:
        # Fallback path: re-fit on the persisted slice. Safe because
        # the persisted slice itself is train-side (post-cutoff rows
        # were already dropped above), so the geometry still excludes
        # holdout rows.
        pca_mean, pca_components = fit_pca_axes(kept_embeddings)
        projected = project_with_axes(
            kept_embeddings, mean=pca_mean, components=pca_components
        )
    if projected.shape[0] != len(kept_meetings):
        projected = np.zeros((len(kept_meetings), 2), dtype=np.float32)

    parquet_path = out_dir / "embedding_index.parquet"
    rows = []
    for idx, row in enumerate(kept_meetings):
        rows.append(
            {
                "event_date": row.event_date,
                "text_hash": row.text_hash,
                "axis_stance": row.axis_stance,
                "embedding_2d_x": float(projected[idx, 0]) if idx < projected.shape[0] else 0.0,
                "embedding_2d_y": float(projected[idx, 1]) if idx < projected.shape[0] else 0.0,
                "pre_meeting_trailing_2y_yield_change_5d_bps": (
                    float(row.trailing_2y_yield_change_5d_bps)
                    if row.trailing_2y_yield_change_5d_bps is not None
                    else None
                ),
                "vix_close": (
                    float(row.vix_close)
                    if row.vix_close is not None
                    else None
                ),
            }
        )
    metadata_df = pd.DataFrame(rows)
    _atomic_write_parquet(metadata_df, parquet_path)

    npz_path = out_dir / "embedding_index.npz"
    npz_arrays: dict[str, np.ndarray] = {
        "embeddings": (
            kept_embeddings.astype(np.float32)
            if kept_embeddings.size
            else np.zeros((0, raw_embeddings.shape[1] if raw_embeddings.ndim == 2 else 0), dtype=np.float32)
        ),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }
    if pca_mean is not None:
        npz_arrays["pca_mean"] = pca_mean.astype(np.float32)
    if pca_components is not None:
        npz_arrays["pca_components"] = pca_components.astype(np.float32)
    _atomic_save_npz(npz_path, **npz_arrays)

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
        "row_count": int(len(kept_meetings)),
        "training_package_id": training_package_id,
        "stance_classes": list(STANCE_CLASSES),
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config.to_dict(),
        # Calibration convention: the calibration partition is a
        # temporal tail of the train slice (last ``calibration_share``
        # of pre-train_end sequences). Holdout = the post-train_end
        # meetings up to the fold's ``test_end`` boundary. See
        # ``train_and_persist`` for the carving logic.
        "calibration_convention": "tail_of_train_slice",
    }
    if model_parameter_count is not None:
        manifest["model_parameter_count"] = int(model_parameter_count)
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
    enforce_param_cap: bool = True,
    parameter_count_cap: int = DEFAULT_PARAMETER_COUNT_CAP,
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

    # Build BOTH the pre-train_end training pool (for fit) AND the
    # post-train_end holdout pool (for walk-forward metrics) in a
    # single pass over the meetings panel. Targets whose ``event_date``
    # equals or exceeds ``train_end`` are walk-forward holdout; the
    # rest are train + calibration. ``build_training_sequences`` with
    # ``train_end=None`` returns every labelled target; we filter
    # downstream with explicit boundaries so both slices share the
    # same upstream pipeline.
    all_sequences = build_training_sequences(
        meetings,
        embeddings=raw_embeddings,
        history_length=history_length,
        train_end=None,
    )
    if not all_sequences:
        raise RuntimeError(
            "training sequence build yielded zero rows — verify the parquet "
            "carries statement rows with non-null axis_stance."
        )

    # Holdout boundary from the fold manifest when available. The
    # walk-forward holdout for fold F is meetings with
    # ``train_end <= event_date < test_end`` — exactly the slice the
    # rest of the project's metrics benchmark against. When the
    # manifest omits a ``test_end`` (or no fold was supplied) the
    # holdout extends to end-of-history.
    resolved_test_end: str | None = None
    if fold_id is not None:
        try:
            resolved_test_end = resolve_test_end_from_fold(
                events_parquet=Path(events_parquet), fold_id=fold_id
            )
        except Exception:  # pragma: no cover — guarded so a missing manifest never crashes the trainer
            resolved_test_end = None

    pre_cutoff: list[TrainingSequence] = []
    post_cutoff: list[TrainingSequence] = []
    if resolved_train_end is None:
        pre_cutoff = list(all_sequences)
    else:
        for seq in all_sequences:
            if seq.target_event_date < resolved_train_end:
                pre_cutoff.append(seq)
            else:
                if (
                    resolved_test_end is None
                    or seq.target_event_date < resolved_test_end
                ):
                    post_cutoff.append(seq)
    if not pre_cutoff:
        raise RuntimeError(
            "pre-train_end sequence pool is empty — verify train_end leaves "
            "at least one labelled target inside the train slice."
        )

    # Calibration partition = TEMPORAL TAIL of the train slice (last
    # ``calibration_share`` of pre-train_end sequences). Keeps the
    # calibration pool walk-forward-correct without consuming holdout
    # rows. ``holdout_share`` is preserved as a knob but only affects
    # the legacy fallback path when no post-train_end slice exists.
    n_pre = len(pre_cutoff)
    n_calibration = max(0, int(n_pre * calibration_share))
    train_sequences = pre_cutoff[: max(1, n_pre - n_calibration)]
    calibration_sequences = pre_cutoff[max(1, n_pre - n_calibration) :]
    if post_cutoff:
        holdout_sequences = post_cutoff
    else:
        # Legacy / smoke fallback: no fold manifest and no train_end →
        # carve a tail-of-train holdout so the metrics block is
        # non-empty. This path is never the canonical walk-forward
        # report and is documented as such in manifest.json.
        n_holdout = max(0, int(n_pre * holdout_share))
        if n_holdout > 0 and len(train_sequences) > n_holdout:
            holdout_sequences = train_sequences[-n_holdout:]
            train_sequences = train_sequences[:-n_holdout]
        else:
            holdout_sequences = []

    # Standardise: fit mean / std on TRAIN ONLY, apply to all three
    # partitions + the persisted raw_embeddings pathway (the inference
    # path z-scores embeddings via the same stats stored in the npz).
    feature_mean, feature_std = fit_standardisation_stats(
        train_sequences, embedding_dim=embedding_dim
    )
    train_sequences = apply_standardisation(
        train_sequences,
        embedding_dim=embedding_dim,
        mean=feature_mean,
        std=feature_std,
    )
    calibration_sequences = apply_standardisation(
        calibration_sequences,
        embedding_dim=embedding_dim,
        mean=feature_mean,
        std=feature_std,
    )
    holdout_sequences = apply_standardisation(
        holdout_sequences,
        embedding_dim=embedding_dim,
        mean=feature_mean,
        std=feature_std,
    )

    # Fit the PCA projection axes on the TRAIN-SLICE embeddings only,
    # so the 2D anchors that ship in the bundle never depend on the
    # holdout slice's embedding geometry. The persisted parquet is
    # filtered to the same train-slice rows in ``persist_bundle``.
    if resolved_train_end is not None:
        train_embedding_mask = np.array(
            [row.event_date < resolved_train_end for row in meetings], dtype=bool
        )
    else:
        train_embedding_mask = np.ones(len(meetings), dtype=bool)
    train_embeddings_slice = (
        raw_embeddings[train_embedding_mask]
        if raw_embeddings.size
        else raw_embeddings
    )
    pca_mean, pca_components = fit_pca_axes(train_embeddings_slice)

    # Parameter-count cap (#332) — runs BEFORE the training loop so a
    # mis-configured architecture surfaces in seconds rather than after
    # a multi-epoch fit. The cap is opt-out via ``enforce_param_cap``;
    # the CLI exposes the override as ``--no-param-cap``.
    if enforce_param_cap:
        probe_model = build_model(config)
        assert_parameter_count_within_cap(probe_model, cap=parameter_count_cap)
        del probe_model

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
            "holdout_slice": (
                "post_train_end_to_test_end" if post_cutoff else "tail_of_train_fallback"
            ),
        }
        # Naive-prior baseline: predict the modal class of the train slice.
        modal: int | None = None
        if train_sequences:
            train_labels = [seq.label_index for seq in train_sequences]
            modal = max(set(train_labels), key=train_labels.count)
            baseline_pred = [modal] * len(holdout_sequences)
            metrics_payload["baseline_modal"] = evaluate_metrics(
                evaluation["labels"].tolist(),
                baseline_pred,
                seed=seed,
            )

        # Naive baselines + small-LSTM (#332). Each consumes the stance
        # sequence of the same history window the Transformer arm sees,
        # so the comparison is on the canonical fold protocol. The
        # results land both on ``metrics_payload['baselines']`` (for
        # downstream reporting) and on ``metrics_payload['lift_check']``
        # (the lift / no-lift verdict the API endpoint surfaces).
        holdout_histories = _history_label_lists(holdout_sequences, meetings)
        holdout_truths = evaluation["labels"].tolist()
        train_histories = _history_label_lists(train_sequences, meetings)

        prev_result = evaluate_previous_stance(
            holdout_histories, holdout_truths, fallback_modal=modal
        )
        rolling_result = evaluate_rolling_majority(
            holdout_histories,
            holdout_truths,
            n=DEFAULT_ROLLING_WINDOW,
            fallback_modal=modal,
        )
        try:
            lstm_result = evaluate_small_lstm(
                train_histories,
                holdout_histories,
                holdout_truths,
                seed=seed,
                fallback_modal=modal,
            )
        except Exception:  # pragma: no cover — guarded so a torch import / fit failure never crashes the trainer
            _logger.warning("trajectory_small_lstm_baseline_failed", exc_info=True)
            lstm_result = None

        baseline_results = [prev_result, rolling_result]
        if lstm_result is not None:
            baseline_results.append(lstm_result)
        metrics_payload["baselines"] = {
            b.name: {
                "directional_accuracy": float(b.directional_accuracy)
                if not (b.directional_accuracy != b.directional_accuracy)  # noqa: PLR0124 — nan check
                else None,
                "confusion_matrix": [list(row) for row in b.confusion_matrix],
                "n": int(b.n),
            }
            for b in baseline_results
        }
        transformer_dir_acc = metrics_payload["holdout"]["directional_accuracy"]["point"]
        if transformer_dir_acc == transformer_dir_acc:  # noqa: PLR0124 — finite check
            metrics_payload["lift_check"] = compare_against_transformer(
                transformer_dir_acc, baseline_results
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

    # Parameter count — methodological hygiene for the LSTM vs
    # Transformer comparison. Logged on both the per-architecture
    # metrics block and the manifest so a future reviewer can confirm
    # the two arms ran with comparable capacity.
    model_parameter_count: int | None = None
    try:
        model_parameter_count = int(sum(p.numel() for p in model.parameters()))
        if metrics_payload is not None:
            metrics_payload["model_parameter_count"] = model_parameter_count
    except Exception:  # pragma: no cover — extremely defensive
        model_parameter_count = None

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
        pca_mean=pca_mean,
        pca_components=pca_components,
        model_parameter_count=model_parameter_count,
    )
    _logger.info(
        "trajectory_train_done architecture=%s train_rows=%d cal_rows=%d "
        "holdout_rows=%d holdout_slice=%s out_dir=%s",
        architecture,
        len(train_sequences),
        len(calibration_sequences),
        len(holdout_sequences),
        "post_train_end" if post_cutoff else "tail_of_train_fallback",
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
    parser.add_argument(
        "--no-param-cap",
        action="store_true",
        help=(
            "Bypass the trajectory parameter-count cap (#332). Use only "
            "when the >=5pp lift threshold over the strongest naive "
            "baseline has been demonstrated on the canonical fold "
            "protocol."
        ),
    )
    parser.add_argument(
        "--parameter-count-cap",
        type=int,
        default=DEFAULT_PARAMETER_COUNT_CAP,
        help="Override the default parameter-count cap (#332).",
    )
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
        enforce_param_cap=not args.no_param_cap,
        parameter_count_cap=args.parameter_count_cap,
    )
    print(f"[trajectory.train] saved bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
