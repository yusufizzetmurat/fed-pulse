"""Naive + small-LSTM baselines for the trajectory head (#332).

The Transformer arm shipped in #296 runs at 4 layers x 64 d_model x 4 heads
(~200k parameters) against ~250 historical statements. Stance sequences carry
strong autocorrelation: hawkish meetings cluster, dovish meetings cluster, and
the modal class accounts for a non-trivial share of the corpus. Predictors
that simply echo the last stance or the rolling majority therefore score well
by default — without a published comparison cell, an absolute Transformer
directional-accuracy number does not tell the reviewer whether the arm
extracted a real signal.

This module publishes three baselines on the same fold protocol the train
script uses (event-date sorted statements, history of length N, target is the
next meeting's stance):

* ``previous_stance(history)`` — predicts ``history[-1]``.
* ``rolling_majority(history, n=3)`` — predicts the modal label over the last
  ``n`` real meetings (ties broken by the most recent label).
* ``small_lstm_baseline(history)`` — a 1-layer x 16-hidden LSTM trained on
  stance-index sequences. Caps at <= 5k parameters so the comparison cell is
  unambiguously "small honest baseline" rather than "transformer-lite".

Each baseline ships a :func:`evaluate_baseline_*` helper that consumes the
same ``TrainingSequence`` list the trainer carves out (so train / cal /
holdout slices stay aligned with the Transformer arm) and emits a payload
with directional accuracy + a 3x3 confusion matrix. The
:func:`compare_against_transformer` helper assembles the "lift / no-lift"
verdict per the >= 5pp threshold the issue spells out.

Stance indices follow ``app.trajectory.model.STANCE_CLASSES`` so the integer
labels match whatever the Transformer head emits.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from app.trajectory.model import N_STANCE_CLASSES, STANCE_CLASSES

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_ROLLING_WINDOW: int = 3
DEFAULT_LSTM_HIDDEN: int = 16
DEFAULT_LSTM_LAYERS: int = 1
DEFAULT_LSTM_EPOCHS: int = 40
DEFAULT_LSTM_LR: float = 1e-2
DEFAULT_LSTM_SEED: int = 11
DEFAULT_LSTM_PARAM_CAP: int = 5_000

# Threshold (in directional-accuracy points) the Transformer arm must beat
# the best naive baseline by before /analyze/trajectory drops the "no-lift"
# badge. Spelled out in the issue (#332) as 5pp.
LIFT_THRESHOLD_POINTS: float = 0.05


@dataclass(frozen=True)
class BaselineResult:
    """Per-baseline metrics block + the integer predictions for each holdout row.

    The frozen dataclass keeps the structure trivially serialisable into
    ``metrics.json`` blocks without an extra ``to_dict`` plumbing layer —
    callers can ``dataclasses.asdict(result)`` directly when they need a
    JSON-friendly payload.
    """

    name: str
    predictions: tuple[int, ...]
    truths: tuple[int, ...]
    directional_accuracy: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    n: int


# ---------------------------------------------------------------------------
# Naive predictors
# ---------------------------------------------------------------------------


def _coerce_index(value: Any) -> int | None:
    """Accept either ``"hawkish"`` (string) or ``0`` (int) and return the index.

    Mixed-type inputs are common: the train script keeps history as
    ``TrainingSequence.label_index`` (int) but the runtime singleton reads
    stance strings off the parquet. A single helper means callers do not
    have to branch on which path is producing the history list.
    """

    if value is None:
        return None
    if isinstance(value, (int, np.integer)):  # noqa: UP038 — keep tuple form; numpy + builtin int via X | Y breaks isinstance on some numpy versions
        idx = int(value)
        if 0 <= idx < N_STANCE_CLASSES:
            return idx
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        for i, label in enumerate(STANCE_CLASSES):
            if cleaned == label:
                return i
    return None


def previous_stance(history: Sequence[Any]) -> int | None:
    """Predict the last observed stance in ``history``.

    Returns ``None`` if the entire history is unrecognised — the caller is
    expected to fall back to the train-slice modal class in that case. The
    contract matches what the existing ``baseline_modal`` block in
    ``train.evaluate_metrics`` does for the modal-class fallback.
    """

    for raw in reversed(list(history)):
        idx = _coerce_index(raw)
        if idx is not None:
            return idx
    return None


def rolling_majority(
    history: Sequence[Any], *, n: int = DEFAULT_ROLLING_WINDOW
) -> int | None:
    """Predict the modal label over the last ``n`` real meetings in ``history``.

    Ties are broken by recency — the most recent stance wins, so this
    degenerates to :func:`previous_stance` when every label in the window
    is distinct. ``n`` clamps to the history length so a short prefix does
    not return ``None`` purely because the window is wider than the
    available data.
    """

    if n <= 0:
        raise ValueError(f"rolling_majority window must be positive; got {n}")
    indices: list[int] = []
    for raw in reversed(list(history)):
        idx = _coerce_index(raw)
        if idx is not None:
            indices.append(idx)
        if len(indices) >= n:
            break
    if not indices:
        return None
    counts = Counter(indices)
    top = counts.most_common()
    best_freq = top[0][1]
    # Recency tie-break: ``indices`` is ordered most-recent first, so the
    # first label with the best frequency is the most recent winner.
    for idx in indices:
        if counts[idx] == best_freq:
            return idx
    return top[0][0]


# ---------------------------------------------------------------------------
# Small-LSTM baseline (stance sequences only — no text embeddings)
# ---------------------------------------------------------------------------


def _stance_indices_from_history(history: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for raw in history:
        idx = _coerce_index(raw)
        if idx is not None:
            out.append(idx)
    return out


def build_small_lstm(
    *,
    hidden_size: int = DEFAULT_LSTM_HIDDEN,
    num_layers: int = DEFAULT_LSTM_LAYERS,
    n_classes: int = N_STANCE_CLASSES,
) -> Any:
    """Build a 1-layer x ``hidden_size`` LSTM over one-hot stance inputs.

    Layout::

        Input: (B, T, n_classes)  one-hot stance indices
        LSTM:  hidden_size, num_layers
        Head:  Linear(hidden_size, n_classes)

    The module ships in eval mode. :func:`train_small_lstm` flips it to
    train mode for the fit loop. Parameter count is verified against
    ``DEFAULT_LSTM_PARAM_CAP`` by :func:`assert_param_count_within_cap`.
    """

    import torch  # type: ignore[import-not-found,unused-ignore]

    nn = torch.nn

    class _SmallLSTM(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.n_classes = int(n_classes)
            self.lstm = nn.LSTM(
                input_size=n_classes,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, inputs: Any) -> Any:
            outputs, _ = self.lstm(inputs)
            # Read the final timestep — the trajectory head shares the same
            # "decode from the last real position" convention so the two
            # arms are wired the same way.
            last = outputs[:, -1, :]
            return self.head(last)

    model = _SmallLSTM()
    model.eval()
    return model


def assert_param_count_within_cap(model: Any, *, cap: int) -> int:
    """Hard-fail when a baseline / arm exceeds the parameter-count cap.

    Returns the actual parameter count so callers can record it in their
    metrics payload. The error is an :class:`AssertionError` so the
    failure mode is unmissable in CI and is straightforward to override
    with ``-O`` only in adversarial test paths (the trainer surfaces a
    ``--no-param-cap`` CLI flag for legitimate overrides).
    """

    actual = int(sum(p.numel() for p in model.parameters()))
    if actual > int(cap):
        raise AssertionError(
            f"parameter count {actual} exceeds cap {cap}; "
            "shrink the architecture or pass --no-param-cap to override"
        )
    return actual


def _one_hot(indices: Sequence[int], *, n_classes: int) -> np.ndarray:
    arr = np.zeros((len(indices), n_classes), dtype=np.float32)
    for t, idx in enumerate(indices):
        if 0 <= int(idx) < n_classes:
            arr[t, int(idx)] = 1.0
    return arr


def _build_lstm_training_pairs(
    stance_sequences: Iterable[Sequence[int]],
) -> list[tuple[np.ndarray, int]]:
    """Carve ``(history_one_hot, next_label)`` rows out of stance sequences.

    Each input sequence ``[s0, s1, ..., sK]`` yields ``K`` training rows
    of shape ``(t+1, n_classes)`` paired with label ``s_{t+1}``. Empty /
    length-1 sequences contribute zero rows.
    """

    pairs: list[tuple[np.ndarray, int]] = []
    for seq in stance_sequences:
        indices = [int(s) for s in seq if 0 <= int(s) < N_STANCE_CLASSES]
        if len(indices) < 2:
            continue
        for cut in range(1, len(indices)):
            history = indices[:cut]
            target = indices[cut]
            pairs.append((_one_hot(history, n_classes=N_STANCE_CLASSES), target))
    return pairs


def _pad_right(
    arrays: Sequence[np.ndarray], *, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad variable-length one-hot histories into a dense ``(B, T, C)`` tensor.

    Returns ``(padded, lengths)``. The small-LSTM forward pass reads the
    final timestep, so the actual padded slots after the true history
    end never propagate into the prediction.
    """

    if not arrays:
        return (
            np.zeros((0, 1, n_classes), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    lengths = np.asarray([arr.shape[0] for arr in arrays], dtype=np.int64)
    max_len = int(lengths.max())
    padded = np.zeros((len(arrays), max_len, n_classes), dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, : arr.shape[0], :] = arr
    return padded, lengths


def train_small_lstm(  # noqa: PLR0913 — keyword-only knobs mirror the CLI surface.
    stance_sequences: Iterable[Sequence[int]],
    *,
    hidden_size: int = DEFAULT_LSTM_HIDDEN,
    num_layers: int = DEFAULT_LSTM_LAYERS,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    learning_rate: float = DEFAULT_LSTM_LR,
    seed: int = DEFAULT_LSTM_SEED,
    param_cap: int = DEFAULT_LSTM_PARAM_CAP,
) -> tuple[Any, int]:
    """Fit the small-LSTM baseline on a list of stance-only sequences.

    Returns ``(model, parameter_count)``. The model is returned in eval
    mode so the caller can run :func:`predict_small_lstm` directly. An
    empty pair list short-circuits training and returns the untrained
    module — the caller's evaluation path then falls back to the modal
    class on the holdout slice.
    """

    import torch  # type: ignore[import-not-found,unused-ignore]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = build_small_lstm(hidden_size=hidden_size, num_layers=num_layers)
    parameter_count = assert_param_count_within_cap(model, cap=param_cap)

    pairs = _build_lstm_training_pairs(stance_sequences)
    if not pairs:
        return model, parameter_count

    inputs, _lengths = _pad_right([p[0] for p in pairs], n_classes=N_STANCE_CLASSES)
    labels = np.asarray([p[1] for p in pairs], dtype=np.int64)

    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs_t)
        loss = loss_fn(logits, labels_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, parameter_count


def predict_small_lstm(model: Any, history: Sequence[Any]) -> int | None:
    """Run the trained small-LSTM on one history window.

    Returns the predicted stance index or ``None`` when ``history``
    carries no recognised labels.
    """

    import torch  # type: ignore[import-not-found,unused-ignore]

    indices = _stance_indices_from_history(history)
    if not indices:
        return None
    arr = _one_hot(indices, n_classes=N_STANCE_CLASSES)
    inputs_t = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(inputs_t)
        prediction = int(torch.argmax(logits, dim=-1).item())
    return prediction


# ---------------------------------------------------------------------------
# Metrics + comparison
# ---------------------------------------------------------------------------


def directional_accuracy(
    truths: Sequence[int], predictions: Sequence[int]
) -> float:
    if len(truths) == 0:
        return float("nan")
    matches = sum(1 for t, p in zip(truths, predictions) if int(t) == int(p))
    return matches / len(truths)


def confusion_matrix(
    truths: Sequence[int], predictions: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Build a 3x3 confusion matrix indexed by ``STANCE_CLASSES``.

    Rows are truth labels, columns are predictions — the standard
    sklearn convention. Returns a tuple-of-tuples so the result is
    hashable + immutable; callers serialise it as a list-of-lists in
    JSON payloads.
    """

    rows: list[list[int]] = [
        [0 for _ in range(N_STANCE_CLASSES)] for _ in range(N_STANCE_CLASSES)
    ]
    for t, p in zip(truths, predictions):
        if 0 <= int(t) < N_STANCE_CLASSES and 0 <= int(p) < N_STANCE_CLASSES:
            rows[int(t)][int(p)] += 1
    return tuple(tuple(row) for row in rows)


def evaluate_previous_stance(
    history_label_lists: Sequence[Sequence[Any]],
    truths: Sequence[int],
    *,
    fallback_modal: int | None = None,
) -> BaselineResult:
    """Run :func:`previous_stance` on a list of (history, truth) rows."""

    predictions: list[int] = []
    for history in history_label_lists:
        pred = previous_stance(history)
        if pred is None:
            pred = int(fallback_modal) if fallback_modal is not None else 0
        predictions.append(pred)
    return BaselineResult(
        name="previous_stance",
        predictions=tuple(predictions),
        truths=tuple(int(t) for t in truths),
        directional_accuracy=directional_accuracy(truths, predictions),
        confusion_matrix=confusion_matrix(truths, predictions),
        n=len(truths),
    )


def evaluate_rolling_majority(
    history_label_lists: Sequence[Sequence[Any]],
    truths: Sequence[int],
    *,
    n: int = DEFAULT_ROLLING_WINDOW,
    fallback_modal: int | None = None,
) -> BaselineResult:
    """Run :func:`rolling_majority` on a list of (history, truth) rows."""

    predictions: list[int] = []
    for history in history_label_lists:
        pred = rolling_majority(history, n=n)
        if pred is None:
            pred = int(fallback_modal) if fallback_modal is not None else 0
        predictions.append(pred)
    return BaselineResult(
        name=f"rolling_majority_n{n}",
        predictions=tuple(predictions),
        truths=tuple(int(t) for t in truths),
        directional_accuracy=directional_accuracy(truths, predictions),
        confusion_matrix=confusion_matrix(truths, predictions),
        n=len(truths),
    )


def evaluate_small_lstm(  # noqa: PLR0913 — kw-only baseline-evaluation config; collapsing into a dataclass would obscure the (train / history / target / hidden_size / lr / epochs / seed / n_classes) contract
    train_stance_sequences: Sequence[Sequence[int]],
    history_label_lists: Sequence[Sequence[Any]],
    truths: Sequence[int],
    *,
    hidden_size: int = DEFAULT_LSTM_HIDDEN,
    num_layers: int = DEFAULT_LSTM_LAYERS,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    learning_rate: float = DEFAULT_LSTM_LR,
    seed: int = DEFAULT_LSTM_SEED,
    fallback_modal: int | None = None,
) -> BaselineResult:
    """Fit + evaluate the small-LSTM baseline on the canonical fold protocol.

    ``train_stance_sequences`` carries the stance-only history the LSTM
    is fit on (one sequence per training sample — the trainer's
    pre-train_end pool). ``history_label_lists`` + ``truths`` is the
    holdout slice the baseline is evaluated on.
    """

    model, _params = train_small_lstm(
        train_stance_sequences,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    predictions: list[int] = []
    for history in history_label_lists:
        pred = predict_small_lstm(model, history)
        if pred is None:
            pred = int(fallback_modal) if fallback_modal is not None else 0
        predictions.append(pred)
    return BaselineResult(
        name=f"small_lstm_h{hidden_size}_l{num_layers}",
        predictions=tuple(predictions),
        truths=tuple(int(t) for t in truths),
        directional_accuracy=directional_accuracy(truths, predictions),
        confusion_matrix=confusion_matrix(truths, predictions),
        n=len(truths),
    )


def compare_against_transformer(
    transformer_dir_acc: float,
    baselines: Sequence[BaselineResult],
    *,
    threshold: float = LIFT_THRESHOLD_POINTS,
) -> dict[str, Any]:
    """Assemble the lift / no-lift verdict the API surface ships.

    The baseline the badge compares against is the strongest one (max
    directional accuracy) — the issue is explicit that the Transformer
    arm must clear the *best* naive baseline, not an average. Returns a
    payload sized for the ``TrajectoryResponse`` extension fields:

        {
            "baseline_used": str,        # name of the strongest baseline
            "baseline_dir_acc": float,   # its directional accuracy
            "delta_dir_acc": float,      # transformer - baseline
            "lift_vs_baseline": bool,    # delta >= threshold
            "threshold": float,
            "per_baseline": [BaselineResult-as-dict, ...]
        }

    All ``float`` values are emitted unconditionally so the JSON payload
    has stable keys regardless of which baseline lands on top.
    """

    finite_baselines = [
        b for b in baselines if not _is_nan(b.directional_accuracy)
    ]
    if not finite_baselines:
        return {
            "baseline_used": None,
            "baseline_dir_acc": None,
            "delta_dir_acc": None,
            "lift_vs_baseline": False,
            "threshold": float(threshold),
            "per_baseline": [_baseline_to_dict(b) for b in baselines],
        }
    best = max(finite_baselines, key=lambda b: b.directional_accuracy)
    delta = float(transformer_dir_acc) - float(best.directional_accuracy)
    return {
        "baseline_used": best.name,
        "baseline_dir_acc": float(best.directional_accuracy),
        "delta_dir_acc": float(delta),
        "lift_vs_baseline": bool(delta >= float(threshold)),
        "threshold": float(threshold),
        "per_baseline": [_baseline_to_dict(b) for b in baselines],
    }


def _is_nan(value: float) -> bool:
    try:
        return value != value  # noqa: PLR0124 — standard nan check
    except TypeError:
        return False


def _baseline_to_dict(result: BaselineResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "directional_accuracy": float(result.directional_accuracy)
        if not _is_nan(result.directional_accuracy)
        else None,
        "confusion_matrix": [list(row) for row in result.confusion_matrix],
        "n": int(result.n),
    }


__all__ = [
    "BaselineResult",
    "DEFAULT_LSTM_HIDDEN",
    "DEFAULT_LSTM_LAYERS",
    "DEFAULT_LSTM_PARAM_CAP",
    "DEFAULT_ROLLING_WINDOW",
    "LIFT_THRESHOLD_POINTS",
    "assert_param_count_within_cap",
    "build_small_lstm",
    "compare_against_transformer",
    "confusion_matrix",
    "directional_accuracy",
    "evaluate_previous_stance",
    "evaluate_rolling_majority",
    "evaluate_small_lstm",
    "predict_small_lstm",
    "previous_stance",
    "rolling_majority",
    "train_small_lstm",
]
