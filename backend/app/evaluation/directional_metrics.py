"""Directional classification metrics derived from continuous forecaster output.

The Phase 9 reframe (#184) shifts the headline metric from `close_rmse`
(a noisy continuous target dominated by random-walk variance) to the
sign of the next-day move. This module turns the existing regression
head's output into a directional classification view, so we can read a
``direction_accuracy`` number off any sweep without retraining a
classification head.

Decision: predicted direction is ``sign(predicted_close - prev_close)``.
Ground-truth direction is whatever the caller passes in via
``true_direction`` -- typically ``direction_t1d`` from
``events.parquet`` (the t+1 trading-day signed move). For continuous
inputs there is no tie-breaking ambiguity because the float space is
dense enough that ``pred == prev`` is essentially measure-zero; the
helper still defends with an explicit ``epsilon`` floor for
numerical-noise rows.

The helper is dependency-light by design: it takes plain Python /
numpy / torch arrays and returns plain floats, so it slots into both
the training-time eval loop and a post-hoc CLI without dragging in
the rest of the training-loop module.
"""

from __future__ import annotations

from typing import Any


def compute_directional_metrics(
    pred_close: Any,
    true_close: Any,
    prev_close: Any,
    *,
    epsilon: float = 0.0,
) -> dict[str, float | None]:
    """Return ``direction_accuracy`` + ``f1_macro`` + ``direction_auc``.

    Parameters
    ----------
    pred_close, true_close, prev_close :
        1-D iterables (numpy array, torch tensor, or list) of length N.
        ``prev_close`` is the closing price the model was conditioned on
        (the last bar of the input sequence on the unscaled price
        axis). ``pred_close`` and ``true_close`` are the corresponding
        predicted vs realised closes for the same N events.
    epsilon :
        Magnitude threshold below which a predicted change is treated
        as ``0`` (flat). Defaults to ``0.0`` so any non-zero predicted
        delta picks a side. Set to a small positive number (e.g.
        ``1e-6``) if numerical noise rows should map to "flat" rather
        than picking an arbitrary sign.

    Returns
    -------
    dict with three floats:

    - ``direction_accuracy`` -- fraction of events where
      ``sign(pred_close - prev_close) == sign(true_close - prev_close)``.
      Baseline = the majority-class fraction in the realised set (the
      project's events.parquet sits at ~53.7% +1 so the bar is 0.537).
    - ``f1_macro`` -- unweighted-mean F1 across the three classes
      {-1, 0, +1}. Falls back to the two-class f1 when no events in
      the realised set land in class 0.
    - ``direction_auc`` -- binary ROC-AUC for "up vs not-up", scored
      by the continuous predicted delta. Returns ``None`` when only
      one class is present in ``true_close`` (AUC undefined).

    Edge cases
    ----------
    - Empty arrays -> all three return ``None``.
    - Single-class ground truth -> ``direction_auc`` is ``None`` while
      accuracy / f1 still compute on the degenerate distribution.
    - Different array lengths -> ``ValueError``.
    """

    import numpy as np

    pred_close_arr = _to_numpy_1d(pred_close)
    true_close_arr = _to_numpy_1d(true_close)
    prev_close_arr = _to_numpy_1d(prev_close)

    if not (pred_close_arr.shape == true_close_arr.shape == prev_close_arr.shape):
        raise ValueError(
            "pred_close / true_close / prev_close must share shape; got "
            f"{pred_close_arr.shape}, {true_close_arr.shape}, {prev_close_arr.shape}"
        )

    if pred_close_arr.size == 0:
        return {
            "direction_accuracy": None,
            "f1_macro": None,
            "direction_auc": None,
        }

    pred_delta = pred_close_arr - prev_close_arr
    true_delta = true_close_arr - prev_close_arr
    pred_dir = _signed(pred_delta, epsilon=epsilon)
    true_dir = _signed(true_delta, epsilon=epsilon)

    accuracy = float((pred_dir == true_dir).mean())
    f1 = _macro_f1_three_class(true_dir, pred_dir)
    auc = _binary_auc_up(true_dir, pred_delta)

    return {
        "direction_accuracy": accuracy,
        "f1_macro": f1,
        "direction_auc": auc,
    }


def _to_numpy_1d(values: Any) -> "Any":
    """Coerce list / numpy array / torch tensor into a 1-D numpy float array."""

    import numpy as np

    try:
        import torch

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
    except ImportError:
        pass

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr


def _signed(delta: "Any", *, epsilon: float) -> "Any":
    """Map ``delta`` to {-1, 0, +1} with a tolerance band of size ``epsilon``."""

    import numpy as np

    out = np.zeros_like(delta, dtype=np.int8)
    out[delta > epsilon] = 1
    out[delta < -epsilon] = -1
    return out


def _macro_f1_three_class(y_true: "Any", y_pred: "Any") -> float:
    """Unweighted-mean F1 over the three classes {-1, 0, +1}.

    Sklearn would do this in one call but pulling it into the training
    loop on every eval pass would be a noticeable cost; an inline
    computation is fast and avoids the import in places it does not
    need to be. Classes absent from both ``y_true`` and ``y_pred`` are
    skipped from the macro average so f1_macro on a two-class problem
    is the actual two-class macro F1, not 2/3 of it.
    """

    classes = (-1, 0, 1)
    f1s: list[float] = []
    for cls in classes:
        true_pos = int(((y_true == cls) & (y_pred == cls)).sum())
        false_pos = int(((y_true != cls) & (y_pred == cls)).sum())
        false_neg = int(((y_true == cls) & (y_pred != cls)).sum())
        support_true = true_pos + false_neg
        support_pred = true_pos + false_pos
        if support_true == 0 and support_pred == 0:
            # Class never appears in either the truth or the
            # predictions. Excluding it from the macro average keeps
            # the metric meaningful on the dataset's actual class
            # distribution (the realised ``direction_t1d`` has only
            # 4 zero-rows out of ~4k so the flat class is essentially
            # absent in practice).
            continue
        precision = true_pos / support_pred if support_pred > 0 else 0.0
        recall = true_pos / support_true if support_true > 0 else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        f1s.append(f1)
    if not f1s:
        return 0.0
    return float(sum(f1s) / len(f1s))


def _binary_auc_up(y_true: "Any", scores: "Any") -> float | None:
    """ROC-AUC for the binary view ``up vs not-up``.

    Maps ``y_true`` to ``{1 if up else 0}``. Falls back to a manual
    Mann-Whitney-U computation so the helper stays dependency-light;
    sklearn is widely available in the project image but the import
    cost on every eval pass is non-trivial. Returns ``None`` when
    only one class is present (AUC undefined).
    """

    import numpy as np

    truth_up = (y_true == 1).astype(np.int64)
    n_pos = int(truth_up.sum())
    n_neg = int(truth_up.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank-based AUC = (sum_of_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    # Use average-rank tie handling so ties between pos and neg
    # contribute the expected 0.5 each. ``scipy.stats.rankdata`` would
    # do this cleanly but numpy alone covers it via argsort + index
    # average over ties.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # Tie correction
    unique_scores, inverse_indices, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inverse_indices == idx
            ranks[mask] = ranks[mask].mean()

    sum_ranks_pos = float(ranks[truth_up == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


__all__ = ["compute_directional_metrics"]
