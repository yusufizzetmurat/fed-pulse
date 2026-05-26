"""Recall@k evaluator for the retrieval encoder (#329).

The retrieval rebuild needs a quality gate the runtime path can read
back: only promote a fine-tuned encoder to ``role: retrieval`` if its
recall@k on the hand-labelled probe set beats the FinBERT-FedAdjacent
cosine baseline. This module is the measurement primitive; the
promotion decision lives in the registry pin.

The helper is deliberately a pure function over numpy arrays:

* no torch dep beyond plain cosine similarity (the index path already
  normalises embeddings; this module re-normalises defensively so it
  also accepts raw encoder output)
* no HF / sentence-transformers import — callers feed pre-computed
  embeddings, which lets the test path stub the encoder out entirely
* no I/O — the eval-set fixture loader lives next to the test, not
  here, so the helper survives changes to the fixture format

Contract:

* ``index_embeddings`` is the ``(N_index, d)`` embedding matrix of the
  retrieval corpus (one row per indexed statement).
* ``query_embeddings`` is the ``(N_queries, d)`` embedding matrix of
  the anchor probes.
* ``ground_truth_matches`` is a length-``N_queries`` sequence; each
  entry is the integer row index into ``index_embeddings`` of the
  single ground-truth match for that query. A negative value (e.g.
  ``-1``) marks a probe whose ground-truth row is not in the index;
  those probes are skipped so a partial fixture does not deflate the
  reported recall.
* ``k_values`` is the cut-off ladder; defaults to ``(1, 3, 5)``.

Returns ``dict[int, float]`` mapping ``k -> recall@k`` averaged over
the scored probes. Empty input yields the all-zeros dict so the
caller can log a clean number rather than handle ``NaN``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)


def _l2_normalise(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation so dot products become cosine similarities.

    Mirrors the convention in :mod:`app.retrieval.index` so a caller
    can feed either raw or pre-normalised embeddings without surprise.
    """

    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    arr = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def compute_recall_at_k(  # noqa: C901 — multi-input validation + multi-k pass; collapsing would hide the recall@k semantics
    index_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    ground_truth_matches: Sequence[int],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[int, float]:
    """Compute recall@k for a batch of (query, ground-truth) probes.

    Recall@k for a single probe is 1.0 if the ground-truth row sits
    inside the top-k cosine-similarity matches against the query;
    otherwise 0.0. The reported number is the mean over scored probes
    (probes with a negative ground-truth index are skipped).

    Empty inputs return ``{k: 0.0 for k in k_values}`` rather than
    raising — callers log the all-zeros result to make the empty case
    visible, then move on.
    """

    if not k_values:
        return {}
    k_clean = tuple(int(k) for k in k_values if int(k) > 0)
    if not k_clean:
        return {int(k): 0.0 for k in k_values}

    if (
        index_embeddings is None
        or query_embeddings is None
        or index_embeddings.size == 0
        or query_embeddings.size == 0
        or len(ground_truth_matches) == 0
    ):
        return {k: 0.0 for k in k_clean}

    if len(ground_truth_matches) != query_embeddings.shape[0]:
        raise ValueError(
            "ground_truth_matches length "
            f"{len(ground_truth_matches)} does not match query count "
            f"{query_embeddings.shape[0]}"
        )

    if index_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            "index_embeddings dim "
            f"{index_embeddings.shape[1]} does not match query "
            f"dim {query_embeddings.shape[1]}"
        )

    index_norm = _l2_normalise(index_embeddings)
    query_norm = _l2_normalise(query_embeddings)
    # (N_queries, N_index) cosine-similarity matrix. The probe set is
    # small (~30) and the corpus is small (~250), so a dense matmul
    # beats argpartition on per-row clarity.
    sims = query_norm @ index_norm.T

    n_index = index_norm.shape[0]
    max_k = min(max(k_clean), n_index)

    # Sort each row in descending similarity once and slice for every
    # ``k`` from the same ranking — keeps the helper a single pass
    # over the similarity matrix.
    ranked = np.argsort(-sims, axis=1)[:, :max_k]

    hits: dict[int, int] = {k: 0 for k in k_clean}
    scored = 0
    for row_idx, gt in enumerate(ground_truth_matches):
        gt_int = int(gt)
        if gt_int < 0 or gt_int >= n_index:
            continue
        scored += 1
        for k in k_clean:
            top_k = ranked[row_idx, : min(k, n_index)]
            if gt_int in top_k:
                hits[k] += 1

    if scored == 0:
        return {k: 0.0 for k in k_clean}
    return {k: hits[k] / scored for k in k_clean}


__all__ = ["DEFAULT_K_VALUES", "compute_recall_at_k"]
