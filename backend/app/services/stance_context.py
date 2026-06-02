"""Rolling stance-score context for the multi-axis dashboard tile.

The multi-axis stance classifier emits a per-class distribution. The
underlying score the validity study anchored against the Fed's
policy moves is ``s = P(hawkish) - P(dovish)``. The instrument is
*valid* (Spearman +0.283, AUC hike-vs-cut 0.80) but *narrow* — the
absolute level is mis-centred (dovish bias) and the dovish end can't
tell hold from cut. Per Lead 2 of the validity write-up the dashboard
should surface stance as a rolling z-score against recent meetings,
not the raw absolute number, so the displayed claim matches the
instrument's validated scope.

This module reads the past ``n`` runs for a symbol off
``analysis_runs.payload``, extracts the stance distribution per run,
computes the trailing mean / std of ``s``, and returns the package the
StanceTile renders. The current run is intentionally excluded from the
trailing window when ``exclude_run_id`` is supplied so the z-score is
computed against history, not against itself.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import AnalysisRun, _extract_stance_score


@dataclass(frozen=True)
class StanceContextPoint:
    """One historical (date, score) pair."""

    document_date: str
    stance_score: float


@dataclass(frozen=True)
class StanceContext:
    """Trailing stance-score summary for the dashboard tile."""

    n: int
    mean: float | None
    std: float | None
    history: list[StanceContextPoint]


def build_stance_context(  # noqa: PLR0913 - five flags is the natural shape here
    session: Session,
    *,
    symbol: str,
    horizon: str | None = None,
    n: int = 12,
    exclude_run_id: str | None = None,
    leave_one_out: bool = True,
) -> StanceContext:
    """Read the trailing ``n`` stance scores for ``symbol`` and summarize.

    Filters out runs whose payload lacks a usable stance distribution
    so a small backlog of regression-mode rows can't poison the mean.
    Returns mean=None / std=None when fewer than two usable rows are
    found — the caller renders ``insufficient history`` rather than
    surfacing a misleading z-score off a one-sample baseline.

    ``leave_one_out`` defaults to True so the contract is safe-by-default
    even when the caller forgets to pass ``exclude_run_id``: the most-
    recent persisted row (which IS the run that triggered the fetch)
    is dropped from the trailing window so the z-score is computed
    against history, not against itself. Pass False only for diagnostic
    endpoints that genuinely want the full window.
    """

    stmt = select(AnalysisRun).where(AnalysisRun.symbol == symbol)
    if horizon:
        stmt = stmt.where(AnalysisRun.horizon == horizon)
    if exclude_run_id:
        stmt = stmt.where(AnalysisRun.id != exclude_run_id)
    # Pull one extra row over the target n so the implicit
    # leave-one-out below can drop the newest without shrinking the
    # window. ``n * 2`` keeps the budget for any rows we'll filter out
    # for missing distributions.
    stmt = stmt.order_by(AnalysisRun.created_at.desc()).limit(max(n * 2 + 1, n + 1))

    rows = list(session.execute(stmt).scalars().all())
    if leave_one_out and exclude_run_id is None and rows:
        # Drop the most-recent row implicitly — it IS the just-persisted
        # run that the caller is about to z-score, so including it
        # would deflate the magnitude.
        rows = rows[1:]
    scored: list[StanceContextPoint] = []
    for row in rows:
        if len(scored) >= n:
            break
        score = _extract_stance_score(row.payload)
        if score is None or not math.isfinite(score):
            continue
        scored.append(StanceContextPoint(document_date=str(row.document_date), stance_score=score))

    if len(scored) < 2:
        return StanceContext(n=len(scored), mean=None, std=None, history=scored)

    scores = [p.stance_score for p in scored]
    mean = statistics.fmean(scores)
    # Use the unbiased (sample) standard deviation. statistics.stdev
    # requires at least two points; the < 2 guard above ensures we get
    # there. A degenerate constant series (or a series that rounds to
    # zero spread under float arithmetic) cannot produce a meaningful
    # z-score, so std is collapsed to None — the frontend reads None
    # as "fall back to raw rendering" and a future consumer cannot
    # accidentally divide a finite value into the near-zero residue.
    std_value = float(statistics.stdev(scores))
    std: float | None = std_value if std_value > 0.0 else None
    return StanceContext(n=len(scored), mean=float(mean), std=std, history=scored)
