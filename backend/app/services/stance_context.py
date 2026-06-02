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
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import AnalysisRun


def _extract_stance_score(payload: Any) -> float | None:
    """``P(hawkish) - P(dovish)`` from a persisted /analyze response.

    Returns ``None`` when the payload is missing the multi-axis block,
    when the distribution is empty, or when both class probabilities
    are absent. A neutral-only distribution (hawkish + dovish = 0)
    correctly returns 0.0, not None — that IS the score in that case.
    """

    if not isinstance(payload, dict):
        return None
    multi_axis = payload.get("multi_axis")
    if not isinstance(multi_axis, dict):
        return None
    stance = multi_axis.get("stance")
    if not isinstance(stance, dict):
        return None
    distribution = stance.get("distribution")
    if not isinstance(distribution, dict):
        return None
    hawk = distribution.get("hawkish")
    dove = distribution.get("dovish")
    if not isinstance(hawk, int | float) and not isinstance(dove, int | float):
        return None
    h = float(hawk) if isinstance(hawk, int | float) else 0.0
    d = float(dove) if isinstance(dove, int | float) else 0.0
    return h - d


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


def build_stance_context(
    session: Session,
    *,
    symbol: str,
    horizon: str | None = None,
    n: int = 12,
    exclude_run_id: str | None = None,
) -> StanceContext:
    """Read the trailing ``n`` stance scores for ``symbol`` and summarize.

    Filters out runs whose payload lacks a usable stance distribution
    so a small backlog of regression-mode rows can't poison the mean.
    Returns mean=None / std=None when fewer than two usable rows are
    found — the caller renders ``insufficient history`` rather than
    surfacing a misleading z-score off a one-sample baseline.
    """

    stmt = select(AnalysisRun).where(AnalysisRun.symbol == symbol)
    if horizon:
        stmt = stmt.where(AnalysisRun.horizon == horizon)
    if exclude_run_id:
        stmt = stmt.where(AnalysisRun.id != exclude_run_id)
    stmt = stmt.order_by(AnalysisRun.created_at.desc()).limit(max(n * 2, n))

    rows = list(session.execute(stmt).scalars().all())
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
    # there. A degenerate constant series produces std=0.0; the
    # frontend treats that as "no meaningful spread" and falls back to
    # the raw value rendering.
    std = statistics.stdev(scores)
    return StanceContext(n=len(scored), mean=float(mean), std=float(std), history=scored)
