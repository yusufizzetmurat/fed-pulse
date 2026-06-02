from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CredibilityVector:
    """Four-axis credibility check on the issuing institution at one FOMC date.

    Drift, realized-vs-stated gap, and market-implied gap are all in `[-1, 1]`
    so the downstream MLP doesn't need per-axis normalisation. `months_since_reversal`
    is a raw integer; consumers should bucket it when feeding the model.
    """

    drift_score: float
    realized_vs_stated_gap: float
    market_implied_gap: float
    months_since_reversal: int

    def as_list(self) -> list[float]:
        return [
            float(self.drift_score),
            float(self.realized_vs_stated_gap),
            float(self.market_implied_gap),
            float(self.months_since_reversal),
        ]


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    similarity = dot / (na * nb)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def drift_vs_prior(
    current_embedding: Sequence[float],
    prior_embeddings: Sequence[Sequence[float]],
) -> float:
    """Cosine distance between the current statement embedding and the mean
    of the prior ``len(prior_embeddings)`` statements (typically 4).

    Returns 0.0 when no prior context exists — caller should treat that as
    "drift unknown" rather than "no drift".
    """

    if not prior_embeddings:
        return 0.0
    dim = len(current_embedding)
    if dim == 0:
        return 0.0
    mean: list[float] = [0.0] * dim
    counted = 0
    for vec in prior_embeddings:
        if len(vec) != dim:
            continue
        for i, value in enumerate(vec):
            mean[i] += float(value)
        counted += 1
    if counted == 0:
        return 0.0
    mean = [value / counted for value in mean]
    return cosine_distance(list(current_embedding), mean)


def months_since_last_reversal(stance_history: Sequence[float], window: int = 12) -> int:
    """Count whole months since the sign of `stance_history` last flipped.

    Positive values mean hawkish, negative dovish, zero neutral. Resets when
    the sign of two consecutive entries differs. `window` caps the lookback so
    the feature stays bounded; the function returns `window` if no reversal is
    found in that span (i.e. the stance has been stable for at least `window`
    months).
    """

    if len(stance_history) < 2:
        return 0
    series = list(stance_history)[-window:]
    for offset, (prev, curr) in enumerate(
        zip(reversed(series), list(reversed(series))[1:]), start=1
    ):
        if (prev > 0 > curr) or (prev < 0 < curr):
            return offset
    return min(len(series), window)


def realized_vs_stated_gap(
    stated_path: Sequence[float],
    realized_path: Sequence[float],
    *,
    window: int = 90,
) -> float:
    """Pearson correlation between stated tone (a numeric stance score) and
    realized effective fed funds change over the trailing ``window`` days.

    Returns a value in ``[-1, 1]``; 0.0 when either series is too short.
    """

    if len(stated_path) < 2 or len(realized_path) < 2:
        return 0.0
    n = min(len(stated_path), len(realized_path), window)
    stated = list(stated_path)[-n:]
    realized = list(realized_path)[-n:]
    mean_s = sum(stated) / n
    mean_r = sum(realized) / n
    num = sum((s - mean_s) * (r - mean_r) for s, r in zip(stated, realized))
    var_s = sum((s - mean_s) ** 2 for s in stated)
    var_r = sum((r - mean_r) ** 2 for r in realized)
    denom = math.sqrt(var_s * var_r)
    if denom <= 1e-12:
        return 0.0
    correlation = num / denom
    return max(-1.0, min(1.0, correlation))


def market_implied_gap(sep_terminal: float | None, ois_terminal: float | None) -> float:
    """SEP-implied terminal rate minus market-implied terminal rate, clipped
    to ``[-1, 1]`` (scaled by 1/4 since the gap rarely exceeds 4pp in
    observed history).

    The Fed publishes a long-run median fed-funds projection in its quarterly
    Summary of Economic Projections; the market-implied long-run is
    approximated by the 5-year Treasury yield (DGS5) where a clean OIS
    forward is unavailable on FRED. Both lookups live in
    :mod:`app.services.credibility_loader` and are strict ``< as_of`` so a
    same-day FOMC release cannot leak into its own credibility feature.

    Returns 0.0 if either value is missing so downstream consumers don't have
    to special-case None.
    """

    if sep_terminal is None or ois_terminal is None:
        return 0.0
    raw = float(sep_terminal) - float(ois_terminal)
    scaled = raw / 4.0
    return max(-1.0, min(1.0, scaled))


def compute_credibility_vector(
    *,
    current_embedding: Sequence[float] = (),
    prior_embeddings: Sequence[Sequence[float]] = (),
    stance_history: Sequence[float] = (),
    stated_path: Sequence[float] = (),
    realized_path: Sequence[float] = (),
    sep_terminal: float | None = None,
    ois_terminal: float | None = None,
) -> CredibilityVector:
    """Aggregate the four axes into a single vector ready for the credibility
    MLP. Each axis degrades to 0.0 when its inputs are unavailable rather than
    raising — the model's loss surface stays well-defined."""

    return CredibilityVector(
        drift_score=drift_vs_prior(current_embedding, prior_embeddings),
        realized_vs_stated_gap=realized_vs_stated_gap(stated_path, realized_path),
        market_implied_gap=market_implied_gap(sep_terminal, ois_terminal),
        months_since_reversal=months_since_last_reversal(stance_history),
    )
