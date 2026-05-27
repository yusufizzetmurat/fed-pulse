"""Retrieval-augmented input features for the forecaster (#306).

The on-disk retrieval index (#294) already serves the historical-analog
panel at request time, exposing top-K past FOMC statements with their
stance and a coarse post-event vol-regime bucket. #306 wires the same
analog matches into the forecaster's training-input pipeline as a small
contextual feature block: for each event, retrieve the top-K analogs at
training time and append derived summary stats over the analog set onto
every bar of the supervised sequence.

The summary stats are *contextual* — they describe the retrieval
result, not the analog's post-event observed move. Including the
analog's `forward_realized_vol_10d` (or the `subsequent_vol_regime` it
buckets into) as an input feature would be a label leak via similarity:
two near-identical past statements share most of the surprise direction
and a non-trivial fraction of the post-event vol response, so a feature
that admitted the analog's post-event outcome would let the forecaster
read a lossy copy of its own target through the cosine-similarity gate.
The features here are limited to similarity moments and stance-agreement
counts so the strict-backward `analog_event_date < event_date` filter
on the retrieval call is the only temporal contract that needs holding.
See ADR 0028 and the per-feature row in
``docs/feature-provenance-audit.md``.

Designed to be plain-Python so it imports without pulling torch /
transformers at module-import time — the loader calls into the runtime
analogs singleton (``app.services.analogs``) only when
``use_retrieval_analogs=True`` is threaded through.
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass
from typing import Any

from app.models.config import RICH_RETRIEVAL_ANALOG_DIM

_logger = logging.getLogger(__name__)


# The similarity floor mirrors the #295 panel default: any analog hit
# scoring below this is considered too weak to count as a contextual
# signal. The loader floor is plumbed off this constant so a future
# tweak ripples through both the panel and the trainer in lockstep.
DEFAULT_SIMILARITY_FLOOR: float = 0.40

# The K choice matches the #295 panel's display top-K so the input
# features summarise the same set of analog matches a human reviewer
# sees on the UI. Documented in ADR 0028.
DEFAULT_TOP_K: int = 3

# Number of derived analog summary features the loader writes onto each
# event. Re-exported as ``RICH_RETRIEVAL_ANALOG_DIM`` from
# ``app.models.config`` so the schema constants stay one source of
# truth; this alias just gives the helper a local name.
ANALOG_FEATURE_DIM = RICH_RETRIEVAL_ANALOG_DIM


@dataclass(frozen=True)
class AnalogSummaryFeatures:
    """Per-event summary stats over a top-K analog result set.

    The five fields below are the entries written into the
    ``analog_features`` slot on every bar of a supervised sequence, in
    the documented order. ``analog_count_above_floor`` is normalised
    against ``DEFAULT_TOP_K`` so the model sees a value in ``[0, 1]``
    that does not blow up when a future configuration bumps K.
    """

    analog_max_similarity: float
    analog_mean_similarity: float
    analog_similarity_dispersion: float
    analog_count_above_floor: float
    analog_max_stance_score: float

    def as_list(self) -> list[float]:
        return [
            float(self.analog_max_similarity),
            float(self.analog_mean_similarity),
            float(self.analog_similarity_dispersion),
            float(self.analog_count_above_floor),
            float(self.analog_max_stance_score),
        ]


def _norm_stance(value: Any) -> str | None:
    """Canonicalise a stance label to one of ``hawkish`` / ``dovish`` / ``neutral``."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned in ("hawkish", "dovish", "neutral"):
        return cleaned
    return None


def compute_analog_summary_features(
    hits: list[Any],
    *,
    event_stance: str | None,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    top_k: int = DEFAULT_TOP_K,
) -> AnalogSummaryFeatures:
    """Derive the 5-dim analog summary block from a list of top-K hits.

    Each entry in ``hits`` must expose a ``similarity`` float and an
    ``axis_stance`` attribute (``str | None``); the production path
    feeds :class:`app.retrieval.index.AnalogHit` instances, the test
    path feeds plain dataclasses with the same fields. The
    ``event_stance`` is the canonical stance of the current event under
    audit; the stance-agreement score is computed as the fraction of
    analogs whose stance matches the current event's stance, scaled
    onto ``[0, 1]``. An unknown current-event stance collapses the
    stance-agreement to ``0.0``.

    Empty ``hits`` returns all-zero features (the loader's caller flips
    the missing flag to 1.0 in that case so the model can tell "no
    retrieval signal" from "retrieval signal of zero magnitude").
    """

    if not hits:
        return AnalogSummaryFeatures(
            analog_max_similarity=0.0,
            analog_mean_similarity=0.0,
            analog_similarity_dispersion=0.0,
            analog_count_above_floor=0.0,
            analog_max_stance_score=0.0,
        )

    sims = [float(getattr(h, "similarity", 0.0)) for h in hits]
    finite_sims = [s for s in sims if math.isfinite(s)]
    if not finite_sims:
        return AnalogSummaryFeatures(
            analog_max_similarity=0.0,
            analog_mean_similarity=0.0,
            analog_similarity_dispersion=0.0,
            analog_count_above_floor=0.0,
            analog_max_stance_score=0.0,
        )

    n = len(finite_sims)
    s_max = max(finite_sims)
    s_mean = sum(finite_sims) / n
    # Population std (not sample std): we want a deterministic dispersion
    # measure that does not blow up at n=1 (where sample std is
    # undefined). Numerically clamped to >= 0 to absorb any floating-point
    # residual that pushes the sum-of-squares fractionally negative.
    variance = sum((s - s_mean) ** 2 for s in finite_sims) / n
    s_disp = math.sqrt(max(variance, 0.0))

    floor = float(similarity_floor)
    above = sum(1 for s in finite_sims if s >= floor)
    k_norm = max(int(top_k), 1)
    count_above_norm = float(above) / float(k_norm)

    # Stance-agreement score: fraction of hits whose stance matches the
    # current event's stance. Unknown current-event stance collapses to
    # 0.0 ("no stance signal to agree on"). Unknown analog stances do
    # not contribute to the numerator. Both gates are leak-clean: stance
    # is itself a T-snapshot text-level signal in the audit.
    current_stance = _norm_stance(event_stance)
    if current_stance is None:
        stance_score = 0.0
    else:
        agreements = sum(
            1
            for h in hits
            if _norm_stance(getattr(h, "axis_stance", None)) == current_stance
        )
        stance_score = float(agreements) / float(n)

    return AnalogSummaryFeatures(
        analog_max_similarity=float(s_max),
        analog_mean_similarity=float(s_mean),
        analog_similarity_dispersion=float(s_disp),
        analog_count_above_floor=float(count_above_norm),
        analog_max_stance_score=float(stance_score),
    )


def lookup_analog_hits(
    *,
    text: str,
    event_date: datetime.date,
    top_k: int = DEFAULT_TOP_K,
) -> list[Any] | None:
    """Query the runtime analogs singleton with a strict-backward filter.

    Returns ``None`` when the bundle is absent on disk (graceful
    degrade: the loader then leaves every event in the all-zeros +
    missing-flag-1.0 state). Returns an empty list when the bundle is
    present but no row clears the ``as_of_date`` cutoff (first event in
    the corpus). Otherwise returns the top-K hits ranked by cosine
    similarity, each carrying ``event_date`` < the supplied
    ``event_date`` by construction of the index ``query`` cutoff.

    The lookup uses the existing ``app.services.analogs.find_analogs``
    entry point so the trainer reuses the same retrieval pipeline the
    panel serves at request time. The ``as_of_date`` argument enforces
    a strict-backward filter (``event_date_in_index < event_date``);
    self-match suppression by ``text_hash`` is included via the same
    helper.
    """

    from app.services.analogs import find_analogs

    cleaned = (text or "").strip()
    if not cleaned:
        return None
    result = find_analogs(cleaned, k=int(top_k), as_of_date=event_date)
    if result is None:
        return None
    hits = result.get("hits") or []
    return list(hits)


__all__ = [
    "ANALOG_FEATURE_DIM",
    "AnalogSummaryFeatures",
    "DEFAULT_SIMILARITY_FLOOR",
    "DEFAULT_TOP_K",
    "compute_analog_summary_features",
    "lookup_analog_hits",
]
