"""Semantic-diff service for the Workspace descriptive panel.

Composes two views of the change between a pasted FOMC statement and
the most recent strictly-prior statement on disk:

- ``token_spans`` — token-level redline built on top of
  :func:`app.data.statement_delta.compute_delta_spans` (which already
  runs the ``difflib.SequenceMatcher`` opcode walk). Surfaced as an
  ordered list of ``SemanticDiffSpan`` rows so the frontend can render
  the redline in document order: unchanged stretches keep the
  ``unchanged`` kind so the panel can collapse long equal runs with an
  ellipsis, and ``replace`` opcodes land as a single ``substituted``
  span with the prior text on ``paired_text`` so the substitution can
  render as a paired chip rather than two unrelated add/remove rows.

- ``topic_deltas`` — emphasis shift across six canonical topics
  (inflation, labor, growth, financial conditions, policy stance,
  balance sheet). Each topic ships a small hand-curated phrase list;
  the emphasis score for one document is the share of total topic
  hits that landed on that topic, so the six numbers sum to ~1.0
  before NaN-handling. The delta is ``current - prior`` and the list
  is returned sorted by ``abs(delta)`` descending so the panel can
  rank the biggest emphasis shifts at the top.

Cold-start contract: when no prior statement is available
(:func:`load_prior_statement` returns ``None``) the response carries
empty span and topic lists with an explanatory summary; the panel
renders the cold-start banner only.

Descriptive only — this surface is never wired into any forecast head.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from app.config import DATA_DIR
from app.schemas import (
    SemanticDiffResponse,
    SemanticDiffSpan,
    SemanticDiffTopic,
)

# Reuse the tokenisation contract from ``app.data.statement_delta``:
# lowercase + whitespace split, no stemming, no stopword removal. The
# private import keeps a single source of truth so the topic scorer
# and the diff opcodes see identical token boundaries. If the upstream
# helper changes shape the type-checker will catch it here.
from app.data.statement_delta import _normalise as _whitespace_normalise  # noqa: PLC2701


# Canonical statements file the dashboard reads from. The path follows
# the same convention as :func:`app.main.list_documents` — host
# ``./data`` is mounted into the container at ``/data`` and the
# fallback is resolved by :data:`app.config.DATA_DIR` on bare-metal
# runs.
DEFAULT_STATEMENTS_PATH: Path = DATA_DIR / "fomc_statements.json"


# Hand-curated phrase list per canonical topic. The phrases are kept
# short (1–3 tokens) and lowercase; matching runs on the normalised
# token stream so the counts are stable across casing and whitespace
# variants. Six phrases per topic keeps the surface honest — this is
# explicitly NOT a learned topic model, it is a transparent keyword
# scorer the panel can explain in a tooltip.
TOPIC_PHRASES: dict[str, tuple[str, ...]] = {
    "Inflation": (
        "inflation",
        "price stability",
        "price pressures",
        "core pce",
        "disinflation",
        "2 percent",
    ),
    "Labor": (
        "labor market",
        "employment",
        "unemployment",
        "payrolls",
        "job gains",
        "labor demand",
    ),
    "Growth": (
        "economic activity",
        "gdp",
        "spending",
        "consumer spending",
        "business investment",
        "expansion",
    ),
    "Financial conditions": (
        "financial conditions",
        "credit conditions",
        "tighter credit",
        "lending standards",
        "market functioning",
        "financial markets",
    ),
    "Policy stance": (
        "federal funds rate",
        "target range",
        "monetary policy",
        "policy stance",
        "restrictive",
        "additional firming",
    ),
    "Balance sheet": (
        "securities holdings",
        "balance sheet",
        "treasury securities",
        "agency mortgage",
        "runoff",
        "reinvestment",
    ),
}


_TOPIC_ORDER: tuple[str, ...] = tuple(TOPIC_PHRASES.keys())

# Truncation threshold for unchanged runs in the rendered redline.
# Kept here so the backend can collapse very long equal stretches
# before the wire shape goes out — the frontend still applies its
# own ellipsis on top for runs that survive this guard. 60 tokens
# is roughly two sentences of statement text, which lines up with
# how the FOMC writes paragraphs.
UNCHANGED_RUN_KEEP_TOKENS: int = 60


@dataclass(frozen=True)
class PriorStatement:
    """In-memory tuple for a single statement row.

    Carries the ISO event date and the verbatim statement body so
    callers (the diff helpers, the response composer, the prior-lookup
    tests) can pass one value around rather than juggling a tuple.
    """

    event_date: str
    text: str


def _read_statements(path: Path) -> list[PriorStatement]:
    """Load and normalise the on-disk statements JSON.

    Returns an empty list when the file is missing so cold-start works
    on a fresh clone without raising.
    """

    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    rows: list[PriorStatement] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        event_date = str(item.get("date") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        if not event_date or not text:
            continue
        rows.append(PriorStatement(event_date=event_date, text=text))
    return rows


def load_prior_statement(
    current_date: str,
    *,
    path: Path | str | None = None,
) -> PriorStatement | None:
    """Return the most recent FOMC statement strictly before ``current_date``.

    ``current_date`` is parsed as ISO-8601 (the leading ten characters
    are enough). Rows dated on or after ``current_date`` are filtered
    out so a same-day prior cannot fold the current event into its own
    diff — this matches the strict-prior contract documented on
    :func:`app.data.statement_delta.select_prior_statement_text`.
    """

    target = _dt.date.fromisoformat(current_date[:10])
    statements_path = Path(path) if path is not None else DEFAULT_STATEMENTS_PATH
    candidates = _read_statements(statements_path)
    best: PriorStatement | None = None
    best_date: _dt.date | None = None
    for row in candidates:
        try:
            row_date = _dt.date.fromisoformat(row.event_date[:10])
        except ValueError:
            continue
        if row_date >= target:
            continue
        if best_date is None or row_date > best_date:
            best = row
            best_date = row_date
    return best


def _truncate_unchanged(tokens: list[str]) -> str:
    """Collapse very long equal runs to a head/tail with an ellipsis.

    The frontend has its own ellipsis logic for unchanged runs > 25
    words; this helper is the backend-side belt-and-braces for runs
    that would otherwise blow past 60 tokens. Short runs (<=
    ``UNCHANGED_RUN_KEEP_TOKENS``) are returned verbatim.
    """

    if len(tokens) <= UNCHANGED_RUN_KEEP_TOKENS:
        return " ".join(tokens)
    head = " ".join(tokens[: UNCHANGED_RUN_KEEP_TOKENS // 2])
    tail = " ".join(tokens[-(UNCHANGED_RUN_KEEP_TOKENS // 2):])
    return f"{head} … {tail}"


def compute_token_spans(
    prior_text: str,
    current_text: str,
) -> list[SemanticDiffSpan]:
    """Build the ordered redline span list for the panel.

    Wraps :class:`difflib.SequenceMatcher` directly so the unchanged
    runs ride alongside the added / removed / substituted spans in
    document order. The reuse contract with
    :func:`app.data.statement_delta.compute_delta_spans` is in the
    shared tokenisation: both helpers call ``_normalise`` so the
    diff opcodes line up across the two surfaces.

    Returns an empty list when either side is empty (cold-start —
    callers translate that into the explanatory banner).
    """

    if not prior_text or not current_text:
        return []
    prior_tokens = _whitespace_normalise(prior_text)
    current_tokens = _whitespace_normalise(current_text)
    if not prior_tokens or not current_tokens:
        return []
    matcher = difflib.SequenceMatcher(
        a=prior_tokens, b=current_tokens, autojunk=False
    )
    spans: list[SemanticDiffSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            spans.append(
                SemanticDiffSpan(
                    kind="unchanged",
                    text=_truncate_unchanged(current_tokens[j1:j2]),
                )
            )
        elif tag == "insert":
            spans.append(
                SemanticDiffSpan(
                    kind="added",
                    text=" ".join(current_tokens[j1:j2]),
                )
            )
        elif tag == "delete":
            spans.append(
                SemanticDiffSpan(
                    kind="removed",
                    text=" ".join(prior_tokens[i1:i2]),
                )
            )
        elif tag == "replace":
            spans.append(
                SemanticDiffSpan(
                    kind="substituted",
                    text=" ".join(current_tokens[j1:j2]),
                    paired_text=" ".join(prior_tokens[i1:i2]),
                )
            )
    return spans


def _topic_hits(text: str) -> dict[str, int]:
    """Count phrase occurrences per topic on a normalised string.

    Phrase matching runs on the lowercase, whitespace-collapsed text
    to keep the scorer simple and reproducible. Multi-token phrases
    are counted with ``str.count`` after the same normalisation, so
    "labor market" and "Labor market" both register.
    """

    normalised = " ".join(_whitespace_normalise(text))
    hits: dict[str, int] = {topic: 0 for topic in _TOPIC_ORDER}
    if not normalised:
        return hits
    for topic, phrases in TOPIC_PHRASES.items():
        for phrase in phrases:
            if not phrase:
                continue
            hits[topic] += normalised.count(phrase)
    return hits


def _emphasis_shares(hits: dict[str, int]) -> dict[str, float]:
    """Convert raw counts into shares of total topic mass.

    When a document has zero topic hits the shares all collapse to
    zero — the panel renders a "no topic mentions detected" stub in
    that case rather than dividing by zero.
    """

    total = float(sum(hits.values()))
    if total <= 0.0:
        return {topic: 0.0 for topic in _TOPIC_ORDER}
    return {topic: hits[topic] / total for topic in _TOPIC_ORDER}


def _sample_phrases(text: str, topic: str, *, limit: int = 3) -> list[str]:
    """Return up to ``limit`` phrases for ``topic`` that occur in ``text``.

    Used for the per-topic sparkline tooltip so the panel can show
    *which* phrases drove the emphasis number without re-running the
    matcher on the frontend.
    """

    normalised = " ".join(_whitespace_normalise(text))
    if not normalised:
        return []
    hits: list[str] = []
    for phrase in TOPIC_PHRASES.get(topic, ()):  # canonical order
        if phrase and phrase in normalised:
            hits.append(phrase)
        if len(hits) >= limit:
            break
    return hits


def compute_topic_deltas(
    prior_text: str,
    current_text: str,
) -> list[SemanticDiffTopic]:
    """Score the six canonical topics and return them ranked by |delta|.

    Each row carries the current/prior emphasis share, the signed
    delta, and a small sample-phrase list (the phrases that landed
    in the current document for that topic, in canonical order).

    Returns an empty list when both sides are blank.
    """

    if not current_text:
        return []
    current_hits = _topic_hits(current_text)
    current_shares = _emphasis_shares(current_hits)
    prior_hits = _topic_hits(prior_text or "")
    prior_shares = _emphasis_shares(prior_hits)
    rows: list[SemanticDiffTopic] = []
    for topic in _TOPIC_ORDER:
        current = current_shares[topic]
        prior = prior_shares[topic]
        rows.append(
            SemanticDiffTopic(
                topic=topic,
                prior_emphasis=prior,
                current_emphasis=current,
                delta=current - prior,
                sample_phrases=_sample_phrases(current_text, topic),
            )
        )
    rows.sort(key=lambda row: abs(row.delta), reverse=True)
    return rows


def build_response(
    current_date: str,
    current_text: str,
    *,
    path: Path | str | None = None,
) -> SemanticDiffResponse:
    """Compose the wire response for ``POST /fomc/semantic-diff``.

    Cold-start (no strict-prior on disk) returns an empty
    ``token_spans`` list, an empty ``topic_deltas`` list, and an
    explanatory summary. Callers and the frontend panel both rely on
    the empty-list shape to drive the cold-start banner.
    """

    prior = load_prior_statement(current_date, path=path)
    if prior is None:
        return SemanticDiffResponse(
            current_date=current_date,
            prior_date="",
            token_spans=[],
            topic_deltas=[],
            summary="Earliest statement in dataset; no prior to compare.",
        )
    spans = compute_token_spans(prior.text, current_text)
    topics = compute_topic_deltas(prior.text, current_text)
    top = topics[0] if topics else None
    if top is not None and abs(top.delta) > 0.0:
        direction = "more" if top.delta > 0 else "less"
        summary = (
            f"Largest emphasis shift: {direction} weight on "
            f"{top.topic.lower()} vs the {prior.event_date} statement."
        )
    else:
        summary = (
            f"No material topic-emphasis shift vs the {prior.event_date} "
            "statement."
        )
    return SemanticDiffResponse(
        current_date=current_date,
        prior_date=prior.event_date,
        token_spans=spans,
        topic_deltas=topics,
        summary=summary,
    )


__all__ = [
    "DEFAULT_STATEMENTS_PATH",
    "PriorStatement",
    "TOPIC_PHRASES",
    "UNCHANGED_RUN_KEEP_TOKENS",
    "build_response",
    "compute_token_spans",
    "compute_topic_deltas",
    "load_prior_statement",
]
