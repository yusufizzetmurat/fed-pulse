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
from typing import Literal

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


# Minimum whitespace-split token count for the diff to run. Anything
# shorter is treated as ``no_input`` — the redline and topic-emphasis
# views require enough surface area to be meaningful, and the FOMC
# statement boilerplate runs hundreds of tokens, so 5 is a generous
# floor that only rejects truly degenerate inputs.
MIN_INPUT_TOKENS: int = 5


# Latin-1 ratio gate for the non-English short-circuit. Statements
# are English-only on the FOMC surface, so a body whose characters
# are majority outside the basic Latin-1 range almost certainly
# isn't a statement and can't be diffed against the English-only
# topic phrase list. 0.5 is a deliberately loose threshold so
# pasted text with light unicode punctuation (curly quotes, em-dashes)
# still parses; the gate only trips on majority non-Latin scripts.
LATIN_RATIO_THRESHOLD: float = 0.5


def _is_majority_non_latin(text: str) -> bool:
    """Return True when ``text`` is majority outside basic Latin-1.

    A lightweight non-English detector that avoids the langdetect
    dependency. Counts characters with ``ord(ch) < 256`` (basic
    Latin-1) against the total character count after stripping
    whitespace. When the Latin-1 share falls below
    :data:`LATIN_RATIO_THRESHOLD` the caller short-circuits with
    ``status="non_english"``.
    """

    stripped = "".join(text.split())
    if not stripped:
        return False
    latin = sum(1 for ch in stripped if ord(ch) < 256)
    return (latin / len(stripped)) < LATIN_RATIO_THRESHOLD


DegradedStatus = Literal["no_input", "non_english", "no_prior"]


def _classify_input(text: str) -> Literal["no_input", "non_english"] | None:
    """Bucket ``text`` for the silent-null edge cases.

    Returns one of ``"no_input"`` / ``"non_english"`` when the diff
    should short-circuit, or ``None`` when the body is healthy enough
    to run through the full pipeline. Order matters: non-Latin is
    checked before the token-count gate so a long block of CJK still
    reports as ``non_english`` rather than ``no_input``.
    """

    if not text or not text.strip():
        return "no_input"
    if _is_majority_non_latin(text):
        return "non_english"
    tokens = text.split()
    if len(tokens) < MIN_INPUT_TOKENS:
        return "no_input"
    return None


def _degraded_response(
    current_date: str,
    status: DegradedStatus,
    summary: str,
) -> SemanticDiffResponse:
    """Build the empty-payload response used by every edge-case path.

    Keeps the wire shape stable: empty ``token_spans`` + empty
    ``topic_deltas`` so any client that ignores ``status`` still
    falls through to the existing cold-start renderer.
    """

    return SemanticDiffResponse(
        current_date=current_date,
        prior_date="",
        token_spans=[],
        topic_deltas=[],
        summary=summary,
        status=status,
    )


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
    tail = " ".join(tokens[-(UNCHANGED_RUN_KEEP_TOKENS // 2) :])
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
    callers translate that into the explanatory banner) or when the
    current body trips the silent-null edge-case guard (empty,
    whitespace-only, < ``MIN_INPUT_TOKENS`` tokens, or majority
    non-Latin). The function never raises on these inputs so the
    orchestrator always receives a parseable shape.
    """

    if not prior_text or not current_text:
        return []
    if _classify_input(current_text) is not None:
        return []
    prior_tokens = _whitespace_normalise(prior_text)
    current_tokens = _whitespace_normalise(current_text)
    if not prior_tokens or not current_tokens:
        return []
    matcher = difflib.SequenceMatcher(a=prior_tokens, b=current_tokens, autojunk=False)
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
    hits: dict[str, int] = dict.fromkeys(_TOPIC_ORDER, 0)
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
        return dict.fromkeys(_TOPIC_ORDER, 0.0)
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

    Returns an empty list when both sides are blank or when the
    current body trips the silent-null edge-case guard (the topic
    scorer is English-keyword based and would produce uninformative
    all-zero rows on non-Latin or near-empty input).
    """

    if not current_text:
        return []
    if _classify_input(current_text) is not None:
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

    Silent-null contract: every edge case returns a parseable
    response with a ``status`` field rather than raising. The
    orchestrator can always parse the wire shape; clients that
    ignore ``status`` see the existing empty-list cold-start view.

    Status values:

    - ``no_input`` — current body is empty, whitespace-only, or
      under :data:`MIN_INPUT_TOKENS` whitespace-split tokens.
    - ``non_english`` — current body is majority outside basic
      Latin-1 (see :func:`_is_majority_non_latin`).
    - ``no_prior`` — no strict-prior FOMC statement on disk for
      ``current_date`` (the original cold-start case).
    - ``ok`` — full diff produced; spans + topic deltas populated.
    """

    text_status = _classify_input(current_text)
    if text_status == "no_input":
        token_count = len((current_text or "").split())
        summary = (
            f"Input too short to diff (n={token_count} "
            f"{'token' if token_count == 1 else 'tokens'})."
        )
        return _degraded_response(current_date, "no_input", summary)
    if text_status == "non_english":
        summary = "Non-Latin text — diff not run."
        return _degraded_response(current_date, "non_english", summary)

    prior = load_prior_statement(current_date, path=path)
    if prior is None:
        return SemanticDiffResponse(
            current_date=current_date,
            prior_date="",
            token_spans=[],
            topic_deltas=[],
            summary="Earliest statement in dataset; no prior to compare.",
            status="no_prior",
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
        summary = f"No material topic-emphasis shift vs the {prior.event_date} " "statement."
    return SemanticDiffResponse(
        current_date=current_date,
        prior_date=prior.event_date,
        token_spans=spans,
        topic_deltas=topics,
        summary=summary,
        status="ok",
    )


__all__ = [
    "DEFAULT_STATEMENTS_PATH",
    "LATIN_RATIO_THRESHOLD",
    "MIN_INPUT_TOKENS",
    "PriorStatement",
    "TOPIC_PHRASES",
    "UNCHANGED_RUN_KEEP_TOKENS",
    "build_response",
    "compute_token_spans",
    "compute_topic_deltas",
    "load_prior_statement",
]
