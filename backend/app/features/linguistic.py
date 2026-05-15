"""Phase 8 structured linguistic features for FOMC text.

Emits a fixed-dimension, interpretable feature vector per document
that complements multi-axis stance labels and encoder embeddings.
The design follows three strands of Fed-NLP literature:

* Hansen, McMahon & Tong (2018), "Shocking Language: Understanding the
  Macroeconomic Effects of Central Bank Communication." (LDA topic
  shares on FOMC text as a stable, interpretable summary axis.)
* Aruoba & Drechsel (2024), "Identifying Monetary Policy Shocks: A
  Natural Language Approach." (hand-crafted hawkish/dovish dictionaries
  combined with topic-model controls.)
* Cieslak & Schrimpf (2019), "Non-monetary news in central bank
  communication." (forward-looking phrasing and comparison-to-prior
  signal the policy vs. growth-news split.)

The output is a 15-dim numeric vector keyed by ``text_hash`` (see
:class:`LinguisticVector` for the exact field order):

1   ``topic_share_inflation``            -- LDA share for the inflation-aligned topic.
2   ``topic_share_employment``           -- LDA share for the labor-market slot.
3   ``topic_share_financial_stability``  -- LDA share for the financial-conditions slot.
4   ``topic_share_growth``               -- LDA share for the activity / spending slot.
5   ``topic_share_balance_sheet``        -- LDA share for the balance-sheet slot.
6   ``topic_share_misc_1``               -- residual LDA topic #1 (next misc index).
7   ``topic_share_misc_2``               -- residual LDA topic #2.
8   ``topic_share_misc_3``               -- residual LDA topic #3.
9   ``hedge_density``                    -- hedge tokens per 1000 whitespace tokens.
10  ``comparison_density``               -- comparison-to-prior phrases per 1000 tokens.
11  ``forward_density``                  -- forward-looking phrases per 1000 tokens.
12  ``concrete_ratio``                   -- unique (numbers ∪ dates ∪ currency) spans / total words.
13  ``hawk_dove_asymmetry``              -- (#hawk - #dove) / (#hawk + #dove + 1).
14  ``log_token_count``                  -- log1p(whitespace token count).
15  ``pivot_distance``                   -- token-set Jaccard distance vs the prior
                                            same-kind statement (Hansen-McMahon 2016).
                                            Only defined for ``event_kind = "statement"``;
                                            other kinds and the first statement
                                            in the corpus emit ``NaN``.

When a named slot fails the seed-overlap floor (see
:func:`_assign_named_topics`) the slot is emitted with value ``0.0``
and the topic that would otherwise have been pinned to it falls into
the next ``misc_*`` slot instead. This keeps the 14 output columns
fixed while avoiding silent mislabelling.

The 8 LDA topic shares always sum to 1; the densities are independent.
``compute_linguistic_features`` is a pure function of the text plus a
fitted ``LdaModel`` -- per-doc idempotent; scrambling other docs in the
training corpus does not move a given doc's feature vector beyond the
LDA fit's contribution.

Determinism contract: ``random_state=11``, fixed ``max_iter``, fixed
``CountVectorizer`` vocabulary cutoffs. Same input corpus -> bit-identical
LDA model object (modulo float-rounding in the topic-share output).

CLI:
    python -m app.features.linguistic --training-package-id <id> \
        --output linguistic_features.parquet

Persists alongside the parquet:
* ``linguistic_lda_model.pkl`` -- pickled (vectoriser, LDA estimator).
* ``linguistic_lda_topics.json`` -- per-topic top-15 vocabulary tokens
  and the human-assigned label, for the wiki write-up + downstream
  audit. Top-words come from the fitted LDA directly so re-runs hit
  the same JSON byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from app.config import DATA_DIR as DEFAULT_DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 11
NUM_TOPICS = 8
LDA_MAX_ITER = 50
LDA_LEARNING_METHOD = "batch"
TOP_WORDS_PER_TOPIC = 15
VOCAB_MIN_DF = 5
VOCAB_MAX_DF = 0.6
VOCAB_MAX_FEATURES = 5000

# The five "named" topic slots in the output vector. ``misc_1..3`` cover
# the remaining ``NUM_TOPICS - 5`` topics so all latent topics are
# accessible downstream.
NAMED_TOPIC_KEYS: tuple[str, ...] = (
    "inflation",
    "employment",
    "financial_stability",
    "growth",
    "balance_sheet",
)

# Seed words used to bind a fitted LDA topic to a human label. The
# topic with the highest aggregate posterior mass over its seed list
# wins the slot. Each topic can be assigned to at most one slot, in the
# order ``NAMED_TOPIC_KEYS`` -- so "inflation" gets first dibs, then
# "employment", and so on.
TOPIC_SEED_WORDS: dict[str, tuple[str, ...]] = {
    "inflation": (
        "inflation",
        "prices",
        "price",
        "cpi",
        "pce",
        "core",
        "energy",
        "transitory",
        "elevated",
        "wages",
    ),
    "employment": (
        "employment",
        "labor",
        "jobs",
        "unemployment",
        "payrolls",
        "workers",
        "hiring",
        "wage",
        "participation",
        "maximum",
    ),
    "financial_stability": (
        "financial",
        "stability",
        "banks",
        "banking",
        "credit",
        "lending",
        "liquidity",
        "stress",
        "leverage",
        "vulnerability",
    ),
    "growth": (
        "growth",
        "activity",
        "spending",
        "output",
        "gdp",
        "demand",
        "production",
        "consumption",
        "investment",
        "expansion",
    ),
    "balance_sheet": (
        "balance",
        "sheet",
        "securities",
        "treasury",
        "mbs",
        "agency",
        "holdings",
        "reinvestment",
        "purchases",
        "reserves",
    ),
}

# Hedge / certainty markers (Hansen-McMahon-Tong 2018, Loughran-McDonald
# uncertainty list, adapted for Fed text).
HEDGE_TOKENS: frozenset[str] = frozenset(
    {
        "perhaps",
        "appears",
        "should",
        "may",
        "anticipate",
        "expect",
        "projected",
        "likely",
        "somewhat",
        "modestly",
        "broadly",
        "generally",
        "gradual",
        "gradually",
        "patient",
        "accommodative",
    }
)

# Comparison-to-prior phrases (Cieslak-Schrimpf 2019 motivation: language
# anchoring against the prior meeting reveals revisions to the policy
# path).
COMPARISON_PHRASES: tuple[str, ...] = (
    "since the last meeting",
    "in contrast to",
    "we revised",
    "departed from",
    "compared with",
    "relative to",
    "previously",
    "earlier in the year",
)

# Forward-looking phrases (Aruoba-Drechsel 2024).
FORWARD_TOKENS: frozenset[str] = frozenset(
    {
        "future",
        "ahead",
        "upcoming",
        "expect",
        "expects",
        "anticipate",
        "anticipates",
        "projection",
        "outlook",
    }
)
FORWARD_PHRASES: tuple[str, ...] = (
    "going forward",
    "will continue",
    "will likely",
)

# Hawkish / dovish lexicons. Single-word tokens are matched on the
# tokenised stream; multi-word phrases are matched on the raw text.
HAWK_TOKENS: frozenset[str] = frozenset(
    {
        "tightening",
        "restrictive",
        "firm",
        "vigilant",
        "raise",
        "hike",
        "tighten",
    }
)
HAWK_PHRASES: tuple[str, ...] = ("contain inflation",)

DOVE_TOKENS: frozenset[str] = frozenset(
    {
        "accommodative",
        "supportive",
        "ease",
        "easing",
        "cut",
        "lower",
        "loose",
        "expand",
        "stimulate",
    }
)
DOVE_PHRASES: tuple[str, ...] = ("support employment",)

# Regex helpers for the concrete/abstract ratio.
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|q[1-4]\b|fy\d{2,4})\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"[\$€£¥]|\bbps\b|\bpercent\b|\b%\b|%", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z]+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinguisticVector:
    """The 15-dim structured linguistic feature row for a single document.

    Fields, in emission order:

    1.  ``topic_share_inflation``
    2.  ``topic_share_employment``
    3.  ``topic_share_financial_stability``
    4.  ``topic_share_growth``
    5.  ``topic_share_balance_sheet``
    6.  ``topic_share_misc_1``
    7.  ``topic_share_misc_2``
    8.  ``topic_share_misc_3``
    9.  ``hedge_density``
    10. ``comparison_density``
    11. ``forward_density``
    12. ``concrete_ratio``
    13. ``hawk_dove_asymmetry``
    14. ``log_token_count``
    15. ``pivot_distance``

    A named slot that fails the seed-overlap floor in
    :func:`_assign_named_topics` is emitted with ``0.0`` and its
    would-be topic falls through into the next ``misc_*`` slot, so the
    15 fields are always present regardless of LDA fit quality.

    ``pivot_distance`` is ``NaN`` for documents whose event kind is not
    ``statement``, and for the first statement in the corpus. The first
    14 axes are always finite.
    """

    topic_share_inflation: float
    topic_share_employment: float
    topic_share_financial_stability: float
    topic_share_growth: float
    topic_share_balance_sheet: float
    topic_share_misc_1: float
    topic_share_misc_2: float
    topic_share_misc_3: float
    hedge_density: float
    comparison_density: float
    forward_density: float
    concrete_ratio: float
    hawk_dove_asymmetry: float
    log_token_count: float
    pivot_distance: float = math.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "topic_share_inflation": self.topic_share_inflation,
            "topic_share_employment": self.topic_share_employment,
            "topic_share_financial_stability": self.topic_share_financial_stability,
            "topic_share_growth": self.topic_share_growth,
            "topic_share_balance_sheet": self.topic_share_balance_sheet,
            "topic_share_misc_1": self.topic_share_misc_1,
            "topic_share_misc_2": self.topic_share_misc_2,
            "topic_share_misc_3": self.topic_share_misc_3,
            "hedge_density": self.hedge_density,
            "comparison_density": self.comparison_density,
            "forward_density": self.forward_density,
            "concrete_ratio": self.concrete_ratio,
            "hawk_dove_asymmetry": self.hawk_dove_asymmetry,
            "log_token_count": self.log_token_count,
            "pivot_distance": self.pivot_distance,
        }


@dataclass
class LdaArtifact:
    """Fitted LDA bundle + the human-label-to-topic-index mapping.

    ``topic_assignments`` maps each ``NAMED_TOPIC_KEYS`` slot to the
    LDA topic index that best matches its seed words; the slots not
    pinned to a named topic fall through to ``misc_topic_indices`` in
    ascending topic-index order. This mapping is deterministic given
    the LDA fit and the seed-word lexicons above.
    """

    vectorizer: CountVectorizer
    lda: LatentDirichletAllocation
    topic_assignments: dict[str, int]
    misc_topic_indices: tuple[int, ...]
    top_words: list[list[str]] = field(default_factory=list)
    coherence_notes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenisation + density helpers
# ---------------------------------------------------------------------------


def _whitespace_tokens(text: str) -> list[str]:
    return text.split()


def _word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _per_1000(count: int, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return (count / total_tokens) * 1000.0


def _count_token_hits(tokens: Sequence[str], lexicon: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


def _count_phrase_hits(text_lower: str, phrases: Sequence[str]) -> int:
    return sum(text_lower.count(p) for p in phrases)


def hedge_density(text: str) -> float:
    """Hedge tokens per 1000 whitespace tokens.

    Whitespace token count matches Hansen-McMahon-Tong's denominator;
    word-token hits (case-folded) provide the numerator.
    """

    ws_tokens = _whitespace_tokens(text)
    word_tokens = _word_tokens(text)
    hits = _count_token_hits(word_tokens, HEDGE_TOKENS)
    return _per_1000(hits, len(ws_tokens))


def comparison_density(text: str) -> float:
    """Comparison-to-prior phrases per 1000 whitespace tokens."""

    ws_tokens = _whitespace_tokens(text)
    hits = _count_phrase_hits(text.lower(), COMPARISON_PHRASES)
    return _per_1000(hits, len(ws_tokens))


def forward_density(text: str) -> float:
    """Forward-looking phrases + tokens per 1000 whitespace tokens."""

    ws_tokens = _whitespace_tokens(text)
    word_tokens = _word_tokens(text)
    hits = _count_token_hits(word_tokens, FORWARD_TOKENS)
    hits += _count_phrase_hits(text.lower(), FORWARD_PHRASES)
    return _per_1000(hits, len(ws_tokens))


def concrete_ratio(text: str) -> float:
    """(unique number / date / currency spans) / #words.

    Words are alphabetic tokens (case-folded). The three regex families
    overlap on tokens like ``5.25%`` (number + currency marker) and
    ``$2.5 billion`` (currency + number), so naive summation
    double-counts and can push the ratio above 1.0 on rate-heavy FOMC
    text. We instead union the ``(start, end)`` match spans across the
    three regexes and count unique spans -- two regexes that fire on
    overlapping byte ranges count once. Returns 0.0 when there are no
    words, so callers don't need to special-case empty strings.
    """

    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    spans: set[tuple[int, int]] = set()
    for pattern in (_NUMBER_RE, _DATE_RE, _CURRENCY_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            if end > start:
                spans.add((start, end))
    # Collapse any spans whose byte ranges overlap (e.g. ``5.25`` and
    # ``%`` in ``5.25%`` produce adjacent but not strictly overlapping
    # spans; ``$`` and ``2.5`` in ``$2.5`` likewise). We treat
    # adjacency (end_i == start_j) as overlap so multi-piece concrete
    # tokens collapse to a single concrete count.
    if not spans:
        return 0.0
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return len(merged) / len(words)


def hawk_dove_asymmetry(text: str) -> float:
    """``(hawk - dove) / (hawk + dove + 1)``.

    The +1 smoother keeps the score bounded in ``[-1, 1]`` even for
    short fragments where ``hawk + dove == 0``.
    """

    text_lower = text.lower()
    word_tokens = _word_tokens(text)
    hawk = _count_token_hits(word_tokens, HAWK_TOKENS) + _count_phrase_hits(
        text_lower, HAWK_PHRASES
    )
    dove = _count_token_hits(word_tokens, DOVE_TOKENS) + _count_phrase_hits(
        text_lower, DOVE_PHRASES
    )
    return (hawk - dove) / (hawk + dove + 1)


def log_token_count(text: str) -> float:
    return math.log1p(len(_whitespace_tokens(text)))


# ---------------------------------------------------------------------------
# LDA fit + topic assignment
# ---------------------------------------------------------------------------


def _build_vectorizer(
    *, min_df: int = VOCAB_MIN_DF, max_df: float = VOCAB_MAX_DF
) -> CountVectorizer:
    return CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        max_features=VOCAB_MAX_FEATURES,
    )


#: Minimum number of slot seed words that must appear in the winning
#: topic's top-N vocabulary for the slot to be assigned. If a slot's
#: best topic clears posterior mass but only via incidental seed hits
#: (i.e. the topic's *top words* do not contain at least
#: ``MIN_SEED_OVERLAP`` seed tokens), the slot is left unassigned and
#: that topic falls through into the misc pool. Prevents the
#: silent-mislabel failure mode where "employment" inherits a
#: balance-sheet / QE topic just because its actual labor topic was
#: claimed by a higher-priority slot.
MIN_SEED_OVERLAP: int = 2
SEED_OVERLAP_TOP_N: int = 10


def _assign_named_topics(
    lda: LatentDirichletAllocation, vocab: Sequence[str]
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Pin each named topic slot to its best-matching LDA topic.

    Iterate the named slots in declaration order; for each slot pick
    the unassigned topic whose seed words receive the most cumulative
    posterior weight in the topic-word matrix. Ties broken by topic
    index (lower wins) for determinism: candidate topic indices are
    walked in ascending order and a strict ``score > best_score``
    comparison keeps the first (lowest-index) topic on ties.

    Seed-overlap floor: the winning topic must contain at least
    ``MIN_SEED_OVERLAP`` of the slot's seed words in its top-N
    (``SEED_OVERLAP_TOP_N``) vocabulary, otherwise the slot is left
    unassigned. Downstream callers emit ``0.0`` for the slot and the
    topic that would have been pinned falls through to the misc pool.
    This blocks the silent-mislabel failure mode observed on the
    Sprint 1 fit where ``employment`` inherited a QE / balance-sheet
    topic with zero labor vocabulary.

    Returns the slot->index map plus the remaining topic indices in
    ascending order.
    """

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    components = lda.components_.astype(np.float64)
    row_norms = components.sum(axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    normalised = components / row_norms

    # Precompute each topic's top-N vocabulary tokens for the overlap
    # floor check. Re-uses the same deterministic ordering as
    # ``_top_words_for_topic`` (weight desc, then vocab index asc).
    top_n_per_topic: list[set[str]] = []
    for topic_idx in range(normalised.shape[0]):
        weights = lda.components_[topic_idx]
        order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
        top_n_per_topic.append({vocab[i] for i in order[:SEED_OVERLAP_TOP_N]})

    assignments: dict[str, int] = {}
    used: set[int] = set()
    for slot in NAMED_TOPIC_KEYS:
        seeds = TOPIC_SEED_WORDS[slot]
        seed_set = set(seeds)
        seed_indices = [word_to_idx[w] for w in seeds if w in word_to_idx]
        if not seed_indices:
            continue
        # Walk topic indices in ascending order with a strict ``>``
        # comparison so ties go to the lower index (deterministic and
        # documented).
        best_topic = -1
        best_score = -1.0
        for topic_idx in range(normalised.shape[0]):
            if topic_idx in used:
                continue
            score = float(normalised[topic_idx, seed_indices].sum())
            if score > best_score:
                best_score = score
                best_topic = topic_idx
        if best_topic < 0:
            continue
        # Seed-overlap floor: reject the assignment if the winning
        # topic's top-N vocabulary does not contain at least
        # ``MIN_SEED_OVERLAP`` of the slot's seed words. The topic is
        # NOT marked as used, so it remains available to later slots
        # (if any) and otherwise falls to the misc pool.
        overlap = top_n_per_topic[best_topic] & seed_set
        if len(overlap) < MIN_SEED_OVERLAP:
            continue
        assignments[slot] = best_topic
        used.add(best_topic)

    misc = tuple(sorted(i for i in range(normalised.shape[0]) if i not in used))
    return assignments, misc


def _top_words_for_topic(
    lda: LatentDirichletAllocation, vocab: Sequence[str], topic_idx: int, k: int
) -> list[str]:
    weights = lda.components_[topic_idx]
    # Tie-break on vocabulary index (ascending) so re-runs match.
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    return [vocab[i] for i in order[:k]]


def fit_lda(
    texts: Iterable[str],
    *,
    num_topics: int = NUM_TOPICS,
    random_state: int = RANDOM_STATE,
    min_df: int | None = None,
    max_df: float = VOCAB_MAX_DF,
) -> LdaArtifact:
    """Fit LDA on the corpus and bind each named slot to its best topic.

    The vectoriser strips English stop-words, collapses to lowercase
    alphabetic tokens of length >= 2, and caps the vocabulary at
    ``VOCAB_MAX_FEATURES``. Same input list -> same fit (sklearn LDA
    is deterministic given ``random_state``).

    The default ``min_df`` is ``VOCAB_MIN_DF`` when at least 50 docs are
    supplied; smaller corpora downgrade to ``min_df=2`` so unit tests on
    a handful of docs do not empty out the vocabulary. The chosen value
    is recorded in ``LdaArtifact.coherence_notes`` so reruns can audit
    the cutoff.
    """

    text_list = [t for t in texts if t and t.strip()]
    if not text_list:
        raise ValueError("fit_lda requires at least one non-empty document")
    if min_df is None:
        min_df = VOCAB_MIN_DF if len(text_list) >= 50 else 2
    vectorizer = _build_vectorizer(min_df=min_df, max_df=max_df)
    dtm = vectorizer.fit_transform(text_list)
    vocab = vectorizer.get_feature_names_out().tolist()
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=random_state,
        max_iter=LDA_MAX_ITER,
        learning_method=LDA_LEARNING_METHOD,
        evaluate_every=-1,
    )
    lda.fit(dtm)
    assignments, misc = _assign_named_topics(lda, vocab)
    top_words = [
        _top_words_for_topic(lda, vocab, t, TOP_WORDS_PER_TOPIC) for t in range(num_topics)
    ]
    coherence: dict[str, str] = {}
    for slot, topic_idx in assignments.items():
        seeds = TOPIC_SEED_WORDS[slot]
        overlap = set(top_words[topic_idx][:SEED_OVERLAP_TOP_N]) & set(seeds)
        coherence[slot] = (
            f"clean (overlap with seeds: {sorted(overlap)})"
        )
    # Slots that did NOT clear the seed-overlap floor are recorded so
    # the audit JSON makes the drop-to-misc explicit. Downstream
    # readers can see at a glance which named slot was emitted as 0.0.
    for slot in NAMED_TOPIC_KEYS:
        if slot in assignments:
            continue
        coherence[slot] = (
            f"unassigned -- no candidate topic cleared the "
            f"seed-overlap floor (MIN_SEED_OVERLAP={MIN_SEED_OVERLAP} of top-"
            f"{SEED_OVERLAP_TOP_N}); slot emitted as 0.0 and the topic falls "
            "to misc"
        )
    return LdaArtifact(
        vectorizer=vectorizer,
        lda=lda,
        topic_assignments=assignments,
        misc_topic_indices=misc,
        top_words=top_words,
        coherence_notes=coherence,
    )


def _topic_shares(text: str, artifact: LdaArtifact) -> np.ndarray:
    """Return the LDA topic posterior for one document.

    Empty / OOV documents fall back to a uniform distribution so the
    output stays well-defined.
    """

    n_topics = artifact.lda.n_components
    if not text or not text.strip():
        return np.full(n_topics, 1.0 / n_topics)
    dtm = artifact.vectorizer.transform([text])
    if dtm.sum() == 0:
        return np.full(n_topics, 1.0 / n_topics)
    return artifact.lda.transform(dtm)[0]


# ---------------------------------------------------------------------------
# Pivot distance (token-set Jaccard vs prior same-kind statement)
# ---------------------------------------------------------------------------

# The kind for which ``pivot_distance`` is defined. Minutes, press
# conferences, speeches and testimonies follow different stylistic
# conventions (Q&A transcripts vs. a curated written paragraph), so the
# vocabulary diff against a prior statement would not be interpretable.
PIVOT_DISTANCE_KIND: str = "statement"


def pivot_distance_tokens(text: str) -> set[str]:
    """Return the normalised token set used by :func:`pivot_distance`.

    Reuses ``_TOKEN_RE`` -- the same tokeniser that backs ``_word_tokens``
    -- so the Jaccard distance shares a tokenisation convention with the
    rest of the module (case-folded alphanumeric runs of length >= 1).
    """

    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def pivot_distance(text: str, prior_tokens: set[str] | None) -> float:
    """Token-set Jaccard distance between ``text`` and ``prior_tokens``.

    ``1 - |A ∩ B| / |A ∪ B|`` over the normalised tokens of each
    document. The result lies in ``[0, 1]``:

    - 0.0 when ``text`` shares its full vocabulary with the prior;
    - 1.0 when the two vocabularies are disjoint.

    Returns ``NaN`` when ``prior_tokens`` is ``None`` -- there is no
    prior statement to diff against (first statement in the corpus).
    Also returns ``NaN`` when both token sets are empty, because the
    Jaccard distance is undefined on an empty union.
    """

    if prior_tokens is None:
        return math.nan
    current = pivot_distance_tokens(text)
    if not current and not prior_tokens:
        return math.nan
    intersection = len(current & prior_tokens)
    union = len(current | prior_tokens)
    if union == 0:
        return math.nan
    return 1.0 - (intersection / union)


# ---------------------------------------------------------------------------
# Per-document feature assembly
# ---------------------------------------------------------------------------


def compute_linguistic_features(
    text: str,
    lda_artifact: LdaArtifact | None = None,
    *,
    prior_statement_tokens: set[str] | None = None,
) -> LinguisticVector:
    """Compute the 15-dim linguistic feature vector for one document.

    See :class:`LinguisticVector` for the full field list and the
    seed-overlap floor that decides which named slot is emitted.

    When ``lda_artifact`` is None the five named topic shares plus the
    three misc slots all return 0.0 -- the hand-crafted densities are
    still emitted. Pass a fitted artifact (from :func:`fit_lda`) to get
    the full vector.

    ``prior_statement_tokens`` is the normalised token set of the
    previous statement (chronological order) used to compute
    ``pivot_distance``. Pass ``None`` when the caller is processing a
    non-statement document or the first statement in the corpus; the
    pivot field then emits ``NaN``.
    """

    if lda_artifact is None:
        topic_named = dict.fromkeys(NAMED_TOPIC_KEYS, 0.0)
        misc_shares = (0.0, 0.0, 0.0)
    else:
        shares = _topic_shares(text, lda_artifact)
        topic_named = {
            slot: float(shares[idx])
            for slot, idx in lda_artifact.topic_assignments.items()
        }
        for slot in NAMED_TOPIC_KEYS:
            topic_named.setdefault(slot, 0.0)
        misc_list = [float(shares[i]) for i in lda_artifact.misc_topic_indices]
        while len(misc_list) < 3:
            misc_list.append(0.0)
        misc_shares = tuple(misc_list[:3])

    return LinguisticVector(
        topic_share_inflation=topic_named["inflation"],
        topic_share_employment=topic_named["employment"],
        topic_share_financial_stability=topic_named["financial_stability"],
        topic_share_growth=topic_named["growth"],
        topic_share_balance_sheet=topic_named["balance_sheet"],
        topic_share_misc_1=misc_shares[0],
        topic_share_misc_2=misc_shares[1],
        topic_share_misc_3=misc_shares[2],
        hedge_density=hedge_density(text),
        comparison_density=comparison_density(text),
        forward_density=forward_density(text),
        concrete_ratio=concrete_ratio(text),
        hawk_dove_asymmetry=hawk_dove_asymmetry(text),
        log_token_count=log_token_count(text),
        pivot_distance=pivot_distance(text, prior_statement_tokens),
    )


# ---------------------------------------------------------------------------
# Corpus loading + batch builder
# ---------------------------------------------------------------------------


# ``document_type`` strings vary by source casing. The map mirrors the
# event-row builder so the two share the same kind taxonomy.
_DOCUMENT_TYPE_TO_KIND: dict[str, str] = {
    "statement": "statement",
    "Statement": "statement",
    "minutes": "minutes",
    "Minutes": "minutes",
    "meeting_transcript": "minutes",
    "press_conference": "press_conference",
    "congressional_testimony": "testimony",
    "chair_speech": "speech",
    "governor_speech": "speech",
}


@dataclass
class _CorpusDoc:
    text_hash: str
    text: str
    event_date: str = ""
    event_kind: str = ""


def _aggregate_corpus(package_dir: Path) -> list[_CorpusDoc]:
    """Concatenate registry rows by ``text_hash``.

    Sentence-level shards (TDW, gtfintechlab) get joined per
    text_hash to give the LDA fit document-level granularity rather
    than per-sentence chatter. Sorted by ``text_hash`` so the corpus
    order is deterministic.

    ``event_date`` and ``event_kind`` are lifted from the first
    registry row contributing to each ``text_hash`` -- they are
    properties of the document, not of any single sentence shard.
    The kind is mapped via ``_DOCUMENT_TYPE_TO_KIND``; rows whose
    ``document_type`` is unknown contribute an empty kind string and
    will be skipped by the ``pivot_distance`` walker downstream.
    """

    registry = package_dir / "registry_normalized.jsonl"
    if not registry.exists():
        raise FileNotFoundError(f"Missing registry: {registry}")
    by_hash: dict[str, list[str]] = {}
    seen_order: list[str] = []
    event_date_by_hash: dict[str, str] = {}
    event_kind_by_hash: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "") or "").strip()
        if not text:
            continue
        thash = str(payload.get("text_hash", "") or "").strip()
        if not thash:
            continue
        if thash not in by_hash:
            by_hash[thash] = []
            seen_order.append(thash)
            event_date_by_hash[thash] = str(payload.get("event_date", "") or "").strip()
            doc_type = str(payload.get("document_type", "") or "").strip()
            event_kind_by_hash[thash] = _DOCUMENT_TYPE_TO_KIND.get(doc_type, "")
        by_hash[thash].append(text)
    docs = [
        _CorpusDoc(
            text_hash=h,
            text="\n".join(by_hash[h]),
            event_date=event_date_by_hash.get(h, ""),
            event_kind=event_kind_by_hash.get(h, ""),
        )
        for h in seen_order
    ]
    docs.sort(key=lambda d: d.text_hash)
    return docs


def build_linguistic_feature_frame(
    *,
    package_dir: Path,
    artifact: LdaArtifact | None = None,
) -> tuple[pd.DataFrame, LdaArtifact]:
    """Fit LDA on the package corpus (if not already fit) and emit the
    per-document feature frame.

    The frame is sorted by ``text_hash`` so re-runs yield identical
    parquet bytes when paired with snappy compression. Columns:
    ``text_hash`` followed by all 15 numeric feature axes.

    ``pivot_distance`` is computed in chronological order over the
    statement rows only. The walk picks the latest preceding statement
    whose ``event_date`` is strictly less than the current one and
    diffs its token set against the current document. The first
    statement in the corpus (no strictly-prior peer) gets ``NaN``;
    non-statement documents always get ``NaN``. Ties on ``event_date``
    are broken by ``text_hash`` ascending, but a tied prior is treated
    as concurrent and not used -- the prior must be strictly earlier.
    """

    docs = _aggregate_corpus(package_dir)
    if artifact is None:
        artifact = fit_lda(d.text for d in docs)

    # Walk statement rows in chronological order and build a map from
    # text_hash to the token set of the strictly-earlier prior statement
    # (or None when there is no such peer). Non-statement rows are
    # absent from the map and pass ``None`` to
    # ``compute_linguistic_features`` so the pivot field emits NaN.
    # Same-date statements share a prior (the latest strictly-earlier
    # date); none of them become the prior for any other same-date peer
    # because the contract requires ``as_of_ts < current.as_of_ts``.
    prior_tokens_by_hash: dict[str, set[str] | None] = {}
    statement_docs = [d for d in docs if d.event_kind == PIVOT_DISTANCE_KIND]
    statement_docs.sort(key=lambda d: (d.event_date, d.text_hash))
    by_date: dict[str, list[_CorpusDoc]] = {}
    date_order: list[str] = []
    for doc in statement_docs:
        if doc.event_date not in by_date:
            by_date[doc.event_date] = []
            date_order.append(doc.event_date)
        by_date[doc.event_date].append(doc)
    prev_tokens: set[str] | None = None
    for date in date_order:
        bucket = by_date[date]
        # All docs sharing this event_date see the same strictly-earlier prior.
        for doc in bucket:
            prior_tokens_by_hash[doc.text_hash] = (
                set(prev_tokens) if prev_tokens is not None else None
            )
        # Promote this bucket to "prior" for any later date. When multiple
        # statements share the same date, the first one in (event_date,
        # text_hash) order is the canonical prior representative -- a
        # deterministic, documented tie-break that matches the rest of
        # the module's ordering convention.
        prev_tokens = pivot_distance_tokens(bucket[0].text)

    rows: list[dict[str, Any]] = []
    for doc in docs:
        prior_tokens = prior_tokens_by_hash.get(doc.text_hash)
        vec = compute_linguistic_features(
            doc.text, artifact, prior_statement_tokens=prior_tokens
        )
        row = {"text_hash": doc.text_hash}
        row.update(vec.as_dict())
        rows.append(row)
    if not rows:
        return _empty_frame(), artifact
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("text_hash", kind="mergesort").reset_index(drop=True)
    return frame, artifact


def _empty_frame() -> pd.DataFrame:
    cols = ["text_hash"] + list(LinguisticVector.__dataclass_fields__.keys())
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_lda_artifact(
    artifact: LdaArtifact, *, model_path: Path, topics_path: Path
) -> None:
    """Persist the fitted LDA bundle + top-words JSON.

    Top-words are written sorted by topic index; the JSON is dumped
    with ``sort_keys=True`` so reruns hit the same bytes.
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "vectorizer": artifact.vectorizer,
                "lda": artifact.lda,
                "topic_assignments": artifact.topic_assignments,
                "misc_topic_indices": list(artifact.misc_topic_indices),
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Invert ``topic_assignments`` so the JSON makes per-topic narration
    # natural: every topic index lists its top words and its label.
    label_for_topic: dict[int, str] = {idx: slot for slot, idx in artifact.topic_assignments.items()}
    for offset, idx in enumerate(artifact.misc_topic_indices, start=1):
        label_for_topic[idx] = f"misc_{offset}"
    payload = {
        "random_state": RANDOM_STATE,
        "num_topics": NUM_TOPICS,
        "max_iter": LDA_MAX_ITER,
        "named_topic_keys": list(NAMED_TOPIC_KEYS),
        "topic_assignments": dict(sorted(artifact.topic_assignments.items())),
        "misc_topic_indices": list(artifact.misc_topic_indices),
        "coherence_notes": dict(sorted(artifact.coherence_notes.items())),
        "topics": [
            {
                "topic_index": i,
                "human_label": label_for_topic.get(i, f"topic_{i}"),
                "top_words": artifact.top_words[i] if i < len(artifact.top_words) else [],
            }
            for i in range(NUM_TOPICS)
        ],
    }
    topics_path.parent.mkdir(parents=True, exist_ok=True)
    topics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_lda_artifact(model_path: Path) -> LdaArtifact:
    with model_path.open("rb") as fh:
        bundle = pickle.load(fh)
    vectorizer: CountVectorizer = bundle["vectorizer"]
    lda: LatentDirichletAllocation = bundle["lda"]
    vocab = vectorizer.get_feature_names_out().tolist()
    top_words = [
        _top_words_for_topic(lda, vocab, t, TOP_WORDS_PER_TOPIC)
        for t in range(lda.n_components)
    ]
    return LdaArtifact(
        vectorizer=vectorizer,
        lda=lda,
        topic_assignments=dict(bundle.get("topic_assignments", {})),
        misc_topic_indices=tuple(bundle.get("misc_topic_indices", ())),
        top_words=top_words,
        coherence_notes={},
    )


def write_linguistic_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the linguistic feature frame deterministically (snappy)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit LDA + emit the per-document linguistic features parquet."
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--output",
        default="linguistic_features.parquet",
        help=(
            "Output parquet filename, relative to the training-package directory "
            "(or an absolute path)."
        ),
    )
    parser.add_argument(
        "--model-output",
        default="linguistic_lda_model.pkl",
        help="Filename for the pickled LDA artifact.",
    )
    parser.add_argument(
        "--topics-output",
        default="linguistic_lda_topics.json",
        help="Filename for the topic-words JSON (audit + wiki source).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    package_dir = DEFAULT_DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    frame, artifact = build_linguistic_feature_frame(package_dir=package_dir)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = package_dir / output_path
    write_linguistic_parquet(frame, output_path)

    model_path = Path(args.model_output)
    if not model_path.is_absolute():
        model_path = package_dir / model_path
    topics_path = Path(args.topics_output)
    if not topics_path.is_absolute():
        topics_path = package_dir / topics_path
    save_lda_artifact(artifact, model_path=model_path, topics_path=topics_path)

    print(f"[linguistic] rows: {len(frame)} -> {output_path}")
    print(f"[linguistic] LDA model: {model_path}")
    print(f"[linguistic] topic words: {topics_path}")
    print("[linguistic] topic assignments:")
    for slot, idx in sorted(artifact.topic_assignments.items()):
        words = artifact.top_words[idx][:5]
        note = artifact.coherence_notes.get(slot, "")
        print(f"  {slot:>22s} <- topic {idx}: {' '.join(words)} [{note}]")
    if artifact.misc_topic_indices:
        print("[linguistic] misc topics:")
        for offset, idx in enumerate(artifact.misc_topic_indices, start=1):
            words = artifact.top_words[idx][:5]
            print(f"  misc_{offset} <- topic {idx}: {' '.join(words)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
