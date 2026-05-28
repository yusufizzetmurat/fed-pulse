"""Statement-delta (redline) feature builder (#443).

Token-level diff between a supervised FOMC statement and the immediately
preceding statement, materialised as three text spans per event:

- ``inserted_text`` -- tokens present in ``statement_t`` but absent
  from ``statement_{t-1}``. Joined as whitespace-separated text so the
  encoder pass can mean-pool over the span without extra tokenisation
  bookkeeping.
- ``deleted_text`` -- tokens present in ``statement_{t-1}`` but absent
  from ``statement_t``. Same join convention.
- ``substituted_pairs`` -- list of ``(old, new)`` token strings for the
  ``replace`` opcodes returned by ``difflib.SequenceMatcher``. The
  encoder pass concatenates the two halves around an ``[OLD]``/``[NEW]``
  separator so the diff direction is preserved at pool time.

Strict-prior contract: the prior statement is selected from the
``preferred`` document set with ``prior.event_date < this.event_date``,
asserted by the builder helper before the diff is computed. The strict
inequality is the contract this module enforces — a prior dated equal
to the supervised event would fold the as-of statement into its own
diff and create a same-day leak.

The block is opt-in via ``--use-statement-delta`` on
``app.train_forecaster``. When the flag is off, the loader leaves the
``statement_delta_embedding`` slot ``None`` and
``FeatureVector.as_rich_list`` does NOT append the block, so the default
per-bar feature size stays byte-identical to pre-#443. See ADR 0036.

The encoder pass over the three spans is a separate ops question --
this module produces the text spans + (optionally, when an encoder
callable is wired in via the build CLI) the mean-pooled embedding. The
build path that mounts an encoder is GPU-blocked behind the canonical
sweep return; the parquet column is nullable so an events.parquet built
without the encoder lands with NULL embeddings and the feature block
collapses to the missing-1.0 slot at load time.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


# Mean-pooled embedding width the loader expects when the slot is
# populated. Matches the encoder's hidden width on the canonical
# FinBERT-Fed-Adjacent checkpoint (768). Held here so the loader, the
# scaler, and the parquet schema agree without crossing module boundaries.
STATEMENT_DELTA_EMBEDDING_DIM: int = 768

# Whitespace + lowercase normalisation. The diff runs on a token list
# produced by this pre-processor; keeping the helper small and pure makes
# the contract explicit (no stemming, no stopword removal -- a single-word
# change should land as a single opcode).
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StatementDelta:
    """Three text spans plus optional embedding for one event's redline.

    ``inserted_text`` / ``deleted_text`` are whitespace-joined token
    strings; ``substituted_pairs`` is a list of ``(old, new)`` strings.
    ``embedding`` is ``None`` when the build pass did not invoke an
    encoder; the loader collapses ``None`` to the all-zeros + missing-1.0
    slot so the column stays nullable on disk.
    """

    inserted_text: str
    deleted_text: str
    substituted_pairs: list[tuple[str, str]]
    embedding: list[float] | None


def _normalise(text: str) -> list[str]:
    """Lowercase + whitespace-tokenise. Empty input → empty list."""

    if not text:
        return []
    return _WHITESPACE_RE.split(text.strip().lower())


def compute_delta_spans(
    *,
    current_text: str,
    prior_text: str | None,
) -> tuple[str, str, list[tuple[str, str]]] | None:
    """Run the token-level diff and return the three text spans.

    Returns ``None`` when ``prior_text`` is empty (cold-start at the
    beginning of the corpus) so the caller flips the missing flag instead
    of writing an empty triple that the encoder would mis-pool over.
    """

    if not prior_text:
        return None
    current_tokens = _normalise(current_text)
    prior_tokens = _normalise(prior_text)
    if not current_tokens or not prior_tokens:
        return None
    matcher = difflib.SequenceMatcher(
        a=prior_tokens, b=current_tokens, autojunk=False
    )
    inserted: list[str] = []
    deleted: list[str] = []
    substituted: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            inserted.extend(current_tokens[j1:j2])
        elif tag == "delete":
            deleted.extend(prior_tokens[i1:i2])
        elif tag == "replace":
            old_span = " ".join(prior_tokens[i1:i2])
            new_span = " ".join(current_tokens[j1:j2])
            substituted.append((old_span, new_span))
    return (
        " ".join(inserted),
        " ".join(deleted),
        substituted,
    )


def _mean_pool(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Element-wise mean over a sequence of equal-length float vectors."""

    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0:
        return []
    sums = [0.0] * width
    for vec in vectors:
        if len(vec) != width:
            raise ValueError(
                "_mean_pool received vectors of unequal width "
                f"({len(vec)} vs {width}); cannot pool"
            )
        for i, v in enumerate(vec):
            sums[i] += float(v)
    n = float(len(vectors))
    return [s / n for s in sums]


def compute_delta_for_event(
    *,
    current_text: str,
    prior_text: str | None,
    encode_text: Callable[[str], list[float]] | None = None,
) -> StatementDelta | None:
    """End-to-end: diff the two statements and (optionally) embed the spans.

    When ``encode_text`` is ``None`` the embedding stays at ``None``;
    the events.parquet writer encodes that as a NULL list and the
    loader collapses to the missing-1.0 slot.

    When ``encode_text`` is supplied it is called over each of the three
    spans (``inserted``, ``deleted``, and the concatenation of the
    substituted pair texts) and the three outputs are mean-pooled to a
    single fixed-dim vector. An empty span yields no encoder call (so a
    diff that is pure insertions still produces a mean-pool over the
    one non-empty channel).
    """

    spans = compute_delta_spans(current_text=current_text, prior_text=prior_text)
    if spans is None:
        return None
    inserted_text, deleted_text, substituted_pairs = spans
    embedding: list[float] | None = None
    if encode_text is not None:
        channels: list[list[float]] = []
        if inserted_text:
            channels.append(list(encode_text(inserted_text)))
        if deleted_text:
            channels.append(list(encode_text(deleted_text)))
        if substituted_pairs:
            sub_text = " ".join(
                f"[OLD] {old} [NEW] {new}" for old, new in substituted_pairs
            )
            channels.append(list(encode_text(sub_text)))
        if channels:
            embedding = _mean_pool(channels)
    return StatementDelta(
        inserted_text=inserted_text,
        deleted_text=deleted_text,
        substituted_pairs=substituted_pairs,
        embedding=embedding,
    )


def select_prior_statement_text(
    *,
    event_date: str,
    prior_index: Iterable[tuple[str, str]],
) -> str | None:
    """Return the text of the most recent statement strictly before ``event_date``.

    ``prior_index`` is an iterable of ``(event_date_iso, text)`` pairs
    over the preferred-statement set. The strict ``<`` filter is the
    contract this helper enforces — a same-date prior would fold the
    supervised event into its own diff, so any entry with
    ``prior.event_date >= target`` is silently dropped.
    """

    target = _dt.date.fromisoformat(event_date[:10])
    best_date: _dt.date | None = None
    best_text: str | None = None
    for date_str, text in prior_index:
        if not date_str or not text:
            continue
        prior_date = _dt.date.fromisoformat(date_str[:10])
        # Strict-prior contract: same-date rows are silently filtered
        # so a caller can pass the full preferred-statement index
        # (which includes the supervised event's own row) without a
        # pre-filter step. The ``<`` filter rejects every row dated
        # at or after ``target``, so the strict-prior guarantee holds
        # without an additional caller-side filter.
        if prior_date >= target:
            continue
        if best_date is None or prior_date > best_date:
            best_date = prior_date
            best_text = text
    if best_text is not None and best_date is not None:
        assert best_date < target, (
            "select_prior_statement_text post-condition violated: "
            f"selected prior dated {best_date} for event {target}"
        )
    return best_text


__all__ = [
    "STATEMENT_DELTA_EMBEDDING_DIM",
    "StatementDelta",
    "compute_delta_for_event",
    "compute_delta_spans",
    "select_prior_statement_text",
]
