from __future__ import annotations

import re
from dataclasses import dataclass


# Sentence segmentation is intentionally simple — splits on `. ! ?` followed by
# whitespace. FOMC statements are well-punctuated so this avoids dragging
# spaCy in for a one-line dependency.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Keyword dictionary lifted from the standard FinBERT / Loughran-McDonald
# hawkish-dovish word lists, trimmed to the high-signal terms that survive
# token-rank thresholds in the FOMC corpus. Positive scores → hawkish, negative
# → dovish. Term weights are coarse — sufficient for a sentence-rank salience
# heuristic, not for a calibrated regression.
_HAWKISH_WEIGHTS: dict[str, float] = {
    "tighten": 0.9, "tightening": 0.9, "raise": 0.7, "raised": 0.7,
    "increase": 0.6, "increases": 0.6, "elevated": 0.7, "inflation": 0.5,
    "persistent": 0.6, "strong": 0.5, "robust": 0.5, "firmly": 0.6,
    "decisively": 0.7, "committed": 0.6, "vigilant": 0.7, "above": 0.4,
    "overheating": 0.9, "expansion": 0.4, "solid": 0.4,
    "hike": 0.9, "hikes": 0.9, "restrictive": 0.8,
}
_DOVISH_WEIGHTS: dict[str, float] = {
    "ease": 0.9, "easing": 0.9, "cut": 0.9, "cuts": 0.9, "lower": 0.6,
    "reduce": 0.6, "reduces": 0.6, "weakening": 0.7, "weakness": 0.7,
    "downside": 0.7, "soften": 0.7, "softer": 0.6, "moderate": 0.4,
    "moderating": 0.5, "patient": 0.5, "gradual": 0.4, "accommodative": 0.9,
    "support": 0.4, "decline": 0.5, "stimulus": 0.7, "transitory": 0.6,
}


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    weight: float

    def to_dict(self) -> dict[str, float | str]:
        return {"token": self.token, "weight": float(self.weight)}


@dataclass(frozen=True)
class SentenceAttribution:
    text: str
    score: float
    top_tokens: list[TokenAttribution]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "score": float(self.score),
            "topTokens": [token.to_dict() for token in self.top_tokens],
        }


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = _SENT_SPLIT.split(text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _tokenise(sentence: str) -> list[str]:
    """Lowercase alphanumeric token list — matches the keyword dictionary."""

    return re.findall(r"[A-Za-z]+", sentence.lower())


def attribute_sentence(sentence: str, *, top_k: int = 5) -> SentenceAttribution:
    """Score one sentence on the hawkish (+) ↔ dovish (-) axis using a
    weighted keyword count, and emit the top-`top_k` tokens by absolute
    contribution. Score is bounded to `[-1, 1]` via a soft tanh-like clip on
    the cumulative weight.
    """

    tokens = _tokenise(sentence)
    if not tokens:
        return SentenceAttribution(text=sentence, score=0.0, top_tokens=[])

    contributions: list[TokenAttribution] = []
    cumulative = 0.0
    for token in tokens:
        weight = _HAWKISH_WEIGHTS.get(token, 0.0) - _DOVISH_WEIGHTS.get(token, 0.0)
        if weight == 0.0:
            continue
        cumulative += weight
        contributions.append(TokenAttribution(token=token, weight=float(weight)))

    contributions.sort(key=lambda item: abs(item.weight), reverse=True)
    top = contributions[:top_k]
    # Soft squash via x / (1 + |x|) so the score sits in (-1, 1).
    score = cumulative / (1.0 + abs(cumulative))
    return SentenceAttribution(text=sentence, score=float(score), top_tokens=top)


def attribute_text(text: str, *, top_k_per_sentence: int = 5) -> list[SentenceAttribution]:
    """Per-sentence attribution covering the whole document body."""

    return [
        attribute_sentence(sentence, top_k=top_k_per_sentence)
        for sentence in split_sentences(text)
    ]


def to_response(attributions: list[SentenceAttribution], *, method: str = "keyword_salience_v1") -> dict[str, object]:
    """Shape the attribution output to match the frontend `XaiResponse` type
    (`{ sentences: [{ text, score, topTokens: [{ token, weight }] }], method }`).
    """

    return {
        "method": method,
        "sentences": [item.to_dict() for item in attributions],
    }
