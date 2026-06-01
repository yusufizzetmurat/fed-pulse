"""Text hygiene utilities for FOMC corpus documents.

The Fed's HTML carries navigation chrome (Share/Print/RSS), footnote-return
markers, "Implementation Note issued <date>" trailers, the Board-of-Governors
boilerplate footer, and a "Voting for the FOMC monetary policy action were
<member names>" paragraph. The first four are pure boilerplate and add noise
to embedding / classification pipelines. The voting block is partly signal —
the dissent phrasing ("voted unanimously" / "Voting against this action
was X, who preferred ...") encodes hawkish/dovish disagreement we want to
keep — and partly chrome (the member-name roster).

``clean_fomc_text`` applies the transforms in a fixed order so the
sanitization stays reproducible. Each transform is exposed as a private
helper so unit tests can pin a single behaviour in isolation.
"""

from __future__ import annotations

import re
from typing import Literal

DocumentKind = Literal["statement", "minutes", "press_conference"]


# 1. "Return to text" markers in footnotes (with or without surrounding brackets).
_RETURN_TO_TEXT_RE = re.compile(r"\s*\[?\s*Return to text\s*\]?", flags=re.IGNORECASE)


# 2. "Last Update: <date>" trailer line that introduces the footer block.
_LAST_UPDATE_RE = re.compile(
    r"\s*(?:Back to Top\s*)?Last Update[:\s]\s*[A-Za-z]+ \d{1,2},\s*\d{4}\b",
    flags=re.IGNORECASE,
)


# 3. The Board-of-Governors postal-address footer signature. The minutes
# pages also embed the "Board of Governors of the Federal Reserve System"
# string in their TOP-OF-PAGE banner, so anchoring the cut to the bare
# heading would erase the entire document body. The Fed's real footer
# always carries the postal address "20th Street and Constitution Avenue
# N.W., Washington, DC 20551". We cut in two passes:
#   1. The address-anchored cut removes "Board of Governors ... 20th
#      Street ..." to end-of-text. This is the canonical footer slug and
#      only ever appears at the very bottom.
#   2. The chrome-anchored cut catches "Board of Governors of the Federal
#      Reserve System" when it's immediately followed by the footer nav
#      menu ("About the Fed", "News & Events", "Monetary Policy",
#      "Supervision", "Stay Connected") — that combo is the chunk between
#      the body and the address slug. Both passes anchor on tail-only
#      patterns so the top-of-page banner stays out of scope.
_BOARD_FOOTER_ADDRESS_RE = re.compile(
    r"\s*Board of Governors of the Federal Reserve System\s+"
    r"20th Street and Constitution Avenue\b.*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_BOARD_FOOTER_NAV_RE = re.compile(
    r"\s*Board of Governors of the Federal Reserve System\s+"
    r"(?:About the Fed|News\s*&\s*Events|Monetary Policy|Supervision|Stay Connected)\b.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


# 4. Tail navigation chrome: "For release at ... Share", "Share Print PDF",
# "Subscribe to RSS", "Subscribe to Email", and the social-media link
# enumeration. These are short fragments scattered through the trailing
# window; remove them individually rather than as a single greedy block so
# we don't accidentally eat policy text that mentions one of the keywords.
_NAV_FRAGMENT_RES = (
    # "For release at 2:00 p.m. EST Share" / "For immediate release ... Share".
    # The release-time clause is short and bounded by "Share"; allow periods
    # inside the time stamp ("p.m.") but cap the span so a legitimate
    # sentence ending in "Share" elsewhere in the body cannot be eaten.
    re.compile(r"\bFor (?:release|immediate release)[^\n]{0,120}?\bShare\b\.?", flags=re.IGNORECASE),
    re.compile(r"\b(?:Share\s+Print\s+PDF|Share\s+Print|Share)\b\s*(?:PDF)?", flags=re.IGNORECASE),
    re.compile(r"\bSubscribe to (?:RSS|Email|the FRB)\b[^.\n]*", flags=re.IGNORECASE),
    re.compile(
        r"\bFederal Reserve (?:Facebook|Instagram|YouTube|Flickr|LinkedIn|Threads)\s+Page\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bLink to Federal Reserve (?:X|Bluesky)\s+Page\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bStay Connected\b", flags=re.IGNORECASE),
)


# 5. "Implementation Note issued <date>" — everything from that line to
# end-of-text is operational chatter (which desks should buy/sell what) and
# is appended after the policy paragraphs, never inside them.
_IMPLEMENTATION_NOTE_RE = re.compile(
    r"\s*Implementation Note issued\b.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


# 6. Voting block. The roster runs from "Voting for [the FOMC] monetary
# policy action were [:] <names>." We KEEP the dissent half — "Voting
# against this action was/were ... who preferred ..." carries the
# hawkish/dovish disagreement signal — and DROP the rest of the names.
#
# The roster sentence ends at the period before the next discourse marker
# ("Voting against", "Implementation Note", "In a related", or a newline).
# A naive ``[^.]*\.`` would stop at "Jerome H." because the roster is full
# of "Name H." abbreviations. The lookahead-anchored form below skips past
# those internal periods until the real end-of-sentence boundary is hit.
_VOTING_FOR_RE = re.compile(
    r"\s*Voting for the (?:FOMC )?monetary policy action were[:]?\s.*?"
    r"(?=\s*Voting against\b|\s*Implementation Note\b|\s*In a related\b|\n\s*\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Lone "Mr./Ms./Mrs. X voted as an alternate member at this meeting." tails
# that follow the roster sentence in a few statements.
_ALTERNATE_VOTER_RE = re.compile(
    r"\s*(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\s+voted as an alternate member at this meeting\.",
)


# 7. Whitespace normalisation: non-breaking spaces, repeated whitespace,
# and trailing blank lines.
_NBSP_RE = re.compile(r"[   ]")
_WS_RUN_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINE_RUN_RE = re.compile(r"\n\s*\n\s*\n+")


def _strip_return_to_text(text: str) -> str:
    return _RETURN_TO_TEXT_RE.sub(" ", text)


def _strip_last_update(text: str) -> str:
    return _LAST_UPDATE_RE.sub("", text)


def _strip_board_footer(text: str) -> str:
    cleaned = _BOARD_FOOTER_ADDRESS_RE.sub("", text)
    cleaned = _BOARD_FOOTER_NAV_RE.sub("", cleaned)
    return cleaned


def _strip_nav_chrome(text: str) -> str:
    # Apply to the trailing window only when possible, otherwise the whole
    # string. The "tail window" heuristic matches the scout report
    # observation that these fragments cluster near the end of the article.
    cleaned = text
    for pattern in _NAV_FRAGMENT_RES:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _strip_implementation_note(text: str) -> str:
    return _IMPLEMENTATION_NOTE_RE.sub("", text)


def _strip_voting_roster(text: str) -> str:
    """Drop the member-name roster while preserving the dissent sentence.

    The "Voting for the [FOMC] monetary policy action were <names>."
    sentence is the chrome we don't want. Any sentence that follows and
    begins with "Voting against ..." contains the dissent signal we KEEP.
    The simple sub() below removes only the "Voting for ..." sentence
    because the regex stops at the first period; the "Voting against ..."
    sentence is left untouched.
    """

    cleaned = _VOTING_FOR_RE.sub(" ", text)
    cleaned = _ALTERNATE_VOTER_RE.sub("", cleaned)
    return cleaned


def _collapse_whitespace(text: str) -> str:
    cleaned = _NBSP_RE.sub(" ", text)
    cleaned = _WS_RUN_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINE_RUN_RE.sub("\n\n", cleaned)
    # Trim each line so leading/trailing whitespace from the substitutions
    # above doesn't pile up at line ends.
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    return cleaned.strip()


def clean_fomc_text(raw: str, *, kind: DocumentKind = "statement") -> str:
    """Sanitize an FOMC document body in-place.

    Parameters
    ----------
    raw:
        The document body as scraped (or as cached on disk from a prior
        scrape). May contain navigation chrome, footnote-return markers,
        the Board-of-Governors footer, "Implementation Note issued ..."
        trailers, and a voting roster sentence.
    kind:
        Hint for which patterns are most relevant. Today the cleaning
        pipeline is the same for all kinds — statements, minutes, and
        press conferences each carry overlapping chrome — but accepting
        the argument lets future per-kind tweaks land without changing
        the call sites.

    Returns
    -------
    str
        The cleaned text. Whitespace is normalised and trimmed.
    """

    if not raw:
        return ""

    cleaned = raw
    cleaned = _strip_return_to_text(cleaned)
    cleaned = _strip_last_update(cleaned)
    cleaned = _strip_board_footer(cleaned)
    cleaned = _strip_nav_chrome(cleaned)
    cleaned = _strip_implementation_note(cleaned)
    cleaned = _strip_voting_roster(cleaned)
    cleaned = _collapse_whitespace(cleaned)

    # ``kind`` is accepted for future per-kind extensions; today it does
    # not branch behaviour. Reference it to keep linters quiet.
    del kind
    return cleaned


__all__ = [
    "clean_fomc_text",
    "DocumentKind",
]
