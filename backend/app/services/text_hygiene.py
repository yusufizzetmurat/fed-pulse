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


# Per-kind feature flag for the top-of-page banner scrub. The chrome lives
# on every Fed page we ingest, so all three kinds default to ``True``; if a
# regression shows up on one kind we can drop that kind from the set
# without disturbing the other transforms.
_TOP_BANNER_KINDS: frozenset[str] = frozenset(
    {"statement", "minutes", "press_conference"}
)


# 0. Top-of-page banner — UTF-8 BOM plus the Fed's site-wide navigation
# header. The minutes scraper carries the literal BOM (or its mojibake
# 0xEF 0xBB 0xBF Latin-1 form) followed by "Skip to main content", the
# "An official website of the United States government" .gov banner, the
# "Stay Connected" social-media row, the "Subscribe to RSS / Email" menu,
# and the global site map BEFORE the article body starts. The footer
# regexes downstream can't reach this because the postal-address anchor
# only fires at the real footer thousands of characters later. We cut
# from the start of the document up to the first real body marker.
_LEADING_BOM_RE = re.compile(
    # Real UTF-8 BOM (﻿) plus the mojibake form (ï»¿)
    # that shows up when the source page was decoded as Latin-1 before
    # being re-saved as UTF-8.
    r"^[﻿ï»¿\s]+",
)

# The first body marker after the chrome banner — anchor the cut here.
# Only phrase-anchors are admitted. An earlier draft accepted a fallback
# calendar-date alternation ("January 25-26, 2011", "March 15, 2011",
# ...), but pre-2012 minutes carry the page title "FRB: FOMC Minutes,
# <date>" AT THE VERY TOP — ahead of the chrome — so the date alternation
# matched the title-line date and left all the nav chrome ("skip to main
# navigation ... Site Map ... A-Z Index ... Advanced Search ...") in the
# cleaned body for 33/148 minutes docs (the 2007-2011 cohort). Every
# banner-prefixed document in the corpus already carries one of the
# phrase anchors below downstream of the chrome (verified across all
# 148 minutes + 64 statements), so the date fallback is unnecessary
# in addition to being unsafe.
_BODY_ANCHOR_RE = re.compile(
    r"(?:"
    r"For release at\b"
    r"|For immediate release\b"
    r"|Information received since\b"
    r"|The Federal Open Market Committee\b"
    r"|A (?:joint )?meeting of the Federal Open Market Committee\b"
    r"|Minutes of the Federal Open Market Committee\b"
    r")",
    flags=re.IGNORECASE,
)

# Telltale tokens that the scraped HTML carried the site-wide nav banner
# at the top of the document. We only invoke the body-anchor cut when
# one of these is present in the leading window, so a clean document
# whose first sentence happens to start with a calendar date is left
# untouched.
_TOP_BANNER_SIGNALS_RE = re.compile(
    r"(?:"
    r"Skip to main (?:content|navigation)"
    r"|An official website of the United States [Gg]overnment"
    r"|Official websites use \.gov"
    r"|Secure \.gov websites use HTTPS"
    r"|Share sensitive information only on official"
    r")",
    flags=re.IGNORECASE,
)


# 1. "Return to text" markers in footnotes (with or without surrounding brackets).
_RETURN_TO_TEXT_RE = re.compile(r"\s*\[?\s*Return to text\s*\]?", flags=re.IGNORECASE)


# 2. "Last Update: <date>" trailer line that introduces the footer block.
_LAST_UPDATE_RE = re.compile(
    r"\s*(?:Back to Top\s*)?Last Update[:\s]\s*[A-Za-z]+ \d{1,2},\s*\d{4}\b",
    flags=re.IGNORECASE,
)


# 3. The Board-of-Governors postal-address footer signature. The minutes
# pages embed "Board of Governors of the Federal Reserve System" plus the
# nav-menu chrome ("Stay Connected", "About the Fed", "News & Events", ...)
# in their TOP-OF-PAGE banner AS WELL AS in the real footer, so a heading-
# anchored greedy cut would erase the entire document body. The Fed's
# real footer is unique in that it terminates with the postal address
# "20th Street and Constitution Avenue N.W., Washington, DC 20551". We
# anchor every footer cut to that address (either directly via the
# address regex, or via a lookahead that requires the address downstream
# in the chrome-anchored regex). The top-of-page banner does not carry
# the address, so it stays out of scope of these cuts and is handled by
# the per-fragment nav-chrome stripping further down.
_BOARD_FOOTER_ADDRESS_RE = re.compile(
    r"\s*Board of Governors of the Federal Reserve System\s+"
    r"20th Street and Constitution Avenue\b.*$",
    flags=re.IGNORECASE | re.DOTALL,
)
# Match the chrome banner ONLY when the postal-address slug follows
# within ~1500 chars. The real footer chrome ("Board of Governors ...
# About the Fed ... Stay Connected ... Subscribe to RSS ... Board of
# Governors ... 20th Street ...") spans a few hundred characters in
# practice. The top-of-page banner contains the same prefix but the
# address only appears at the very end of the document, tens of
# thousands of characters later, so the bounded lookahead never matches
# for the top banner.
_BOARD_FOOTER_NAV_RE = re.compile(
    r"\s*Board of Governors of the Federal Reserve System\s+"
    r"(?:About the Fed|News\s*&\s*Events|Monetary Policy|Supervision|Stay Connected)\b"
    r"[\s\S]{0,1500}?(?=Board of Governors of the Federal Reserve System\s+"
    r"20th Street and Constitution Avenue\b)",
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
    # The Fed nav-chrome row reads "Share Print PDF" (or "Share Print"). Anchor
    # to that adjacency — never strip a bare "Share", which is a common noun
    # in policy text ("market share", "Structure and Share Data", "share of
    # workers employed part time", and the .gov banner "Share sensitive
    # information only on official sites").
    re.compile(r"\bShare\s+Print(?:\s+PDF)?\b", flags=re.IGNORECASE),
    re.compile(r"\bSubscribe to (?:RSS|Email|the FRB)\b[^.\n]*", flags=re.IGNORECASE),
    re.compile(
        r"\bFederal Reserve (?:Facebook|Instagram|YouTube|Flickr|LinkedIn|Threads)\s+Page\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bLink to Federal Reserve (?:X|Bluesky)\s+Page\b",
        flags=re.IGNORECASE,
    ),
    # "Stay Connected" is the social-media banner heading the Fed footer.
    # Strip it only when it sits adjacent to footer-chrome tokens (the
    # "Federal Reserve <network> Page" enumeration, "Subscribe to ...", or
    # the "Board of Governors of the Federal Reserve System" banner) — that
    # adjacency confirms the chrome context. Bare "stay connected" used as
    # a verb phrase elsewhere in the body is left intact. Both forms below
    # use lookaround so the surrounding chrome stays in scope of the other
    # nav-fragment patterns and gets stripped on its own pass.
    re.compile(
        r"(?<=Reserve System)\s+Stay Connected\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bStay Connected\b(?=\s+(?:Federal Reserve (?:Facebook|Instagram|YouTube|Flickr|LinkedIn|Threads|X|Bluesky)\s+Page|Link to Federal Reserve|Subscribe to (?:RSS|Email)))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?<=Page)\s+Stay Connected\b",
        flags=re.IGNORECASE,
    ),
)


# 5. "Implementation Note issued <Month> <day>, <year>" — the chrome
# trailer that closes every FOMC statement (the desk-operations chatter is
# appended after the policy paragraphs). Minutes also reference prior
# Implementation Notes mid-body ("...operations in the Implementation Note
# issued following the December 2025 meeting."), so we anchor the cut to
# the trailer form ONLY — phrase + an actual date (Month <day>, <year>).
# Any non-date continuation ("issued following ...") stays in place.
_IMPLEMENTATION_NOTE_RE = re.compile(
    r"\s*Implementation Note issued\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s*\d{4}\b.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


# 6. Voting block. Two real shapes:
#
#   statement: "Voting for [the FOMC] monetary policy action were <names>."
#   minutes:   "Voting for this action: <names>. Voting against this
#              action: <None | name [who ...]>."
#
# We KEEP the dissent half — "Voting against this action was/were ... who
# preferred ..." carries the hawkish/dovish disagreement signal — and DROP
# the rest of the names.
#
# The roster sentence ends at the period before the next discourse marker
# ("Voting against", "Implementation Note", "In a related", or a blank
# line). A naive ``[^.]*\.`` would stop at "Jerome H." because the roster
# is full of "Name H." abbreviations. The lookahead below skips past those
# internal periods until the real end-of-sentence boundary is hit. We do
# NOT fall back to end-of-text — if none of the anchors appear (e.g., a
# single-newline-separated continuation sentence follows the roster), the
# regex must NOT match, otherwise we silently amputate post-roster prose.
_VOTING_FOR_RE = re.compile(
    # ``Voting (by notation) for the ...`` is a real corpus variant
    # used when the action is taken via written vote rather than at the
    # FOMC meeting itself; admit the optional parenthetical so those
    # rosters get cleaned alongside the standard form.
    r"\s*Voting(?:\s+\([^)]+\))? for the (?:FOMC )?monetary policy action were[:]?\s.*?"
    r"(?="
    r"\s*Voting against\b"
    r"|\s*Implementation Note issued\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s*\d{4}\b"
    r"|\s*In a related\b"
    r"|\s*(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\s+voted as an alternate member\b"
    r"|\n\s*\n"
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)

# Minutes form: "Voting for this action: <names>." — stop at the
# "Voting against this action" anchor. As above, do not fall back to
# end-of-text.
_VOTING_FOR_ACTION_RE = re.compile(
    r"\s*Voting for this action[:]\s.*?"
    r"(?=\s*Voting against this action\b)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Minutes form: "Voting against this action: None." (the no-dissent
# variant). When the dissent column literally says "None", there is no
# signal to keep — strip the bare line so the embedding side does not
# treat it as content. We do NOT touch "Voting against this action: <name>"
# variants — those carry the dissent signal.
_VOTING_AGAINST_NONE_RE = re.compile(
    r"\s*Voting against this action[:]\s*None\.?",
    flags=re.IGNORECASE,
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


def _strip_top_banner(text: str) -> str:
    """Strip the leading BOM and the Fed's site-wide navigation banner.

    The minutes scraper writes the article HTML verbatim, so every
    page carries a UTF-8 BOM followed by "Skip to main content", the
    .gov boilerplate ("An official website of the United States
    government" / "Secure .gov websites use HTTPS" / "Share sensitive
    information only on official, secure websites."), the "Stay
    Connected" social-media row, the "Subscribe to RSS / Email" menu,
    and the global site map BEFORE the article body starts. We cut
    from the start of the document up to the first real body marker
    — one of the "For release at" / "Information received" / "The
    Federal Open Market Committee" / "Minutes of the Federal Open
    Market Committee" phrasings.

    The cut only runs when the leading window carries at least one
    chrome signal ("Skip to main content" / .gov boilerplate). A
    clean document whose first sentence happens to start with a
    calendar date is left untouched.
    """

    if not text:
        return text

    # Always strip a leading BOM (real or mojibake) so the byte does
    # not leak into downstream tokenisers even on clean docs.
    cleaned = _LEADING_BOM_RE.sub("", text)

    # Only invoke the body-anchor cut when the leading window carries
    # a chrome signal. Bound the window at 4k chars — the longest
    # banner observed in the corpus runs ~3.5k characters.
    head_window = cleaned[:4000]
    if not _TOP_BANNER_SIGNALS_RE.search(head_window):
        return cleaned

    # Find the first body anchor anywhere in the document. The chrome
    # banner can stretch past 4k on a few minutes pages (the global
    # site map runs long), so do not bound the anchor search.
    anchor = _BODY_ANCHOR_RE.search(cleaned)
    if anchor is None:
        return cleaned
    return cleaned[anchor.start():]


def _strip_return_to_text(text: str) -> str:
    return _RETURN_TO_TEXT_RE.sub(" ", text)


def _strip_last_update(text: str) -> str:
    return _LAST_UPDATE_RE.sub("", text)


def _strip_board_footer(text: str) -> str:
    # The nav-anchored cut uses a lookahead requiring the postal-address
    # banner downstream — that lookahead must resolve BEFORE the address
    # cut consumes it. Run nav first, then address.
    cleaned = _BOARD_FOOTER_NAV_RE.sub("", text)
    cleaned = _BOARD_FOOTER_ADDRESS_RE.sub("", cleaned)
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

    The "Voting for ..." enumeration (statement and minutes forms) is the
    chrome we don't want. Any sentence that follows and begins with
    "Voting against ... <name>" contains the dissent signal we KEEP — its
    presence is what anchors the roster cut. A "Voting against this
    action: None." line carries no signal and is stripped on its own.
    """

    cleaned = _VOTING_FOR_RE.sub(" ", text)
    cleaned = _VOTING_FOR_ACTION_RE.sub(" ", cleaned)
    cleaned = _VOTING_AGAINST_NONE_RE.sub(" ", cleaned)
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
    # The top-of-page banner cut runs FIRST so every subsequent
    # transform sees the already-de-headered text. Gated by the
    # per-kind flag so we can dial back a single kind if a regression
    # appears without disturbing the others.
    if kind in _TOP_BANNER_KINDS:
        cleaned = _strip_top_banner(cleaned)
    cleaned = _strip_return_to_text(cleaned)
    cleaned = _strip_last_update(cleaned)
    cleaned = _strip_board_footer(cleaned)
    cleaned = _strip_nav_chrome(cleaned)
    # The voting-roster cut anchors on the "Implementation Note issued ..."
    # trailer (among other tokens). Run it BEFORE _strip_implementation_note
    # so that anchor is still present when the lookahead resolves.
    cleaned = _strip_voting_roster(cleaned)
    cleaned = _strip_implementation_note(cleaned)
    cleaned = _collapse_whitespace(cleaned)

    return cleaned


__all__ = [
    "clean_fomc_text",
    "DocumentKind",
]
