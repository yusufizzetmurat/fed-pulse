"""FOMC vote-tally parser (#444).

Extracts the structured vote block from an FOMC statement body and emits
five scalars per event:

- ``votes_for`` -- count of members who voted with the action.
- ``votes_against`` -- count of dissenters.
- ``dissent_count`` -- alias of ``votes_against``; kept as a separate
  column so the downstream feature block reads cleanly.
- ``is_unanimous`` -- ``True`` iff ``votes_against == 0``.
- ``dissent_direction`` -- ``"hawkish_dissent"`` / ``"dovish_dissent"``
  / ``None``. Hawkish when the dissenter's stated preference is a
  higher target range (or a smaller cut / no cut when the action is to
  ease). Dovish when the stated preference is a lower target range.
  Mixed dissents (multiple dissenters with opposing preferred actions,
  rare but historical) resolve to ``None``.

Provenance contract: the FOMC statement IS the structured release that
carries the vote block. Parsing it from ``doc.text`` adds zero new
upstream dependencies and the values are by definition observable on
``T`` (the vote IS the event). No leak surface — see the audit row in
``docs/feature-provenance-audit.md``.

The block is opt-in via ``--use-vote-features`` on
``app.train_forecaster``. When the flag is off, the loader leaves the
``vote_features`` slot ``None`` and ``FeatureVector.as_rich_list`` does
NOT append the block, so the default per-bar feature size stays
byte-identical to pre-#444. See ADR 0036.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Sentinel for "no vote block parseable on this row" -- the parser
# returns ``None`` and the loader collapses to the all-zeros + missing-1.0
# slot. Distinct from ``is_unanimous=True``, which is a parsed, populated
# row with a zero dissent count.

# Heuristic regex set. The FOMC has used the same standardised template
# since the early 1990s with minor wording drift: "Voting for the FOMC
# monetary policy action were:" / "Voting against the action were:" /
# "Voting against this action:" (post-2008 variant). The patterns below
# tolerate the documented variants without over-broadening into prose
# matches.
_FOR_HEADERS: tuple[str, ...] = (
    r"voting for (?:the )?(?:fomc )?(?:monetary )?(?:policy )?action(?:s)?",
    r"voting for this action",
)
_AGAINST_HEADERS: tuple[str, ...] = (
    r"voting against (?:the )?(?:fomc )?(?:monetary )?(?:policy )?action(?:s)?",
    r"voting against this action",
)

# Direction-inference cues. Hawkish dissent: preferred a higher / larger
# / no-cut / smaller-cut path. Dovish: lower / larger-cut / smaller-hike.
# Captured on the trailing prose after the dissenter's name. The phrasing
# tracks the standard FOMC dissent template ("preferred at this meeting
# to..." / "preferred a [larger|smaller] [increase|reduction] in the
# target range" / "preferred to maintain the target range").
_HAWKISH_CUES: tuple[str, ...] = (
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?higher\s+target",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?to\s+(?:raise|increase)\s+the\s+target",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?larger\s+(?:increase|hike)",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?smaller\s+(?:decrease|reduction|cut)",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?to\s+maintain\s+the\s+target\s+range",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:to\s+)?(?:not|no)\s+(?:cut|reduce|decrease)",
)
_DOVISH_CUES: tuple[str, ...] = (
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?lower\s+target",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?to\s+(?:lower|reduce|cut|decrease)\s+the\s+target",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?larger\s+(?:decrease|reduction|cut)",
    r"preferred\s+(?:at\s+this\s+meeting\s+)?(?:a\s+)?smaller\s+(?:increase|hike)",
)


@dataclass(frozen=True)
class VoteTally:
    """Parsed vote tally for one FOMC statement.

    ``dissent_direction`` is ``None`` when the vote was unanimous, when
    no parseable dissent prose follows the "voting against" header, or
    when multiple dissenters pulled in opposing directions.
    """

    votes_for: int
    votes_against: int
    dissent_direction: str | None  # "hawkish_dissent" | "dovish_dissent" | None

    @property
    def dissent_count(self) -> int:
        return int(self.votes_against)

    @property
    def is_unanimous(self) -> bool:
        return self.votes_against == 0


def _find_section(text: str, header_patterns: tuple[str, ...]) -> tuple[int, int] | None:
    """Return the (start, end) span of a vote section in ``text``.

    The section runs from the header match to the next blank line, the
    next "Voting" header (so the for-section doesn't bleed into the
    against-section), or the end of the text.
    """

    lowered = text.lower()
    header_match: re.Match[str] | None = None
    for pattern in header_patterns:
        match = re.search(pattern, lowered)
        if match is not None:
            if header_match is None or match.start() < header_match.start():
                header_match = match
    if header_match is None:
        return None
    start = header_match.start()
    rest = lowered[header_match.end():]
    # End at the next "voting" header (for/against partition) or a
    # paragraph break that runs into the next prose block.
    next_voting = re.search(r"\n\s*voting", rest)
    paragraph_break = re.search(r"\n\s*\n", rest)
    candidates: list[int] = []
    if next_voting is not None:
        candidates.append(header_match.end() + next_voting.start())
    if paragraph_break is not None:
        candidates.append(header_match.end() + paragraph_break.start())
    end = min(candidates) if candidates else len(text)
    return start, end


def _count_named_members(section: str) -> int:
    """Count distinct named members in a vote section.

    The FOMC template lists members separated by semicolons (modern
    template) or by " and " before the last name (older / shorter
    lists / single dissenters: "voting against was Esther L. George, who
    preferred..."). Splitting on semicolons keeps the internal commas
    that follow each name's role label ("Powell, Chair") attached to
    the right member; the parser then strips role-only suffixes before
    confirming the token looks like a personal name. Returns ``0`` when
    the section is empty after the header.
    """

    # Strip the header itself before counting tokens. The header line
    # ends with ":" in the standard template.
    payload = section.split(":", 1)
    body = payload[1] if len(payload) > 1 else payload[0]
    # Collapse "and" before the final name into a semicolon so list-of-
    # two patterns ("X and Y, who preferred...") split uniformly with
    # the modern semicolon-delimited template.
    normalised = re.sub(r"\s+and\s+", "; ", body)
    raw_tokens = re.split(r"[;\n]", normalised)
    count = 0
    seen: set[str] = set()
    for token in raw_tokens:
        token = token.strip().rstrip(".")
        if not token:
            continue
        # Drop the trailing "(...)" gloss (some statements append a
        # parenthetical "(alternate)" on substitute voters).
        token = re.sub(r"\s*\(.*?\)\s*", "", token).strip()
        if not token:
            continue
        # Trim the dissent-explanation tail: the prose for a single
        # dissenter often runs "Esther L. George, who preferred at this
        # meeting to maintain the target range..." -- everything from
        # "who" / "because" / "preferred" onward is explanatory, not a
        # second name.
        token = re.split(
            r"\s*\b(?:who|because|preferred)\b\s*", token, maxsplit=1
        )[0].strip().rstrip(",")
        if not token:
            continue
        # Strip the role suffix after the name: "Jerome H. Powell, Chair"
        # and "John C. Williams, Vice Chair" both collapse to the bare
        # name part. The role is dropped after the first comma so a
        # legitimate name with an internal comma (extremely rare on the
        # FOMC roster) is preserved up to that comma.
        bare_name = token.split(",", 1)[0].strip()
        if not bare_name:
            continue
        # A member name is at least two capitalised words ("J. Smith"
        # counts; "Jerome H. Powell" counts; "Vice Chair" does not, but
        # is already stripped above as a role suffix).
        words = bare_name.split()
        if len(words) < 2:
            continue
        cap_words = [w for w in words if w[:1].isupper()]
        if len(cap_words) < 2:
            continue
        # Dedup on the bare name so a roster that double-lists a member
        # (e.g. "and" + earlier comma) does not over-count.
        key = bare_name.lower()
        if key in seen:
            continue
        seen.add(key)
        count += 1
    return count


def _infer_direction(text: str, header_patterns: tuple[str, ...]) -> str | None:
    """Infer hawkish / dovish from the trailing prose after the against header.

    The FOMC template puts the dissenter's stated preference in a
    sentence right after the "voting against" name list, e.g. "Voting
    against the action: Esther L. George, who preferred at this meeting
    to maintain the target range for the federal funds rate at 0 to 1/4
    percent." Returns ``None`` when no cue matches or when both hawkish
    AND dovish cues fire (mixed dissent).
    """

    lowered = text.lower()
    header_match: re.Match[str] | None = None
    for pattern in header_patterns:
        match = re.search(pattern, lowered)
        if match is not None:
            if header_match is None or match.start() < header_match.start():
                header_match = match
    if header_match is None:
        return None
    # Read up to ~600 chars after the against header -- typical dissent
    # explanation paragraph plus a small buffer.
    tail = lowered[header_match.end(): header_match.end() + 600]
    hawkish = any(re.search(p, tail) for p in _HAWKISH_CUES)
    dovish = any(re.search(p, tail) for p in _DOVISH_CUES)
    if hawkish and not dovish:
        return "hawkish_dissent"
    if dovish and not hawkish:
        return "dovish_dissent"
    return None


def parse_vote_tally(text: str | None) -> VoteTally | None:
    """Parse the vote-tally block from an FOMC statement body.

    Returns ``None`` when the text is empty / non-string or no
    "voting for ... voting against" structure is found. The caller
    treats a ``None`` return as "no parseable vote" and flips the
    missing flag to 1.0; this is also the default behaviour for event
    kinds other than ``statement`` (the vote lives in the statement
    document, not in minutes or speeches).
    """

    if not text or not isinstance(text, str):
        return None
    for_span = _find_section(text, _FOR_HEADERS)
    if for_span is None:
        return None
    against_span = _find_section(text, _AGAINST_HEADERS)
    for_section = text[for_span[0]: for_span[1]]
    votes_for = _count_named_members(for_section)
    if votes_for == 0:
        # The header matched but the name list did not parse; treat as
        # unparseable so the missing flag fires rather than emitting a
        # zero count that the model would read as "unanimous against".
        return None
    if against_span is None:
        return VoteTally(votes_for=votes_for, votes_against=0, dissent_direction=None)
    against_section = text[against_span[0]: against_span[1]]
    votes_against = _count_named_members(against_section)
    if votes_against == 0:
        return VoteTally(votes_for=votes_for, votes_against=0, dissent_direction=None)
    direction = _infer_direction(text, _AGAINST_HEADERS)
    return VoteTally(
        votes_for=votes_for,
        votes_against=votes_against,
        dissent_direction=direction,
    )


__all__ = ["VoteTally", "parse_vote_tally"]
