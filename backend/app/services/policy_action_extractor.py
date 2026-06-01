"""Extract the mechanical policy decision from an FOMC statement (#446).

Surfaces four structured signals the analyst-style dashboard renders on
the `/analyze` Policy Action card:

- ``target_range_low_bp`` / ``target_range_high_bp``: the named target
  range for the federal funds rate, in basis points (3.50%-3.75% → 350
  / 375). Handles the three phrasings the statements use: decimal
  (``3.50 to 3.75 percent``), hyphen-decimal (``3-1/4 to 3-1/2``), and
  the older ``X to Y percent`` form.
- ``change_direction``: ``hike`` / ``hold`` / ``cut`` derived from the
  verb the Committee uses ("decided to raise" / "decided to maintain"
  / "decided to lower"). When the verb is absent, the sign of the
  current-vs-prior target-range midpoint takes over (and ``hold`` is
  the default for an exact match).
- ``change_magnitude_bp``: signed int. Pulled from the in-prose phrase
  ("by 1/4 percentage point" / "by 25 basis points") when present;
  otherwise derived as ``this_mid - prior_mid`` when the caller
  supplies a prior target-range midpoint in bps.
- ``balance_sheet_state``: ``expansion`` / ``tapering`` / ``runoff`` /
  ``None``. Regex / keyword pass over the balance-sheet paragraph.

The module is pure-function and dependency-free so the call site stays
trivial to test and never blocks /analyze on an extractor failure --
the higher-level helper in ``app.main`` wraps every call in try/except.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ChangeDirection = Literal["hike", "hold", "cut"]
BalanceSheetState = Literal["expansion", "tapering", "runoff"]


@dataclass(frozen=True)
class PolicyAction:
    """Structured policy decision extracted from a single FOMC statement.

    All fields are optional: a statement that names no target range
    (press conference Q&A, speech excerpt, scraping miss) yields a
    payload with every field ``None``. The caller treats that as
    "no policy action surfaced" and the frontend renders an empty card.
    """

    target_range_low_bp: int | None = None
    target_range_high_bp: int | None = None
    change_direction: ChangeDirection | None = None
    change_magnitude_bp: int | None = None
    balance_sheet_state: BalanceSheetState | None = None


# ---------------------------------------------------------------------------
# Target range extraction
# ---------------------------------------------------------------------------


# Match a percent range: ``3.50 to 3.75 percent`` / ``3-1/4 to 3-1/2
# percent`` / ``0 to 1/4 percent``. Two number tokens (decimal,
# whole+fraction, or bare fraction), separated by ``to``/``-``/``–``,
# followed by ``percent``. The number tokens are an alternation so the
# bare-fraction form (``1/4`` on its own) parses without forcing a
# leading integer; ``\d+(?:\.\d+)?`` covers the decimal + integer
# cases.
_NUMBER_TOKEN = r"(?:\d+\s*-\s*\d+/\d+|\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
_RANGE_DECIMAL_RE = re.compile(
    rf"({_NUMBER_TOKEN})\s*(?:to|-|–|—)\s*({_NUMBER_TOKEN})\s*percent",
    flags=re.IGNORECASE,
)


def _fraction_to_decimal(piece: str) -> float | None:
    """Coerce ``3-1/4`` / ``3 1/4`` / ``1/4`` / ``3.5`` to a percent float.

    Returns ``None`` on an unparseable input rather than raising so the
    extractor degrades to a partial payload on malformed prose.
    """

    text = piece.strip().replace(" ", "")
    # Whole + fraction form (``3-1/4``).
    match = re.fullmatch(r"(\d+)-(\d+)/(\d+)", text)
    if match:
        whole, num, denom = (int(g) for g in match.groups())
        if denom == 0:
            return None
        return whole + num / denom
    # Bare fraction (``1/4``).
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if match:
        num, denom = (int(g) for g in match.groups())
        if denom == 0:
            return None
        return num / denom
    # Decimal form (``3.5``, ``3``).
    try:
        return float(text)
    except ValueError:
        return None


def _percent_to_bp(value: float) -> int:
    """Round percent-as-float to integer basis points (3.75 → 375)."""

    return int(round(value * 100))


_TARGET_RANGE_ANCHOR = re.compile(r"target\s+range", flags=re.IGNORECASE)


def extract_target_range_bp(text: str) -> tuple[int, int] | None:
    """Return ``(low_bp, high_bp)`` for the named target range, or ``None``.

    The strategy: locate the first ``target range`` anchor and look for
    the closest ``X to Y percent`` form within the next ~200 chars.
    The statements use one of three phrasings inside that window:
    ``target range ... at X to Y percent`` (hold), ``target range ...
    to X to Y percent`` (raise / lower with a magnitude phrase), and
    the bare ``target range ... X to Y percent`` (no phrase between).
    Anchoring on ``target range`` keeps the regex from locking onto a
    historical comparison ("up from the 2.25 to 2.50 percent range")
    later in the prose.
    """

    anchor = _TARGET_RANGE_ANCHOR.search(text)
    if not anchor:
        return None
    # Look in a window after the anchor so the regex doesn't reach
    # across paragraphs into unrelated numeric prose.
    window = text[anchor.end() : anchor.end() + 240]
    match = _RANGE_DECIMAL_RE.search(window)
    if not match:
        return None
    low_pct = _fraction_to_decimal(match.group(1))
    high_pct = _fraction_to_decimal(match.group(2))
    if low_pct is None or high_pct is None:
        return None
    if low_pct > high_pct:
        # Mis-parse (the order on a real statement is always low-then-
        # high). Bail rather than emit an inverted range.
        return None
    return _percent_to_bp(low_pct), _percent_to_bp(high_pct)


# ---------------------------------------------------------------------------
# Change verb + magnitude extraction
# ---------------------------------------------------------------------------


_HIKE_VERBS = re.compile(
    r"decided\s+to\s+(?:raise|increase)\s+the\s+target\s+range",
    flags=re.IGNORECASE,
)
_CUT_VERBS = re.compile(
    r"decided\s+to\s+(?:lower|reduce|cut)\s+the\s+target\s+range",
    flags=re.IGNORECASE,
)
_HOLD_VERBS = re.compile(
    r"decided\s+to\s+(?:maintain|keep)\s+the\s+target\s+range",
    flags=re.IGNORECASE,
)


def extract_change_direction(text: str) -> ChangeDirection | None:
    """Pull the policy verb. ``None`` when no verb phrase is named."""

    if _HIKE_VERBS.search(text):
        return "hike"
    if _CUT_VERBS.search(text):
        return "cut"
    if _HOLD_VERBS.search(text):
        return "hold"
    return None


# Match the magnitude phrase: ``by 25 basis points`` / ``by 1/4
# percentage point`` / ``by 1/2 percentage point``.
_BP_MAGNITUDE_RE = re.compile(
    r"by\s+(\d+)\s+basis\s+points?",
    flags=re.IGNORECASE,
)
_PCT_MAGNITUDE_RE = re.compile(
    r"by\s+(\d+(?:\s*-\s*\d+/\d+)?|\d+/\d+)\s+percentage\s+points?",
    flags=re.IGNORECASE,
)


def extract_change_magnitude_bp(text: str) -> int | None:
    """Pull the unsigned magnitude of the change in bps from the prose.

    Returns the absolute magnitude only; the sign is applied by the
    caller against ``extract_change_direction`` (or against the
    prior-mid delta when the verb is absent).
    """

    match = _BP_MAGNITUDE_RE.search(text)
    if match:
        return int(match.group(1))
    match = _PCT_MAGNITUDE_RE.search(text)
    if match:
        pct = _fraction_to_decimal(match.group(1))
        if pct is None:
            return None
        return _percent_to_bp(pct)
    return None


# ---------------------------------------------------------------------------
# Balance-sheet posture extraction
# ---------------------------------------------------------------------------


_BALANCE_SHEET_ANCHOR = re.compile(
    r"(?:balance\s+sheet|Treasury\s+securities|agency\s+mortgage[- ]backed\s+securities|"
    r"Standing\s+Repo\s+Facility)",
    flags=re.IGNORECASE,
)

# Order matters: tapering is the more specific phrase and must win over
# the generic "reduce holdings" runoff phrasing when both are present.
_TAPER_KEYWORDS = re.compile(
    r"slow(?:ing)?\s+the\s+pace|reduce\s+the\s+monthly\s+(?:cap|pace)|"
    r"taper|slowing\s+the\s+pace\s+of\s+(?:its\s+)?(?:decline|run[- ]?off)",
    flags=re.IGNORECASE,
)
_RUNOFF_KEYWORDS = re.compile(
    r"reduc(?:e|ing)\s+its\s+holdings|run(?:ning)?[- ]?off|"
    r"continue\s+(?:to\s+)?reduc(?:e|ing)|decline\s+in\s+the\s+balance\s+sheet",
    flags=re.IGNORECASE,
)
_EXPANSION_KEYWORDS = re.compile(
    r"increase\s+(?:its\s+)?holdings|purchas(?:e|ing)\s+(?:additional\s+)?"
    r"(?:Treasury\s+securities|agency\s+mortgage)|expand\s+(?:its\s+)?(?:holdings|balance\s+sheet)|"
    r"net\s+asset\s+purchases",
    flags=re.IGNORECASE,
)


def _balance_sheet_paragraph(text: str) -> str | None:
    """Return the paragraph that names balance-sheet operations, or None.

    Statements are split on blank lines; we keep the FIRST paragraph
    whose body matches the anchor regex. Returning the paragraph (not
    the whole document) keeps the keyword pass from cross-contaminating
    with vocabulary that happens to repeat elsewhere ("reducing
    inflation" inside an inflation paragraph, say).
    """

    if not text:
        return None
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        if _BALANCE_SHEET_ANCHOR.search(para):
            return para
    return None


def extract_balance_sheet_state(text: str) -> BalanceSheetState | None:
    """Return the balance-sheet posture named in the statement, or None.

    The order of checks is deliberate: tapering ("slowing the pace of
    decline") is a narrower phrasing than runoff ("continuing to reduce
    holdings") and must take precedence when both surface.
    """

    paragraph = _balance_sheet_paragraph(text)
    if paragraph is None:
        return None
    if _TAPER_KEYWORDS.search(paragraph):
        return "tapering"
    if _EXPANSION_KEYWORDS.search(paragraph):
        return "expansion"
    if _RUNOFF_KEYWORDS.search(paragraph):
        return "runoff"
    return None


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


def _midpoint_bp(low_bp: int, high_bp: int) -> int:
    return (low_bp + high_bp) // 2


def extract_policy_action(
    text: str,
    prior_target_range_mid_bp: int | None = None,
) -> PolicyAction:
    """Return the structured policy decision named in ``text``.

    ``prior_target_range_mid_bp`` is the midpoint of the prior meeting's
    target range in basis points (e.g. 363 for a 3.50%-3.75% range).
    When supplied AND the current statement names a target range, the
    helper derives a signed ``change_magnitude_bp`` from the midpoint
    delta even if the in-prose magnitude phrase is absent. When the
    delta and the verb disagree (rare; a "decided to maintain" with a
    midpoint move would be a parse bug), the verb wins and the magnitude
    falls back to the unsigned in-prose value.
    """

    target_range = extract_target_range_bp(text)
    direction = extract_change_direction(text)
    in_prose_magnitude = extract_change_magnitude_bp(text)
    balance_sheet = extract_balance_sheet_state(text)

    low_bp = target_range[0] if target_range else None
    high_bp = target_range[1] if target_range else None

    signed_magnitude: int | None = None
    if direction == "hold":
        signed_magnitude = 0
    elif in_prose_magnitude is not None and direction in {"hike", "cut"}:
        signed_magnitude = in_prose_magnitude if direction == "hike" else -in_prose_magnitude
    elif prior_target_range_mid_bp is not None and low_bp is not None and high_bp is not None:
        delta = _midpoint_bp(low_bp, high_bp) - prior_target_range_mid_bp
        signed_magnitude = delta
        if direction is None:
            if delta > 0:
                direction = "hike"
            elif delta < 0:
                direction = "cut"
            else:
                direction = "hold"

    return PolicyAction(
        target_range_low_bp=low_bp,
        target_range_high_bp=high_bp,
        change_direction=direction,
        change_magnitude_bp=signed_magnitude,
        balance_sheet_state=balance_sheet,
    )
