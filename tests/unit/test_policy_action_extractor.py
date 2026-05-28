"""Cover the policy-action extractor + /analyze ``policy_action`` block (#446).

The extractor pulls four structured signals from an FOMC statement:

- ``target_range_low_bp`` / ``target_range_high_bp``: the named target
  range, in basis points;
- ``change_direction``: ``hike`` / ``hold`` / ``cut`` derived from the
  policy verb the Committee uses ("decided to raise" / "decided to
  maintain" / "decided to lower");
- ``change_magnitude_bp``: signed int. Pulled from the in-prose
  magnitude phrase first; falls back to ``this_mid - prior_mid`` when
  the caller supplies a prior midpoint;
- ``balance_sheet_state``: ``expansion`` / ``tapering`` / ``runoff`` /
  ``None``, regex over the balance-sheet paragraph.

These tests pin the extractor against representative statement texts
(hike / hold / cut / runoff / tapering) and pin the higher-level
:func:`app.main._build_policy_action_card` helper against the empty-
text short-circuit.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Representative statement-like texts
#
# Phrasing mirrors the canonical FOMC statement structure: a sentence
# naming the target range, an optional in-prose magnitude phrase, and a
# separate paragraph naming balance-sheet operations.


HIKE_STATEMENT_25BP = (
    "Recent indicators suggest that economic activity has continued to expand at a "
    "modest pace.\n\n"
    "In support of these goals, the Committee decided to raise the target range for "
    "the federal funds rate by 1/4 percentage point to 5-1/4 to 5-1/2 percent.\n\n"
    "In addition, the Committee will continue reducing its holdings of Treasury "
    "securities and agency mortgage-backed securities, as described in its previously "
    "announced plans."
)

HOLD_STATEMENT_5_25_5_50 = (
    "Recent indicators suggest that economic activity has continued to expand at a "
    "solid pace. Job gains have moderated.\n\n"
    "In support of its goals, the Committee decided to maintain the target range for "
    "the federal funds rate at 5-1/4 to 5-1/2 percent.\n\n"
    "In addition, the Committee will continue reducing its holdings of Treasury "
    "securities and agency mortgage-backed securities."
)

CUT_STATEMENT_50BP = (
    "Recent indicators suggest that economic activity has continued to expand at a "
    "solid pace.\n\n"
    "In light of the progress on inflation and the balance of risks, the Committee "
    "decided to lower the target range for the federal funds rate by 1/2 percentage "
    "point to 4-3/4 to 5 percent.\n\n"
    "The Committee will continue reducing its holdings of Treasury securities and "
    "agency mortgage-backed securities."
)

TAPER_STATEMENT = (
    "Recent indicators suggest that economic activity has continued to expand at a "
    "solid pace.\n\n"
    "The Committee decided to maintain the target range for the federal funds rate "
    "at 5-1/4 to 5-1/2 percent.\n\n"
    "Beginning in June, the Committee will slow the pace of decline of its securities "
    "holdings by reducing the monthly redemption cap on Treasury securities from "
    "$60 billion to $25 billion."
)

EXPANSION_STATEMENT = (
    "The coronavirus outbreak has caused tremendous human and economic hardship.\n\n"
    "The Committee decided to maintain the target range for the federal funds rate "
    "at 0 to 1/4 percent.\n\n"
    "To support the flow of credit to households and businesses, the Federal Reserve "
    "will increase its holdings of Treasury securities and agency mortgage-backed "
    "securities by at least $80 billion and $40 billion per month, respectively."
)

NON_POLICY_TEXT = (
    "The Chair will take questions from reporters. Members of the press, please "
    "identify yourselves before asking a question."
)


# ---------------------------------------------------------------------------
# Schema surface


def test_analyze_response_carries_policy_action_field() -> None:
    """:class:`AnalyzeResponse` must declare the new sibling field as optional."""

    from app.schemas import AnalyzeResponse

    assert "policy_action" in AnalyzeResponse.model_fields
    field = AnalyzeResponse.model_fields["policy_action"]
    assert field.default is None


def test_policy_action_card_validates_empty_payload() -> None:
    """An all-None card (non-policy text) must validate cleanly."""

    from app.schemas import PolicyActionCard

    card = PolicyActionCard()
    assert card.target_range_low_bp is None
    assert card.target_range_high_bp is None
    assert card.change_direction is None
    assert card.change_magnitude_bp is None
    assert card.balance_sheet_state is None


def test_policy_action_card_validates_full_payload() -> None:
    """A fully populated card must validate against the schema."""

    from app.schemas import PolicyActionCard

    card = PolicyActionCard(
        target_range_low_bp=525,
        target_range_high_bp=550,
        change_direction="hike",
        change_magnitude_bp=25,
        balance_sheet_state="runoff",
    )
    assert card.target_range_low_bp == 525
    assert card.target_range_high_bp == 550
    assert card.change_direction == "hike"
    assert card.change_magnitude_bp == 25
    assert card.balance_sheet_state == "runoff"


# ---------------------------------------------------------------------------
# Extractor: target range


def test_extract_target_range_hike_statement() -> None:
    from app.services.policy_action_extractor import extract_target_range_bp

    bounds = extract_target_range_bp(HIKE_STATEMENT_25BP)
    assert bounds == (525, 550)


def test_extract_target_range_hold_statement() -> None:
    from app.services.policy_action_extractor import extract_target_range_bp

    bounds = extract_target_range_bp(HOLD_STATEMENT_5_25_5_50)
    assert bounds == (525, 550)


def test_extract_target_range_cut_statement() -> None:
    from app.services.policy_action_extractor import extract_target_range_bp

    bounds = extract_target_range_bp(CUT_STATEMENT_50BP)
    assert bounds == (475, 500)


def test_extract_target_range_zlb_statement() -> None:
    """The 0 to 1/4 percent ZLB phrasing must parse cleanly."""

    from app.services.policy_action_extractor import extract_target_range_bp

    bounds = extract_target_range_bp(EXPANSION_STATEMENT)
    assert bounds == (0, 25)


def test_extract_target_range_returns_none_on_non_policy_text() -> None:
    from app.services.policy_action_extractor import extract_target_range_bp

    assert extract_target_range_bp(NON_POLICY_TEXT) is None


# ---------------------------------------------------------------------------
# Extractor: direction + magnitude


def test_extract_direction_hike() -> None:
    from app.services.policy_action_extractor import extract_change_direction

    assert extract_change_direction(HIKE_STATEMENT_25BP) == "hike"


def test_extract_direction_hold() -> None:
    from app.services.policy_action_extractor import extract_change_direction

    assert extract_change_direction(HOLD_STATEMENT_5_25_5_50) == "hold"


def test_extract_direction_cut() -> None:
    from app.services.policy_action_extractor import extract_change_direction

    assert extract_change_direction(CUT_STATEMENT_50BP) == "cut"


def test_extract_direction_none_on_non_policy_text() -> None:
    from app.services.policy_action_extractor import extract_change_direction

    assert extract_change_direction(NON_POLICY_TEXT) is None


def test_extract_magnitude_percentage_point_phrase() -> None:
    """'by 1/4 percentage point' -> 25 bps."""

    from app.services.policy_action_extractor import extract_change_magnitude_bp

    assert extract_change_magnitude_bp(HIKE_STATEMENT_25BP) == 25


def test_extract_magnitude_half_point_phrase() -> None:
    from app.services.policy_action_extractor import extract_change_magnitude_bp

    assert extract_change_magnitude_bp(CUT_STATEMENT_50BP) == 50


def test_extract_magnitude_basis_points_phrase() -> None:
    """The explicit 'by N basis points' phrasing must parse."""

    from app.services.policy_action_extractor import extract_change_magnitude_bp

    text = "The Committee decided to raise the target range by 75 basis points."
    assert extract_change_magnitude_bp(text) == 75


def test_extract_magnitude_none_when_phrase_absent() -> None:
    from app.services.policy_action_extractor import extract_change_magnitude_bp

    assert extract_change_magnitude_bp(HOLD_STATEMENT_5_25_5_50) is None


# ---------------------------------------------------------------------------
# Extractor: balance sheet


def test_extract_balance_sheet_runoff() -> None:
    from app.services.policy_action_extractor import extract_balance_sheet_state

    assert extract_balance_sheet_state(HOLD_STATEMENT_5_25_5_50) == "runoff"


def test_extract_balance_sheet_tapering_wins_over_runoff() -> None:
    """Tapering is a narrower phrasing and must take precedence."""

    from app.services.policy_action_extractor import extract_balance_sheet_state

    assert extract_balance_sheet_state(TAPER_STATEMENT) == "tapering"


def test_extract_balance_sheet_expansion() -> None:
    from app.services.policy_action_extractor import extract_balance_sheet_state

    assert extract_balance_sheet_state(EXPANSION_STATEMENT) == "expansion"


def test_extract_balance_sheet_none_on_non_policy_text() -> None:
    from app.services.policy_action_extractor import extract_balance_sheet_state

    assert extract_balance_sheet_state(NON_POLICY_TEXT) is None


# ---------------------------------------------------------------------------
# Top-level extract_policy_action


def test_extract_policy_action_hike_with_in_prose_magnitude() -> None:
    """In-prose magnitude wins even without a prior midpoint."""

    from app.services.policy_action_extractor import extract_policy_action

    action = extract_policy_action(HIKE_STATEMENT_25BP)
    assert action.target_range_low_bp == 525
    assert action.target_range_high_bp == 550
    assert action.change_direction == "hike"
    assert action.change_magnitude_bp == 25
    assert action.balance_sheet_state == "runoff"


def test_extract_policy_action_hold_emits_zero_magnitude() -> None:
    from app.services.policy_action_extractor import extract_policy_action

    action = extract_policy_action(HOLD_STATEMENT_5_25_5_50)
    assert action.change_direction == "hold"
    assert action.change_magnitude_bp == 0


def test_extract_policy_action_cut_signs_magnitude_negative() -> None:
    from app.services.policy_action_extractor import extract_policy_action

    action = extract_policy_action(CUT_STATEMENT_50BP)
    assert action.change_direction == "cut"
    assert action.change_magnitude_bp == -50


def test_extract_policy_action_falls_back_to_prior_midpoint() -> None:
    """When the magnitude phrase is absent, ``this_mid - prior_mid`` rides."""

    from app.services.policy_action_extractor import extract_policy_action

    text = (
        "The Committee decided to raise the target range for the federal funds "
        "rate to 5-1/4 to 5-1/2 percent."
    )
    # Prior mid was 5.125% -> 513 bps; this mid is 537 bps; delta = 24.
    action = extract_policy_action(text, prior_target_range_mid_bp=513)
    assert action.change_direction == "hike"
    assert action.change_magnitude_bp == 24


def test_extract_policy_action_returns_empty_on_non_policy_text() -> None:
    """A non-policy excerpt must yield an all-None payload, not raise."""

    from app.services.policy_action_extractor import extract_policy_action

    action = extract_policy_action(NON_POLICY_TEXT)
    assert action.target_range_low_bp is None
    assert action.target_range_high_bp is None
    assert action.change_direction is None
    assert action.change_magnitude_bp is None
    assert action.balance_sheet_state is None


# ---------------------------------------------------------------------------
# main._build_policy_action_card helper


def test_build_policy_action_card_short_circuits_on_empty_text() -> None:
    """Whitespace-only text must surface ``None`` rather than empty dict."""

    from app.main import _build_policy_action_card
    from app.schemas import AnalyzeRequest

    payload = AnalyzeRequest(text="   ", date="2024-01-31")
    assert _build_policy_action_card(payload) is None


def test_build_policy_action_card_populates_on_hike_statement() -> None:
    from app.main import _build_policy_action_card
    from app.schemas import AnalyzeRequest, PolicyActionCard

    payload = AnalyzeRequest(text=HIKE_STATEMENT_25BP, date="2023-07-26")
    block = _build_policy_action_card(payload)
    assert block is not None
    # The block must validate against the schema in the response shape.
    card = PolicyActionCard(**block)
    assert card.target_range_low_bp == 525
    assert card.target_range_high_bp == 550
    assert card.change_direction == "hike"
    assert card.change_magnitude_bp == 25
    assert card.balance_sheet_state == "runoff"
