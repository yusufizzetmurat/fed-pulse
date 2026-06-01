from __future__ import annotations

import re

from app.services.text_hygiene import (
    _collapse_whitespace,
    _strip_board_footer,
    _strip_implementation_note,
    _strip_last_update,
    _strip_nav_chrome,
    _strip_return_to_text,
    _strip_voting_roster,
    clean_fomc_text,
)


# ---------------------------------------------------------------------------
# Per-transform isolation tests
# ---------------------------------------------------------------------------


def test_strip_return_to_text_removes_bracketed_and_bare_markers():
    raw = (
        "1. The Federal Open Market Committee is referenced as the FOMC. "
        "Return to text 2. Attended Tuesday's session only. [Return to text]"
    )
    cleaned = _strip_return_to_text(raw)
    assert "Return to text" not in cleaned
    assert "[Return to text]" not in cleaned
    # Body text survives.
    assert "Federal Open Market Committee" in cleaned
    assert "Attended Tuesday" in cleaned


def test_strip_last_update_drops_trailer_line():
    raw = "policy text. Back to Top Last Update: February 19, 2020 Board of Governors"
    cleaned = _strip_last_update(raw)
    assert "Last Update" not in cleaned
    assert "policy text." in cleaned


def test_strip_board_footer_removes_address_and_everything_after():
    raw = (
        "policy text that should survive. "
        "Board of Governors of the Federal Reserve System 20th Street and "
        "Constitution Avenue N.W., Washington, DC 20551"
    )
    cleaned = _strip_board_footer(raw)
    assert "Board of Governors" not in cleaned
    assert "Constitution Avenue" not in cleaned
    assert "20551" not in cleaned
    assert "policy text that should survive." in cleaned


def test_strip_nav_chrome_removes_share_print_rss():
    raw = (
        "For release at 2:00 p.m. EST Share. The Committee judges that the "
        "current stance of monetary policy is appropriate. Subscribe to RSS "
        "Federal Reserve Facebook Page Federal Reserve YouTube Page Stay Connected"
    )
    cleaned = _strip_nav_chrome(raw)
    assert "Subscribe to RSS" not in cleaned
    assert "Facebook Page" not in cleaned
    assert "YouTube Page" not in cleaned
    assert "Stay Connected" not in cleaned
    # The release-time chrome ("For release at ... Share") is removed too.
    assert "For release at" not in cleaned
    # Policy sentence survives intact.
    assert "current stance of monetary policy is appropriate" in cleaned


def test_strip_implementation_note_kills_trailer_to_end_of_text():
    raw = (
        "Voting against this action was Loretta J. Mester, who preferred to "
        "reduce the target range. Implementation Note issued March 15, 2020 "
        "Federal Reserve actions to support the flow of credit"
    )
    cleaned = _strip_implementation_note(raw)
    assert "Implementation Note" not in cleaned
    assert "actions to support the flow of credit" not in cleaned
    # The dissent sentence above the trailer survives — this is signal we
    # want to keep, not chrome.
    assert "Voting against this action" in cleaned
    assert "Mester" in cleaned


def test_strip_voting_roster_removes_names_keeps_dissent_signal():
    raw = (
        "The Committee decided to maintain the target range. "
        "Voting for the monetary policy action were Jerome H. Powell, Chair; "
        "John C. Williams, Vice Chair; Michelle W. Bowman; Lael Brainard; "
        "Richard H. Clarida; and Randal K. Quarles. Voting against this "
        "action was Loretta J. Mester, who preferred to reduce the target "
        "range for the federal funds rate to 1/2 to 3/4 percent at this "
        "meeting."
    )
    cleaned = _strip_voting_roster(raw)
    # Member roster is gone.
    assert "Jerome H. Powell" not in cleaned
    assert "John C. Williams" not in cleaned
    assert "Vice Chair" not in cleaned
    # Dissent sentence is preserved — that is the signal the corpus needs.
    assert "Voting against this action was Loretta J. Mester" in cleaned
    assert "preferred to reduce the target range" in cleaned
    # Lead-in policy sentence survives.
    assert "Committee decided to maintain the target range" in cleaned


def test_strip_voting_roster_handles_fomc_variant_and_alternate_member():
    raw = (
        "Voting for the FOMC monetary policy action were: Jerome H. Powell, "
        "Chairman; John C. Williams, Vice Chairman; and Loretta J. Mester. "
        "Ms. Daly voted as an alternate member at this meeting."
    )
    cleaned = _strip_voting_roster(raw)
    assert "Jerome H. Powell" not in cleaned
    assert "voted as an alternate member" not in cleaned


def test_strip_voting_roster_handles_by_notation_variant():
    """Inter-meeting written votes carry a 'Voting (by notation) for ...'
    header — the same roster must be scrubbed."""

    raw = (
        "Voting (by notation) for the monetary policy action were Jerome H. "
        "Powell, Chairman; John C. Williams, Vice Chairman; and Lael "
        "Brainard. "
        "Implementation Note issued March 15, 2020."
    )
    cleaned = _strip_voting_roster(raw)
    assert "Jerome H. Powell" not in cleaned
    assert "Lael Brainard" not in cleaned
    # The trailing Implementation Note line must remain intact for the
    # downstream _strip_implementation_note pass to find it.
    assert "Implementation Note issued March 15, 2020." in cleaned


def test_collapse_whitespace_normalises_nbsp_and_runs():
    raw = "policy\xa0statement     with    runs\n\n\n\nand blank lines"
    cleaned = _collapse_whitespace(raw)
    assert "\xa0" not in cleaned
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("policy statement with runs")


# ---------------------------------------------------------------------------
# End-to-end realistic-document tests
# ---------------------------------------------------------------------------


REALISTIC_STATEMENT = (
    "January 29, 2020\n"
    "For release at 2:00 p.m. EST Share\n"
    "Information received since the Federal Open Market Committee met in "
    "December indicates that the labor market remains strong and that "
    "economic activity has been rising at a moderate rate.\n"
    "Consistent with its statutory mandate, the Committee seeks to foster "
    "maximum employment and price stability. The Committee decided to "
    "maintain the target range for the federal funds rate at 1-1/2 to 1-3/4 "
    "percent.\n"
    "Voting for the monetary policy action were Jerome H. Powell, Chair; "
    "John C. Williams, Vice Chair; Michelle W. Bowman; Lael Brainard; "
    "Richard H. Clarida; Patrick Harker; Robert S. Kaplan; Neel Kashkari; "
    "Loretta J. Mester; and Randal K. Quarles.\n"
    "Implementation Note issued January 29, 2020"
)


REALISTIC_DISSENT_STATEMENT = (
    "March 15, 2020\n"
    "For release at 5:00 p.m. EST Share\n"
    "The Committee decided to lower the target range for the federal funds "
    "rate to 0 to 1/4 percent.\n"
    "Voting for the monetary policy action were Jerome H. Powell, Chair; "
    "and Randal K. Quarles. Voting against this action was Loretta J. "
    "Mester, who was fully supportive of all of the actions taken to "
    "promote the smooth functioning of markets and the flow of credit to "
    "households and businesses but preferred to reduce the target range "
    "for the federal funds rate to 1/2 to 3/4 percent at this meeting.\n"
    "Implementation Note issued March 15, 2020"
)


REALISTIC_MINUTES_TAIL = (
    "The meeting adjourned at 9:50 a.m. on January 29, 2020. Notation Vote "
    "By notation vote completed on January 2, 2020, the Committee "
    "unanimously approved the minutes of the Committee meeting held on "
    "December 10-11, 2019. _______________________ James A. Clouse "
    "Secretary 1. The Federal Open Market Committee is referenced as the "
    'FOMC and the Committee in these minutes. Return to text 2. Attended '
    "Tuesday's session only. Return to text Back to Top Last Update: "
    "February 19, 2020 Board of Governors of the Federal Reserve System "
    "About the Fed News & Events Monetary Policy Federal Reserve Facebook "
    "Page Federal Reserve YouTube Page Subscribe to RSS Subscribe to Email "
    "Board of Governors of the Federal Reserve System 20th Street and "
    "Constitution Avenue N.W., Washington, DC 20551"
)


def test_clean_statement_end_to_end_removes_all_chrome_keeps_policy():
    cleaned = clean_fomc_text(REALISTIC_STATEMENT, kind="statement")

    # Chrome is gone.
    assert "Implementation Note" not in cleaned
    assert "Jerome H. Powell" not in cleaned
    assert "Vice Chair" not in cleaned
    assert "Loretta J. Mester" not in cleaned
    assert "Subscribe to RSS" not in cleaned
    assert "For release at" not in cleaned

    # Policy-relevant content survives.
    assert "labor market remains strong" in cleaned
    assert "maximum employment and price stability" in cleaned
    assert "target range for the federal funds rate" in cleaned


def test_clean_statement_preserves_dissent_drops_member_list():
    cleaned = clean_fomc_text(REALISTIC_DISSENT_STATEMENT, kind="statement")

    # The dissent half is the signal we keep.
    assert "Voting against this action was Loretta J. Mester" in cleaned
    assert "preferred to reduce the target range" in cleaned
    # The dissent sentence keeps the dissenter's name by design — that name
    # IS part of the signal (which member dissented, in which direction).
    # The pre-dissent roster is what we drop.
    assert "Jerome H. Powell, Chair" not in cleaned
    assert "Randal K. Quarles" not in cleaned
    assert "Voting for the monetary policy action" not in cleaned
    # Policy sentence survives.
    assert "lower the target range for the federal funds rate" in cleaned
    # Implementation Note trailer gone.
    assert "Implementation Note" not in cleaned


def test_clean_minutes_end_to_end_strips_footer_and_return_markers():
    cleaned = clean_fomc_text(REALISTIC_MINUTES_TAIL, kind="minutes")

    assert "Return to text" not in cleaned
    assert "Last Update" not in cleaned
    assert "Board of Governors" not in cleaned
    assert "20551" not in cleaned
    assert "Facebook Page" not in cleaned
    assert "Subscribe to RSS" not in cleaned
    # Substantive content is preserved.
    assert "meeting adjourned" in cleaned
    assert "Notation Vote" in cleaned
    assert "James A. Clouse" in cleaned


def test_clean_empty_text_is_noop():
    assert clean_fomc_text("", kind="statement") == ""
    assert clean_fomc_text("", kind="minutes") == ""


def test_clean_press_conference_kind_accepted():
    # Press-conference kind is plumbed through but currently uses the same
    # pipeline. This test pins the contract so a future per-kind branch
    # has to deliberately change the assertion.
    raw = "Chair Powell: The Committee remains data-dependent. Subscribe to RSS"
    cleaned = clean_fomc_text(raw, kind="press_conference")
    assert "Subscribe to RSS" not in cleaned
    assert "data-dependent" in cleaned


def test_clean_is_idempotent():
    once = clean_fomc_text(REALISTIC_STATEMENT, kind="statement")
    twice = clean_fomc_text(once, kind="statement")
    assert once == twice


def test_clean_collapses_whitespace_no_double_spaces():
    cleaned = clean_fomc_text(REALISTIC_STATEMENT, kind="statement")
    # No runs of >1 internal space, no leading/trailing whitespace.
    assert "  " not in cleaned
    assert cleaned == cleaned.strip()
    assert not re.search(r"\n{3,}", cleaned)


# ---------------------------------------------------------------------------
# Regression tests for over-cut / over-strip bugs caught in review
# ---------------------------------------------------------------------------


def test_implementation_note_inline_reference_is_preserved():
    """Minutes legitimately reference prior Implementation Notes mid-body.

    The on-disk 2026-01-28 minutes carry
    "...operations in the Implementation Note issued following the December
    2025 meeting. All participants indicated support for ..." 42k characters
    before the real footer. A DOTALL ``.*$`` cut anchored on the bare phrase
    truncates ~80% of the document. The trailer form always carries an
    explicit date (Month <day>, <year>); the inline reference does not.
    """

    raw = (
        "The Desk continues to conduct standing repurchase agreement "
        "operations in the Implementation Note issued following the "
        "December 2025 meeting. All participants indicated support for, "
        "and agreed to abide by, the FOMC Policy on Investment and "
        "Trading for Committee Participants."
    )
    cleaned = _strip_implementation_note(raw)
    assert "Implementation Note issued following" in cleaned
    assert "All participants indicated support" in cleaned
    assert "Trading for Committee Participants" in cleaned


def test_implementation_note_trailer_with_date_is_still_stripped():
    """The dated trailer form is the only thing we want to cut."""

    raw = (
        "Voting against this action was Loretta J. Mester. "
        "Implementation Note issued March 15, 2020 Federal Reserve actions "
        "to support the flow of credit to households and businesses."
    )
    cleaned = _strip_implementation_note(raw)
    assert "Implementation Note" not in cleaned
    assert "Federal Reserve actions" not in cleaned
    assert "Voting against this action" in cleaned


def test_share_as_ordinary_noun_is_preserved():
    """'share' is a common noun in FOMC prose. Only chrome-adjacent
    occurrences (Share + Print [+ PDF], or "For release at ... Share")
    should be stripped — never bare 'share'.
    """

    raw = (
        "The average share of workers employed part time for economic "
        "reasons declined in the quarter. Market share among the largest "
        "banks remained stable. The Structure and Share Data report was "
        "discussed. Share sensitive information only on official, secure "
        "websites."
    )
    cleaned = _strip_nav_chrome(raw)
    assert "average share of workers" in cleaned
    assert "Market share among the largest" in cleaned
    assert "Structure and Share Data" in cleaned
    assert "Share sensitive information" in cleaned


def test_share_print_pdf_chrome_is_still_stripped():
    raw = "policy text. Share Print PDF after the headline."
    cleaned = _strip_nav_chrome(raw)
    assert "Share Print PDF" not in cleaned
    assert "policy text." in cleaned
    assert "after the headline." in cleaned


def test_voting_roster_does_not_swallow_post_roster_sentence():
    """Without a recognized trailing anchor, the roster cut must NOT match.

    A single-newline-separated continuation sentence after the roster is
    real FOMC text (the HTML normalises to plain text with no blank
    lines). The previous '.*?$' fallback amputated the tail; the new
    behaviour is to leave the entire string alone if no anchor is found,
    which is a safer failure mode than silently deleting policy prose.
    """

    raw = (
        "Voting for the monetary policy action were Jerome H. Powell, "
        "Chair; John C. Williams, Vice Chair; and Loretta J. Mester. "
        "The Committee judges that the current stance of monetary policy "
        "is appropriate to return inflation to 2 percent over time."
    )
    cleaned = _strip_voting_roster(raw)
    # The post-roster sentence MUST be preserved. We accept that the
    # roster names may still be present in this anchor-less variant —
    # the alternative (greedy '.*$' eating policy text) is worse.
    assert "The Committee judges that the current stance" in cleaned
    assert "return inflation to 2 percent" in cleaned


def test_voting_roster_minutes_form_is_stripped():
    """Minutes write the voting block as 'Voting for this action: ...
    Voting against this action: <None|name>'. The statement-only regex
    never matched this form. The minutes-form regex below covers it.
    """

    raw = (
        "The Committee decided to maintain the target range. "
        "Voting for this action: Jerome H. Powell, John C. Williams, "
        "Michelle W. Bowman, Lael Brainard, Richard H. Clarida, Patrick "
        "Harker, Robert S. Kaplan, Neel Kashkari, Loretta J. Mester, and "
        "Randal K. Quarles. Voting against this action: None. Consistent "
        "with the Committee's decision, the Board of Governors voted "
        "unanimously to raise the interest rate."
    )
    cleaned = _strip_voting_roster(raw)
    # Member roster is gone.
    assert "Jerome H. Powell" not in cleaned
    assert "Randal K. Quarles" not in cleaned
    # The "None" dissent line is dropped too — no signal to keep.
    assert "Voting against this action: None" not in cleaned
    # The lead-in policy sentence and the post-roster continuation both
    # survive intact.
    assert "Committee decided to maintain the target range" in cleaned
    assert "Board of Governors voted unanimously to raise" in cleaned


def test_voting_roster_minutes_form_preserves_dissent():
    raw = (
        "Voting for this action: Jerome H. Powell, John C. Williams, and "
        "Randal K. Quarles. Voting against this action: Loretta J. Mester "
        "President Mester was fully supportive of all of the actions taken "
        "to promote the smooth functioning of markets and the flow of "
        "credit to households and businesses but preferred to reduce the "
        "target range for the federal funds rate to 1/2 to 3/4 percent."
    )
    cleaned = _strip_voting_roster(raw)
    assert "Jerome H. Powell" not in cleaned
    assert "Randal K. Quarles" not in cleaned
    # Dissent half is kept verbatim.
    assert "Voting against this action: Loretta J. Mester" in cleaned
    assert "preferred to reduce the target range" in cleaned


def test_stay_connected_as_verb_phrase_is_preserved():
    """'stay connected' is ordinary prose when not adjacent to the
    social-media footer chrome.
    """

    raw = (
        "Regional Reserve Bank presidents continue to stay connected with "
        "community banks and small business owners across their districts."
    )
    cleaned = _strip_nav_chrome(raw)
    assert "stay connected with" in cleaned
    assert "community banks" in cleaned


def test_stay_connected_footer_banner_is_still_stripped():
    raw = (
        "Board of Governors of the Federal Reserve System Stay Connected "
        "Federal Reserve Facebook Page Federal Reserve YouTube Page"
    )
    cleaned = _strip_nav_chrome(raw)
    assert "Stay Connected" not in cleaned
    assert "Facebook Page" not in cleaned
    assert "YouTube Page" not in cleaned
