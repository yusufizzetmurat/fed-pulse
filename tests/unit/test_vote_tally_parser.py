"""Unit coverage for :mod:`app.data.vote_tally` (#444)."""

from __future__ import annotations

import pytest

from app.data.vote_tally import VoteTally, parse_vote_tally


# Standard FOMC statement vote-block templates. The wording follows the
# real post-2008 statements (vote tally + dissent explanation paragraph).


UNANIMOUS_BODY = """
The Committee maintained the target range for the federal funds rate.

Voting for the FOMC monetary policy action were: Jerome H. Powell,
Chair; John C. Williams, Vice Chair; Michael S. Barr; Michelle W. Bowman;
Lisa D. Cook; Austan D. Goolsbee; Patrick Harker; Philip N. Jefferson;
Adriana D. Kugler; Lorie K. Logan; Christopher J. Waller.
"""

HAWKISH_DISSENT_BODY = """
The Committee decided to maintain the target range for the federal funds
rate at 5-1/4 to 5-1/2 percent.

Voting for the action were: Jerome H. Powell, Chair; John C. Williams,
Vice Chair; Lisa D. Cook; Mary C. Daly; Beth M. Hammack; Philip N.
Jefferson; Adriana D. Kugler; Christopher J. Waller.

Voting against the action was Michelle W. Bowman, who preferred a higher
target range for the federal funds rate at this meeting.
"""

DOVISH_DISSENT_BODY = """
The Committee decided to maintain the target range for the federal funds
rate at 5-1/4 to 5-1/2 percent.

Voting for the FOMC monetary policy action were: Jerome H. Powell, Chair;
John C. Williams, Vice Chair; Michael S. Barr; Michelle W. Bowman;
Lisa D. Cook; Mary C. Daly; Philip N. Jefferson; Lorie K. Logan;
Christopher J. Waller.

Voting against this action: Austan D. Goolsbee, who preferred a lower
target range for the federal funds rate at this meeting.
"""


def test_unanimous_vote_yields_zero_dissent() -> None:
    tally = parse_vote_tally(UNANIMOUS_BODY)
    assert tally is not None
    assert tally.is_unanimous is True
    assert tally.dissent_count == 0
    assert tally.dissent_direction is None
    # 11 voters listed above; the parser counts named members.
    assert tally.votes_for == 11
    assert tally.votes_against == 0


def test_hawkish_dissent_classified_correctly() -> None:
    tally = parse_vote_tally(HAWKISH_DISSENT_BODY)
    assert tally is not None
    assert tally.is_unanimous is False
    assert tally.votes_against == 1
    assert tally.dissent_count == 1
    assert tally.dissent_direction == "hawkish_dissent"


def test_dovish_dissent_classified_correctly() -> None:
    tally = parse_vote_tally(DOVISH_DISSENT_BODY)
    assert tally is not None
    assert tally.votes_against == 1
    assert tally.dissent_direction == "dovish_dissent"


def test_empty_text_returns_none() -> None:
    assert parse_vote_tally("") is None
    assert parse_vote_tally(None) is None  # type: ignore[arg-type]


def test_missing_vote_block_returns_none() -> None:
    text = (
        "The Federal Open Market Committee continues to anticipate that "
        "appropriate monetary policy stance will remain accommodative."
    )
    assert parse_vote_tally(text) is None


def test_mixed_dissent_collapses_to_none_direction() -> None:
    """A statement with two dissenters pulling in opposite directions
    must surface ``dissent_direction=None`` rather than committing to
    one side. The parser sees both a hawkish cue and a dovish cue in
    the trailing prose and treats the row as ambiguous."""

    body = """
    Voting for the action were: Jerome H. Powell, Chair; John C. Williams,
    Vice Chair; Lisa D. Cook; Adriana D. Kugler; Christopher J. Waller.

    Voting against the action were Beth M. Hammack, who preferred a
    higher target range, and Austan D. Goolsbee, who preferred a lower
    target range for the federal funds rate.
    """
    tally = parse_vote_tally(body)
    assert tally is not None
    assert tally.votes_against == 2
    assert tally.dissent_direction is None


def test_dataclass_helpers_align() -> None:
    tally = VoteTally(votes_for=9, votes_against=2, dissent_direction="dovish_dissent")
    assert tally.dissent_count == 2
    assert tally.is_unanimous is False
