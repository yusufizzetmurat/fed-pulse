"""#443 / #444 default-off byte-identity contract on ``FeatureVector.as_rich_list``.

When ``use_statement_delta`` and ``use_vote_features`` are both off, the
``FeatureVector`` default constructor leaves the two new slots at
``None`` and ``as_rich_list`` must NOT append the two new tails. The
per-bar feature width therefore stays at the legacy ``RICH_FEATURE_SIZE``
(or the regime / SEP widened width when those legacy opt-ins are on),
which is the structural lock the canonical-sweep byte-identity contract
relies on for #443 / #444 to ship without invalidating pre-merge
checkpoints.
"""

from __future__ import annotations

from app.models.config import (
    FeatureVector,
    RICH_FEATURE_SIZE,
    RICH_MACRO_REGIME_DIM,
    RICH_MACRO_REGIME_MISSING_DIM,
    RICH_PRESS_CONF_DIM,
    RICH_SEP_DIM,
    RICH_SEP_MISSING_DIM,
    RICH_STATEMENT_DELTA_DIM,
    RICH_STATEMENT_DELTA_MISSING_DIM,
    RICH_VOTE_FEATURES_DIM,
    RICH_VOTE_FEATURES_MISSING_DIM,
)


def _bare_vector() -> FeatureVector:
    return FeatureVector(
        date="2024-05-01",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.012,
    )


def test_default_vector_emits_rich_feature_size_only() -> None:
    """No opt-in payload → as_rich_list width is RICH_FEATURE_SIZE.

    This is the byte-identity guarantee: a checkpoint trained under
    the default (flags off) sees the exact same per-bar width #443 /
    #444 are added against. Any future change that accidentally appends
    one of the new tails on the default path will break this assertion.
    """

    vector = _bare_vector()
    rich = vector.as_rich_list()
    assert len(rich) == RICH_FEATURE_SIZE


def test_statement_delta_slot_appends_tail_when_populated() -> None:
    """Width-and-position contract for the #443 tail.

    The tail lands at the very end of the emitted list (after any
    regime / SEP tails when those are on); on the bare path it sits
    immediately after the standard RICH_FEATURE_SIZE slice. Width is
    ``RICH_STATEMENT_DELTA_DIM + RICH_STATEMENT_DELTA_MISSING_DIM``.
    """

    vector = _bare_vector()
    vector.statement_delta_embedding = [0.1] * RICH_STATEMENT_DELTA_DIM
    vector.statement_delta_embedding_missing = 0.0
    rich = vector.as_rich_list()
    expected = (
        RICH_FEATURE_SIZE
        + RICH_STATEMENT_DELTA_DIM
        + RICH_STATEMENT_DELTA_MISSING_DIM
    )
    assert len(rich) == expected
    # Last position carries the missing flag.
    assert rich[-1] == 0.0


def test_vote_features_slot_appends_tail_when_populated() -> None:
    vector = _bare_vector()
    vector.vote_features = [10.0 / 12.0, 1.0 / 12.0, 0.0, 1.0]
    vector.vote_features_missing = 0.0
    rich = vector.as_rich_list()
    expected = (
        RICH_FEATURE_SIZE
        + RICH_VOTE_FEATURES_DIM
        + RICH_VOTE_FEATURES_MISSING_DIM
    )
    assert len(rich) == expected
    assert rich[-1] == 0.0


def test_statement_delta_uniform_width_across_populated_and_missing() -> None:
    """Regression: opt-in row with missing data must match populated width.

    Before #524, an event opted-in via ``use_statement_delta`` but
    lacking a strict-prior statement left
    ``statement_delta_embedding=None`` and ``as_rich_list`` skipped the
    tail entirely. Statement events emitted +769 dims, non-statement
    events emitted +0 → ragged ``torch.tensor`` build at sweep time.
    The loader now zero-fills the slot whenever the flag is on, so the
    two FeatureVectors below must produce identical-width rich lists.
    """

    populated = _bare_vector()
    populated.statement_delta_embedding = [0.1] * RICH_STATEMENT_DELTA_DIM
    populated.statement_delta_embedding_missing = 0.0
    missing = _bare_vector()
    missing.statement_delta_embedding = [0.0] * RICH_STATEMENT_DELTA_DIM
    missing.statement_delta_embedding_missing = 1.0
    assert len(populated.as_rich_list()) == len(missing.as_rich_list())


def test_vote_features_uniform_width_across_populated_and_missing() -> None:
    populated = _bare_vector()
    populated.vote_features = [0.83, 0.08, 0.0, 1.0]
    populated.vote_features_missing = 0.0
    missing = _bare_vector()
    missing.vote_features = [0.0] * RICH_VOTE_FEATURES_DIM
    missing.vote_features_missing = 1.0
    assert len(populated.as_rich_list()) == len(missing.as_rich_list())


def test_press_conf_uniform_width_across_populated_and_missing() -> None:
    populated = _bare_vector()
    populated.press_conf_features = [1.0]
    missing = _bare_vector()
    missing.press_conf_features = [0.0] * RICH_PRESS_CONF_DIM
    assert len(populated.as_rich_list()) == len(missing.as_rich_list())


def test_all_opt_in_tails_compose_in_documented_order() -> None:
    """Regime → SEP → statement-delta → vote-features.

    The append order is the canonical four-block sequence. A consumer
    iterating the rich vector under all four opt-ins on knows where
    each block sits without ambiguity.
    """

    vector = _bare_vector()
    vector.macro_regime_features = [0.0] * RICH_MACRO_REGIME_DIM
    vector.macro_regime_features_missing = 0.0
    vector.sep_features = [0.0] * RICH_SEP_DIM
    vector.sep_features_missing = 0.0
    vector.statement_delta_embedding = [0.0] * RICH_STATEMENT_DELTA_DIM
    vector.statement_delta_embedding_missing = 0.0
    vector.vote_features = [0.0] * RICH_VOTE_FEATURES_DIM
    vector.vote_features_missing = 0.0
    rich = vector.as_rich_list()
    expected = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
        + RICH_STATEMENT_DELTA_DIM
        + RICH_STATEMENT_DELTA_MISSING_DIM
        + RICH_VOTE_FEATURES_DIM
        + RICH_VOTE_FEATURES_MISSING_DIM
    )
    assert len(rich) == expected
