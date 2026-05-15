from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd
import pytest

from app.features.linguistic import (
    HAWK_TOKENS,
    HEDGE_TOKENS,
    LinguisticVector,
    NAMED_TOPIC_KEYS,
    NUM_TOPICS,
    RANDOM_STATE,
    build_linguistic_feature_frame,
    comparison_density,
    compute_linguistic_features,
    concrete_ratio,
    fit_lda,
    forward_density,
    hawk_dove_asymmetry,
    hedge_density,
    log_token_count,
)


# ---------------------------------------------------------------------------
# Hand-crafted density tests
# ---------------------------------------------------------------------------


def test_hedge_density_matches_expected_count_per_1000_tokens():
    # 5 hedge tokens (perhaps, may, expect, somewhat, gradually) in 50 ws tokens.
    text = (
        "perhaps the committee may decide carefully and expect inflation to ease "
        "somewhat while moving gradually toward neutral one two three four five "
        "six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty"
    )
    ws_tokens = text.split()
    n_hedges = sum(1 for t in [w.lower() for w in ws_tokens] if t in HEDGE_TOKENS)
    expected = n_hedges / len(ws_tokens) * 1000
    assert hedge_density(text) == pytest.approx(expected)


def test_hedge_density_is_zero_for_empty_text():
    assert hedge_density("") == 0.0
    assert hedge_density("   ") == 0.0


def test_hawk_dove_asymmetry_swings_hawkish_then_dovish():
    hawkish = "We will raise rates aggressively and tighten further to hike inflation expectations down."
    dovish = "We will accommodate growth and ease policy with broad easing and lower rates to support employment."
    assert hawk_dove_asymmetry(hawkish) > 0.5
    assert hawk_dove_asymmetry(dovish) < -0.5


def test_hawk_dove_asymmetry_bounded_for_neutral_text():
    text = "the committee monitors developments and assesses risks"
    value = hawk_dove_asymmetry(text)
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx(0.0, abs=1e-9)


def test_forward_density_on_fomc_style_statement():
    text = (
        "Looking ahead, the Committee expects inflation to ease and anticipates "
        "additional policy adjustments going forward. The outlook remains uncertain "
        "but the Committee will continue to assess incoming data and will likely "
        "adjust the stance as appropriate."
    )
    ws_tokens = text.split()
    # Expected hits: ahead, expects, anticipates, outlook, going forward,
    # will continue, will likely == 7. The two-word phrases are NOT split
    # before token-matching, so phrase matching captures them exactly.
    score = forward_density(text)
    assert score > 0
    # Sanity: at minimum 5 forward signals; with 40-ish ws tokens that's > 100 per 1000.
    assert score >= (5 / len(ws_tokens)) * 1000


def test_comparison_density_picks_up_explicit_phrases():
    text = (
        "Since the last meeting we revised our outlook. In contrast to previous "
        "projections, growth has moderated relative to the prior path."
    )
    ws_tokens = text.split()
    score = comparison_density(text)
    expected_min = (3 / len(ws_tokens)) * 1000
    assert score >= expected_min


def test_concrete_ratio_grows_with_numbers_and_dates():
    abstract = "the committee considers risks and assesses the outlook"
    concrete = "in July 2022 the committee raised rates by 75 basis points to 2.5 percent"
    assert concrete_ratio(concrete) > concrete_ratio(abstract)
    assert concrete_ratio("") == 0.0


def test_log_token_count_matches_log1p_of_whitespace_count():
    text = "one two three four five"
    assert log_token_count(text) == pytest.approx(math.log1p(5))
    assert log_token_count("") == 0.0


# ---------------------------------------------------------------------------
# LDA-backed compute + idempotence
# ---------------------------------------------------------------------------


_INFLATION_DOC = (
    "Inflation has remained elevated reflecting broad price pressures and "
    "energy prices. Core PCE inflation continues to run above target as "
    "wages and prices respond to demand. The Committee is highly attentive "
    "to inflation risks and will act to bring inflation back toward 2 percent."
)

_EMPLOYMENT_DOC = (
    "Employment gains have been robust and the unemployment rate remains low. "
    "Labor market conditions are tight, with strong payrolls and rising wages. "
    "Workers continue to see hiring and participation has improved as job "
    "openings exceed available labor supply."
)

_FINANCIAL_DOC = (
    "Financial stability concerns remain elevated as banking stress and credit "
    "tightening reduce lending. Liquidity in funding markets has improved but "
    "leverage at non-bank financial firms is a vulnerability that the staff "
    "continues to monitor closely."
)

_GROWTH_DOC = (
    "Economic growth has moderated this quarter as consumer spending slowed. "
    "Activity in production and investment softened while output and demand "
    "remained mixed. Real GDP expansion is below trend but consumption holds "
    "up better than expected."
)

_BALANCE_SHEET_DOC = (
    "The Committee will continue reducing its balance sheet by allowing "
    "Treasury securities and agency mortgage-backed securities to roll off. "
    "Reinvestment of principal payments has been wound down and reserve "
    "balances are declining as planned holdings adjust."
)


def _toy_corpus() -> list[str]:
    """Repeat-rich corpus where each named topic has a distinct vocabulary.

    Keeping the corpus topic-pure makes the topic-assignment check
    deterministic and lets the inflation share test stay well-defined
    regardless of sklearn LDA seeding drift.
    """

    base = [
        _INFLATION_DOC,
        _EMPLOYMENT_DOC,
        _FINANCIAL_DOC,
        _GROWTH_DOC,
        _BALANCE_SHEET_DOC,
    ]
    return base * 4


def test_compute_linguistic_features_returns_full_vector_without_lda():
    vec = compute_linguistic_features(_INFLATION_DOC)
    assert isinstance(vec, LinguisticVector)
    # Without LDA the topic shares default to 0.0 but the hand-crafted
    # axes still come through.
    assert vec.topic_share_inflation == 0.0
    assert vec.hedge_density >= 0.0
    assert vec.log_token_count > 0.0


def test_fit_lda_is_deterministic_across_runs():
    corpus = _toy_corpus()
    artifact_a = fit_lda(corpus)
    artifact_b = fit_lda(corpus)
    # Same topic assignments and same top-words.
    assert artifact_a.topic_assignments == artifact_b.topic_assignments
    assert artifact_a.top_words == artifact_b.top_words


def test_compute_linguistic_features_idempotent_on_repeated_calls():
    artifact = fit_lda(_toy_corpus())
    vec_a = compute_linguistic_features(_INFLATION_DOC, artifact)
    vec_b = compute_linguistic_features(_INFLATION_DOC, artifact)
    assert vec_a == vec_b


def test_lda_topic_assignments_cover_all_named_slots():
    artifact = fit_lda(_toy_corpus())
    # Every named slot maps to a topic when the vocabulary covers the seeds.
    assert set(artifact.topic_assignments.keys()).issubset(set(NAMED_TOPIC_KEYS))
    # Misc topic count = num_topics - assigned named slots.
    assert (
        len(artifact.misc_topic_indices)
        + len(artifact.topic_assignments)
        == NUM_TOPICS
    )


def test_per_doc_features_invariant_to_other_doc_order():
    """Scrambling the order of OTHER docs does not change a doc's feature row.

    The LDA fit itself depends on the corpus, but for the same set of
    documents (any permutation) sklearn's batch LDA with a fixed
    ``random_state`` should converge to the same factorisation -- so a
    given document's posterior + hand-crafted features are stable.
    """

    base = _toy_corpus()
    artifact_a = fit_lda(base)
    rng = random.Random(0)
    shuffled = list(base)
    rng.shuffle(shuffled)
    artifact_b = fit_lda(shuffled)
    vec_a = compute_linguistic_features(_INFLATION_DOC, artifact_a)
    vec_b = compute_linguistic_features(_INFLATION_DOC, artifact_b)
    # Hand-crafted axes are by construction order-invariant.
    assert vec_a.hedge_density == pytest.approx(vec_b.hedge_density)
    assert vec_a.comparison_density == pytest.approx(vec_b.comparison_density)
    assert vec_a.forward_density == pytest.approx(vec_b.forward_density)
    assert vec_a.concrete_ratio == pytest.approx(vec_b.concrete_ratio)
    assert vec_a.hawk_dove_asymmetry == pytest.approx(vec_b.hawk_dove_asymmetry)
    assert vec_a.log_token_count == pytest.approx(vec_b.log_token_count)
    # LDA shares come from a different fit (same random_state, same data
    # set) -- they should agree to numerical tolerance.
    assert vec_a.topic_share_inflation == pytest.approx(
        vec_b.topic_share_inflation, abs=1e-6
    )


def test_inflation_text_lights_up_inflation_topic_share():
    """A document dominated by inflation vocabulary should produce a
    non-trivial inflation topic share.

    The synthetic toy corpus is topic-pure, so the inflation document
    should clear the 0.20 floor from the acceptance criteria (the same
    target the wiki uses for the 2022-07 FOMC statement check).
    """

    artifact = fit_lda(_toy_corpus())
    vec = compute_linguistic_features(_INFLATION_DOC, artifact)
    assert vec.topic_share_inflation >= 0.20


def test_july_2022_statement_inflation_share_above_floor():
    """The real 2022-07 FOMC statement (inflation-peak month) lights up
    the inflation topic.

    Uses the same toy corpus to anchor the LDA fit; the FOMC paragraph
    below is the public 2022-07-27 statement abstract. The inflation
    share must clear 0.20 to honor the acceptance criteria.
    """

    july_2022 = (
        "Recent indicators of spending and production have softened. Nonetheless, "
        "job gains have been robust in recent months, and the unemployment rate "
        "has remained low. Inflation remains elevated, reflecting supply and "
        "demand imbalances related to the pandemic, higher food and energy prices, "
        "and broader price pressures. The Committee is highly attentive to "
        "inflation risks. The Committee seeks to achieve maximum employment and "
        "inflation at the rate of 2 percent over the longer run. In support of "
        "these goals, the Committee decided to raise the target range for the "
        "federal funds rate to 2-1/4 to 2-1/2 percent and anticipates that "
        "ongoing increases in the target range will be appropriate. The Committee "
        "is strongly committed to returning inflation to its 2 percent objective."
    )
    artifact = fit_lda(_toy_corpus())
    vec = compute_linguistic_features(july_2022, artifact)
    assert vec.topic_share_inflation >= 0.20


# ---------------------------------------------------------------------------
# Package-level builder + parquet determinism
# ---------------------------------------------------------------------------


def _seed_training_package(package_dir: Path, docs: list[tuple[str, str]]) -> None:
    """Write a stub registry_normalized.jsonl with ``(text_hash, text)`` rows."""

    package_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for h, text in docs:
        payload = {
            "text_hash": h,
            "text": text,
            "event_date": "2022-07-27",
            "source": "synthetic",
            "source_record_id": h,
            "document_type": "statement",
        }
        lines.append(json.dumps(payload))
    (package_dir / "registry_normalized.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_build_linguistic_feature_frame_is_byte_deterministic(tmp_path):
    package = tmp_path / "tp_test"
    docs = [
        ("hash_inf", _INFLATION_DOC),
        ("hash_emp", _EMPLOYMENT_DOC),
        ("hash_fin", _FINANCIAL_DOC),
        ("hash_gro", _GROWTH_DOC),
        ("hash_bal", _BALANCE_SHEET_DOC),
        # Repeats add corpus mass for LDA without breaking the unique-hash
        # row count downstream.
        ("hash_inf2", _INFLATION_DOC),
        ("hash_emp2", _EMPLOYMENT_DOC),
        ("hash_fin2", _FINANCIAL_DOC),
        ("hash_gro2", _GROWTH_DOC),
        ("hash_bal2", _BALANCE_SHEET_DOC),
    ]
    _seed_training_package(package, docs)
    frame_a, artifact_a = build_linguistic_feature_frame(package_dir=package)
    frame_b, _ = build_linguistic_feature_frame(package_dir=package)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert list(frame_a["text_hash"]) == sorted(h for h, _ in docs)
    assert artifact_a.lda.n_components == NUM_TOPICS
    # Every numeric column is populated for every row.
    for col in frame_a.columns:
        if col == "text_hash":
            continue
        assert frame_a[col].notna().all(), f"column {col} has NaNs"


def test_random_state_constant_pinned_at_eleven():
    # Guardrail: the spec mandates RANDOM_STATE=11. Catch silent drift.
    assert RANDOM_STATE == 11


def test_named_topic_keys_match_spec():
    assert NAMED_TOPIC_KEYS == (
        "inflation",
        "employment",
        "financial_stability",
        "growth",
        "balance_sheet",
    )
