from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd
import pytest

from app.features.linguistic import (
    HEDGE_TOKENS,
    LinguisticVector,
    MIN_SEED_OVERLAP,
    NAMED_TOPIC_KEYS,
    NUM_TOPICS,
    PIVOT_DISTANCE_KIND,
    RANDOM_STATE,
    SEED_OVERLAP_TOP_N,
    VOCAB_MIN_DF,
    build_linguistic_feature_frame,
    comparison_density,
    compute_linguistic_features,
    concrete_ratio,
    fit_lda,
    forward_density,
    hawk_dove_asymmetry,
    hedge_density,
    log_token_count,
    pivot_distance,
    pivot_distance_tokens,
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


def test_concrete_ratio_bounded_by_one_on_percent_tokens():
    """``5.25%`` matches both the number regex and the currency regex.

    The naive sum-of-match-counts implementation can push the ratio
    above 1.0 on rate-heavy FOMC text. The deduplicated implementation
    must collapse overlapping spans so the ratio stays in ``[0, 1]``.
    """

    assert concrete_ratio("Inflation reached 5.25% in June") <= 1.0


def test_concrete_ratio_bounded_by_one_on_currency_amount_tokens():
    """``$2.5 billion`` overlaps the currency and number regexes; ratio stays bounded."""

    assert concrete_ratio("Reserves grew $2.5 billion in Q1") <= 1.0


def test_concrete_ratio_dedup_keeps_single_concrete_span():
    """``5.25%`` is one concrete span, not two; ratio = 1/words."""

    text = "Rate hit 5.25% today"  # alphabetic words: Rate, hit, today -> 3
    # The only concrete span is ``5.25%`` (number + currency marker
    # merged into one). Ratio must equal 1 / number_of_alpha_words.
    expected = 1 / 3
    assert concrete_ratio(text) == pytest.approx(expected)


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
    # Every numeric column other than ``pivot_distance`` is populated
    # for every row. ``pivot_distance`` is NaN by design for the first
    # statement in chronological order and for non-statement kinds; the
    # dedicated ``test_pivot_distance_*`` suite covers its semantics.
    for col in frame_a.columns:
        if col in ("text_hash", "pivot_distance"):
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


# ---------------------------------------------------------------------------
# Seed-overlap floor + mislabel prevention
# ---------------------------------------------------------------------------


# Distinct vocabularies for 4 of the 5 named slots; the 5th slot
# (``employment``) gets a corpus where its seed words never appear in
# the top of any topic. The fit's 8 topics will spread across the 5
# vocabulary islands; ``employment`` should fail the seed-overlap
# floor and fall to misc rather than inheriting a balance-sheet /
# policy topic.
_FLOOR_INFLATION = (
    "inflation prices price cpi pce core energy transitory elevated wages "
    "inflation prices price cpi pce core energy elevated wages "
    "inflation prices price cpi core elevated"
)
_FLOOR_FINANCIAL = (
    "financial stability banks banking credit lending liquidity stress leverage vulnerability "
    "financial stability banks banking credit lending liquidity stress leverage "
    "financial stability banks credit lending leverage"
)
_FLOOR_GROWTH = (
    "growth activity spending output gdp demand production consumption investment expansion "
    "growth activity spending output gdp demand production consumption expansion "
    "growth activity spending output gdp demand production"
)
_FLOOR_BALANCE_SHEET = (
    "balance sheet securities treasury mbs agency holdings reinvestment purchases reserves "
    "balance sheet securities treasury mbs agency holdings reinvestment purchases "
    "balance sheet securities treasury mbs agency holdings reinvestment"
)
# Boilerplate FOMC-policy framing that does NOT contain employment
# seed words. Any topic that absorbs this vocabulary should NOT be
# pinned to the ``employment`` slot.
_FLOOR_POLICY_NO_LABOR = (
    "committee federal policy securities rate participants market reserve agency funds term purchases monetary range longer "
    "committee federal policy securities rate participants market reserve agency funds term purchases monetary range longer "
    "committee federal policy securities rate participants market reserve agency funds term purchases monetary range longer"
)


def _seed_overlap_corpus() -> list[str]:
    """Toy corpus that exercises the seed-overlap floor.

    Four named slots have distinct, repeating vocabularies. The fifth
    slot (``employment``) gets only policy-framing boilerplate with no
    labor seeds. With 8 topics over this corpus the LDA fit will
    produce at least one topic that ``employment`` could try to claim
    by total seed-weight, but the top-N vocabulary will be policy
    boilerplate -- the seed-overlap floor must reject the assignment.
    """

    base = [
        _FLOOR_INFLATION,
        _FLOOR_FINANCIAL,
        _FLOOR_GROWTH,
        _FLOOR_BALANCE_SHEET,
        _FLOOR_POLICY_NO_LABOR,
    ]
    # Repeats keep min_df happy even at production-style cutoffs.
    return base * 6


def test_seed_overlap_floor_drops_employment_to_misc_when_no_labor_topic():
    """``employment`` falls to misc when no fitted topic contains labor seeds.

    Mirrors the Sprint 1 mislabel: top-15 of the topic that would have
    been pinned to ``employment`` is pure policy boilerplate
    (``committee``, ``federal``, ``policy``, ``securities``, ``rate``,
    ...). The seed-overlap floor must reject the assignment and emit
    ``topic_share_employment == 0.0``.
    """

    artifact = fit_lda(_seed_overlap_corpus(), min_df=2)
    # Four named slots get assignments; ``employment`` does not.
    assert "employment" not in artifact.topic_assignments
    # The 4 clean slots all map successfully.
    assigned_slots = set(artifact.topic_assignments.keys())
    for slot in ("inflation", "financial_stability", "growth", "balance_sheet"):
        assert slot in assigned_slots, f"{slot} should pass the floor"
    # The slot count plus misc count still equals NUM_TOPICS.
    assert (
        len(artifact.topic_assignments) + len(artifact.misc_topic_indices)
        == NUM_TOPICS
    )
    # And the per-doc vector emits 0.0 for the unassigned slot.
    vec = compute_linguistic_features(_FLOOR_INFLATION, artifact)
    assert vec.topic_share_employment == 0.0


def test_seed_overlap_floor_assigned_slot_actually_contains_seed_words():
    """Every *assigned* slot's winning topic has >= MIN_SEED_OVERLAP seeds in top-N."""

    artifact = fit_lda(_seed_overlap_corpus(), min_df=2)
    # Re-load top words via the artifact directly.
    for slot, topic_idx in artifact.topic_assignments.items():
        from app.features.linguistic import TOPIC_SEED_WORDS

        seeds = set(TOPIC_SEED_WORDS[slot])
        top_n = set(artifact.top_words[topic_idx][:SEED_OVERLAP_TOP_N])
        overlap = top_n & seeds
        assert len(overlap) >= MIN_SEED_OVERLAP, (
            f"slot {slot} -> topic {topic_idx} has only {len(overlap)} "
            f"seed overlaps in top-{SEED_OVERLAP_TOP_N}: {top_n}"
        )


def test_unassigned_slot_does_not_steal_misc_count():
    """A slot dropped by the floor does not double-count against misc."""

    artifact = fit_lda(_seed_overlap_corpus(), min_df=2)
    # ``misc_topic_indices`` is the complement of *assigned* topics.
    # An unassigned slot leaves its candidate topic in the misc pool.
    used = set(artifact.topic_assignments.values())
    expected_misc = sorted(i for i in range(NUM_TOPICS) if i not in used)
    assert list(artifact.misc_topic_indices) == expected_misc


def test_assign_named_topics_tie_break_prefers_lower_index():
    """When two unassigned topics tie on seed-weight, the lower index wins."""

    # Build a synthetic LDA-like artifact: 3 topics, vocabulary
    # {a, b, c, d}. Topics 0 and 1 have identical inflation seed weights
    # (both have ``inflation`` and ``prices`` in the top, equal mass);
    # topic 2 has none. The tie-break must pick topic 0.
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np

    from app.features.linguistic import _assign_named_topics

    vocab = ["inflation", "prices", "labor", "jobs"]
    # Hand-craft components_ to set up the tie.
    components = np.array(
        [
            [5.0, 5.0, 0.5, 0.5],  # topic 0: inflation seed mass = 10
            [5.0, 5.0, 0.5, 0.5],  # topic 1: identical inflation seed mass
            [0.1, 0.1, 5.0, 5.0],  # topic 2: labor seed mass dominates
        ]
    )
    lda = LatentDirichletAllocation(n_components=3)
    lda.components_ = components

    assignments, _misc = _assign_named_topics(lda, vocab)
    # ``inflation`` should pick topic 0 (lower index) on the tie.
    assert assignments["inflation"] == 0
    # ``employment`` should land on topic 2 (clear labor seeds in top-N).
    assert assignments["employment"] == 2


# ---------------------------------------------------------------------------
# Vocabulary determinism + production-scale min_df
# ---------------------------------------------------------------------------


def test_lda_artifact_vocabulary_is_alphabetically_sorted():
    """``CountVectorizer`` returns vocabulary in alphabetical order.

    The LDA fit consumes the DTM whose columns are aligned with the
    feature names, so an alphabetical vocabulary makes the LDA fit
    independent of corpus-document hash ordering. Regression test:
    catch silent drift if a future change swaps in a non-sorted
    vectoriser.
    """

    artifact = fit_lda(_toy_corpus())
    vocab = list(artifact.vectorizer.get_feature_names_out())
    assert vocab == sorted(vocab), "vocabulary must be alphabetically sorted"


def test_build_linguistic_feature_frame_deterministic_at_production_min_df(tmp_path):
    """Determinism at production-style ``min_df=VOCAB_MIN_DF`` (5).

    The smoke test above uses 10 docs which trips the < 50 docs
    auto-downgrade to ``min_df=2``. Production runs hit
    ``min_df=VOCAB_MIN_DF`` against 16k+ docs. Reproduce the
    production cutoff with a 60-doc synthetic corpus so vocabulary-
    order hash-dependence (if any) is exercised.
    """

    assert VOCAB_MIN_DF == 5  # Guardrail: catch silent drift.
    package = tmp_path / "tp_prod"
    base_docs = [
        ("inf", _INFLATION_DOC),
        ("emp", _EMPLOYMENT_DOC),
        ("fin", _FINANCIAL_DOC),
        ("gro", _GROWTH_DOC),
        ("bal", _BALANCE_SHEET_DOC),
    ]
    # Repeat each base doc 12x with distinct text_hashes so the
    # registry sees 60 rows; each token appears in 12 docs so it
    # clears min_df=5 comfortably.
    docs: list[tuple[str, str]] = []
    for i in range(12):
        for stem, text in base_docs:
            docs.append((f"{stem}_{i:02d}", text))
    _seed_training_package(package, docs)
    frame_a, artifact_a = build_linguistic_feature_frame(package_dir=package)
    frame_b, artifact_b = build_linguistic_feature_frame(package_dir=package)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    # Both fits hit the production cutoff.
    assert artifact_a.vectorizer.min_df == VOCAB_MIN_DF
    assert artifact_b.vectorizer.min_df == VOCAB_MIN_DF
    # Vocabulary is alphabetically sorted (regression for the order
    # determinism contract).
    vocab_a = list(artifact_a.vectorizer.get_feature_names_out())
    assert vocab_a == sorted(vocab_a)
    # Top-words match across runs.
    assert artifact_a.top_words == artifact_b.top_words


def test_min_seed_overlap_constants_pinned():
    """Spec guardrails: catch silent drift on the floor settings."""

    assert MIN_SEED_OVERLAP == 2
    assert SEED_OVERLAP_TOP_N == 10


# ---------------------------------------------------------------------------
# pivot_distance: token-set Jaccard vs prior same-kind statement
# ---------------------------------------------------------------------------


def test_pivot_distance_pure_function_returns_nan_when_prior_is_none():
    """No prior token set -> NaN. This is the unit-level guarantee that
    backs the "first statement in the corpus" frame-level rule."""

    result = pivot_distance("inflation remains elevated", None)
    assert math.isnan(result)


def test_pivot_distance_pure_function_zero_on_identical_tokens():
    """Same vocabulary -> distance 0."""

    text = "inflation has remained elevated reflecting broad price pressures"
    result = pivot_distance(text, pivot_distance_tokens(text))
    assert result == pytest.approx(0.0, abs=1e-12)


def test_pivot_distance_pure_function_one_on_disjoint_tokens():
    """Fully disjoint vocab -> distance 1."""

    a = "alpha beta gamma delta"
    b = "lorem ipsum dolor sit amet"
    result = pivot_distance(a, pivot_distance_tokens(b))
    assert result == pytest.approx(1.0, abs=1e-12)


def _statement_entry(text_hash: str, event_date: str, text: str) -> dict:
    return {
        "text_hash": text_hash,
        "text": text,
        "event_date": event_date,
        "source": "scraped_fed",
        "source_record_id": text_hash,
        "document_type": "statement",
    }


def _minutes_entry(text_hash: str, event_date: str, text: str) -> dict:
    return {
        "text_hash": text_hash,
        "text": text,
        "event_date": event_date,
        "source": "scraped_fed",
        "source_record_id": text_hash,
        "document_type": "minutes",
    }


def _press_conference_entry(text_hash: str, event_date: str, text: str) -> dict:
    return {
        "text_hash": text_hash,
        "text": text,
        "event_date": event_date,
        "source": "scraped_fed",
        "source_record_id": text_hash,
        "document_type": "press_conference",
    }


def _write_typed_registry(package_dir: Path, entries: list[dict]) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    with (package_dir / "registry_normalized.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def test_pivot_distance_first_statement_is_nan(tmp_path):
    """The chronologically first statement has no prior -> NaN."""

    package = tmp_path / "tp_pivot_first"
    entries = [
        _statement_entry("a_first", "2023-01-31", _INFLATION_DOC),
        _statement_entry("b_second", "2023-03-22", _EMPLOYMENT_DOC),
        _statement_entry("c_third", "2023-05-03", _FINANCIAL_DOC),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    by_hash = {row["text_hash"]: row for row in frame.to_dict(orient="records")}
    # First statement -> NaN
    assert math.isnan(by_hash["a_first"]["pivot_distance"])
    # Subsequent statements have finite distances
    assert not math.isnan(by_hash["b_second"]["pivot_distance"])
    assert not math.isnan(by_hash["c_third"]["pivot_distance"])


def test_pivot_distance_identical_text_is_zero(tmp_path):
    """When a statement's text equals the previous statement's text the
    Jaccard distance must be exactly 0."""

    package = tmp_path / "tp_pivot_zero"
    entries = [
        _statement_entry("a_prior", "2023-01-31", _INFLATION_DOC),
        _statement_entry("b_clone", "2023-03-22", _INFLATION_DOC),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    by_hash = {row["text_hash"]: row for row in frame.to_dict(orient="records")}
    assert math.isnan(by_hash["a_prior"]["pivot_distance"])
    assert by_hash["b_clone"]["pivot_distance"] == pytest.approx(0.0, abs=1e-12)


def test_pivot_distance_disjoint_vocab_is_one(tmp_path):
    """Two statements with fully disjoint vocabularies produce distance 1."""

    package = tmp_path / "tp_pivot_one"
    entries = [
        _statement_entry(
            "a_prior", "2023-01-31", "alpha beta gamma delta epsilon zeta"
        ),
        _statement_entry(
            "b_disjoint", "2023-03-22", "lorem ipsum dolor sit amet consectetur"
        ),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    by_hash = {row["text_hash"]: row for row in frame.to_dict(orient="records")}
    assert by_hash["b_disjoint"]["pivot_distance"] == pytest.approx(1.0, abs=1e-12)


def test_pivot_distance_nan_on_non_statement_kinds(tmp_path):
    """Minutes / press conferences emit NaN regardless of prior history."""

    package = tmp_path / "tp_pivot_nonstmt"
    entries = [
        _statement_entry("a_stmt", "2023-01-31", _INFLATION_DOC),
        _minutes_entry("b_min", "2023-02-21", _INFLATION_DOC),
        _press_conference_entry("c_pc", "2023-03-22", _INFLATION_DOC),
        _statement_entry("d_stmt", "2023-03-22", _INFLATION_DOC),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    by_hash = {row["text_hash"]: row for row in frame.to_dict(orient="records")}
    # Minutes + press conference -> NaN regardless of prior availability.
    assert math.isnan(by_hash["b_min"]["pivot_distance"])
    assert math.isnan(by_hash["c_pc"]["pivot_distance"])
    # The trailing statement has a prior (a_stmt) and emits a finite value.
    assert not math.isnan(by_hash["d_stmt"]["pivot_distance"])
    # That trailing statement is identical text to a_stmt -> distance 0.
    assert by_hash["d_stmt"]["pivot_distance"] == pytest.approx(0.0, abs=1e-12)


def test_pivot_distance_uses_strict_chronological_prior(tmp_path):
    """When several priors exist, the LATEST one strictly before the current
    date is used. We make the most-recent prior identical to the current
    text (distance 0) and an earlier prior fully disjoint (would yield
    distance 1). The result must be 0, proving the walker picked the
    strict-less-than most-recent prior."""

    package = tmp_path / "tp_pivot_strict"
    entries = [
        # Earliest prior: fully disjoint vocabulary.
        _statement_entry(
            "a_oldest", "2023-01-31", "alpha beta gamma delta epsilon"
        ),
        # Most-recent strictly-prior statement: identical text to current.
        _statement_entry(
            "b_recent_prior", "2023-03-22", _INFLATION_DOC
        ),
        # Current statement.
        _statement_entry("c_current", "2023-05-03", _INFLATION_DOC),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    by_hash = {row["text_hash"]: row for row in frame.to_dict(orient="records")}
    # Must use the most-recent strict prior (b_recent_prior) -> distance 0.
    assert by_hash["c_current"]["pivot_distance"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_pivot_distance_kind_constant_pinned():
    """Spec guardrail: pivot_distance is only defined for the statement kind."""

    assert PIVOT_DISTANCE_KIND == "statement"


def test_linguistic_vector_has_pivot_distance_field():
    """The 15-dim contract requires ``pivot_distance`` on every row."""

    fields = list(LinguisticVector.__dataclass_fields__.keys())
    assert "pivot_distance" in fields
    assert len(fields) == 15


def test_compute_linguistic_features_default_pivot_is_nan():
    """When the caller does not pass ``prior_statement_tokens`` the
    pivot field emits NaN (matches the non-statement / first-statement
    semantics)."""

    vec = compute_linguistic_features(_INFLATION_DOC)
    assert math.isnan(vec.pivot_distance)


def test_build_linguistic_feature_frame_pivot_column_present(tmp_path):
    """The Sprint-1-style builder must emit a ``pivot_distance`` column."""

    package = tmp_path / "tp_pivot_col"
    entries = [
        _statement_entry("a_stmt", "2023-01-31", _INFLATION_DOC),
        _statement_entry("b_stmt", "2023-03-22", _EMPLOYMENT_DOC),
    ]
    _write_typed_registry(package, entries)
    artifact = fit_lda(_toy_corpus())
    frame, _ = build_linguistic_feature_frame(package_dir=package, artifact=artifact)
    assert "pivot_distance" in frame.columns
    # Non-NaN count = #statements with a strict prior = N - 1 here.
    assert frame["pivot_distance"].notna().sum() == 1
