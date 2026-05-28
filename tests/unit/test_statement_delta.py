"""Unit coverage for :mod:`app.data.statement_delta` (#443)."""

from __future__ import annotations

import pytest

from app.data.statement_delta import (
    compute_delta_for_event,
    compute_delta_spans,
    select_prior_statement_text,
)


def test_simple_token_substitution_lands_in_substituted_pairs() -> None:
    prior = "Economic activity has been expanding at a solid pace"
    current = "Economic activity has been expanding at a moderate pace"
    spans = compute_delta_spans(current_text=current, prior_text=prior)
    assert spans is not None
    inserted, deleted, substituted = spans
    assert inserted == ""
    assert deleted == ""
    assert substituted == [("solid", "moderate")]


def test_insert_only_diff_populates_inserted_text() -> None:
    prior = "Inflation remains elevated"
    current = "Inflation has eased but remains elevated"
    spans = compute_delta_spans(current_text=current, prior_text=prior)
    assert spans is not None
    inserted, deleted, substituted = spans
    # "remains" is shared; the prefix "has eased but" is the new span.
    assert "has eased" in inserted
    assert deleted == ""


def test_empty_prior_returns_none() -> None:
    assert compute_delta_spans(current_text="any text", prior_text=None) is None
    assert compute_delta_spans(current_text="any text", prior_text="") is None


def test_compute_delta_for_event_with_encoder_pools_mean() -> None:
    """When ``encode_text`` is supplied, the per-span outputs are
    mean-pooled and surfaced on ``StatementDelta.embedding``. The mean
    is taken over the non-empty channels only."""

    def fake_encoder(text: str) -> list[float]:
        # Returns a deterministic 3-vector based on token count so the
        # test can assert the mean-pool math without standing up the
        # real FinBERT pipeline.
        n_tokens = len(text.split())
        return [float(n_tokens), float(n_tokens) * 2.0, 0.5]

    delta = compute_delta_for_event(
        current_text="rates remained accommodative this meeting",
        prior_text="rates remained restrictive this meeting",
        encode_text=fake_encoder,
    )
    assert delta is not None
    # Substituted span: ("restrictive", "accommodative"). The encoder
    # gets called once on the "[OLD] restrictive [NEW] accommodative"
    # concatenation; no inserted or deleted side fires.
    assert delta.substituted_pairs == [("restrictive", "accommodative")]
    assert delta.embedding is not None
    assert len(delta.embedding) == 3


def test_compute_delta_for_event_without_encoder_leaves_embedding_none() -> None:
    delta = compute_delta_for_event(
        current_text="rates rose",
        prior_text="rates fell",
        encode_text=None,
    )
    assert delta is not None
    assert delta.embedding is None


def test_select_prior_returns_most_recent_strict_prior() -> None:
    index = [
        ("2024-01-31", "first statement body"),
        ("2024-03-20", "second statement body"),
        ("2024-05-01", "third statement body"),
    ]
    prior = select_prior_statement_text(
        event_date="2024-05-01", prior_index=index
    )
    # 2024-05-01 itself is excluded by the strict ``<`` filter.
    assert prior == "second statement body"


def test_select_prior_returns_none_when_cold_start() -> None:
    index = [("2024-03-20", "later statement")]
    assert (
        select_prior_statement_text(event_date="2024-01-31", prior_index=index)
        is None
    )


def test_select_prior_silently_filters_same_date_prior() -> None:
    """The caller passes the full preferred-statement index, which
    includes the supervised event's own row. The selector silently
    skips same-date entries so it can run as a pre-filter step --
    the strict ``<`` filter is the guarantee this helper enforces."""

    index = [
        ("2024-03-20", "older prior body"),
        ("2024-05-01", "same-date body (the event itself)"),
    ]
    prior = select_prior_statement_text(
        event_date="2024-05-01", prior_index=index
    )
    assert prior == "older prior body"


def test_select_prior_returns_none_when_only_same_date_available() -> None:
    """No strict-prior entry → None (cold-start path)."""

    index = [("2024-05-01", "same-date only")]
    assert (
        select_prior_statement_text(event_date="2024-05-01", prior_index=index)
        is None
    )
