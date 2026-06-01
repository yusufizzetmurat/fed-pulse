"""Tests for the Workspace-spine semantic-diff serving wrapper.

The service composes two views of the change between a pasted FOMC
statement and the most recent strictly-prior statement on disk:

- ``compute_token_spans`` — ordered redline spans across
  ``unchanged | added | removed | substituted``
- ``compute_topic_deltas`` — six-topic emphasis deltas sorted by
  ``abs(delta)`` desc
- ``build_response`` — composes the wire response, including the
  cold-start (no strict-prior) banner case
- ``load_prior_statement`` — strict-prior selector against the
  on-disk statements JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.semantic_diff import (
    TOPIC_PHRASES,
    build_response,
    compute_token_spans,
    compute_topic_deltas,
    load_prior_statement,
)


def _write_statements(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_prior_returns_most_recent_strict_prior(tmp_path: Path) -> None:
    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [
            {"date": "2026-01-28", "text": "older body"},
            {"date": "2026-03-18", "text": "middle body"},
            {"date": "2026-05-01", "text": "same-day body"},
        ],
    )

    prior = load_prior_statement("2026-05-01", path=path)

    assert prior is not None
    # 2026-05-01 itself is excluded by the strict ``<`` filter.
    assert prior.event_date == "2026-03-18"
    assert prior.text == "middle body"


def test_load_prior_returns_none_for_cold_start(tmp_path: Path) -> None:
    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "later body"}],
    )

    assert load_prior_statement("2026-01-28", path=path) is None


def test_load_prior_handles_missing_file(tmp_path: Path) -> None:
    """A missing statements JSON is the fresh-clone case; load returns None."""

    missing = tmp_path / "nope.json"
    assert load_prior_statement("2026-05-01", path=missing) is None


def test_compute_token_spans_emits_unchanged_added_removed_substituted() -> None:
    # Crafted so each opcode lands at least once: ``has eased`` is a
    # pure insertion (sits between equal runs), ``slowly`` is a pure
    # deletion (also sits between equal runs), and ``solid`` ->
    # ``moderate`` is the substitution at the tail.
    prior = "Inflation remains elevated slowly and labor market is solid"
    current = "Inflation has eased remains elevated and labor market is moderate"

    spans = compute_token_spans(prior, current)

    kinds = [span.kind for span in spans]
    assert "unchanged" in kinds
    assert any(span.kind == "added" and "eased" in span.text for span in spans)
    assert any(
        span.kind == "removed" and "slowly" in span.text for span in spans
    )
    assert any(
        span.kind == "substituted"
        and "moderate" in span.text
        and span.paired_text is not None
        and "solid" in span.paired_text
        for span in spans
    )


def test_compute_token_spans_returns_empty_for_empty_inputs() -> None:
    assert compute_token_spans("", "current text") == []
    assert compute_token_spans("prior text", "") == []
    assert compute_token_spans("", "") == []


def test_compute_topic_deltas_returns_six_canonical_topics() -> None:
    prior = (
        "Inflation has eased but remains elevated. The labor market is strong "
        "and economic activity continues to expand at a moderate pace."
    )
    current = (
        "Inflation remains elevated. The Committee is firmly committed to "
        "returning inflation to its 2 percent objective. The labor market is "
        "strong."
    )

    topics = compute_topic_deltas(prior, current)

    # All six canonical topics, ordered by abs(delta) desc.
    assert len(topics) == len(TOPIC_PHRASES)
    assert {row.topic for row in topics} == set(TOPIC_PHRASES.keys())
    deltas = [abs(row.delta) for row in topics]
    assert deltas == sorted(deltas, reverse=True)
    # Each emphasis share is in [0, 1].
    for row in topics:
        assert 0.0 <= row.current_emphasis <= 1.0
        assert 0.0 <= row.prior_emphasis <= 1.0


def test_compute_topic_deltas_with_no_topic_hits_emits_zero_shares() -> None:
    """A document with no topic phrases still emits the six rows at zero."""

    # Both bodies clear the MIN_INPUT_TOKENS gate but use vocabulary
    # the canonical topic phrase list doesn't cover, so every row
    # should land at zero rather than tripping the silent-null guard.
    prior = "alpha beta gamma delta epsilon zeta"
    current = "eta theta iota kappa lambda mu"

    topics = compute_topic_deltas(prior, current)

    assert len(topics) == len(TOPIC_PHRASES)
    for row in topics:
        assert row.prior_emphasis == 0.0
        assert row.current_emphasis == 0.0
        assert row.delta == 0.0


def test_compute_topic_deltas_sample_phrases_come_from_current_document() -> None:
    """The phrase chips reflect *what's in the current statement*, not the prior."""

    prior = "Inflation remains elevated."
    current = "Labor market and employment data show payrolls held steady."

    topics = compute_topic_deltas(prior, current)
    labor = next(row for row in topics if row.topic == "Labor")

    assert labor.sample_phrases  # at least one phrase
    # Phrases all come from the labor topic's curated list.
    for phrase in labor.sample_phrases:
        assert phrase in TOPIC_PHRASES["Labor"]


def test_build_response_cold_start_returns_banner_summary(tmp_path: Path) -> None:
    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "later only"}],
    )

    out = build_response(
        "2026-01-28",
        # Past MIN_INPUT_TOKENS so cold-start fires, not the no_input
        # short-circuit.
        "Some pasted statement text that runs past the gate.",
        path=path,
    )

    assert out.prior_date == ""
    assert out.token_spans == []
    assert out.topic_deltas == []
    assert "Earliest statement" in out.summary
    assert out.status == "no_prior"


def test_build_response_with_prior_populates_spans_and_topics(tmp_path: Path) -> None:
    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [
            {
                "date": "2026-03-18",
                "text": (
                    "Inflation remains elevated. Labor market is strong. "
                    "Economic activity continues to expand."
                ),
            },
        ],
    )

    out = build_response(
        "2026-05-01",
        (
            "Inflation has eased but remains elevated. The Committee is "
            "firmly committed to returning inflation to its 2 percent "
            "objective. The labor market is strong."
        ),
        path=path,
    )

    assert out.prior_date == "2026-03-18"
    # Spans were produced from the strict-prior body.
    assert out.token_spans
    assert any(span.kind == "added" for span in out.token_spans)
    # Six canonical topics ride alongside.
    assert len(out.topic_deltas) == len(TOPIC_PHRASES)
    # The summary mentions the prior date for context.
    assert "2026-03-18" in out.summary


def test_build_response_with_missing_file_returns_cold_start(tmp_path: Path) -> None:
    """When the statements JSON is absent the response degrades to cold-start."""

    out = build_response(
        "2026-05-01",
        "any current body that exceeds the minimum token gate",
        path=tmp_path / "missing.json",
    )

    assert out.prior_date == ""
    assert out.token_spans == []
    assert out.topic_deltas == []
    assert out.status == "no_prior"


def test_build_response_with_empty_input_returns_no_input_status(
    tmp_path: Path,
) -> None:
    """Empty current_text is silent-null with status=no_input."""

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "Inflation remains elevated."}],
    )

    out = build_response("2026-05-01", "", path=path)

    assert out.status == "no_input"
    assert out.prior_date == ""
    assert out.token_spans == []
    assert out.topic_deltas == []
    # Summary surfaces the token count so the panel can render
    # "Input too short to diff (n=0 tokens)" rather than blanking.
    assert "0 tokens" in out.summary


def test_build_response_with_whitespace_only_input_returns_no_input(
    tmp_path: Path,
) -> None:
    """Whitespace-only current_text never raises and reports no_input."""

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "Inflation remains elevated."}],
    )

    out = build_response("2026-05-01", "   \n\t  ", path=path)

    assert out.status == "no_input"
    assert out.token_spans == []
    assert out.topic_deltas == []


def test_build_response_with_single_token_returns_no_input(
    tmp_path: Path,
) -> None:
    """Single-token bodies trip the MIN_INPUT_TOKENS gate."""

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "Inflation remains elevated."}],
    )

    out = build_response("2026-05-01", "hawkish", path=path)

    assert out.status == "no_input"
    assert out.token_spans == []
    # Summary uses the singular noun on a one-token input.
    assert "1 token" in out.summary
    assert "tokens" not in out.summary.replace("1 token", "")


def test_build_response_with_non_ascii_input_returns_non_english(
    tmp_path: Path,
) -> None:
    """Majority-non-Latin input short-circuits with status=non_english."""

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "Inflation remains elevated."}],
    )

    # A CJK block well past MIN_INPUT_TOKENS — the non-Latin gate
    # must fire before the token-count gate.
    cjk_body = "  ".join(["通货膨胀"] * 8)

    out = build_response("2026-05-01", cjk_body, path=path)

    assert out.status == "non_english"
    assert out.token_spans == []
    assert out.topic_deltas == []
    assert "Non-Latin" in out.summary


def test_compute_token_spans_with_short_input_returns_empty() -> None:
    """compute_token_spans must not raise on the edge-case inputs."""

    assert compute_token_spans("prior body", "hi") == []
    # Non-Latin current body short-circuits to the same empty shape.
    assert compute_token_spans("prior body", "通货膨胀 通货膨胀 通货膨胀") == []


def test_compute_topic_deltas_with_short_input_returns_empty() -> None:
    """compute_topic_deltas must not raise on the edge-case inputs."""

    assert compute_topic_deltas("inflation labor", "hi") == []
    assert compute_topic_deltas("inflation labor", "通货膨胀 通货膨胀 通货膨胀") == []


def test_build_response_ok_path_carries_ok_status(tmp_path: Path) -> None:
    """The happy path tags the response with status=ok."""

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [
            {
                "date": "2026-03-18",
                "text": (
                    "Inflation remains elevated. Labor market is strong. "
                    "Economic activity continues to expand."
                ),
            },
        ],
    )

    out = build_response(
        "2026-05-01",
        (
            "Inflation has eased but remains elevated. The labor market is "
            "strong and economic activity is expanding."
        ),
        path=path,
    )

    assert out.status == "ok"


def test_endpoint_returns_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.services import semantic_diff as semantic_diff_mod

    path = _write_statements(
        tmp_path / "fomc_statements.json",
        [{"date": "2026-03-18", "text": "Inflation remains elevated."}],
    )
    monkeypatch.setattr(
        semantic_diff_mod, "DEFAULT_STATEMENTS_PATH", path
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/fomc/semantic-diff",
        json={
            "current_date": "2026-05-01",
            "current_text": (
                "Inflation has eased but remains elevated. The labor "
                "market is strong."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["current_date"] == "2026-05-01"
    assert body["prior_date"] == "2026-03-18"
    assert isinstance(body["token_spans"], list)
    assert isinstance(body["topic_deltas"], list)
    assert len(body["topic_deltas"]) == len(TOPIC_PHRASES)


def test_endpoint_rejects_bad_iso_date() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main_mod

    client = TestClient(main_mod.app)
    response = client.post(
        "/fomc/semantic-diff",
        json={"current_date": "not-a-date", "current_text": "x"},
    )

    assert response.status_code == 422


def test_endpoint_cold_start_returns_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.services import semantic_diff as semantic_diff_mod

    # Point the service at an empty corpus so any date triggers cold-start.
    path = _write_statements(tmp_path / "fomc_statements.json", [])
    monkeypatch.setattr(semantic_diff_mod, "DEFAULT_STATEMENTS_PATH", path)

    client = TestClient(main_mod.app)
    response = client.post(
        "/fomc/semantic-diff",
        json={
            "current_date": "2026-05-01",
            # Past MIN_INPUT_TOKENS so the no_input gate doesn't fire
            # ahead of the cold-start path.
            "current_text": (
                "anything that exceeds the minimum input gate token count"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prior_date"] == ""
    assert body["token_spans"] == []
    assert body["topic_deltas"] == []
    assert "Earliest statement" in body["summary"]
    assert body["status"] == "no_prior"
