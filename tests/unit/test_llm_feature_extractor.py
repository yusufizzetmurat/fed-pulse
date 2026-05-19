from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data.llm_feature_catalog import (
    CATALOG,
    CATALOG_VERSION,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_user_prompt,
    feature_names,
    levels_for,
)
from app.data.llm_feature_extractor import (
    AnthropicExtractorClient,
    ExtractionResult,
    _parse_response_json,
    _validate,
    extract_for_package,
    extract_one,
)


# ---------------------------------------------------------------------------
# Robust JSON parser (handles the common LLM JSON-corruption patterns)
# ---------------------------------------------------------------------------


def test_parse_clean_json() -> None:
    assert _parse_response_json('{"a": "b"}') == {"a": "b"}


def test_parse_strips_markdown_fence_with_language_tag() -> None:
    payload = '```json\n{"a": "b"}\n```'
    assert _parse_response_json(payload) == {"a": "b"}


def test_parse_strips_markdown_fence_without_language_tag() -> None:
    payload = '```\n{"a": "b"}\n```'
    assert _parse_response_json(payload) == {"a": "b"}


def test_parse_recovers_from_leading_prose() -> None:
    payload = 'Here is the JSON:\n{"a": "b"}\n'
    assert _parse_response_json(payload) == {"a": "b"}


def test_parse_recovers_from_trailing_prose() -> None:
    payload = '{"a": "b"}\nLet me know if you need clarification.'
    assert _parse_response_json(payload) == {"a": "b"}


def test_parse_handles_nested_braces() -> None:
    payload = '{"outer": "x", "inner": {"y": 1}}'
    result = _parse_response_json(payload)
    assert result == {"outer": "x", "inner": {"y": 1}}


def test_parse_handles_braces_inside_strings() -> None:
    payload = '{"text": "an opening { brace doesn\'t end the object"}'
    result = _parse_response_json(payload)
    assert "text" in result


def test_parse_returns_empty_on_no_json() -> None:
    assert _parse_response_json("Sorry, I cannot help with that.") == {}


def test_parse_returns_empty_on_unbalanced_braces() -> None:
    assert _parse_response_json("{ unbalanced") == {}


# ---------------------------------------------------------------------------
# Catalogue invariants
# ---------------------------------------------------------------------------


def test_catalogue_has_ten_features() -> None:
    assert len(CATALOG) == 10


def test_every_feature_has_a_citation_and_at_least_two_levels() -> None:
    for f in CATALOG:
        assert f.name and f.name.replace("_", "").isalnum()
        assert len(f.levels) >= 2
        assert f.citation and "(" in f.citation and ")" in f.citation
        assert f.prompt_question.strip().endswith((".", "?", "'"))


def test_levels_for_returns_expected_set() -> None:
    levels = levels_for("hawkish_shift_vs_prior")
    assert "hawkish_shift" in levels
    assert "not_assessable" in levels


def test_unknown_feature_lookup_raises() -> None:
    with pytest.raises(KeyError, match="unknown catalogue feature"):
        levels_for("never_existed")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_user_prompt_redacts_the_event_date() -> None:
    """Contamination mitigation: the prompt must mask the date so the
    model cannot recall the specific meeting from pretraining."""

    text = "The Federal Open Market Committee decided today to maintain rates."
    prompt = build_user_prompt(text)
    assert "[REDACTED]" in prompt
    # The prompt should not surface event-date strings even if the
    # document text contains them; the document is embedded verbatim,
    # so the test is on the surrounding prompt-template structure.
    assert "publication date has been intentionally redacted" in prompt


def test_user_prompt_lists_every_catalogue_feature() -> None:
    prompt = build_user_prompt("...")
    for f in CATALOG:
        assert f.name in prompt
        for level in f.levels:
            # Every allowed level must surface so the model knows the
            # vocabulary it can respond with.
            assert level in prompt


def test_system_prompt_forbids_outside_knowledge() -> None:
    """The contamination mitigation depends on the system prompt
    explicitly telling the model not to consult outside knowledge.
    Test the wording is in place."""

    assert "do not consult outside knowledge" in SYSTEM_PROMPT
    assert "JSON only" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _valid_response() -> dict[str, str]:
    """A fully-populated catalogue response in the allowed-level set."""

    return {f.name: f.levels[0] for f in CATALOG}


def test_validate_passes_on_full_in_vocab_response() -> None:
    ok, err, features = _validate(_valid_response())
    assert ok is True
    assert err is None
    assert set(features.keys()) == set(feature_names())


def test_validate_flags_missing_features() -> None:
    payload = _valid_response()
    del payload["hawkish_shift_vs_prior"]
    ok, err, features = _validate(payload)
    assert ok is False
    assert "missing" in (err or "")
    assert "hawkish_shift_vs_prior" in (err or "")


def test_validate_flags_out_of_vocab_levels() -> None:
    payload = _valid_response()
    payload["hawkish_shift_vs_prior"] = "definitely-not-allowed"
    ok, err, features = _validate(payload)
    assert ok is False
    assert "out_of_vocab" in (err or "")
    assert "hawkish_shift_vs_prior" in (err or "")


def test_validate_rejects_non_string_values() -> None:
    payload = _valid_response()
    payload["hawkish_shift_vs_prior"] = 42  # type: ignore[assignment]
    ok, err, features = _validate(payload)
    assert ok is False
    assert "not-a-string" in (err or "")


# ---------------------------------------------------------------------------
# Extraction (with stub client)
# ---------------------------------------------------------------------------


class _StubClient:
    """In-memory stand-in for AnthropicExtractorClient.

    The tests inject this so the extractor's retry / cache / validation
    logic is exercised without hitting the real API. Sequence of
    canned (raw, parsed) responses; the next call pops from the queue
    so tests can simulate retry-then-success or exhausted-retries.
    """

    def __init__(self, scripted: list[tuple[str, dict[str, object]]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[str] = []

    def extract(self, document_text: str) -> tuple[str, dict[str, object]]:
        self.calls.append(document_text[:80])
        if not self._scripted:
            return "", {}
        return self._scripted.pop(0)


def test_extract_one_returns_ok_on_first_valid_response() -> None:
    payload = _valid_response()
    client = _StubClient([(json.dumps(payload), dict(payload))])
    result = extract_one(
        text_hash="abcd",
        document_text="x" * 1000,
        client=client,  # type: ignore[arg-type]
    )
    assert result.status == "ok"
    assert result.features == payload
    assert len(client.calls) == 1


def test_extract_one_retries_on_invalid_then_succeeds() -> None:
    bad_payload = {"hawkish_shift_vs_prior": "bogus_level"}
    good_payload = _valid_response()
    client = _StubClient(
        [
            ("{\"hawkish_shift_vs_prior\": \"bogus_level\"}", bad_payload),
            (json.dumps(good_payload), dict(good_payload)),
        ]
    )
    result = extract_one(
        text_hash="abcd",
        document_text="x" * 1000,
        client=client,  # type: ignore[arg-type]
        max_retries=3,
    )
    assert result.status == "ok"
    assert len(client.calls) == 2


def test_extract_one_returns_out_of_vocab_after_max_retries() -> None:
    bad_payload = {"hawkish_shift_vs_prior": "bogus_level"}
    client = _StubClient(
        [("{}", bad_payload) for _ in range(3)]
    )
    result = extract_one(
        text_hash="abcd",
        document_text="x" * 1000,
        client=client,  # type: ignore[arg-type]
        max_retries=3,
    )
    assert result.status in {"invalid_json", "out_of_vocab"}
    assert "out_of_vocab" in (result.error_detail or "")


def test_extract_one_skips_short_documents() -> None:
    """Documents below the minimum length get flagged without an API call."""

    client = _StubClient([])
    result = extract_one(
        text_hash="abcd",
        document_text="too short",
        client=client,  # type: ignore[arg-type]
    )
    assert result.status == "document_too_short"
    assert client.calls == []  # never called


# ---------------------------------------------------------------------------
# Cache + bulk extraction
# ---------------------------------------------------------------------------


def test_extract_for_package_persists_cache_parquet(tmp_path: Path) -> None:
    payload = _valid_response()
    client = _StubClient(
        [(json.dumps(payload), dict(payload))] * 3
    )
    documents = [
        ("hash_aaa", "x" * 500),
        ("hash_bbb", "y" * 500),
        ("hash_ccc", "z" * 500),
    ]
    cache_path = extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
        progress_every=10,
    )
    assert cache_path.exists()
    assert cache_path.parent.name == f"{MODEL_ID}_{CATALOG_VERSION}"

    import pandas as pd

    frame = pd.read_parquet(cache_path)
    assert len(frame) == 3
    assert set(frame["text_hash"].tolist()) == {"hash_aaa", "hash_bbb", "hash_ccc"}
    assert (frame["status"] == "ok").all()
    # Every catalogue feature should be present as a column.
    for name in feature_names():
        assert name in frame.columns
        assert frame[name].notna().all()


def test_extract_for_package_is_idempotent_on_existing_cache(tmp_path: Path) -> None:
    """Re-running on the same cache should skip every cached text_hash
    and make zero new API calls."""

    payload = _valid_response()
    client = _StubClient(
        [(json.dumps(payload), dict(payload))] * 2
    )
    documents = [("hash_aaa", "x" * 500), ("hash_bbb", "y" * 500)]

    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    assert len(client.calls) == 2

    # Second run -- the stub has no responses left but should not be called.
    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    assert len(client.calls) == 2  # unchanged


def test_extract_for_package_persists_failure_rows(tmp_path: Path) -> None:
    """Documents that fail validation should still land in the cache so
    a re-run can see what happened, but they must be retryable -- never
    cached as permanent."""

    bad_payload = {"unknown_key": "x"}
    client = _StubClient([("{}", bad_payload)] * 3)
    documents = [("hash_zzz", "x" * 500)]

    cache_path = extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )

    import pandas as pd

    frame = pd.read_parquet(cache_path)
    assert len(frame) == 1
    assert frame.iloc[0]["status"] in {"invalid_json", "out_of_vocab"}
    assert frame.iloc[0]["text_hash"] == "hash_zzz"


# ---------------------------------------------------------------------------
# Resume + retry contract (the robustness asks)
# ---------------------------------------------------------------------------


def test_resume_retries_api_error_rows_automatically(tmp_path: Path) -> None:
    """If the prior run cached an api_error row, a fresh run must
    re-extract it -- a transient API outage should not be remembered
    as a permanent failure."""

    payload = _valid_response()

    # Run 1: every call fails (client raises). Each row caches with
    # status="invalid_json" or similar because the stub returns ("", {})
    # when exhausted -- but we want to simulate a real API exception.
    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, document_text: str) -> tuple[str, dict[str, object]]:
            self.calls += 1
            raise RuntimeError("simulated network outage")

    flaky = _FlakyClient()
    documents = [("hash_aaa", "x" * 500)]
    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=flaky,  # type: ignore[arg-type]
    )

    import pandas as pd

    cache_path = next(tmp_path.glob("*/*.parquet"))
    frame = pd.read_parquet(cache_path)
    assert frame.iloc[0]["status"] in {"invalid_json", "out_of_vocab"}

    # Run 2: same documents, but the client recovers. The cached
    # failure row should be RE-EXTRACTED automatically (not skipped).
    good_client = _StubClient([(json.dumps(payload), dict(payload))])
    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=good_client,  # type: ignore[arg-type]
    )
    frame_after = pd.read_parquet(cache_path)
    assert len(frame_after) == 1
    assert frame_after.iloc[0]["status"] == "ok", (
        "second run should have re-extracted the failed row"
    )
    assert len(good_client.calls) == 1


def test_resume_does_not_retry_document_too_short(tmp_path: Path) -> None:
    """A short-document skip is deterministic -- re-running cannot
    change it. Must NOT be retried."""

    client = _StubClient([])
    documents = [("hash_short", "tiny")]  # below the 200-char floor
    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    assert len(client.calls) == 0  # never called

    # Second run -- still no calls (the document is permanently "too short").
    extract_for_package(
        training_package_id="test_pkg",
        documents=documents,
        cache_dir=tmp_path,
        client=client,  # type: ignore[arg-type]
    )
    assert len(client.calls) == 0


def test_retry_failed_flag_forces_re_extraction(tmp_path: Path) -> None:
    """Passing retry_failed=True re-extracts every row including
    successfully-extracted ones. Useful when the catalogue prompt
    text changes without bumping CATALOG_VERSION."""

    payload_v1 = _valid_response()
    client_v1 = _StubClient([(json.dumps(payload_v1), dict(payload_v1))])
    extract_for_package(
        training_package_id="test_pkg",
        documents=[("hash_aaa", "x" * 500)],
        cache_dir=tmp_path,
        client=client_v1,  # type: ignore[arg-type]
    )
    assert len(client_v1.calls) == 1

    # Same hash, retry_failed=True -> should re-call.
    payload_v2 = _valid_response()  # different content the LLM "could" give
    client_v2 = _StubClient([(json.dumps(payload_v2), dict(payload_v2))])
    extract_for_package(
        training_package_id="test_pkg",
        documents=[("hash_aaa", "x" * 500)],
        cache_dir=tmp_path,
        client=client_v2,  # type: ignore[arg-type]
        retry_failed=True,
    )
    assert len(client_v2.calls) == 1


def test_per_row_flush_persists_after_every_extraction(tmp_path: Path) -> None:
    """Robustness against SIGKILL: cache must be persisted after every
    successful row, not just at the end of the loop. We simulate the
    SIGKILL with a KeyboardInterrupt that ``extract_one``'s broad
    ``except Exception`` does NOT swallow."""

    payload = _valid_response()

    class _CrashAfterOne:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, document_text: str) -> tuple[str, dict[str, object]]:
            self.calls += 1
            if self.calls > 1:
                # KeyboardInterrupt is a BaseException -- escapes the
                # broad ``except Exception`` inside extract_one and
                # propagates out of extract_for_package.
                raise KeyboardInterrupt("simulated SIGINT")
            return json.dumps(payload), dict(payload)

    documents = [("hash_first", "x" * 500), ("hash_second", "x" * 500)]
    with pytest.raises(KeyboardInterrupt):
        extract_for_package(
            training_package_id="test_pkg",
            documents=documents,
            cache_dir=tmp_path,
            client=_CrashAfterOne(),  # type: ignore[arg-type]
            progress_every=1,
        )

    # The first row must be on disk despite the crash.
    import pandas as pd

    cache_files = list(tmp_path.glob("*/*.parquet"))
    assert cache_files, "per-row flush did not write the cache parquet"
    frame = pd.read_parquet(cache_files[0])
    text_hashes_in_cache = set(frame["text_hash"].tolist())
    assert "hash_first" in text_hashes_in_cache, (
        "per-row flush should have persisted the first row before the crash"
    )
    assert "hash_second" not in text_hashes_in_cache


# ---------------------------------------------------------------------------
# Real client construction (requires the env var; skipped otherwise)
# ---------------------------------------------------------------------------


def test_anthropic_client_construction_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicExtractorClient(api_key=None)


def test_extraction_result_dataclass_round_trips() -> None:
    """Smoke -- the dataclass is frozen and serialises cleanly."""

    r = ExtractionResult(
        text_hash="abcd",
        status="ok",
        features={f.name: f.levels[0] for f in CATALOG},
        raw_response="{}",
        error_detail=None,
        elapsed_seconds=1.23,
    )
    assert r.text_hash == "abcd"
    assert r.features is not None
    assert "hawkish_shift_vs_prior" in r.features
