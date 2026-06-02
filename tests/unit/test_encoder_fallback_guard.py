"""Fail-closed encoder fallback + provenance guard (MLC1 hardening).

The silent fallback from the primary FOMC sentiment model to the generic
distilbert-sst-2 classifier could contaminate cached embeddings undetected.
These tests lock the hardened contract: refuse the fallback by default, allow
it only via an explicit opt-out, enforce the guard at the encode boundary, and
stamp encoder provenance into built artifacts.
"""

from __future__ import annotations

import pytest

import app.services.text_encoder as te


@pytest.fixture
def reset_classifier(monkeypatch):
    monkeypatch.setattr(te, "_classifier", None)
    monkeypatch.setattr(te, "_loaded_model_id", None)
    yield


def _build_factory(primary_id: str):
    """Fake _build_pipeline: primary 404s, fallback loads."""

    def _fake(model_id: str, device: int):
        if model_id == primary_id:
            raise RuntimeError("primary unavailable (simulated 404)")
        return object()  # stand-in fallback pipeline

    return _fake


def test_get_classifier_raises_on_fallback_by_default(monkeypatch, reset_classifier):
    monkeypatch.delenv("FED_PULSE_ALLOW_SENTIMENT_FALLBACK", raising=False)
    monkeypatch.delenv("FED_PULSE_REQUIRE_PRIMARY_SENTIMENT", raising=False)
    monkeypatch.setattr(te, "MODEL_ID", "primary/stance-model")
    monkeypatch.setattr(te, "_build_pipeline", _build_factory("primary/stance-model"))

    with pytest.raises(RuntimeError, match="fall"):
        te.get_classifier()


def test_get_classifier_allows_fallback_with_explicit_optout(monkeypatch, reset_classifier):
    monkeypatch.setenv("FED_PULSE_ALLOW_SENTIMENT_FALLBACK", "1")
    monkeypatch.delenv("FED_PULSE_REQUIRE_PRIMARY_SENTIMENT", raising=False)
    monkeypatch.setattr(te, "MODEL_ID", "primary/stance-model")
    sentinel = object()

    def _fake(model_id: str, device: int):
        if model_id == "primary/stance-model":
            raise RuntimeError("primary down")
        return sentinel

    monkeypatch.setattr(te, "_build_pipeline", _fake)

    clf = te.get_classifier()
    assert clf is sentinel
    assert te.get_loaded_model_id() == te.FALLBACK_MODEL_ID


def test_require_primary_env_overrides_optout(monkeypatch, reset_classifier):
    # Back-compat: REQUIRE wins even if ALLOW is also set.
    monkeypatch.setenv("FED_PULSE_ALLOW_SENTIMENT_FALLBACK", "1")
    monkeypatch.setenv("FED_PULSE_REQUIRE_PRIMARY_SENTIMENT", "1")
    monkeypatch.setattr(te, "MODEL_ID", "primary/stance-model")
    monkeypatch.setattr(te, "_build_pipeline", _build_factory("primary/stance-model"))

    with pytest.raises(RuntimeError, match="fall"):
        te.get_classifier()


def test_encode_chunks_enforces_primary_guard_before_load(monkeypatch):
    # encode_chunks must delegate to the primary guard BEFORE loading any model,
    # so ANY caller is protected (not just the build scripts that call it today).
    # Both deps are stubbed so this never touches the network in either state.
    def _guard() -> None:
        raise RuntimeError("primary guard tripped")

    def _should_not_load() -> object:
        raise AssertionError("get_classifier reached before the primary guard")

    monkeypatch.setattr(te, "assert_primary_model_loaded", _guard)
    monkeypatch.setattr(te, "get_classifier", _should_not_load)

    with pytest.raises(RuntimeError, match="primary guard tripped"):
        te.encode_chunks("a sufficiently long FOMC statement for encoding")


def test_loaded_encoder_provenance_reports_active_encoder(monkeypatch):
    class _Cfg:
        hidden_size = 768

    class _Model:
        config = _Cfg()

    class _Clf:
        model = _Model()

    monkeypatch.setattr(te, "get_classifier", lambda: _Clf())
    monkeypatch.setattr(te, "get_loaded_model_id", lambda: "gtfintechlab/FOMC-RoBERTa")
    monkeypatch.setattr(te, "revision_for", lambda _m: "rev-xyz")

    prov = te.loaded_encoder_provenance()

    assert prov == {
        "model_id": "gtfintechlab/FOMC-RoBERTa",
        "revision": "rev-xyz",
        "hidden_size": 768,
    }
