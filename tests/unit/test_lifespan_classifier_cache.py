from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("transformers")


def test_classifier_loads_exactly_once_across_many_get_classifier_calls(monkeypatch):
    """The lazy module-level cache should keep `classifier_load_count` at 1
    no matter how many callers ask for the classifier. This is the invariant
    the FastAPI lifespan relies on."""

    from app.services import text_encoder

    monkeypatch.setattr(text_encoder, "_classifier", None)
    monkeypatch.setattr(text_encoder, "_classifier_load_count", 0)

    build_calls = {"n": 0}

    class _FakePipeline:
        tokenizer = None
        model = None

        def __call__(self, *_args, **_kwargs):
            return [{"label": "POSITIVE", "score": 0.5}]

    def _fake_build(model_id, device):  # noqa: ARG001
        build_calls["n"] += 1
        return _FakePipeline()

    monkeypatch.setattr(text_encoder, "_build_pipeline", _fake_build)

    for _ in range(50):
        text_encoder.get_classifier()

    assert text_encoder.classifier_load_count() == 1
    assert build_calls["n"] == 1


def test_warmup_classifier_primes_the_cache(monkeypatch):
    """`warmup_classifier()` (called from the FastAPI lifespan) must load the
    pipeline so the first request doesn't pay the cold-start cost."""

    from app.services import text_encoder

    monkeypatch.setattr(text_encoder, "_classifier", None)
    monkeypatch.setattr(text_encoder, "_classifier_load_count", 0)

    class _FakePipeline:
        tokenizer = None
        model = None

    monkeypatch.setattr(text_encoder, "_build_pipeline", lambda *_a, **_kw: _FakePipeline())

    assert text_encoder.classifier_load_count() == 0
    text_encoder.warmup_classifier()
    assert text_encoder.classifier_load_count() == 1
    # Subsequent calls hit the cache, no rebuild.
    text_encoder.warmup_classifier()
    assert text_encoder.classifier_load_count() == 1
