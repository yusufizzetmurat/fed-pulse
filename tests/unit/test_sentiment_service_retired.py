"""Assert the legacy ``app.services.sentiment`` module is retired (#339).

Deliverable #2 of the encoder-parity audit was to drop the legacy
sentiment service in favour of routing every caller through
``app.services.text_encoder.analyze_text`` (which itself prefers the
``TextMultiAxisClassifier`` stance head when a checkpoint is loaded).
This test pins the contract so a future refactor cannot quietly
re-introduce the legacy shim and split the stance signal across two
encoders again.
"""

from __future__ import annotations

import importlib

import pytest


def test_legacy_sentiment_module_no_longer_importable() -> None:
    """``import app.services.sentiment`` must fail with ``ModuleNotFoundError``."""

    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("app.services.sentiment")


def test_analyze_text_lives_on_text_encoder() -> None:
    """``analyze_text`` must remain available on the text-encoder module so
    the existing /analyze + prepare_training_data + attention_ablation
    callers stay drop-in."""

    text_encoder = importlib.import_module("app.services.text_encoder")
    assert callable(getattr(text_encoder, "analyze_text", None)), (
        "app.services.text_encoder.analyze_text must be callable after the "
        "sentiment-service retirement -- callers re-route to this surface."
    )


def test_main_module_imports_analyze_text_from_text_encoder() -> None:
    """The FastAPI app must take its stance-scoring function from the
    text-encoder module, not from the retired sentiment shim."""

    main_mod = importlib.import_module("app.main")
    analyze_text = getattr(main_mod, "analyze_text", None)
    assert callable(analyze_text)
    # The bound module on the symbol points at text_encoder, not sentiment.
    assert analyze_text.__module__ == "app.services.text_encoder", (
        f"main.analyze_text resolves to {analyze_text.__module__!r}; "
        "the sentiment-service retirement requires the text-encoder path."
    )
