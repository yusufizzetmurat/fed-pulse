"""Behavioural tests for the ``build_stance_daily`` script.

The script reduces the multi-axis classifier output to ``s = P(hawk) -
P(dove)`` for every statement in the FOMC corpus. The unit tests
target the per-document reducer behind ``_score_one`` so a CI run does
not need to load the multi-axis checkpoint or hit yfinance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Load the script module by path so we can monkeypatch its
# ``score_text`` resolution. Adding ``scripts/`` to sys.path leaks
# every script under that tree into pytest's collection, which is
# noisy and unsafe; loading the file directly keeps the surface
# narrow.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_stance_daily.py"


@pytest.fixture()
def builder():
    spec = importlib.util.spec_from_file_location("build_stance_daily", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patched_score(monkeypatch, builder, value: dict[str, Any] | None) -> None:
    """Stub ``app.services.multi_axis_classifier.score_text`` to return ``value``."""

    import app.services.multi_axis_classifier as mod

    monkeypatch.setattr(mod, "score_text", lambda _text: value)


def test_score_one_returns_hawkish_minus_dovish(monkeypatch, builder) -> None:
    _patched_score(
        monkeypatch,
        builder,
        {"stance": {"distribution": {"hawkish": 0.72, "dovish": 0.15, "neutral": 0.13}}},
    )
    assert builder._score_one("any text") == pytest.approx(0.57)


def test_score_one_returns_none_when_classifier_unloaded(monkeypatch, builder) -> None:
    _patched_score(monkeypatch, builder, None)
    assert builder._score_one("any text") is None


def test_score_one_returns_none_when_distribution_empty(monkeypatch, builder) -> None:
    _patched_score(monkeypatch, builder, {"stance": {"distribution": {"neutral": 1.0}}})
    # No hawkish / dovish keys → cannot compute s; reducer must signal
    # missing measurement rather than fabricate 0.0.
    assert builder._score_one("any text") is None


def test_score_one_tolerates_one_sided_distribution(monkeypatch, builder) -> None:
    _patched_score(
        monkeypatch,
        builder,
        {"stance": {"distribution": {"hawkish": 0.6}}},  # only hawkish present
    )
    assert builder._score_one("any text") == pytest.approx(0.6)
    _patched_score(
        monkeypatch,
        builder,
        {"stance": {"distribution": {"dovish": 0.4}}},  # only dovish present
    )
    assert builder._score_one("any text") == pytest.approx(-0.4)


def test_score_one_handles_missing_keys(monkeypatch, builder) -> None:
    for payload in (
        None,
        {},
        {"stance": None},
        {"stance": {}},
        {"stance": {"distribution": None}},
    ):
        _patched_score(monkeypatch, builder, payload)
        assert builder._score_one("any text") is None, payload
