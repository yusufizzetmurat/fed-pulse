"""Inference service for the multi-axis text classifier (#78 follow-up).

The service is intentionally opt-in: when no checkpoint exists at
the configured path, ``score_text`` returns ``None`` and the
/analyze handler falls back to populating the stance card from the
legacy sentiment classifier. The tests pin both halves of that
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import multi_axis_classifier as svc


def test_score_text_returns_none_when_checkpoint_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """Pointing the service at a nonexistent path must NOT raise. The
    /analyze handler relies on a graceful ``None`` so it can route
    around an absent classifier."""

    svc.reset_classifier()
    missing = tmp_path / "no_such_checkpoint.pt"
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(missing))
    assert svc.checkpoint_exists() is False
    assert svc.score_text("any text") is None


def test_checkpoint_exists_returns_true_when_file_present(
    monkeypatch, tmp_path: Path
) -> None:
    """The path probe is a stat-only check; it does NOT validate the
    checkpoint contents."""

    svc.reset_classifier()
    present = tmp_path / "present.pt"
    present.write_bytes(b"\x00" * 4)
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(present))
    assert svc.checkpoint_exists() is True


def test_score_text_returns_none_on_malformed_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """A corrupted / non-torch checkpoint at the configured path
    must degrade gracefully to ``None`` rather than crashing the
    /analyze handler."""

    svc.reset_classifier()
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(corrupt))
    assert svc.score_text("any text") is None


def test_reset_classifier_clears_singleton(monkeypatch, tmp_path: Path) -> None:
    """``reset_classifier`` is the post-train hook the trainer calls
    to force the next /analyze request to rebuild the singleton from
    a fresh checkpoint."""

    svc.reset_classifier()
    missing = tmp_path / "still_missing.pt"
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(missing))
    # Prime the singleton with a None result.
    assert svc.get_classifier() is None
    # Reset must clear the cached None so a future load can rebuild.
    svc.reset_classifier()
    # Without the reset the second call would short-circuit on the
    # cached None; with it we hit the load path again.
    assert svc.get_classifier() is None


def test_score_text_skips_empty_input(monkeypatch, tmp_path: Path) -> None:
    """Empty / whitespace-only text must not invoke the model — the
    cards stay empty rather than the classifier hallucinating a
    label from an empty input window."""

    svc.reset_classifier()
    missing = tmp_path / "no_checkpoint.pt"
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(missing))
    assert svc.score_text("") is None
    assert svc.score_text("   ") is None


def test_load_failure_is_sticky_after_first_call(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """#454: once ``_load_state`` fails, subsequent ``get_classifier``
    calls must NOT re-attempt the load. Pre-fix the singleton cached
    ``None`` (indistinguishable from the initial unset state), so
    every request fell through to the load path again — flooding logs
    with a per-request warning and obscuring the "uninitialised"
    /health status. The sticky ``_LoadFailure`` sentinel breaks the
    cycle: first failure logs once, every later call returns ``None``
    without touching the load path.
    """

    svc.reset_classifier()
    missing = tmp_path / "absent.pt"
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(missing))

    call_count = {"n": 0}
    real_load = svc._load_state

    def _tracking_load() -> "svc._ClassifierState | svc._LoadFailure":
        call_count["n"] += 1
        return real_load()

    monkeypatch.setattr(svc, "_load_state", _tracking_load)

    with caplog.at_level("WARNING"):
        assert svc.get_classifier() is None
        first_warning_count = sum(
            1
            for r in caplog.records
            if "multi_axis_classifier_load_failed" in r.getMessage()
        )
        # Subsequent calls return None without re-loading or re-logging.
        for _ in range(5):
            assert svc.get_classifier() is None

    assert call_count["n"] == 1, (
        "Expected exactly one _load_state call across six get_classifier "
        f"calls; got {call_count['n']}. The sticky-cache contract is broken."
    )
    final_warning_count = sum(
        1
        for r in caplog.records
        if "multi_axis_classifier_load_failed" in r.getMessage()
    )
    assert final_warning_count == first_warning_count == 1, (
        "Expected exactly one warning across six get_classifier calls; "
        f"got {final_warning_count}."
    )


def test_reset_classifier_clears_sticky_load_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """``reset_classifier`` must clear the sticky ``_LoadFailure`` so an
    operator who fixes the underlying breakage can recover without a
    process restart. Pre-#454 fix this worked accidentally because the
    failure was cached as ``None`` (which the initial state also was);
    post-fix the reset must explicitly drop the sentinel."""

    svc.reset_classifier()
    missing = tmp_path / "absent.pt"
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(missing))
    assert svc.get_classifier() is None
    # Internal state: cached as _LoadFailure, not _UNSET.
    assert isinstance(svc._state, svc._LoadFailure)
    svc.reset_classifier()
    assert svc._state is svc._UNSET
