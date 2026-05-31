"""Sticky-cache contract for the trajectory singleton (#454).

Mirror the #410 / multi_axis_classifier pattern. A failed
``_load_state`` must be cached as a :class:`_LoadFailure` sentinel
so subsequent ``get_state`` calls return ``None`` without re-running
the load (which would flood logs on a sweep against a broken bundle).
``reset_state`` must clear the sticky sentinel so an operator can
recover without a process restart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import trajectory as svc


@pytest.fixture(autouse=True)
def _reset_trajectory_state():
    """Reset the singleton both before and after each test.

    A test that crashes mid-run leaves module-level ``_state`` in a
    stale shape (either a real state from a fixture or the sticky
    sentinel from this very PR). Without the after-yield reset the
    next test inherits the pollution; matches the analogs.py test
    isolation pattern surfaced in the #551 review.
    """

    svc.reset_state()
    yield
    svc.reset_state()


def test_get_state_returns_none_when_bundle_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing bundle directory degrades to ``None`` without raising."""

    monkeypatch.setenv("FED_PULSE_TRAJECTORY_DIR", str(tmp_path / "absent"))
    assert svc.get_state() is None


def test_load_failure_is_sticky_after_first_call(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """#454: one ``_load_state`` call across many ``get_state`` calls.

    Pre-fix the singleton cached ``None`` on failure (indistinguishable
    from the initial unset state), so every /analyze/trajectory request
    fell through the full bundle-load path again — logs per request, and
    the /health surface stayed ``uninitialised`` silently. The sticky
    :class:`_LoadFailure` sentinel breaks the loop.
    """

    monkeypatch.setenv(
        "FED_PULSE_TRAJECTORY_DIR", str(tmp_path / "absent")
    )

    call_count = {"n": 0}
    real_load = svc._load_state

    def _tracking_load() -> "svc._TrajectoryState | svc._LoadFailure":
        call_count["n"] += 1
        return real_load()

    monkeypatch.setattr(svc, "_load_state", _tracking_load)

    with caplog.at_level("WARNING"):
        assert svc.get_state() is None
        first_warning_count = sum(
            1
            for r in caplog.records
            if "trajectory_load_failed" in r.getMessage()
        )
        for _ in range(5):
            assert svc.get_state() is None

    assert call_count["n"] == 1, (
        "Expected exactly one _load_state call across six get_state calls; "
        f"got {call_count['n']}. The sticky-cache contract is broken."
    )
    final_warning_count = sum(
        1
        for r in caplog.records
        if "trajectory_load_failed" in r.getMessage()
    )
    assert final_warning_count == first_warning_count == 1, (
        "Expected exactly one warning across six get_state calls; got "
        f"{final_warning_count}."
    )


def test_reset_state_clears_sticky_load_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """``reset_state`` must drop the sticky ``_LoadFailure`` sentinel."""

    monkeypatch.setenv(
        "FED_PULSE_TRAJECTORY_DIR", str(tmp_path / "absent")
    )
    assert svc.get_state() is None
    assert isinstance(svc._state, svc._LoadFailure)
    svc.reset_state()
    assert svc._state is svc._UNSET
