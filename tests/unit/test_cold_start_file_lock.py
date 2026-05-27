"""Cross-process cold-start regression (#383).

PR #382 serialised the bootstrap within one process via
``asyncio.Lock``; that's why ``deploy/s6-services/backend/run``
temporarily pinned ``--workers 1``. #383 wraps the bootstrap body in
an OS-level ``fcntl.flock`` on a sentinel file next to
``BEST_MODEL_PATH`` so the prod deploy can scale back to
``--workers > 1`` without two workers racing on the same checkpoint
write.

This test forks ``multiprocessing.Process`` children that each call
the file-lock + re-check pattern from ``_bootstrap_cold_start``
against a shared temp directory. Exactly one child must do the
"bootstrap" work; the other must block on ``LOCK_EX``, observe the
freshly written checkpoint on the re-check, and return.

POSIX-only: ``fcntl.flock`` does not exist on Windows. The s6 deploy
is POSIX so this matches the prod runtime.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

# Windows lacks ``fcntl``; skip the whole module on non-POSIX hosts.
if sys.platform == "win32":  # pragma: no cover -- repo is POSIX-only
    pytest.skip("fcntl.flock is POSIX-only", allow_module_level=True)


def _bootstrap_worker(
    checkpoint_path: str,
    lock_path: str,
    counter_path: str,
    barrier_path: str,
    bootstrap_sleep_s: float,
) -> None:
    """Subprocess entrypoint.

    Mirrors the file-lock + re-check shape of ``_bootstrap_cold_start``
    without dragging the real ML training stack into the test:

      1. Spin on the barrier file so both children enter the lock
         contention window at the same time.
      2. Acquire the shared ``_bootstrap_file_lock``.
      3. Re-check the checkpoint inside the lock; return if present.
      4. Sleep briefly to widen the race window, then write the
         checkpoint + bump the "did real work" counter.
    """

    # Force-import via the same path layout ``tests/conftest.py`` uses
    # (the conftest is not auto-loaded inside a fresh subprocess).
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    backend_dir = repo_root / "backend"
    for entry in (str(backend_dir), str(repo_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from app.main import _bootstrap_file_lock  # noqa: E402

    barrier = Path(barrier_path)
    while not barrier.exists():
        time.sleep(0.01)

    with _bootstrap_file_lock(Path(lock_path)):
        if Path(checkpoint_path).exists():
            return
        time.sleep(bootstrap_sleep_s)
        Path(checkpoint_path).write_bytes(b"bootstrap-payload")
        with open(counter_path, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")


def test_bootstrap_file_lock_serialises_across_processes(tmp_path: Path) -> None:
    """Two processes hit the bootstrap concurrently. The file lock plus
    the in-lock checkpoint re-check must mean exactly one writes."""

    checkpoint = tmp_path / "forecaster_best.pt"
    lock_file = tmp_path / "forecaster_best.pt.bootstrap.lock"
    counter = tmp_path / "bootstrap_calls.log"
    barrier = tmp_path / "go"

    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(
            target=_bootstrap_worker,
            args=(str(checkpoint), str(lock_file), str(counter), str(barrier), 0.5),
        )
        for _ in range(2)
    ]
    for w in workers:
        w.start()

    # Give both children time to reach the barrier wait, then release
    # them simultaneously. Without the file lock both would proceed
    # into the bootstrap body together and both would write the
    # checkpoint -- which is exactly the regression #383 fixes.
    time.sleep(0.3)
    barrier.write_bytes(b"go")

    for w in workers:
        w.join(timeout=30)
        assert w.exitcode == 0, f"worker pid={w.pid} exited with {w.exitcode}"

    assert checkpoint.exists(), "leader must have written the checkpoint"
    assert lock_file.exists(), "sentinel lock file must survive the run"

    lines = [ln for ln in counter.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, (
        "exactly one process should run the bootstrap body; "
        f"got {len(lines)} entries: {lines!r}"
    )


def test_bootstrap_file_lock_released_on_exception(tmp_path: Path) -> None:
    """A raise inside the ``with`` block must still release the lock so
    the next process can acquire it. Guards against the regression where
    a missing ``try/finally`` would leave a stuck sentinel and wedge
    every subsequent worker."""

    from app.main import _bootstrap_file_lock

    lock_file = tmp_path / "stuck.lock"

    with pytest.raises(RuntimeError, match="boom"):
        with _bootstrap_file_lock(lock_file):
            raise RuntimeError("boom")

    # If the lock were leaked we'd deadlock here -- the second acquire
    # would block forever on ``LOCK_EX``. A successful re-entry proves
    # the ``try/finally`` actually released the kernel-level lock.
    with _bootstrap_file_lock(lock_file):
        pass
