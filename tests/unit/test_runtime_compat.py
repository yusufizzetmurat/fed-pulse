"""Unit tests for the ``torch.compile`` resilience helper.

The helper auto-disables dynamo on pods where the installed triton has
an API too new for torch 2.4.x's inductor backend to import. These
tests stub ``triton.compiler.compiler`` via ``sys.modules`` so the
probe and its fallback can be exercised without a live torch+triton
stack.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.training import runtime_compat


@pytest.fixture(autouse=True)
def _isolated_history_db():
    """Override the suite-wide autouse fixture; this test never touches the db."""

    yield


@pytest.fixture(autouse=True)
def _reset_helper(monkeypatch: pytest.MonkeyPatch):
    """Reset the idempotency latch and clear the env var between cases."""

    runtime_compat._reset_for_testing()
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    yield
    runtime_compat._reset_for_testing()


def _install_triton_stub(monkeypatch: pytest.MonkeyPatch, *, with_key: bool) -> None:
    """Stub the three-level ``triton.compiler.compiler`` import path."""

    triton_pkg = types.ModuleType("triton")
    triton_pkg.__path__ = []  # mark as package
    compiler_pkg = types.ModuleType("triton.compiler")
    compiler_pkg.__path__ = []
    compiler_mod = types.ModuleType("triton.compiler.compiler")
    if with_key:
        compiler_mod.triton_key = lambda: "stub-key"  # type: ignore[attr-defined]
    compiler_mod.__file__ = "/stub/triton/compiler/compiler.py"

    monkeypatch.setitem(sys.modules, "triton", triton_pkg)
    monkeypatch.setitem(sys.modules, "triton.compiler", compiler_pkg)
    monkeypatch.setitem(sys.modules, "triton.compiler.compiler", compiler_mod)


def test_no_op_when_triton_key_present(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    _install_triton_stub(monkeypatch, with_key=True)

    runtime_compat.ensure_compile_safe()

    assert "TORCHDYNAMO_DISABLE" not in __import__("os").environ
    captured = capfd.readouterr()
    assert "compile_fallback_to_eager" not in captured.out
    assert "compile_fallback_to_eager" not in captured.err


def test_disables_dynamo_when_triton_key_missing(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import os

    _install_triton_stub(monkeypatch, with_key=False)

    runtime_compat.ensure_compile_safe()

    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    captured = capfd.readouterr()
    assert "compile_fallback_to_eager" in captured.out
    assert "reason=triton_api_mismatch" in captured.out
    assert "triton_module=" in captured.out


def test_idempotent_second_call_emits_no_extra_warning(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    _install_triton_stub(monkeypatch, with_key=False)

    runtime_compat.ensure_compile_safe()
    first = capfd.readouterr()
    runtime_compat.ensure_compile_safe()
    second = capfd.readouterr()

    assert first.out.count("compile_fallback_to_eager") == 1
    assert "compile_fallback_to_eager" not in second.out


def test_respects_existing_operator_override(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import os

    # Operator already set the env var (to anything, including the
    # explicit "0" some operators use to mean "do not auto-touch").
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    _install_triton_stub(monkeypatch, with_key=False)

    runtime_compat.ensure_compile_safe()

    # Helper leaves the operator value alone, even when the probe would
    # otherwise have flipped it to "1".
    assert os.environ["TORCHDYNAMO_DISABLE"] == "0"
    captured = capfd.readouterr()
    assert "compile_fallback_to_eager" not in captured.out


def test_handles_import_error_path(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Same fallback fires when the triton module is not importable at all."""

    import os

    # Remove any triton hits from sys.modules and block re-import by
    # injecting a finder that raises ImportError.
    for name in ("triton", "triton.compiler", "triton.compiler.compiler"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("triton"):
                raise ImportError(f"blocked {fullname} for test")
            return None

    blocker = _Blocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    runtime_compat.ensure_compile_safe()

    assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    captured = capfd.readouterr()
    assert "compile_fallback_to_eager" in captured.out
    assert "reason=triton_api_mismatch" in captured.out
