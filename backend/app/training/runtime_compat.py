"""Runtime-compatibility guards for the training stack.

The sweep runners are routinely operated on rented GPU pods where the
container's pre-installed torch can be clobbered by a later
``pip install -r requirements.lock``. When that happens, the surviving
``triton`` on disk frequently exposes a 3.x API that torch 2.4.x's
inductor backend cannot talk to, and the very first ``torch.compile``
call dies with::

    torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    ImportError: cannot import name 'triton_key' from
    'triton.compiler.compiler'

The manual fix is ``TORCHDYNAMO_DISABLE=1``. This module gives the
canonical sweep runners a way to detect the mismatch up-front and apply
that fix themselves, so the next operator doesn't get to rediscover the
debugging path.
"""

from __future__ import annotations

import importlib
import os
import sys

__all__ = ["ensure_compile_safe"]


_ALREADY_RAN = False


def ensure_compile_safe() -> None:
    """Disable ``torch.compile`` when the local triton is incompatible.

    Probes ``triton.compiler.compiler.triton_key`` -- the exact attribute
    torch 2.4.x's inductor backend imports on its first compile call. If
    the probe fails (``ImportError`` or ``AttributeError``), sets
    ``TORCHDYNAMO_DISABLE=1`` in the process environment, calls
    ``torch._dynamo.disable()`` if available, and emits a single
    structured WARNING line so operators can grep for it in the pod log.

    The helper is idempotent: the second call is a no-op. It also
    respects an existing operator override -- if ``TORCHDYNAMO_DISABLE``
    is already set (to any value), the helper leaves the env alone and
    treats that as "operator already made the call".
    """

    global _ALREADY_RAN
    if _ALREADY_RAN:
        return
    _ALREADY_RAN = True

    # Operator override wins. If the variable is set to anything (even
    # an empty string), the caller has already made a deliberate choice
    # about dynamo and we don't second-guess it.
    if "TORCHDYNAMO_DISABLE" in os.environ:
        return

    triton_module_name = "triton.compiler.compiler"
    try:
        module = importlib.import_module(triton_module_name)
        getattr(module, "triton_key")
    except (ImportError, AttributeError):
        pass
    else:
        return

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    resolved = getattr(
        sys.modules.get(triton_module_name), "__file__", triton_module_name
    )
    print(
        f"WARNING compile_fallback_to_eager reason=triton_api_mismatch "
        f"triton_module={resolved}",
        flush=True,
    )

    try:
        import torch._dynamo as _dynamo  # noqa: WPS433
    except Exception:
        return
    disable = getattr(_dynamo, "disable", None)
    if callable(disable):
        try:
            disable()
        except Exception:
            # ``torch._dynamo.disable`` exists in a couple of shapes
            # across torch versions; if the call fails we still have
            # the env var set, which is the load-bearing half.
            pass


def _reset_for_testing() -> None:
    """Clear the idempotency latch. Test-only hook."""

    global _ALREADY_RAN
    _ALREADY_RAN = False
