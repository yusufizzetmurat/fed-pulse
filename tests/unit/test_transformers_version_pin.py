"""Guard rail for the `transformers` major-version pin (#409).

`transformers` 5.x dropped `GenerationMixin` from
`transformers.models.auto.auto_factory`, which is the import path the
multi-axis encoder loader resolves at module import time. On the
post_306 Runpod sweep this surfaced as a `ModuleNotFoundError` on a
fresh pod that picked up 5.9.0 via hash-pinning, and the only fix was
a downgrade to the 4.x line. `backend/pyproject.toml` now caps the
constraint at `<5`; this test asserts the constraint is honoured at
runtime so a future bump cannot silently slip through the lock.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip(
    "transformers",
    reason="transformers not installed in this environment",
)


def test_transformers_major_version_is_four() -> None:
    version = str(getattr(transformers, "__version__", "")).strip()
    assert version, "transformers must expose a `__version__` attribute"
    major = version.split(".", 1)[0]
    assert major == "4", (
        f"transformers must remain on the 4.x line (got {version!r}); 5.x "
        "removes `GenerationMixin` from `transformers.models.auto.auto_factory` "
        "and breaks the multi-axis encoder loader (see #409)."
    )
