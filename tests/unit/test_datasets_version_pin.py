"""Guard rail for the `datasets` major-version pin (#465).

`datasets` 4.x removed loading-script APIs the PhraseBank loader (and
the other ingest paths at `app/data/ingest_sources.py`,
`app/data/continued_pretraining.py`) call through `load_dataset`. On
the Phase 1 Runpod sweep batch the resolver picked up a 4.x version on
a fresh pod and the loader broke; the only fix was a downgrade to the
3.x line. `backend/pyproject.toml` now caps the constraint at `<4`;
this test asserts the constraint is honoured at runtime so a future
bump cannot silently slip through the lock.
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip(
    "datasets",
    reason="datasets not installed in this environment",
)


def test_datasets_major_version_is_three() -> None:
    version = str(getattr(datasets, "__version__", "")).strip()
    assert version, "datasets must expose a `__version__` attribute"
    major = version.split(".", 1)[0]
    assert major == "3", (
        f"datasets must remain on the 3.x line (got {version!r}); 4.x "
        "removed loading-script APIs the PhraseBank + ingest loaders rely "
        "on through `load_dataset` (see #465)."
    )
