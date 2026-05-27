"""Pin the ``gtfintechlab/fomc-roberta-any-exp`` audit verdict (#339).

The 2026-05-27 audit found the repo unreachable on Hugging Face
(``HfApi.model_info`` returns 404 under an authenticated token) and
treated it as inheriting the gtfintechlab Trillion Dollar Words
contamination flag against ``hf_fomc_communication`` (R-13). The
encoder is therefore deny-listed in ``finetune_batch.py`` and the
registry entry stays at ``revision: main`` as a deliberate
"not reproducible" marker.

This test guards both halves: the deny-list entry must be present, and
the registry revision must NOT be silently flipped to a pinned SHA
without a follow-up audit. Either deny-list+`main` OR pinned-clean is
an acceptable terminal state per the issue body; what's not acceptable
is the pre-#339 hole where the encoder was both unpinned and absent
from any contamination guard.
"""

from __future__ import annotations

import pytest


def test_fomc_roberta_any_exp_audit_state_pinned() -> None:
    """Encoder must be deny-listed OR pinned to a non-``main`` revision."""

    pytest.importorskip("yaml")
    from app.data.finetune_batch import CONTAMINATED_ENCODER_KEYS
    from app.models.registry import encoder_ref

    deny_listed = "gtfintechlab_fomc_roberta_any_exp" in CONTAMINATED_ENCODER_KEYS
    ref = encoder_ref("gtfintechlab/fomc-roberta-any-exp") or encoder_ref(
        "fomc_roberta_any_exp"
    )
    assert ref is not None, "registry must still know about fomc-roberta-any-exp"
    revision = str(getattr(ref, "revision", "") or "").strip().lower()
    pinned_clean = bool(revision) and revision not in {"main", "master", ""}

    assert deny_listed or pinned_clean, (
        "fomc-roberta-any-exp must either be on the contamination deny-list "
        f"or pinned to a non-``main`` revision; got revision={revision!r}, "
        f"deny_listed={deny_listed}."
    )


def test_fomc_roberta_sibling_still_denied() -> None:
    """The original R-13 entry must not be dropped from the deny-list as a
    side-effect of extending the flag to the sibling repo."""

    from app.data.finetune_batch import CONTAMINATED_ENCODER_KEYS

    assert "gtfintechlab_fomc_roberta" in CONTAMINATED_ENCODER_KEYS
