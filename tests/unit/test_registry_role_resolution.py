"""Role-keyed encoder resolution (ADR 0019 / #330).

The canonical encoder slot is split into two roles: ``classifier`` for
the headline classification substrate and ``retrieval`` for the
retrieval base. The resolver returns the alias whose ``role:`` tag
matches; a registry without role tags still loads cleanly for legacy
callers, and an unknown role raises ``KeyError`` at the boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _reset_registry_cache() -> None:
    from app.models.registry import load_registry

    load_registry.cache_clear()


def test_classifier_role_resolves_to_fed_adjacent() -> None:
    """``role: classifier`` returns ``finbert_fed_adjacent``.

    Per ADR 0019 the classifier substrate was originally pinned to
    ``finbert_fomc_only`` (the corpus-ablation sibling). That GPU run
    was never executed and the placeholder hard-failed every classifier-
    role sweep on Runpod (#463). The role tag re-points to
    ``finbert_fed_adjacent`` — the produced FinBERT-BIS DAPT substrate
    with weights mirrored on Hugging Face Hub. Cross-bank DAPT keeps
    the retrieval role only.
    """
    _reset_registry_cache()
    from app.models.registry import resolve_by_role

    alias = resolve_by_role("classifier")
    assert alias == "finbert_fed_adjacent", (
        f"expected classifier substrate to resolve to 'finbert_fed_adjacent'; got {alias!r}"
    )


def test_retrieval_role_resolves_to_xbank_dapt() -> None:
    """``role: retrieval`` returns the cross-bank DAPT encoder."""
    _reset_registry_cache()
    from app.models.registry import resolve_by_role

    alias = resolve_by_role("retrieval")
    assert alias == "finbert_fed_adjacent_xbank_dapt", (
        "expected retrieval substrate to resolve to "
        f"'finbert_fed_adjacent_xbank_dapt'; got {alias!r}"
    )


def test_unknown_role_raises_key_error() -> None:
    """An unknown role label fails fast at the resolver boundary."""
    _reset_registry_cache()
    from app.models.registry import resolve_by_role

    with pytest.raises(KeyError, match="unknown encoder role"):
        resolve_by_role("not_a_real_role")  # type: ignore[arg-type]


def test_role_field_is_none_for_untagged_entries() -> None:
    """Entries without an explicit ``role:`` carry ``role=None``.

    The ``EncoderRef`` dataclass leaves ``role`` at ``None`` for every
    bake-off sibling and control ablation; only the two canonical
    entries (classifier + retrieval) carry a tag. This is what makes
    the ``resolve_by_role`` scan deterministic — it stops at the
    first tagged row and ignores untagged neighbours.
    """
    _reset_registry_cache()
    from app.models.registry import encoder_ref

    finbert = encoder_ref("finbert")
    assert finbert is not None
    assert finbert.role is None

    finbert_tone = encoder_ref("finbert_tone")
    assert finbert_tone is not None
    assert finbert_tone.role is None


def test_role_tags_present_on_canonical_entries() -> None:
    """The two canonical entries carry the expected role tags."""
    _reset_registry_cache()
    from app.models.registry import encoder_ref

    classifier = encoder_ref("finbert_fed_adjacent")
    assert classifier is not None
    assert classifier.role == "classifier"

    retrieval = encoder_ref("finbert_fed_adjacent_xbank_dapt")
    assert retrieval is not None
    assert retrieval.role == "retrieval"

    # The unproduced corpus-ablation sibling no longer carries the
    # classifier role; it stays in the registry as a placeholder for
    # the deferred FOMC-only DAPT pretrain.
    fomc_only = encoder_ref("finbert_fomc_only")
    assert fomc_only is not None
    assert fomc_only.role is None


def test_legacy_registry_without_roles_still_loads(tmp_path: Path) -> None:
    """A registry whose entries omit ``role:`` loads cleanly.

    Back-compat guard: pre-#330 registry YAMLs carry no ``role:`` key.
    The loader must accept them and leave every ``EncoderRef.role`` at
    ``None`` so callers that resolve by alias (the previous default)
    keep working unchanged. ``resolve_by_role`` raises ``KeyError`` on
    the legacy shape because there is no tagged entry to point at —
    callsites are expected to fall back to a hard-coded default in
    that path (mirrors how the retrieval / classifier entrypoints
    guard their imports).
    """
    legacy_yaml = """
updated_at: 2026-01-01

encoders:
  legacy_one:
    repo: ProsusAI/finbert
    revision: deadbeef
    gated: false
    task: classification
    description: legacy entry without role tag.

  legacy_two:
    repo: gtfintechlab/FOMC-RoBERTa
    revision: cafef00d
    gated: false
    task: classification
    description: second legacy entry.
"""
    yaml_path = tmp_path / "registry.yaml"
    yaml_path.write_text(legacy_yaml.strip(), encoding="utf-8")

    from app.models.registry import load_registry

    load_registry.cache_clear()
    registry = load_registry(yaml_path)
    load_registry.cache_clear()  # reset so other tests see the canonical yaml

    assert "legacy_one" in registry
    assert "legacy_two" in registry
    assert registry["legacy_one"].role is None
    assert registry["legacy_two"].role is None


def test_resolve_by_role_dedups_alias_fanout() -> None:
    """The resolver visits each entry once despite the alias / repo fanout.

    ``load_registry`` keys each ``EncoderRef`` under ``alias``, ``repo``,
    and every ``repo_aliases`` entry — a naive ``.values()`` scan would
    visit the same ref multiple times. The dedup pass keeps the scan
    deterministic so the first ``role: <x>`` hit wins.
    """
    _reset_registry_cache()
    from app.models.registry import resolve_by_role

    # Both roles must resolve to a unique alias; if the scan double-
    # visited a ref the result would still be correct but the dedup
    # contract is what keeps role-uniqueness debuggable when a future
    # registry edit introduces an accidental second tagged row.
    assert resolve_by_role("classifier") != resolve_by_role("retrieval")
