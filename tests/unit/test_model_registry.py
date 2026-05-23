from __future__ import annotations

from pathlib import Path

import pytest


def test_registry_yaml_loads() -> None:
    from app.models.registry import load_registry

    load_registry.cache_clear()
    registry = load_registry()
    assert "finbert" in registry
    assert registry["finbert"].repo == "ProsusAI/finbert"
    assert len(registry["finbert"].revision) == 40
    assert registry["finbert"].revision.isalnum()


def test_revision_for_known_repo_returns_sha() -> None:
    from app.models.registry import revision_for

    sha = revision_for("ProsusAI/finbert")
    assert sha is not None
    assert len(sha) == 40


def test_revision_for_unknown_repo_returns_none() -> None:
    from app.models.registry import revision_for

    assert revision_for("not-a-real-model/does-not-exist") is None


def test_repo_alias_resolves_to_namespaced_revision() -> None:
    from app.models.registry import revision_for

    namespaced = revision_for("google-bert/bert-base-uncased")
    unnamespaced = revision_for("bert-base-uncased")
    assert namespaced is not None
    assert namespaced == unnamespaced


def test_from_pretrained_kwargs_raises_on_unknown() -> None:
    from app.models.registry import from_pretrained_kwargs

    with pytest.raises(ValueError, match="not pinned"):
        from_pretrained_kwargs("not-a-real-model/does-not-exist")


def test_from_pretrained_kwargs_merges_extras() -> None:
    from app.models.registry import from_pretrained_kwargs

    kwargs = from_pretrained_kwargs("ProsusAI/finbert", token="ts-1234")
    assert kwargs["revision"]
    assert kwargs["token"] == "ts-1234"


def test_registry_path_resolves_relative_to_module() -> None:
    from app.models.registry import MODEL_REGISTRY_PATH

    assert MODEL_REGISTRY_PATH.name == "registry.yaml"
    assert MODEL_REGISTRY_PATH.exists()
    assert isinstance(MODEL_REGISTRY_PATH, Path)


def test_round3_corpus_ablation_placeholders_are_registered_but_unpinned() -> None:
    """The Round 3 (#242) FOMC-only and BIS-only encoders ship as
    placeholders so the bake-off CLI knows the aliases exist; the empty
    revision keeps ``encoder_ref`` flagging them as 'unpinned local'
    until the first GPU run fills in a real checkpoint path."""

    from app.models.registry import encoder_ref, is_pinned, load_registry

    load_registry.cache_clear()
    for alias in ("finbert_fomc_only", "finbert_bis_only"):
        ref = encoder_ref(alias)
        assert ref is not None, f"{alias!r} missing from registry"
        assert ref.repo.startswith("local/"), (
            f"{alias!r} should advertise a local path until pretraining lands"
        )
        assert ref.revision == "", (
            f"{alias!r} revision should be empty until first pretrain run pins it"
        )
        assert not is_pinned(alias), (
            f"is_pinned should be False for the placeholder alias {alias!r}"
        )
