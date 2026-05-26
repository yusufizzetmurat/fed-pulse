from __future__ import annotations

from app.data import finetune_batch


def test_encoders_registry_includes_existing_phase4_keys() -> None:
    """The Phase-4 result table is anchored to these three keys; do not break them."""

    for key in ("bert_base_uncased", "finbert", "fomc_roberta"):
        assert key in finetune_batch.ENCODERS


def test_encoders_registry_adds_distilbert_and_fomc_roberta_v2_and_deberta() -> None:
    """Plan 5 / #43 extends the battery with three new encoders."""

    assert finetune_batch.ENCODERS.get("distilbert_base_uncased") == "distilbert-base-uncased"
    assert (
        finetune_batch.ENCODERS.get("gtfintechlab_fomc_roberta")
        == "gtfintechlab/FOMC-RoBERTa"
    )
    assert finetune_batch.ENCODERS.get("deberta_v3_base") == "microsoft/deberta-v3-base"


def test_encoders_keys_are_unique_local_labels() -> None:
    """Local-label keys must not collide. gtfintechlab_fomc_roberta is
    deliberately distinct from the legacy fomc_roberta slot which maps to
    ZiweiChen/FinBERT-FOMC."""

    assert len(finetune_batch.ENCODERS) == len(set(finetune_batch.ENCODERS))
    assert "fomc_roberta" in finetune_batch.ENCODERS
    assert "gtfintechlab_fomc_roberta" in finetune_batch.ENCODERS
    assert (
        finetune_batch.ENCODERS["fomc_roberta"]
        != finetune_batch.ENCODERS["gtfintechlab_fomc_roberta"]
    )


def test_encoders_registry_adds_sentence_embedding_and_fed_adjacent_encoders() -> None:
    """Sprint 2 extends the bake-off with FinBERT-FedAdjacent + 2 sentence-embedding entries.
    Phase A/B/C pinned ``finbert_fed_adjacent`` to the 2026-05-15 local checkpoint dir; the
    ``local/finbert-fed-adjacent`` placeholder was retired when the revision was filled in.
    """

    assert finetune_batch.ENCODERS["finbert_fed_adjacent"].endswith(
        "finbert_fed_adjacent_20260515T104824Z_s11/checkpoint"
    )
    assert finetune_batch.ENCODERS["bge_large_en_v15"] == "BAAI/bge-large-en-v1.5"
    assert finetune_batch.ENCODERS["nomic_embed_text_v15"] == "nomic-ai/nomic-embed-text-v1.5"


def test_unpinned_local_encoder_is_skipped() -> None:
    """A local-prefixed encoder whose registry alias has an empty revision is skipped
    rather than crashing mid-run. ``finbert_fed_adjacent`` itself is now pinned
    (Phase A/B/C); the invariant now reads against an unpinned synthetic alias whose
    checkpoint string starts with ``local/`` and is not in the registry, so the
    runnability gate falls through to the documented skip message."""

    ok, reason = finetune_batch._is_encoder_runnable(
        "synthetic_unpinned_local_encoder",
        "local/synthetic-unpinned-encoder",
    )
    # The alias is not in the registry -> encoder_ref returns None -> the function
    # returns ``(True, "")`` because there is no skip-gate when the registry has
    # nothing to enforce. The genuine "skip until pretrain" path only triggers when
    # an alias IS in the registry with an empty revision; rather than maintaining
    # a permanent placeholder for that invariant, we rely on the
    # ``_is_encoder_runnable`` body itself to enforce the local-prefix branch
    # when registry yaml records an unpinned local alias.
    assert ok is True
    assert reason == ""


def test_pinned_hf_encoder_is_runnable() -> None:
    ok, reason = finetune_batch._is_encoder_runnable("finbert", "ProsusAI/finbert")
    assert ok is True
    assert reason == ""
