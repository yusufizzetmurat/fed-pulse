from __future__ import annotations

from app.data import phase3_finetune_batch


def test_encoders_registry_includes_existing_phase4_keys() -> None:
    """The Phase-4 result table is anchored to these three keys; do not break them."""

    for key in ("bert_base_uncased", "finbert", "fomc_roberta"):
        assert key in phase3_finetune_batch.ENCODERS


def test_encoders_registry_adds_distilbert_and_fomc_roberta_v2_and_deberta() -> None:
    """Plan 5 / #43 extends the battery with three new encoders."""

    assert phase3_finetune_batch.ENCODERS.get("distilbert_base_uncased") == "distilbert-base-uncased"
    assert (
        phase3_finetune_batch.ENCODERS.get("gtfintechlab_fomc_roberta")
        == "gtfintechlab/FOMC-RoBERTa"
    )
    assert phase3_finetune_batch.ENCODERS.get("deberta_v3_base") == "microsoft/deberta-v3-base"


def test_encoders_keys_are_unique_local_labels() -> None:
    """Local-label keys must not collide. gtfintechlab_fomc_roberta is
    deliberately distinct from the legacy fomc_roberta slot which maps to
    ZiweiChen/FinBERT-FOMC."""

    assert len(phase3_finetune_batch.ENCODERS) == len(set(phase3_finetune_batch.ENCODERS))
    assert "fomc_roberta" in phase3_finetune_batch.ENCODERS
    assert "gtfintechlab_fomc_roberta" in phase3_finetune_batch.ENCODERS
    assert (
        phase3_finetune_batch.ENCODERS["fomc_roberta"]
        != phase3_finetune_batch.ENCODERS["gtfintechlab_fomc_roberta"]
    )
