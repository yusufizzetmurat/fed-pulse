"""Bundle A.1: cross-bank supervision gate on the multi-axis trainer.

Pins the per-row mask + weight rewrites that
``train_text_multi_axis_classifier._gtfintechlab_row_to_axis_row``
applies under each arm of ``--cross-bank-supervision``. The collate
contract surfaces the rewrites end-to-end so any downstream layer
(loss, sanity log, eval) reads the corrected mask without further
special-casing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.data.train_text_multi_axis_classifier import (  # noqa: E402
    _AxisRow,
    _collate,
    _gtfintechlab_row_to_axis_row,
    _log_per_axis_provenance_breakdown,
)


def _gtf_row(
    *,
    stance: str = "hawkish",
    certain: str = "certain",
    text: str = "Inflation expectations remain anchored.",
) -> dict[str, object]:
    """Minimal HF gtfintechlab record fixture for the row mapper."""

    return {
        "sentences": text,
        "stance_label": stance,
        "certain_label": certain,
        "time_label": "",
        "year": 2024,
    }


class _FakeTokenizer:
    """Tiny tokenizer stand-in so ``__getitem__`` runs without HF."""

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, "torch.Tensor"]:
        ids = torch.zeros(1, max_length, dtype=torch.long)
        mask = torch.ones(1, max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


def _axis_row_from_gtf(
    *,
    provenance: str,
    mode: str,
    stance_weight: float = 1.0,
    stance: str = "hawkish",
) -> _AxisRow:
    row = _gtfintechlab_row_to_axis_row(
        _gtf_row(stance=stance),
        source="gtfintechlab/bank_of_japan",
        provenance=provenance,
        cross_bank_mode=mode,
        cross_bank_stance_weight=stance_weight,
    )
    assert row is not None
    return row


def test_off_mode_keeps_fomc_row_masks_intact() -> None:
    """``off`` is a no-op for FOMC rows: stance stays masked-in at
    weight 1.0 and the other axes follow their natural per-row mask."""

    fomc = _axis_row_from_gtf(provenance="peer_reviewed", mode="off")
    assert fomc.masks["stance"] is True
    assert fomc.masks["certainty"] is True
    assert fomc.stance_sample_weight == 1.0


def test_stance_masked_mode_drops_stance_for_cross_bank_only() -> None:
    """The ``stance_masked`` arm rewrites the per-row stance mask
    to False for every ``peer_reviewed_cross_bank`` row and leaves
    the other axis masks intact. FOMC rows are untouched."""

    cross_bank = _axis_row_from_gtf(
        provenance="peer_reviewed_cross_bank", mode="stance_masked"
    )
    fomc = _axis_row_from_gtf(provenance="peer_reviewed", mode="stance_masked")

    assert cross_bank.masks["stance"] is False
    assert cross_bank.masks["certainty"] is True
    assert cross_bank.stance_sample_weight == 1.0

    assert fomc.masks["stance"] is True
    assert fomc.masks["certainty"] is True
    assert fomc.stance_sample_weight == 1.0


def test_weighted_mode_scales_only_stance_for_cross_bank() -> None:
    """The ``weighted`` arm leaves every mask intact and scales the
    cross-bank rows' stance weight by ``cross_bank_stance_weight``.
    Other axes (certainty, factor, topic) are NOT scaled — the
    weight rides on the stance branch alone."""

    cross_bank = _axis_row_from_gtf(
        provenance="peer_reviewed_cross_bank",
        mode="weighted",
        stance_weight=0.25,
    )
    fomc = _axis_row_from_gtf(
        provenance="peer_reviewed", mode="weighted", stance_weight=0.25
    )

    assert cross_bank.masks["stance"] is True
    assert cross_bank.masks["certainty"] is True
    assert cross_bank.stance_sample_weight == pytest.approx(0.25)

    # FOMC rows are not in the cross-bank bucket, so the weight stays
    # at 1.0 even when the arm is ``weighted``.
    assert fomc.masks["stance"] is True
    assert fomc.stance_sample_weight == 1.0


def test_collate_emits_per_row_stance_weight_under_weighted_arm() -> None:
    """End-to-end: the collate output surfaces a per-row stance
    sample-weight tensor scaled by 0.25 for cross-bank rows. The
    stance mask is all-True so the head still sees the cross-bank
    stance labels — this is the diagnostic A/B for the
    substitute-vs-complement prior."""

    cross_bank = _axis_row_from_gtf(
        provenance="peer_reviewed_cross_bank",
        mode="weighted",
        stance_weight=0.25,
    )
    fomc = _axis_row_from_gtf(provenance="peer_reviewed", mode="weighted")
    rows = [fomc, cross_bank, cross_bank, fomc]

    tokenizer = _FakeTokenizer()
    from app.data.train_text_multi_axis_classifier import _MultiAxisDataset

    ds = _MultiAxisDataset(rows, tokenizer, max_length=8)
    batch = _collate([ds[i] for i in range(len(rows))])

    assert torch.equal(
        batch["mask_stance"], torch.tensor([True, True, True, True])
    )
    assert torch.allclose(
        batch["stance_sample_weight"],
        torch.tensor([1.0, 0.25, 0.25, 1.0], dtype=torch.float32),
    )
    assert batch["provenance"] == [
        "peer_reviewed",
        "peer_reviewed_cross_bank",
        "peer_reviewed_cross_bank",
        "peer_reviewed",
    ]


def test_collate_masks_stance_for_cross_bank_under_stance_masked_arm() -> None:
    """Companion end-to-end check for the ``stance_masked`` arm: the
    cross-bank rows arrive in the batch with stance False (so the
    masked loss skips them on stance) while their certainty mask
    stays True (so the encoder still trains on the auxiliary axis)."""

    cross_bank = _axis_row_from_gtf(
        provenance="peer_reviewed_cross_bank", mode="stance_masked"
    )
    fomc = _axis_row_from_gtf(provenance="peer_reviewed", mode="stance_masked")
    rows = [cross_bank, fomc, cross_bank]

    from app.data.train_text_multi_axis_classifier import _MultiAxisDataset

    ds = _MultiAxisDataset(rows, _FakeTokenizer(), max_length=8)
    batch = _collate([ds[i] for i in range(len(rows))])

    assert torch.equal(
        batch["mask_stance"], torch.tensor([False, True, False])
    )
    # All other axes stay aligned with the per-row natural mask.
    assert torch.equal(
        batch["mask_certainty"], torch.tensor([True, True, True])
    )
    assert torch.allclose(
        batch["stance_sample_weight"],
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    )


def test_per_axis_provenance_log_splits_by_corpus(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first-epoch sanity log must split per-axis counts by
    provenance bucket so any regression that leaks cross-bank rows
    into the FOMC stance head shows up as a non-zero
    ``from_cross_bank`` column on the ``stance`` line. The
    ``stance_masked`` rewrite makes ``from_cross_bank`` on stance
    zero by construction even when the cross-bank rows are admitted
    into the pool."""

    rows = [
        _axis_row_from_gtf(provenance="peer_reviewed", mode="stance_masked"),
        _axis_row_from_gtf(provenance="peer_reviewed", mode="stance_masked"),
        _axis_row_from_gtf(
            provenance="peer_reviewed_cross_bank", mode="stance_masked"
        ),
        _axis_row_from_gtf(
            provenance="peer_reviewed_cross_bank", mode="stance_masked"
        ),
    ]
    caplog.set_level("INFO", logger="app.data.train_text_multi_axis_classifier")
    _log_per_axis_provenance_breakdown(rows)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert (
        "axis=stance rows_total=2 from_FOMC=2 from_cross_bank=0" in messages
    )
    assert (
        "axis=certainty rows_total=4 from_FOMC=2 from_cross_bank=2"
        in messages
    )
