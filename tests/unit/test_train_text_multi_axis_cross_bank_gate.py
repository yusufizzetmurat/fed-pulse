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
    _compute_weighted_total_loss,
    _evaluate,
    _gtfintechlab_row_to_axis_row,
    _log_per_axis_provenance_breakdown,
)
from app.training.loss import MultiTaskLoss  # noqa: E402


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

    # Pin the natural per-row factor / topic masks on a cross-bank
    # row. The gtfintechlab schema does not carry factor or topic
    # labels (the trainer maps only stance + certainty + time), so
    # the natural masks are False here. Any future change that
    # leaks a cross-bank stance label onto the factor or topic
    # mask under ``stance_masked`` would flip these — the regression
    # would show up as the assertion firing.
    assert cross_bank.masks["factor"] is False
    assert cross_bank.masks["topic"] is False
    # FOMC rows under ``stance_masked`` follow the same natural
    # per-row factor / topic masks the gtfintechlab schema produces;
    # ``stance_masked`` rewrites are scoped to cross-bank rows only.
    assert fomc.masks["factor"] is False
    assert fomc.masks["topic"] is False


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
    into the pool. The non-cross-bank bucket is named ``from_other``
    so a future provenance that is neither FOMC nor the recognised
    cross-bank tag does not silently inflate the counter."""

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
        "axis=stance rows_total=2 from_other=2 from_cross_bank=0" in messages
    )
    assert (
        "axis=certainty rows_total=4 from_other=2 from_cross_bank=2"
        in messages
    )


# --- Gradient-path tests for _compute_weighted_total_loss --------------
#
# These guard against the double-gradient regression that the
# pre-fix helper produced: it subtracted ``breakdown["stance"]`` (a
# detached scalar) from ``total`` and added the new weighted stance
# loss. That subtraction does NOT remove the original
# graph-attached stance loss path, so ``total.backward()``
# accumulated stance gradients from BOTH the unweighted and the
# weighted CE — a silent double-count. The current helper delegates
# to ``MultiTaskLoss.forward(..., stance_sample_weight=...)`` so
# there is exactly one stance loss term in the graph. The first
# test pins the analytical gradient; the others pin the corner
# cases (zero-weight collapse, FOMC-only byte-identity, and
# non-stance axis isolation).
#
# All four tests fail on the pre-fix code (confirmed by stashing
# the fix and re-running them).


def _make_two_row_two_class_logits(
    *, target_classes: tuple[int, int], stance_logits: list[list[float]]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tiny 2-row, 2-class fixture with stance + factor + certainty + topic.

    Only the stance axis is exercised by the gradient assertions; the
    other axes carry False masks so they contribute graph-attached
    zeros (matching the MultiTaskLoss empty-mask contract). The
    factor / certainty / topic tensors are still ``requires_grad``
    so the loss can graph-attach a zero through them — the gradient
    on those tensors must be zero everywhere when their masks are
    all-False.
    """

    stance = torch.tensor(stance_logits, dtype=torch.float32, requires_grad=True)
    factor = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    certainty = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
    topic = torch.zeros(2, 4, dtype=torch.float32, requires_grad=True)
    logits = {
        "stance": stance,
        "factor": factor,
        "certainty": certainty,
        "topic": topic,
    }
    targets = {
        "stance": torch.tensor(list(target_classes), dtype=torch.long),
        "factor": torch.zeros(2, dtype=torch.float32),
        "certainty": torch.zeros(2, dtype=torch.long),
        "topic": torch.zeros(2, dtype=torch.long),
    }
    masks = {
        "stance_mask": torch.tensor([True, True], dtype=torch.bool),
        "factor_mask": torch.tensor([False, False], dtype=torch.bool),
        "certainty_mask": torch.tensor([False, False], dtype=torch.bool),
        "topic_mask": torch.tensor([False, False], dtype=torch.bool),
    }
    return logits, targets, masks


def test_weighted_stance_gradient_matches_analytical_2row_2class() -> None:
    """Gradient on stance logits under ``weighted`` arm matches the
    closed-form weighted CE gradient.

    Setup: 2 rows, 2 classes, no class weights, per-row weights
    ``[1.0, 0.25]``, lambda_stance=1.0 (so the stance branch is the
    full total loss). Stance logits chosen so softmax is exact and
    easy to write down. The expected gradient on row i is
    ``(w_i / sum(w)) * (softmax_i - onehot(target_i))``.
    """

    # Row 0 logits [2.0, 0.0] with target=0 → softmax ≈
    # [e^2/(e^2+1), 1/(e^2+1)] ≈ [0.8807970, 0.1192030].
    # Row 1 logits [0.0, 1.0] with target=1 → softmax ≈
    # [1/(1+e), e/(1+e)] ≈ [0.2689414, 0.7310586].
    logits, targets, masks = _make_two_row_two_class_logits(
        target_classes=(0, 1),
        stance_logits=[[2.0, 0.0], [0.0, 1.0]],
    )
    loss_fn = MultiTaskLoss(lambda_stance=1.0, lambda_factor=0.0,
                            lambda_certainty=0.0, lambda_topic=0.0)
    weights = torch.tensor([1.0, 0.25], dtype=torch.float32)

    total, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits,
        targets=targets,
        masks=masks,
        stance_sample_weight=weights,
    )
    total.backward()

    # Closed-form expected gradient.
    softmax = torch.softmax(
        torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32), dim=-1
    )
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    weight_total = weights.sum()
    expected = (weights.unsqueeze(-1) / weight_total) * (softmax - onehot)

    assert logits["stance"].grad is not None
    torch.testing.assert_close(
        logits["stance"].grad, expected, rtol=1e-6, atol=1e-6
    )


def test_all_zero_weight_batch_emits_zero_stance_gradient() -> None:
    """When every cross-bank row has stance_sample_weight=0.0 and no
    FOMC rows are in the batch, the stance loss must contribute
    exactly zero gradient through ``logits["stance"]`` — backward
    must still succeed (no NaN, no error) because the loss returns a
    graph-attached zero from ``logits.sum() * 0.0``."""

    logits, targets, masks = _make_two_row_two_class_logits(
        target_classes=(0, 1),
        stance_logits=[[2.0, 0.0], [0.0, 1.0]],
    )
    loss_fn = MultiTaskLoss(lambda_stance=1.0, lambda_factor=0.0,
                            lambda_certainty=0.0, lambda_topic=0.0)
    weights = torch.zeros(2, dtype=torch.float32)

    total, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits,
        targets=targets,
        masks=masks,
        stance_sample_weight=weights,
    )
    total.backward()

    assert logits["stance"].grad is not None
    assert torch.equal(
        logits["stance"].grad,
        torch.zeros_like(logits["stance"].grad),
    )
    # And the total itself is exactly zero (the only contributing
    # axis was masked-out by the zero weight).
    assert float(total.detach().item()) == 0.0


def test_fomc_only_batch_byte_identical_to_unweighted_loss() -> None:
    """With all rows FOMC (provenance != cross-bank) and weights all
    1.0, ``_compute_weighted_total_loss`` must return a tensor
    numerically equal to ``MultiTaskLoss(logits, targets, masks)[0]``
    — the FOMC-only training run must reproduce the prior numerics
    byte-for-byte so the strict-FOMC headline does not drift."""

    logits_a, targets_a, masks_a = _make_two_row_two_class_logits(
        target_classes=(0, 1),
        stance_logits=[[2.0, 0.0], [0.0, 1.0]],
    )
    logits_b, targets_b, masks_b = _make_two_row_two_class_logits(
        target_classes=(0, 1),
        stance_logits=[[2.0, 0.0], [0.0, 1.0]],
    )
    loss_fn = MultiTaskLoss(lambda_stance=1.0, lambda_factor=0.3,
                            lambda_certainty=0.3, lambda_topic=0.3)
    weights = torch.ones(2, dtype=torch.float32)

    weighted_total, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits_a,
        targets=targets_a,
        masks=masks_a,
        stance_sample_weight=weights,
    )
    plain_total, _ = loss_fn(logits_b, targets_b, masks_b)

    torch.testing.assert_close(
        weighted_total, plain_total, rtol=0.0, atol=0.0
    )


def test_weighted_arm_leaves_other_axis_gradients_unchanged() -> None:
    """The cross-bank ``weighted`` arm scales only the stance branch.
    Gradients on logits["factor"], logits["certainty"], and
    logits["topic"] must be identical to the gradients produced by
    the ``off`` baseline (i.e. ``MultiTaskLoss`` called without any
    per-row weight). Tested with a populated mask on each non-stance
    axis so the comparison is non-trivial."""

    def _seeded_batch(seed: int) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        torch.manual_seed(seed)
        stance = torch.randn(3, 3, requires_grad=True)
        factor = torch.randn(3, requires_grad=True)
        certainty = torch.randn(3, 3, requires_grad=True)
        topic = torch.randn(3, 4, requires_grad=True)
        return (
            {
                "stance": stance,
                "factor": factor,
                "certainty": certainty,
                "topic": topic,
            },
            {
                "stance": torch.tensor([0, 1, 2], dtype=torch.long),
                "factor": torch.tensor([0.1, -0.4, 0.7], dtype=torch.float32),
                "certainty": torch.tensor([0, 2, 1], dtype=torch.long),
                "topic": torch.tensor([1, 3, 0], dtype=torch.long),
            },
            {
                "stance_mask": torch.tensor([True, True, True], dtype=torch.bool),
                "factor_mask": torch.tensor([True, True, True], dtype=torch.bool),
                "certainty_mask": torch.tensor(
                    [True, True, True], dtype=torch.bool
                ),
                "topic_mask": torch.tensor([True, True, True], dtype=torch.bool),
            },
        )

    loss_fn = MultiTaskLoss(lambda_stance=1.0, lambda_factor=0.3,
                            lambda_certainty=0.3, lambda_topic=0.3)

    # Weighted (cross-bank arm with non-unit weights).
    logits_w, targets_w, masks_w = _seeded_batch(seed=17)
    weights = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32)
    total_w, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits_w,
        targets=targets_w,
        masks=masks_w,
        stance_sample_weight=weights,
    )
    total_w.backward()

    # Baseline (``off`` — same seeded inputs, no per-row weight).
    logits_o, targets_o, masks_o = _seeded_batch(seed=17)
    total_o, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits_o,
        targets=targets_o,
        masks=masks_o,
        stance_sample_weight=None,
    )
    total_o.backward()

    for axis in ("factor", "certainty", "topic"):
        assert logits_w[axis].grad is not None
        assert logits_o[axis].grad is not None
        torch.testing.assert_close(
            logits_w[axis].grad,
            logits_o[axis].grad,
            rtol=1e-6,
            atol=1e-6,
        )


# --- _evaluate train/val parity --------------------------------------
#
# Pins that ``_evaluate`` threads ``stance_sample_weight`` through to
# ``loss_fn`` so the val loss tracks the same objective the optimizer
# minimizes on the train side. Without the thread-through, the train
# arm runs weighted CE on stance while the val arm runs unweighted CE
# under ``--cross-bank-supervision=weighted`` — and best-checkpoint
# selection by val_loss picks against the wrong objective.


class _ConstantLogitsModel(torch.nn.Module):
    """Stand-in module whose ``forward`` returns a fixed logits dict.

    The trainer only reads ``model(input_ids, attention_mask)`` so a
    constant-output model is enough to exercise the loss + weight path
    without the real encoder. The dummy ``_param`` keeps optimizer.step
    happy in the train-arm tests (no real parameter update happens
    since the logits don't depend on it)."""

    def __init__(self, logits: dict[str, "torch.Tensor"]) -> None:
        super().__init__()
        self._logits = logits
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(
        self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> dict[str, "torch.Tensor"]:
        # Re-emit the cached logits each call. ``_evaluate`` is called
        # under ``torch.no_grad()`` so any graph attachment via
        # ``self._param`` would be a no-op on the val side anyway.
        return {k: v.clone() for k, v in self._logits.items()}


def _two_row_batch(weights: list[float]) -> dict[str, "torch.Tensor"]:
    """Build a 2-row collated batch with the given per-row stance weights."""

    return {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "target_stance": torch.tensor([0, 1], dtype=torch.long),
        "target_factor": torch.zeros(2, dtype=torch.float32),
        "target_certainty": torch.zeros(2, dtype=torch.long),
        "target_topic": torch.zeros(2, dtype=torch.long),
        "mask_stance": torch.tensor([True, True], dtype=torch.bool),
        "mask_factor": torch.tensor([False, False], dtype=torch.bool),
        "mask_certainty": torch.tensor([False, False], dtype=torch.bool),
        "mask_topic": torch.tensor([False, False], dtype=torch.bool),
        "stance_sample_weight": torch.tensor(weights, dtype=torch.float32),
        "source": ["fomc", "fomc"],
        "provenance": ["peer_reviewed", "peer_reviewed"],
    }


def test_evaluate_threads_stance_sample_weight_through_loss() -> None:
    """Train and val arms must produce the same loss on the same toy
    batch when the same per-row weights are applied. Pre-fix, the val
    arm ignored ``stance_sample_weight`` and ran unweighted CE — the
    two losses diverged under the ``weighted`` arm. This test fails
    on the old code (sees the unweighted/weighted gap) and passes on
    the fixed code.

    Uses non-uniform weights ``[1.0, 0.25]`` so the weighted vs plain
    mean produce different numerics (uniform weights collapse to the
    plain mean and would mask the bug).
    """

    # Deterministic logits chosen so the two rows have meaningfully
    # different per-row CE — non-uniform weighting then yields a
    # detectably different total from the plain mean.
    stance_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]], dtype=torch.float32
    )
    factor_logits = torch.zeros(2, dtype=torch.float32)
    certainty_logits = torch.zeros(2, 3, dtype=torch.float32)
    topic_logits = torch.zeros(2, 4, dtype=torch.float32)
    logits = {
        "stance": stance_logits,
        "factor": factor_logits,
        "certainty": certainty_logits,
        "topic": topic_logits,
    }
    model = _ConstantLogitsModel(logits)
    loss_fn = MultiTaskLoss(
        lambda_stance=1.0,
        lambda_factor=0.0,
        lambda_certainty=0.0,
        lambda_topic=0.0,
    )

    batch = _two_row_batch([1.0, 0.25])

    class _OneShotLoader:
        def __init__(self, batch: dict[str, "torch.Tensor"]) -> None:
            self._batch = batch

        def __iter__(self):
            return iter([self._batch])

    # Eval loss with the per-row stance weight threaded through.
    val_metrics_weighted = _evaluate(
        model, _OneShotLoader(batch), loss_fn, torch.device("cpu")
    )

    # Train-side reference: ``_compute_weighted_total_loss`` is the
    # train arm's loss-computation helper. Calling it directly on the
    # same batch + same stance weight produces the loss the optimizer
    # actually minimizes. Eval must match.
    targets = {
        "stance": batch["target_stance"],
        "factor": batch["target_factor"],
        "certainty": batch["target_certainty"],
        "topic": batch["target_topic"],
    }
    masks = {
        "stance_mask": batch["mask_stance"],
        "factor_mask": batch["mask_factor"],
        "certainty_mask": batch["mask_certainty"],
        "topic_mask": batch["mask_topic"],
    }
    train_total, _ = _compute_weighted_total_loss(
        loss_fn=loss_fn,
        logits=logits,
        targets=targets,
        masks=masks,
        stance_sample_weight=batch["stance_sample_weight"],
    )

    assert val_metrics_weighted["loss"] == pytest.approx(
        float(train_total.detach().item()), rel=0.0, abs=1e-6
    )

    # And the val_loss under the non-uniform weight differs from the
    # val_loss the pre-fix path would have produced (plain unweighted
    # CE). The gap pins that the kwarg actually rode through to the
    # loss formula.
    plain_total, _ = loss_fn(logits, targets, masks)
    assert val_metrics_weighted["loss"] != pytest.approx(
        float(plain_total.detach().item()), abs=1e-6
    )


def test_evaluate_runs_under_no_grad() -> None:
    """``_evaluate`` must not retain a computation graph on the
    forward pass — ``model.eval()`` flips dropout/BN but does NOT
    disable autograd. The model logits inside the eval forward must
    have ``requires_grad=False`` because the ``@torch.no_grad()``
    decorator suppresses graph construction. We capture the logits
    via a side-effect list and assert their grad state."""

    seen_requires_grad: list[bool] = []
    leaf = torch.nn.Parameter(torch.tensor(1.0))

    class _GraphAttachingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.leaf = leaf

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            # Multiply by the leaf so the output WOULD attach to a
            # grad graph in autograd-enabled mode. Under
            # ``@torch.no_grad()`` the multiplication still produces
            # a tensor but with ``requires_grad=False``.
            scale = self.leaf
            stance = torch.zeros(2, 3) * scale
            seen_requires_grad.append(bool(stance.requires_grad))
            return {
                "stance": stance,
                "factor": torch.zeros(2) * scale,
                "certainty": torch.zeros(2, 3) * scale,
                "topic": torch.zeros(2, 4) * scale,
            }

    loss_fn = MultiTaskLoss(
        lambda_stance=1.0,
        lambda_factor=0.0,
        lambda_certainty=0.0,
        lambda_topic=0.0,
    )

    class _OneShotLoader:
        def __init__(self, batch: dict[str, "torch.Tensor"]) -> None:
            self._batch = batch

        def __iter__(self):
            return iter([self._batch])

    batch = _two_row_batch([1.0, 1.0])
    metrics = _evaluate(
        _GraphAttachingModel(),
        _OneShotLoader(batch),
        loss_fn,
        torch.device("cpu"),
    )
    # Eval forward observed the logits as detached — autograd graph
    # construction was suppressed by ``@torch.no_grad()``. Without
    # the decorator, multiplying ``torch.zeros(...) * leaf`` (where
    # ``leaf.requires_grad=True``) would produce a graph-attached
    # tensor with ``requires_grad=True``.
    assert seen_requires_grad == [False]
    assert "loss" in metrics
    # Logits are all zero → softmax is uniform across 3 classes →
    # CE = ln(3) ≈ 1.0986. Sanity-check the formula matches.
    import math

    assert metrics["loss"] == pytest.approx(math.log(3.0), abs=1e-5)


# --- Checkpoint provenance: effective_cross_bank_supervision ---------
#
# When ``--gtfintechlab-fed-only`` is combined with a non-``off``
# cross-bank arm, the loader silently restricts to FOMC. The
# checkpoint payload must record this resolution honestly:
# ``effective_cross_bank_supervision`` is ``"off"`` regardless of the
# requested arm. The raw ``cross_bank_supervision`` field stays so
# operators can trace what they originally asked for.


class _StubMultiAxisModel(torch.nn.Module):
    """Bare-minimum stand-in for ``TextMultiAxisClassifier`` so the
    checkpoint payload test can exercise ``_save_checkpoint`` without
    pulling a real encoder off HF Hub.

    Only the two surfaces ``_save_checkpoint`` touches are implemented:
    ``state_dict`` (inherited from ``nn.Module``) and ``metadata()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def metadata(self) -> dict[str, object]:
        return {"encoder_alias": "stub", "encoder_revision": "stub"}


def _checkpoint_args(*, fed_only: bool, supervision: str) -> "argparse.Namespace":
    import argparse

    return argparse.Namespace(
        training_package_id="dummy",
        encoder_alias="finbert",
        epochs=1,
        seed=0,
        learning_rate=1e-5,
        batch_size=2,
        val_fraction=0.1,
        max_length=8,
        data_source="gtfintechlab_hf",
        gtfintechlab_fed_only=fed_only,
        cross_bank_supervision=supervision,
        cross_bank_stance_weight=0.25,
    )


def test_checkpoint_payload_records_effective_cross_bank_under_fed_only(
    tmp_path,
) -> None:
    """The checkpoint payload must surface
    ``effective_cross_bank_supervision="off"`` when fed_only is True,
    even if the operator asked for ``weighted`` — and must keep the
    raw request alongside for traceability."""

    from app.data.train_text_multi_axis_classifier import _save_checkpoint

    args = _checkpoint_args(fed_only=True, supervision="weighted")
    ckpt_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        _StubMultiAxisModel(),
        path=ckpt_path,
        metrics={"val_loss": 0.0},
        args=args,
        class_weights={},
        factor_coverage=0.0,
    )
    payload = torch.load(ckpt_path, weights_only=False)
    ta = payload["training_args"]
    # Raw request is preserved for traceability.
    assert ta["cross_bank_supervision"] == "weighted"
    # Effective field reflects what the run actually did — under
    # fed_only the cross-bank pool is empty so the arm reduces to off.
    assert ta["effective_cross_bank_supervision"] == "off"
    assert ta["gtfintechlab_fed_only"] is True
    # The #328 factor-coverage stamp rides on every checkpoint so the
    # inference service can gate the factor card on it.
    assert ta["factor_coverage"] == 0.0


def test_checkpoint_payload_effective_matches_raw_when_not_fed_only(
    tmp_path,
) -> None:
    """When fed_only is False, the effective field mirrors the raw
    request — operator asked for ``weighted``, the run ran
    ``weighted``."""

    from app.data.train_text_multi_axis_classifier import _save_checkpoint

    args = _checkpoint_args(fed_only=False, supervision="weighted")
    ckpt_path = tmp_path / "ckpt.pt"
    _save_checkpoint(
        _StubMultiAxisModel(),
        path=ckpt_path,
        metrics={"val_loss": 0.0},
        args=args,
        class_weights={},
        factor_coverage=0.25,
    )
    payload = torch.load(ckpt_path, weights_only=False)
    ta = payload["training_args"]
    assert ta["cross_bank_supervision"] == "weighted"
    assert ta["effective_cross_bank_supervision"] == "weighted"
    assert ta["gtfintechlab_fed_only"] is False
    # The trainer rounds-trips whatever coverage the caller computed
    # onto the payload — the gate decision lives on the inference side.
    assert ta["factor_coverage"] == 0.25
