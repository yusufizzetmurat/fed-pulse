"""The time axis is in the model head AND supervised by the trainer.

Pins the contract that retired the ``factor`` axis in favour of the
``time`` (forward-looking) axis across the model head, the gtfintechlab
row mapper, the events.parquet row mapper, and the inference service.
"""

from __future__ import annotations

import torch

from app.models.config import MULTI_TASK_TIME_LABELS


def test_multi_task_head_emits_time_branch_with_two_classes() -> None:
    """The head must emit a ``time`` branch of shape ``(B, 2)``."""
    from app.models.multi_task_head import MultiTaskHead

    head = MultiTaskHead(hidden_size=16, head_hidden_size=8, dropout=0.0)
    out = head(torch.zeros(5, 16))
    assert "time" in out
    assert out["time"].shape == (5, 2)
    assert len(MULTI_TASK_TIME_LABELS) == 2


def test_gtfintechlab_row_maps_forward_looking_time_label() -> None:
    """A gtfintechlab row carrying ``time_label="forward looking"`` must
    set ``masks["time"] = True`` and ``targets["time"] == 0``."""
    from app.data.train_text_multi_axis_classifier import (
        _gtfintechlab_row_to_axis_row,
    )

    row = _gtfintechlab_row_to_axis_row(
        {
            "sentences": "Rates are likely to rise next year.",
            "stance_label": "hawkish",
            "certain_label": "certain",
            "time_label": "forward looking",
            "year": 2024,
        }
    )
    assert row is not None
    assert row.masks["time"] is True
    assert row.targets["time"] == MULTI_TASK_TIME_LABELS.index("forward looking")
    assert row.targets["time"] == 0


def test_row_targets_maps_not_forward_looking_from_events_parquet() -> None:
    """``_row_targets`` on an events.parquet row carrying
    ``axis_time_label="not forward looking"`` must set the time mask
    True with index 1."""
    from app.data.train_text_multi_axis_classifier import _row_targets

    targets, masks = _row_targets(
        {
            "axis_stance": "dovish",
            "axis_time_label": "not forward looking",
            "axis_certain_label": "uncertain",
        }
    )
    assert masks["time"] is True
    assert targets["time"] == MULTI_TASK_TIME_LABELS.index("not forward looking")
    assert targets["time"] == 1
    # The retired factor axis must be absent from the target dict.
    assert "factor" not in targets
    assert "factor" not in masks


def test_score_text_emits_a_time_card_with_a_valid_label(monkeypatch) -> None:
    """The inference service output must include a ``time`` dict whose
    ``label`` is one of ``MULTI_TASK_TIME_LABELS`` and no ``factor`` key."""
    from app.services import multi_axis_classifier as svc

    class _StubModel:
        def __call__(self, *, input_ids, attention_mask):
            batch = input_ids.shape[0]
            return {
                "stance": torch.zeros(batch, 3),
                "certainty": torch.zeros(batch, 3),
                "time": torch.tensor([[2.0, 0.0]]).repeat(batch, 1),
            }

    class _StubTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.zeros(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }

    state = svc._ClassifierState(
        model=_StubModel(),  # type: ignore[arg-type]
        tokenizer=_StubTokenizer(),
        device=torch.device("cpu"),
        max_length=8,
        encoder_alias="stub",
    )
    monkeypatch.setattr(svc, "get_classifier", lambda: state)

    out = svc.score_text("Inflation will ease over the coming quarters.")
    assert out is not None
    assert "factor" not in out
    assert "time" in out
    assert out["time"]["label"] in MULTI_TASK_TIME_LABELS
    # Logits [2.0, 0.0] -> argmax 0 -> "forward looking".
    assert out["time"]["label"] == MULTI_TASK_TIME_LABELS[0]
    assert 0.0 <= out["time"]["confidence"] <= 1.0
    assert set(out["time"]["distribution"].keys()) == set(MULTI_TASK_TIME_LABELS)
