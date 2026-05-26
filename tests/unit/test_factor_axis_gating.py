"""Factor-axis gating on the multi-axis classifier (#328).

The text multi-axis classifier ships with four output branches
(stance / factor / certainty / topic), but the canonical training
package today has 0 % ``axis_factor`` label coverage — the factor
regression branch trains almost exclusively on the masked-out path,
so its outputs are noise. ADR 0018 documents the decision to gate
the factor card on the persisted coverage so the /analyze response
emits ``factor=None`` rather than rendering a noise prediction.

These tests pin the gating contract end-to-end on the inference
service:

- A 0 % coverage stamp drops the factor card.
- A pre-#328 checkpoint (no ``factor_coverage`` field on the payload)
  also drops the card — the absent stamp is treated as "unknown",
  consistent with the new default behaviour rather than the legacy
  one.
- A coverage stamp above the gate emits a real factor value.
- The other three cards (stance / certainty / topic) keep emitting
  regardless of the factor stamp — gating is scoped to the factor
  card only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from app.services import multi_axis_classifier as svc


class _StubModel:
    """Stand-in for ``TextMultiAxisClassifier`` in the service tests.

    The unit under test is the gating logic in ``score_text`` /
    ``_build_factor_card``; the actual transformer forward is
    irrelevant. The stub returns deterministic logits so the test
    asserts on the gate, not on softmax numerics.
    """

    def __init__(self, *, factor_value: float = 0.7) -> None:
        self._factor_value = float(factor_value)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del input_ids, attention_mask  # unused — fixture passes shaped tensors
        return {
            # Single-row batches: shape (1, n_classes) for classifier
            # heads, (1,) for the regression head — matches the
            # TextMultiAxisClassifier forward contract.
            "stance": torch.tensor([[0.1, 0.7, 0.2]]),
            "factor": torch.tensor([self._factor_value]),
            "certainty": torch.tensor([[0.6, 0.3, 0.1]]),
            "topic": torch.tensor([[0.5, 0.2, 0.2, 0.1]]),
        }


class _StubTokenizer:
    """Returns a single fixed-shape encoded batch.

    The classifier service tokenizes once per ``score_text`` call;
    the stub keeps the call signature compatible without dragging a
    real tokenizer (and its HF dependency) into the unit test.
    """

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del text, max_length, padding, truncation, return_tensors
        return {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
        }


def _install_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factor_coverage: float | None,
    factor_value: float = 0.7,
    threshold: float = svc.DEFAULT_FACTOR_COVERAGE_GATE,
) -> None:
    """Drop a fully-formed ``_ClassifierState`` into the singleton slot.

    Bypasses ``_load_state`` so the test does not hit the HF
    encoder/tokenizer load path; the gating logic in ``score_text``
    only reads ``factor_coverage`` + ``factor_coverage_threshold``
    off the state, which is what the test exercises.
    """

    state = svc._ClassifierState(
        model=_StubModel(factor_value=factor_value),
        tokenizer=_StubTokenizer(),
        device=torch.device("cpu"),
        max_length=8,
        encoder_alias="stub_encoder",
        factor_coverage=factor_coverage,
        factor_coverage_threshold=threshold,
    )
    svc.reset_classifier()
    monkeypatch.setattr(svc, "_state", state)


def test_factor_card_absent_when_coverage_is_zero(monkeypatch) -> None:
    """0 % training-pool factor coverage trips the gate (factor=None).

    The canonical training package today is exactly this case — the
    factor head emits noise so the /analyze response omits the card."""

    _install_state(monkeypatch, factor_coverage=0.0)
    block = svc.score_text("FOMC statement body")
    assert block is not None
    assert block["factor"] is None
    # The other three axes stay populated — the gate is scoped to factor.
    assert block["stance"]["label"] in {"hawkish", "dovish", "neutral"}
    assert block["certainty"]["label"] in {"certain", "uncertain", "neutral"}
    assert block["topic"]["label"] in {
        "macro",
        "forward_guidance",
        "market_reaction",
        "other",
    }


def test_factor_card_absent_when_coverage_below_threshold(monkeypatch) -> None:
    """Coverage just below the gate also drops the card.

    Pins the strict-less-than comparison in ``_build_factor_card``;
    a coverage of 0.005 (half the default 0.01 gate) is treated the
    same as 0 %."""

    _install_state(monkeypatch, factor_coverage=0.005)
    block = svc.score_text("FOMC statement body")
    assert block is not None
    assert block["factor"] is None


def test_factor_card_absent_when_coverage_missing(monkeypatch) -> None:
    """Pre-#328 checkpoints that did not stamp ``factor_coverage`` are
    treated as unknown and drop the card.

    This is the safer default: rendering a noise prediction off a
    legacy checkpoint would surface the very behaviour ADR 0018
    closes out."""

    _install_state(monkeypatch, factor_coverage=None)
    block = svc.score_text("FOMC statement body")
    assert block is not None
    assert block["factor"] is None


def test_factor_card_emits_when_coverage_above_threshold(monkeypatch) -> None:
    """Real coverage above the gate emits the card with the head's value.

    Sanity check that the gate is the only thing controlling the
    card's emission — the factor head's forward still drives the
    value when the gate is open."""

    _install_state(monkeypatch, factor_coverage=0.5, factor_value=0.42)
    block = svc.score_text("FOMC statement body")
    assert block is not None
    factor = block["factor"]
    assert factor is not None
    assert pytest.approx(factor["value"], abs=1e-6) == 0.42
    # Confidence is the abs-as-confidence proxy documented in the service.
    assert pytest.approx(factor["confidence"], abs=1e-6) == 0.42


def test_factor_card_emits_at_threshold_boundary(monkeypatch) -> None:
    """At exactly the threshold the card emits (``< threshold`` drops).

    Locks the boundary condition: 0.01 coverage with a 0.01 gate
    yields a populated card; only strictly-lower values are gated
    off. Future operators tuning the gate via env override know the
    interval is right-open."""

    _install_state(
        monkeypatch,
        factor_coverage=svc.DEFAULT_FACTOR_COVERAGE_GATE,
        factor_value=-0.31,
    )
    block = svc.score_text("FOMC statement body")
    assert block is not None
    assert block["factor"] is not None
    assert block["factor"]["value"] == pytest.approx(-0.31, abs=1e-6)


def test_factor_card_respects_env_threshold_override(monkeypatch) -> None:
    """Tighter threshold via env knob drops a previously-passing coverage.

    Surfaces the operator-facing escape hatch: a 0.5 gate via the
    ``FED_PULSE_TEXT_MULTI_AXIS_FACTOR_GATE`` env wins over the 0.01
    default; a 0.3 coverage that would otherwise emit gets gated off."""

    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_FACTOR_GATE", "0.5")
    threshold = svc._resolve_factor_coverage_threshold()
    assert threshold == pytest.approx(0.5)
    _install_state(
        monkeypatch,
        factor_coverage=0.3,
        threshold=threshold,
    )
    block = svc.score_text("FOMC statement body")
    assert block is not None
    assert block["factor"] is None


def test_coerce_factor_coverage_reads_training_args() -> None:
    """The loader reads coverage off ``training_args.factor_coverage``.

    Pins the payload contract: the trainer's persistence path writes
    the float under ``training_args``; the inference loader reads it
    back from the same key (with a ``metadata`` fallback for
    forward-compat). Pre-#328 payloads (no key) come back as None."""

    assert svc._coerce_factor_coverage({}) is None
    assert svc._coerce_factor_coverage({"training_args": {}}) is None
    assert (
        svc._coerce_factor_coverage(
            {"training_args": {"factor_coverage": 0.7}}
        )
        == pytest.approx(0.7)
    )
    # Fallback path on the metadata bucket so a future writer that
    # stamps the field there does not break the loader.
    assert (
        svc._coerce_factor_coverage(
            {"metadata": {"factor_coverage": 0.25}}
        )
        == pytest.approx(0.25)
    )
    # Garbage / NaN values clip to None so a malformed payload does
    # not blow up the gate.
    import math

    assert (
        svc._coerce_factor_coverage(
            {"training_args": {"factor_coverage": "not a float"}}
        )
        is None
    )
    assert (
        svc._coerce_factor_coverage(
            {"training_args": {"factor_coverage": math.nan}}
        )
        is None
    )


def test_factor_coverage_clipped_to_unit_interval() -> None:
    """Out-of-range stamps are clipped to [0, 1] rather than rejected.

    Defensive — a malformed payload that wrote 1.7 or -0.2 still
    yields a sane gate decision. 1.7 clips to 1.0 (emits), -0.2 clips
    to 0.0 (gated)."""

    assert svc._coerce_factor_coverage(
        {"training_args": {"factor_coverage": 1.7}}
    ) == pytest.approx(1.0)
    assert svc._coerce_factor_coverage(
        {"training_args": {"factor_coverage": -0.5}}
    ) == pytest.approx(0.0)


def test_multi_axis_state_dataclass_carries_coverage_fields() -> None:
    """The dataclass surface includes the two new fields (#328).

    Touch the dataclass directly so a future refactor that drops
    either field will trip this test loudly rather than silently
    breaking the gate at runtime."""

    state = svc._ClassifierState(
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
        device=torch.device("cpu"),
        max_length=8,
        encoder_alias="stub_encoder",
        factor_coverage=0.0,
        factor_coverage_threshold=svc.DEFAULT_FACTOR_COVERAGE_GATE,
    )
    # ``replace`` is the canonical mutation helper for frozen
    # dataclasses; using it here doubles as a check that both fields
    # are named exactly as the rest of the service expects.
    bumped = replace(state, factor_coverage=0.5)
    assert bumped.factor_coverage == pytest.approx(0.5)
    assert bumped.factor_coverage_threshold == pytest.approx(
        svc.DEFAULT_FACTOR_COVERAGE_GATE
    )


def test_factor_coverage_fraction_helper_on_trainer_rows() -> None:
    """The trainer's row-counting helper matches the gate semantics.

    Mirrors the service-side gate on the trainer-side stamp source:
    rows that flag factor populated count toward the numerator,
    everything else does not, and an empty input list returns 0.0
    so misconfigured runs trip the gate cleanly downstream."""

    from app.data.train_text_multi_axis_classifier import (
        _AxisRow,
        _factor_coverage_fraction,
    )

    def _row(factor_present: bool) -> _AxisRow:
        return _AxisRow(
            text="x",
            targets={"stance": 0, "factor": 0.0, "certainty": 0, "topic": 0},
            masks={
                "stance": True,
                "factor": factor_present,
                "certainty": False,
                "topic": False,
            },
        )

    assert _factor_coverage_fraction([]) == pytest.approx(0.0)
    rows: list[_AxisRow] = [
        _row(False),
        _row(False),
        _row(True),
        _row(False),
    ]
    # 1 of 4 → 0.25.
    assert _factor_coverage_fraction(rows) == pytest.approx(0.25)
    rows_all_populated = [_row(True), _row(True), _row(True)]
    assert _factor_coverage_fraction(rows_all_populated) == pytest.approx(1.0)
