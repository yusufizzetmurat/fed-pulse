"""Artefact-level guard for the `text_path_ab_canonical.json` parity claim (#390).

The canonical sweep records the `per_bar` arm as byte-identical to the
`broadcast_static` arm. That is the expected outcome under the current
loader wiring: `build_per_bar_text_tensor` tile-replicates the anchor's
pooled embedding across every lookback bar (ADR 0017 §"Arm A — per-bar
text features"), and the per-bar adapter (Linear → LayerNorm → GELU)
is stateless, so a tile-replicated `(B, T, in_dim)` input collapses to
the same `(B, T, out_dim)` slot `broadcast_static` produces. The
forward-level invariant is pinned in `tests/unit/test_text_path_arms.py`
(`test_per_bar_parity_with_broadcast_static_when_constant_across_bars`).

This regression test pins the *artefact-level* property: every per-trial
metric value and every summary stat in the `per_bar` block must match
its `broadcast_static` counterpart byte-for-byte. A regeneration that
silently produces a different `per_bar` row without also flipping the
loader contract is either a corpus change worth re-captioning §6.15 or
a data-integrity bug worth investigating. Either way the test forces
the conversation to happen at PR review.

See `backend/artifacts/experiments/text_path_ab_canonical.README.md`
for the architectural reasoning. See issue #390 for the audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_PATH = (
    REPO_ROOT
    / "backend"
    / "artifacts"
    / "experiments"
    / "text_path_ab_canonical.json"
)


@pytest.fixture(scope="module")
def artefact() -> dict[str, object]:
    if not ARTEFACT_PATH.exists():
        pytest.skip(f"sweep artefact missing at {ARTEFACT_PATH}")
    return json.loads(ARTEFACT_PATH.read_text())


def test_artefact_shape(artefact: dict[str, object]) -> None:
    trials = artefact["trials"]
    assert isinstance(trials, dict)
    assert "broadcast_static" in trials and "per_bar" in trials
    bs = trials["broadcast_static"]
    pb = trials["per_bar"]
    assert isinstance(bs, list) and isinstance(pb, list)
    assert len(bs) == len(pb), (len(bs), len(pb))
    assert len(bs) == 5, "canonical sweep pins 5 seeds"


def test_per_bar_trials_byte_identical_to_broadcast_static(
    artefact: dict[str, object],
) -> None:
    """Per-trial metrics in `per_bar` must equal `broadcast_static`.

    See module docstring for the architectural reason. If this fails,
    either the loader started emitting a non-constant per-bar payload
    (in which case the §6.15 caption + the README next to the artefact
    need a refresh) or the sweep produced inconsistent output (in which
    case the regeneration is the bug).
    """

    trials = artefact["trials"]
    bs = trials["broadcast_static"]
    pb = trials["per_bar"]
    for i, (b_trial, p_trial) in enumerate(zip(bs, pb)):
        assert b_trial["seed"] == p_trial["seed"], (i, b_trial["seed"], p_trial["seed"])
        b_folds = b_trial["folds"]
        p_folds = p_trial["folds"]
        assert len(b_folds) == len(p_folds), (i, len(b_folds), len(p_folds))
        for j, (b_fold, p_fold) in enumerate(zip(b_folds, p_folds)):
            assert b_fold["fold_id"] == p_fold["fold_id"], (i, j)
            assert b_fold["metrics"] == p_fold["metrics"], (
                f"trial {i} seed={b_trial['seed']} fold {j} ({b_fold['fold_id']}): "
                f"per_bar diverges from broadcast_static — see "
                f"backend/artifacts/experiments/text_path_ab_canonical.README.md"
            )


def test_per_bar_summary_identical_to_broadcast_static(
    artefact: dict[str, object],
) -> None:
    summary = artefact["summary"]
    bs = summary["broadcast_static"]
    pb = summary["per_bar"]
    assert bs == pb, (
        "per_bar summary stats diverge from broadcast_static — "
        "see backend/artifacts/experiments/text_path_ab_canonical.README.md"
    )


def test_per_bar_config_marks_arm_correctly(artefact: dict[str, object]) -> None:
    """The arms collapse mathematically but the configs must still be distinct.

    The per-trial parity is the result of the loader feeding identical
    inputs to the two arms, not of the runner mis-labelling cells. The
    `text_channel` config key must reflect the arm name so a future
    auditor can re-derive which cell was meant to be which.
    """

    trials = artefact["trials"]
    for trial in trials["broadcast_static"]:
        assert trial["arm"] == "broadcast_static"
        assert trial["config"]["text_channel"] == "scalar"
    for trial in trials["per_bar"]:
        assert trial["arm"] == "per_bar"
        assert trial["config"]["text_channel"] == "per_bar"
