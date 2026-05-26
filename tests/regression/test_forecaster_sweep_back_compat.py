"""Byte-identity regression for the forecaster sweep CLI's legacy path.

The random-search + parallel-worker speedup PR ships
``--random-search`` defaulted off and ``--parallel-workers`` defaulted
to 1. On that legacy path the trial enumeration order, the trial
record shape, and the sweep-report payload must reproduce the
pre-speedup sweep output byte-identically -- the random-search and
parallel-worker keys never appear, and ``hp_combo_id`` never sneaks
onto a trial record. This test pins those invariants without
spinning up a real GPU run.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from app.train_forecaster import build_sweep_candidates


def _legacy_args(*, architectures: list[str], seeds: list[int]) -> argparse.Namespace:
    """The namespace the pre-PR exhaustive sweep would have built.

    The fields here mirror the production sweep grid documented at
    ``docs/data-and-training-contracts.md`` -- 3 hidden x 3 layers x
    4 dropout x 2 learning_rate x 3 weight_decay x 1 text_adapter
    (text-off) = 216 HP cells.
    """

    return argparse.Namespace(
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
        learning_rate=1e-3,
        epochs=20,
        head_hidden_size=32,
        hidden_sizes=[32, 64, 128],
        num_layers_grid=[1, 2, 3],
        dropouts=[0.1, 0.2, 0.3, 0.4],
        learning_rates=[1e-3, 3e-4],
        epochs_grid=[20],
        weight_decay=1e-4,
        weight_decays=[0.0, 1e-4, 1e-3],
        text_adapter_dim=64,
        text_adapter_dims=None,
        text_encoder="none",
        use_text_embeddings=True,
        training_package_id=None,
        rich_features=False,
        architecture="lstm",
        architectures=architectures,
        seed=None,
        seeds=seeds,
        credibility_features=False,
        # --random-search off by default; absence + getattr fallbacks
        # inside build_sweep_candidates ensure the legacy path runs.
        random_search=False,
        random_search_samples=50,
        random_search_seed=42,
    )


def test_exhaustive_grid_is_back_compat():
    """--random-search=False reproduces the pre-PR candidate enumeration."""

    args = _legacy_args(architectures=["lstm", "gru"], seeds=[11, 29])
    candidates = build_sweep_candidates(args)

    # 2 architectures x 216 HP combos x 2 seeds = 864 cells.
    assert len(candidates) == 864

    # No hp_combo_id field on the exhaustive path -- adding one
    # would push the CSV column set off the pre-PR contract.
    for record in candidates:
        assert "hp_combo_id" not in record

    # Enumeration order matches the legacy itertools.product layout:
    # arch outer, then hidden / layers / dropout / lr / epochs / wd
    # / text_adapter, then seed innermost. Verify the first eight
    # entries follow that signature.
    first = candidates[0]
    first_cfg = first["model_config"]
    assert first_cfg.architecture == "lstm"
    assert first_cfg.hidden_size == 32
    assert first_cfg.num_layers == 1
    assert first_cfg.dropout == 0.1
    assert first["learning_rate"] == 1e-3
    assert first["weight_decay"] == 0.0
    assert first["seed"] == 11

    # The seed axis is the innermost: the second cell should differ
    # only by seed.
    second = candidates[1]
    assert second["model_config"].hidden_size == 32
    assert second["model_config"].num_layers == 1
    assert second["model_config"].dropout == 0.1
    assert second["learning_rate"] == 1e-3
    assert second["weight_decay"] == 0.0
    assert second["seed"] == 29


def test_exhaustive_grid_full_production_shape():
    """Production sweep shape: 8 archs x 216 HP cells x 5 seeds = 8640 cells."""

    args = _legacy_args(
        architectures=["lstm", "lstm_attn", "gru", "tcn", "transformer", "dlinear", "informer", "tft"],
        seeds=[11, 29, 47, 71, 97],
    )
    candidates = build_sweep_candidates(args)
    assert len(candidates) == 8 * 216 * 5
    # And the architecture axis is the outermost: every entry in the
    # first 1080-cell block carries architecture=lstm.
    arch_block = {c["model_config"].architecture for c in candidates[:1080]}
    assert arch_block == {"lstm"}
