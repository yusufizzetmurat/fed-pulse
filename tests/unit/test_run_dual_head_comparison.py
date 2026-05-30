"""Regression tests for the canonical sweep runners.

Covers the #305 / #306 / #307 flag-threading contract:

- The three new flags appear on both runners' argparsers with the same
  names + choices + defaults as ``app.train_forecaster``.
- The default-off path constructs the same ``ModelConfig`` + calls
  ``load_walk_forward_split`` with the same kwargs as the pre-PR
  canonical sweep (reproducibility-smoke #335 stays byte-identical).
- The opt-in path forwards each flag through to the trainer call.

The tests stub ``load_walk_forward_split`` and ``train_model`` so the
runner can be exercised end-to-end without GPU or a real training
package on disk.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# argparser surface contract
# ---------------------------------------------------------------------------


def test_dual_head_runner_exposes_new_flags(monkeypatch):
    """The three new flags ship on the canonical sweep runner."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
        ],
    )
    args = _parse_args()
    assert args.rates_target_mode == "raw"
    assert args.use_retrieval_analogs is False
    assert args.use_regime_conditioning is False
    assert args.use_statement_delta is False
    assert args.use_vote_features is False
    assert args.use_press_conf is False


def test_dual_head_runner_accepts_opt_in_flags(monkeypatch):
    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--rates-target-mode",
            "fomc_attributable",
            "--use-retrieval-analogs",
            "--use-regime-conditioning",
            "--use-statement-delta",
            "--use-vote-features",
            "--use-press-conf",
        ],
    )
    args = _parse_args()
    assert args.rates_target_mode == "fomc_attributable"
    assert args.use_retrieval_analogs is True
    assert args.use_regime_conditioning is True
    assert args.use_statement_delta is True
    assert args.use_vote_features is True
    assert args.use_press_conf is True


def test_dual_head_runner_rejects_unknown_rates_target_mode(monkeypatch):
    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--rates-target-mode",
            "definitely_not_a_mode",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_per_family_runner_exposes_new_flags(monkeypatch):
    from scripts.run_per_family_ablation import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_per_family_ablation",
            "--training-package-id",
            "tp_dummy",
        ],
    )
    args = _parse_args()
    assert args.rates_target_mode == "raw"
    assert args.use_retrieval_analogs is False
    assert args.use_regime_conditioning is False


def test_per_family_runner_accepts_opt_in_flags(monkeypatch):
    from scripts.run_per_family_ablation import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_per_family_ablation",
            "--training-package-id",
            "tp_dummy",
            "--rates-target-mode",
            "fomc_attributable",
            "--use-retrieval-analogs",
            "--use-regime-conditioning",
        ],
    )
    args = _parse_args()
    assert args.rates_target_mode == "fomc_attributable"
    assert args.use_retrieval_analogs is True
    assert args.use_regime_conditioning is True


# ---------------------------------------------------------------------------
# trainer-call kwargs contract
# ---------------------------------------------------------------------------


class _StubSplit:
    fold_id = "fold_001"
    protocol = "walk_forward"
    train: list[Any] = []
    val: list[Any] = []
    test: list[Any] = []


def _capture_calls(monkeypatch, runner_module):
    """Stub the heavyweight calls inside one runner module.

    Returns a dict that the test populates: ``loader_calls`` collects
    kwargs passed to ``load_walk_forward_split``; ``train_calls``
    collects kwargs passed to ``train_model``; ``config_calls`` collects
    the ``ModelConfig`` instance the runner builds (via the kwarg
    captured on the trainer call).
    """

    captured: dict[str, list[Any]] = {
        "loader_calls": [],
        "train_calls": [],
    }

    def _fake_load_walk_forward_split(**kwargs):
        captured["loader_calls"].append(kwargs)
        return _StubSplit()

    def _fake_train_model(**kwargs):
        captured["train_calls"].append(kwargs)
        return SimpleNamespace(
            summary=SimpleNamespace(
                test_metrics=SimpleNamespace(
                    regime_f1_macro=0.5,
                    regime_accuracy=0.5,
                    regime_loss=1.0,
                    regression_rmse_log_rv=1.0,
                    regression_mae_log_rv=0.8,
                    regression_loss=1.0,
                ),
            ),
        )

    # The runner imports these lazily inside _run_one_cell, so we
    # monkeypatch on the *source* modules they import from.
    from app.training import loaders as loaders_module
    from app.training import loop as loop_module

    monkeypatch.setattr(
        loaders_module,
        "load_walk_forward_split",
        _fake_load_walk_forward_split,
    )
    monkeypatch.setattr(loop_module, "train_model", _fake_train_model)
    return captured


def test_dual_head_runner_default_off_byte_identity(monkeypatch):
    """Default invocation matches the pre-PR canonical loader kwargs."""

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    result = runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
    )

    assert result["head_mode"] == "dual"
    assert len(captured["loader_calls"]) == 1
    loader_kwargs = captured["loader_calls"][0]
    # Default-off contract: every opt-in loader flag passes False.
    assert loader_kwargs["use_retrieval_analogs"] is False
    assert loader_kwargs["use_regime_conditioning"] is False
    assert loader_kwargs["use_statement_delta"] is False
    assert loader_kwargs["use_vote_features"] is False
    assert loader_kwargs["use_press_conf"] is False
    assert loader_kwargs["rich_features"] is True

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "raw"
    assert model_config.use_regime_conditioning is False


def test_dual_head_runner_opt_in_threads_through(monkeypatch):
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
        rates_target_mode="fomc_attributable",
        use_retrieval_analogs=True,
        use_regime_conditioning=True,
        use_statement_delta=True,
        use_vote_features=True,
        use_press_conf=True,
    )

    loader_kwargs = captured["loader_calls"][0]
    assert loader_kwargs["use_retrieval_analogs"] is True
    assert loader_kwargs["use_regime_conditioning"] is True
    assert loader_kwargs["use_statement_delta"] is True
    assert loader_kwargs["use_vote_features"] is True
    assert loader_kwargs["use_press_conf"] is True

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "fomc_attributable"
    assert model_config.use_regime_conditioning is True


# ---------------------------------------------------------------------------
# #401 follow-up: auto-activate rates heads when rates_target_mode != raw.
# Without rates heads mounted, --rates-target-mode is a no-op and the
# canonical-comparison sweep produces byte-identical output to the default.
# ---------------------------------------------------------------------------


def test_dual_head_runner_default_off_leaves_rates_heads_empty(monkeypatch):
    """Default ``rates_target_mode='raw'`` must keep ``rates_heads=()``.

    Pre-#401 canonical sweep is byte-identical -- no auto-activation.
    """

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_heads == ()


def test_dual_head_runner_fomc_attributable_auto_activates_rates_heads(monkeypatch):
    """``rates_target_mode='fomc_attributable'`` must mount canonical heads."""

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from app.models.rates_heads import RATES_HEAD_NAMES
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
        rates_target_mode="fomc_attributable",
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_heads == tuple(RATES_HEAD_NAMES)
    assert model_config.rates_target_mode == "fomc_attributable"


def test_per_family_runner_default_off_leaves_rates_heads_empty(monkeypatch):
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_per_family_ablation as runner

    captured = _capture_calls(monkeypatch, runner)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        text_encoder="finbert_fed_adjacent",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
        rates_target_mode="raw",
        use_retrieval_analogs=False,
        use_regime_conditioning=False,
    )
    runner._run_one_cell(
        "baseline",
        frozenset(),
        11,
        args,
        fold_ids=["fold_001"],
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_heads == ()


def test_per_family_runner_fomc_attributable_auto_activates_rates_heads(monkeypatch):
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from app.models.rates_heads import RATES_HEAD_NAMES
    from scripts import run_per_family_ablation as runner

    captured = _capture_calls(monkeypatch, runner)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        text_encoder="finbert_fed_adjacent",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
        rates_target_mode="fomc_attributable",
        use_retrieval_analogs=False,
        use_regime_conditioning=False,
    )
    runner._run_one_cell(
        "baseline",
        frozenset(),
        11,
        args,
        fold_ids=["fold_001"],
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_heads == tuple(RATES_HEAD_NAMES)
    assert model_config.rates_target_mode == "fomc_attributable"


def test_per_family_runner_default_off_byte_identity(monkeypatch):
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_per_family_ablation as runner

    captured = _capture_calls(monkeypatch, runner)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        text_encoder="finbert_fed_adjacent",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
        rates_target_mode="raw",
        use_retrieval_analogs=False,
        use_regime_conditioning=False,
    )
    runner._run_one_cell(
        "baseline",
        frozenset(),
        11,
        args,
        fold_ids=["fold_001"],
    )

    loader_kwargs = captured["loader_calls"][0]
    assert loader_kwargs["use_retrieval_analogs"] is False
    assert loader_kwargs["use_regime_conditioning"] is False
    # Per-family default-off must also still pass the pre-PR loader
    # family flags (baseline keeps every family on).
    assert loader_kwargs["use_credibility"] is True
    assert loader_kwargs["use_linguistic"] is True
    assert loader_kwargs["use_mp_surprise"] is True
    assert loader_kwargs["use_multi_axis"] is True
    assert loader_kwargs["use_llm_features"] is True

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "raw"
    assert model_config.use_regime_conditioning is False


def test_per_family_runner_opt_in_threads_through(monkeypatch):
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_per_family_ablation as runner

    captured = _capture_calls(monkeypatch, runner)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        text_encoder="finbert_fed_adjacent",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
        rates_target_mode="fomc_attributable",
        use_retrieval_analogs=True,
        use_regime_conditioning=True,
    )
    runner._run_one_cell(
        "baseline",
        frozenset(),
        11,
        args,
        fold_ids=["fold_001"],
    )

    loader_kwargs = captured["loader_calls"][0]
    assert loader_kwargs["use_retrieval_analogs"] is True
    assert loader_kwargs["use_regime_conditioning"] is True

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "fomc_attributable"
    assert model_config.use_regime_conditioning is True
