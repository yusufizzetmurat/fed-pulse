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

    from app.models.config import SEQUENCE_LENGTH
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
    assert args.vol_target_mode == "raw"
    assert args.target_horizon == 10
    assert args.sequence_length == SEQUENCE_LENGTH
    assert args.use_retrieval_analogs is False
    assert args.use_regime_conditioning is False
    assert args.use_statement_delta is False
    assert args.use_vote_features is False
    assert args.use_press_conf is False
    assert args.text_encoder is None
    assert args.use_text_embeddings is True
    assert args.use_mp_surprise is True


def test_dual_head_runner_no_mp_surprise_flag(monkeypatch):
    """``--no-mp-surprise`` flips the loader toggle to False."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--no-mp-surprise",
        ],
    )
    args = _parse_args()
    assert args.use_mp_surprise is False


def test_dual_head_runner_doc_length_flag(monkeypatch):
    """``--use-doc-length`` opts in; default keeps the flag off."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_dual_head_comparison", "--training-package-id", "tp_dummy"],
    )
    args = _parse_args()
    assert args.use_doc_length is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--use-doc-length",
        ],
    )
    args = _parse_args()
    assert args.use_doc_length is True


def test_dual_head_runner_regime_loss_focal_choice_accepted(monkeypatch):
    """``--regime-loss focal`` parses + ``--focal-gamma`` overrides default."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--regime-loss",
            "focal",
            "--focal-gamma",
            "1.5",
        ],
    )
    args = _parse_args()
    assert args.regime_loss == "focal"
    assert args.focal_gamma == 1.5
    assert args.class_balanced_beta == 0.999


def test_dual_head_runner_regime_loss_class_balanced_choice_accepted(monkeypatch):
    """``--regime-loss class_balanced`` parses + ``--class-balanced-beta``."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--regime-loss",
            "class_balanced",
            "--class-balanced-beta",
            "0.99",
        ],
    )
    args = _parse_args()
    assert args.regime_loss == "class_balanced"
    assert args.class_balanced_beta == 0.99
    assert args.focal_gamma == 2.0


def test_dual_head_runner_text_encoder_opt_in(monkeypatch):
    """``--text-encoder <alias>`` parses; ``--no-text-embeddings`` flips."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--text-encoder",
            "finbert_fed_adjacent_xbank",
            "--no-text-embeddings",
        ],
    )
    args = _parse_args()
    assert args.text_encoder == "finbert_fed_adjacent_xbank"
    assert args.use_text_embeddings is False


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
            "--vol-target-mode",
            "garch_residual",
            "--target-horizon",
            "5",
            "--sequence-length",
            "60",
            "--use-retrieval-analogs",
            "--use-regime-conditioning",
            "--use-statement-delta",
            "--use-vote-features",
            "--use-press-conf",
        ],
    )
    args = _parse_args()
    assert args.rates_target_mode == "fomc_attributable"
    assert args.vol_target_mode == "garch_residual"
    assert args.target_horizon == 5
    assert args.sequence_length == 60
    assert args.use_retrieval_analogs is True
    assert args.use_regime_conditioning is True
    assert args.use_statement_delta is True
    assert args.use_vote_features is True
    assert args.use_press_conf is True


def test_dual_head_runner_rejects_unknown_vol_target_mode(monkeypatch):
    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--vol-target-mode",
            "definitely_not_a_mode",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_dual_head_runner_rejects_unsupported_target_horizon(monkeypatch):
    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--target-horizon",
            "7",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_dual_head_runner_aux_horizons_default_empty(monkeypatch):
    """``--aux-horizons`` defaults to an empty tuple (#471 default-off)."""

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
    assert args.aux_horizons == ()
    assert args.aux_horizon_alpha == 0.3


def test_dual_head_runner_aux_horizons_accepts_csv(monkeypatch):
    """``--aux-horizons 5,20`` parses into a 2-tuple of ints."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--aux-horizons",
            "5,20",
            "--aux-horizon-alpha",
            "0.5",
        ],
    )
    args = _parse_args()
    assert args.aux_horizons == (5, 20)
    assert args.aux_horizon_alpha == 0.5


def test_dual_head_runner_rejects_invalid_aux_horizon(monkeypatch):
    """An aux horizon outside ``{1, 3, 5, 20, 30}`` fails parse."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--aux-horizons",
            "7",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_dual_head_runner_rejects_primary_horizon_in_aux(monkeypatch):
    """10d is the primary and must be rejected from the aux tuple."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--aux-horizons",
            "10",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


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


# ---------------------------------------------------------------------------
# #472 vol-regime label mode: parser surface + ModelConfig wiring
# ---------------------------------------------------------------------------


def test_dual_head_runner_vol_regime_label_mode_defaults_off(monkeypatch):
    """Default invocation keeps the byte-identical per-fold quantile path."""

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
    assert args.vol_regime_label_mode == "per_fold_quantile"
    assert args.absolute_calm_max is None
    assert args.absolute_high_min is None


def test_dual_head_runner_accepts_absolute_label_mode_with_overrides(monkeypatch):
    """Operator can opt into absolute labelling and override the cutoffs."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--vol-regime-label-mode",
            "absolute",
            "--absolute-calm-max",
            "10.0",
            "--absolute-high-min",
            "25.0",
        ],
    )
    args = _parse_args()
    assert args.vol_regime_label_mode == "absolute"
    assert args.absolute_calm_max == pytest.approx(10.0)
    assert args.absolute_high_min == pytest.approx(25.0)


def test_dual_head_runner_rejects_unknown_vol_regime_label_mode(monkeypatch):
    """argparse rejects any value outside the allowed enum."""

    from scripts.run_dual_head_comparison import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dual_head_comparison",
            "--training-package-id",
            "tp_dummy",
            "--vol-regime-label-mode",
            "definitely_not_a_mode",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_dual_head_runner_default_label_mode_threads_per_fold_quantile(monkeypatch):
    """Default invocation builds a ModelConfig with the quantile path on."""

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from app.models.config import DEFAULT_ABSOLUTE_VOL_THRESHOLDS
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
    assert model_config.vol_regime_label_mode == "per_fold_quantile"
    assert model_config.absolute_vol_thresholds == DEFAULT_ABSOLUTE_VOL_THRESHOLDS


def test_dual_head_runner_absolute_label_mode_threads_through(monkeypatch):
    """``absolute`` opts the trainer into the fixed-threshold branch."""

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    thresholds = (0.03, 0.05)
    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
        vol_regime_label_mode="absolute",
        absolute_vol_thresholds=thresholds,
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.vol_regime_label_mode == "absolute"
    assert model_config.absolute_vol_thresholds == thresholds


def test_resolve_absolute_thresholds_converts_annualized_percent(monkeypatch):
    """Annualized percent inputs convert to per-period units via sqrt(25.2)."""

    import math

    from scripts.run_dual_head_comparison import _resolve_absolute_thresholds

    out = _resolve_absolute_thresholds("absolute", 12.0, 22.0)
    assert out is not None
    calm, high = out
    assert calm == pytest.approx(0.12 / math.sqrt(25.2))
    assert high == pytest.approx(0.22 / math.sqrt(25.2))


def test_resolve_absolute_thresholds_returns_none_on_quantile_mode():
    """Quantile mode never converts cutoffs even if values are supplied."""

    from scripts.run_dual_head_comparison import _resolve_absolute_thresholds

    assert (
        _resolve_absolute_thresholds("per_fold_quantile", 12.0, 22.0) is None
    )


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
    from app.models.config import SEQUENCE_LENGTH
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
    assert loader_kwargs["vol_target_horizon"] == 10
    assert loader_kwargs["sequence_length"] == SEQUENCE_LENGTH
    # Default-off: text path stays disabled (text_encoder=None gates the
    # loader's ``use_text_path`` predicate regardless of the toggle).
    assert loader_kwargs["text_encoder"] is None
    assert loader_kwargs["use_text_embeddings"] is True
    # mp_surprise defaults on to match the loader default.
    assert loader_kwargs["use_mp_surprise"] is True

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "raw"
    assert model_config.vol_target_mode == "raw"
    assert model_config.vol_target_horizon == 10
    assert model_config.sequence_length == SEQUENCE_LENGTH
    assert model_config.use_regime_conditioning is False


def test_dual_head_runner_no_mp_surprise_threads_into_loader(monkeypatch):
    """``use_mp_surprise=False`` reaches the loader as the kwarg."""

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
        use_mp_surprise=False,
    )

    loader_kwargs = captured["loader_calls"][0]
    assert loader_kwargs["use_mp_surprise"] is False


def test_dual_head_runner_aux_horizons_threads_through(monkeypatch):
    """``--aux-horizons`` lands on the ModelConfig the runner builds."""

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
        aux_horizons=(5, 20),
        aux_horizon_alpha=0.4,
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.aux_horizons == (5, 20)
    assert model_config.aux_horizon_alpha == 0.4


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
        vol_target_mode="garch_residual",
        vol_target_horizon=5,
        sequence_length=60,
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
    assert loader_kwargs["vol_target_horizon"] == 5
    assert loader_kwargs["sequence_length"] == 60

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.rates_target_mode == "fomc_attributable"
    assert model_config.vol_target_mode == "garch_residual"
    assert model_config.vol_target_horizon == 5
    assert model_config.sequence_length == 60
    assert model_config.use_regime_conditioning is True


def test_dual_head_runner_text_encoder_threads_into_loader(monkeypatch):
    """Setting ``text_encoder`` forwards the alias + toggle to the loader."""

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
        text_encoder="finbert_fed_adjacent_xbank",
        use_text_embeddings=False,
    )

    loader_kwargs = captured["loader_calls"][0]
    assert loader_kwargs["text_encoder"] == "finbert_fed_adjacent_xbank"
    assert loader_kwargs["use_text_embeddings"] is False


def test_dual_head_runner_text_encoder_activates_model_text_channel(monkeypatch):
    """``text_encoder`` + ``use_text_embeddings`` activate the model text channel.

    Pre-#546 runner threaded the loader-side toggle but left ModelConfig at
    text_embedding_dim=0 / text_adapter_dim=0, so the model silently
    ignored the embeddings the loader emitted (the encoder bake-off
    no-op surfaced in §6.38). This pins the regression: when the text
    encoder is set, the model's text channel must be non-zero on both
    dims AND the channel mode must be ``embeddings``.
    """

    pytest.importorskip("torch", reason="train_model import path needs torch")
    pytest.importorskip("transformers", reason="encoder hidden_size resolution")
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
        text_encoder="finbert_fed_adjacent",
        use_text_embeddings=True,
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.text_embedding_dim > 0, (
        "text_encoder set with use_text_embeddings=True must resolve a "
        "positive hidden_size onto ModelConfig.text_embedding_dim; "
        "otherwise ForecasterBase ignores the loader-emitted embeddings."
    )
    assert model_config.text_adapter_dim > 0
    assert model_config.text_channel == "embeddings"


def test_dual_head_runner_text_encoder_unset_keeps_text_channel_off(monkeypatch):
    """Default invocation must keep the text channel at the byte-identical
    no-text path (both dims 0, channel='scalar')."""

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
    assert model_config.text_embedding_dim == 0
    assert model_config.text_adapter_dim == 0
    assert model_config.text_channel == "scalar"


def test_dual_head_runner_api_only_encoder_skips_autoconfig(monkeypatch):
    """#556: api_only encoders must read hidden_size off the registry.

    Voyage is served by the Voyage REST API; ``ref.repo`` is the API
    model name, not an HF artifact. The pre-#556 runner called
    ``AutoConfig.from_pretrained`` against the repo and 404'd. The
    api_only short-circuit reads the registry's explicit
    ``hidden_size`` annotation instead — never instantiates the
    model, never hits the HF Hub.

    This test fakes a minimal EncoderRef-shaped object with the
    api_only + hidden_size fields and confirms the runner builds a
    ModelConfig with the registry's hidden_size on it AND does not
    import / call AutoConfig.
    """

    pytest.importorskip("torch", reason="train_model import path needs torch")
    from scripts import run_dual_head_comparison as runner

    captured = _capture_calls(monkeypatch, runner)

    class _FakeRef:
        repo = "voyageai/voyage-finance-2"
        revision = "voyage-finance-2"
        trust_remote_code = False
        api_only = True
        hidden_size = 1024

    monkeypatch.setattr(
        runner, "_run_one_cell", runner._run_one_cell  # noop, just to verify import order
    )
    # Patch the registry resolver the runner imports lazily.
    import app.models.registry as registry

    monkeypatch.setattr(registry, "encoder_ref", lambda alias: _FakeRef())

    runner._run_one_cell(
        "dual",
        seed=11,
        training_package_id="tp_dummy",
        fold_ids=["fold_001"],
        epochs=1,
        regression_alpha=0.5,
        hidden_size=64,
        text_encoder="voyage_finance_2",
        use_text_embeddings=True,
    )

    train_kwargs = captured["train_calls"][0]
    model_config = train_kwargs["model_config"]
    assert model_config.text_embedding_dim == 1024, (
        "api_only encoder must surface the registry's hidden_size onto "
        "ModelConfig without calling AutoConfig"
    )
    assert model_config.text_adapter_dim == 128
    assert model_config.text_channel == "embeddings"


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
