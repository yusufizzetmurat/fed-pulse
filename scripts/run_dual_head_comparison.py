"""Three-way dual-head comparison runner (#304).

Trains the same model under three head-mode configurations on the same
walk-forward fold protocol and emits a per-trial JSON keyed by
configuration so the §16 finalization-roadmap table can read the
results off a single file. Each configuration runs over the official
seed set and writes the per-fold ``regime_f1_macro`` (classification
surface) and ``regression_rmse_log_rv`` (regression surface) so the
table can compare both axes at a glance.

Usage::

    docker compose run --rm backend python -m scripts.run_dual_head_comparison \\
        --training-package-id <id> \\
        --output artifacts/experiments/dual_head_comparison.json \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --regression-alpha 0.5

The output JSON has the structure::

    {
      "head_modes": ["classification", "regression", "dual"],
      "seeds": [...],
      "trials": {
        "classification": [ { "seed": 11, "metrics": {...}, ... }, ... ],
        "regression":     [ ... ],
        "dual":           [ ... ]
      },
      "summary": {
        "classification": { "regime_f1_macro_mean": float, "regime_f1_macro_std": float, ... },
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from app.config import BACKEND_ROOT
from app.training.runtime_compat import ensure_compile_safe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package ID under ``backend/artifacts/training_packages/<id>``.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "``artifacts/experiments/dual_head_comparison.json``."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 97],
        help="Official seed set. Default mirrors docs/benchmark-policy.md.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epochs per cell (default 40).",
    )
    parser.add_argument(
        "--regression-alpha",
        type=float,
        default=0.5,
        help="alpha for head_mode='dual' joint loss.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across all three head modes.",
    )
    parser.add_argument(
        "--head-modes",
        nargs="+",
        choices=("classification", "regression", "dual"),
        default=["classification", "regression", "dual"],
        help="Subset of head modes to evaluate (defaults to all three).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Subset of walk-forward fold IDs to evaluate. Defaults to "
            "every fold present in the training package's "
            "fold_manifest_expanding_walk_forward.json."
        ),
    )
    # #305 rates-head target derivation. Mirrors the same flag on
    # ``app.train_forecaster`` (name + choices + default). ``raw`` keeps
    # this runner byte-identical to the pre-#305 canonical sweep.
    parser.add_argument(
        "--rates-target-mode",
        type=str,
        choices=("raw", "fomc_attributable"),
        default="raw",
        help=(
            "Rates-head target derivation. ``raw`` (default) keeps the "
            "observed ``yield_<tenor>_change_5d`` bps move; "
            "``fomc_attributable`` predicts the 1-D projection onto the "
            "strict-prior policy-surprise direction. See ADR 0027."
        ),
    )
    # #435 dual-head MSE-branch target derivation. ``raw`` (default,
    # byte-identical) feeds ``log(forward_realized_vol_10d)``;
    # ``garch_residual`` swaps in the GARCH(1,1) residual column with
    # fallback to raw on rows lacking a residual (ADR 0034).
    parser.add_argument(
        "--vol-target-mode",
        type=str,
        choices=("raw", "garch_residual"),
        default="raw",
        help=(
            "Forward-vol regression-target derivation. ``raw`` (default) "
            "feeds the standardised ``log(forward_realized_vol_10d)``; "
            "``garch_residual`` swaps in "
            "``forward_realized_vol_10d_garch_residual`` (signed, no "
            "log). See ADR 0034."
        ),
    )
    # Supervised forward-vol horizon (trading days). ``10`` keeps the
    # canonical y axis; the loader switches to
    # ``forward_realized_vol_<H>d`` for any other supported choice.
    parser.add_argument(
        "--target-horizon",
        type=int,
        choices=(1, 3, 5, 10, 20, 30),
        default=10,
        help=(
            "Supervised forward-vol horizon in trading days. ``10`` "
            "(default) reads the canonical "
            "``forward_realized_vol_10d`` column; other choices read "
            "``forward_realized_vol_<H>d`` from the same events "
            "parquet (populated by the multi-horizon data builder)."
        ),
    )
    # Prior-bar window length. Default matches the canonical 20-bar
    # window so the pre-existing sweep stays byte-identical; override
    # for the longer-window arm (e.g. 60) against a TP whose
    # ``prior_bars_json`` carries the matching wider window.
    from app.models.config import SEQUENCE_LENGTH

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=(
            "Prior-bar window length per supervised event. Default "
            f"{SEQUENCE_LENGTH} matches the canonical lookback. Override "
            "against a TP whose prior_bars_json carries the matching "
            "wider window (e.g. 60)."
        ),
    )
    # #306 retrieval-augmented input features. Off by default so the
    # canonical sweep stays byte-identical.
    parser.add_argument(
        "--use-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_true",
        help=(
            "Attach the 5-dim retrieval-analog summary block to every "
            "supervised event. Default off."
        ),
    )
    parser.add_argument(
        "--no-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_false",
        help="Disable the retrieval-analog block (default).",
    )
    # #307 macro-regime conditioning. Off by default so the canonical
    # sweep stays byte-identical.
    parser.add_argument(
        "--use-regime-conditioning",
        dest="use_regime_conditioning",
        action="store_true",
        help=(
            "Attach the 3-scalar macro-regime indicator block and mount "
            "the multiplicative gate over the rich-feature slice. "
            "Default off."
        ),
    )
    parser.add_argument(
        "--no-regime-conditioning",
        dest="use_regime_conditioning",
        action="store_false",
        help="Disable the macro-regime block + gate (default).",
    )
    # #443 / #444 / press-conf pass-through. Off by default so the
    # canonical sweep stays byte-identical; on, each adds its own
    # feature block to the rich-feature slice via the loader kwargs.
    parser.add_argument(
        "--use-statement-delta",
        dest="use_statement_delta",
        action="store_true",
        help=(
            "Attach the #443 statement-delta embedding (768-d) + missing "
            "flag to every supervised event. Reads the "
            "``statement_delta_embedding`` column on events.parquet; rows "
            "without a strict-prior statement carry zeros + missing=1.0."
        ),
    )
    parser.add_argument(
        "--use-vote-features",
        dest="use_vote_features",
        action="store_true",
        help=(
            "Attach the #444 vote-tally feature block (votes_for_norm, "
            "votes_against_norm, is_unanimous, direction_sign) to each "
            "statement event. Non-statement rows carry zeros + missing=1.0."
        ),
    )
    parser.add_argument(
        "--use-press-conf",
        dest="use_press_conf",
        action="store_true",
        help=(
            "Attach the press-conference Q&A slot to each statement event "
            "from press_conf_qa.parquet under the training package. "
            "Statements without a matching press-conf row carry the "
            "missing flag."
        ),
    )
    # #478 VIX term-structure + VRP block. Off by default so the
    # canonical sweep stays byte-identical.
    parser.add_argument(
        "--use-vix-features",
        dest="use_vix_features",
        action="store_true",
        help=(
            "Attach the 6-scalar VIX term-structure + VRP block "
            "(vix, vix1m, vix3m, vix6m, vix_3m_over_1m_slope, vrp) at "
            "T-1 to every supervised event. Reads the strict-prior "
            "vix_*_t_minus_1 columns on events.parquet; pre-coverage "
            "events (^VIX1M / ^VIX3M / ^VIX6M before 2008) carry zeros "
            "+ missing=1.0."
        ),
    )
    # mp_surprise rich-feature block toggle. Default True matches
    # ``load_walk_forward_split`` so canonical runs stay byte-identical.
    # ``--no-mp-surprise`` zeros the mp_surprise block (and its missing
    # flags) end-to-end, isolating the feature's contribution to the
    # joint loss at the loader boundary rather than the model.
    parser.add_argument(
        "--use-mp-surprise",
        dest="use_mp_surprise",
        action="store_true",
        help=(
            "Attach the mp_surprise (monetary-policy surprise) feature "
            "block from data/external/fred/mp_surprises.parquet to each "
            "statement event. On by default."
        ),
    )
    parser.add_argument(
        "--no-mp-surprise",
        dest="use_mp_surprise",
        action="store_false",
        help=(
            "Zero the mp_surprise feature block end-to-end so the "
            "loader returns the same shape with zeros in the mp_surprise "
            "slot. Use to isolate the block's contribution."
        ),
    )
    parser.set_defaults(use_mp_surprise=True)
    # #470 regime-loss variant. ``ce`` keeps the standard CE on the
    # 3-class regime head; ``ordinal_ce`` swaps in bin-distance-weighted
    # CE so a calm->high miss costs 2x a calm->normal miss.
    parser.add_argument(
        "--regime-loss",
        dest="regime_loss",
        type=str,
        choices=("ce", "ordinal_ce", "focal", "class_balanced"),
        default="ce",
        help=(
            "Regime-axis loss kernel. ``ce`` (default) keeps the standard "
            "cross-entropy byte-identical to the pre-#470 canonical "
            "sweep; ``ordinal_ce`` weights each row's CE by "
            "``1 + |true - argmax|`` so far-bin confusions pay more; "
            "``focal`` applies Lin et al. 2017's (1-p)**gamma modulator; "
            "``class_balanced`` uses Cui et al. 2019 effective-number "
            "reweighting on the existing inverse-frequency path."
        ),
    )
    # #502 focal-loss hyperparameter. Only consulted when
    # ``--regime-loss focal`` is set; ignored under every other mode.
    parser.add_argument(
        "--focal-gamma",
        dest="focal_gamma",
        type=float,
        default=2.0,
        help=(
            "Focusing parameter for ``--regime-loss focal``. Default 2.0 "
            "matches Lin et al. 2017. Higher values down-weight easy "
            "examples more aggressively. Ignored under other loss modes."
        ),
    )
    # #502 class-balanced hyperparameter. Only consulted when
    # ``--regime-loss class_balanced`` is set; ignored under every other
    # mode.
    parser.add_argument(
        "--class-balanced-beta",
        dest="class_balanced_beta",
        type=float,
        default=0.999,
        help=(
            "Effective-number reweighting hyperparameter for "
            "``--regime-loss class_balanced``. Default 0.999 mirrors Cui "
            "et al. 2019's CIFAR-LT recipe. Ignored under other loss modes."
        ),
    )
    # #471 multi-horizon auxiliary regression heads. Empty default
    # (no aux heads mounted) keeps every existing canonical sweep
    # byte-identical. ``--aux-horizons 5,20`` mounts one parallel
    # regression head per listed horizon; each head shares the
    # encoder + recurrent core with the primary log-RV head and is
    # supervised against ``forward_realized_vol_<H>d``.
    parser.add_argument(
        "--aux-horizons",
        dest="aux_horizons",
        type=_parse_aux_horizons,
        default=(),
        help=(
            "Comma-separated forward-vol horizons to mount as auxiliary "
            "regression targets alongside the canonical 10d primary. "
            "Example: '--aux-horizons 5,20'. Each entry must be a "
            "supported horizon other than 10. Empty default leaves the "
            "joint loss byte-identical to the pre-#471 path."
        ),
    )
    parser.add_argument(
        "--aux-horizon-alpha",
        type=float,
        default=0.3,
        help=(
            "Weight on each auxiliary horizon's MSE term in the joint "
            "loss (single scalar shared across every aux horizon). "
            "Default 0.3 mirrors the multi-task aux convention."
        ),
    )
    # Pooled-text path. Default ``None`` keeps the text path off so the
    # canonical sweep stays byte-identical; set to an encoder alias
    # (resolved through ``app.models.registry.encoder_ref``) or a HF
    # repo id to consume that encoder's pooled embeddings as a feature
    # family. The companion ``--no-text-embeddings`` toggle zeros the
    # slot while keeping input shape constant; together they match the
    # loader predicate ``bool(text_encoder) and bool(use_text_embeddings)``.
    parser.add_argument(
        "--text-encoder",
        dest="text_encoder",
        type=str,
        default=None,
        help=(
            "Encoder alias or HF repo id whose pooled FOMC statement "
            "embeddings the loader attaches as a per-event feature "
            "family. Default ``None`` keeps the text path off so the "
            "canonical sweep stays byte-identical."
        ),
    )
    parser.add_argument(
        "--use-text-embeddings",
        dest="use_text_embeddings",
        action="store_true",
        help=(
            "Enable the pooled-text embedding slot when "
            "``--text-encoder`` is set (default). No-op when "
            "``--text-encoder`` is unset."
        ),
    )
    parser.add_argument(
        "--no-text-embeddings",
        dest="use_text_embeddings",
        action="store_false",
        help=(
            "Zero the pooled-text embedding slot while keeping the "
            "model input shape constant."
        ),
    )
    # #472 vol-regime labelling mode. ``per_fold_quantile`` (default,
    # byte-identical) keeps the per-fold (q33, q67) cutoffs the canonical
    # sweep fits each fold; ``absolute`` swaps in the fixed
    # ``(calm_max, high_min)`` pair so every fold's calm / normal / high
    # cells refer to the same economic vol level.
    parser.add_argument(
        "--vol-regime-label-mode",
        dest="vol_regime_label_mode",
        type=str,
        choices=("per_fold_quantile", "absolute"),
        default="per_fold_quantile",
        help=(
            "Vol-regime labelling mode. ``per_fold_quantile`` (default) "
            "fits per-fold quantile cutoffs on the train slice each fold; "
            "``absolute`` uses a fixed (calm_max, high_min) pair so every "
            "fold's calm / normal / high cells refer to the same vol "
            "level."
        ),
    )
    parser.add_argument(
        "--absolute-calm-max",
        dest="absolute_calm_max",
        type=float,
        default=None,
        help=(
            "calm_max boundary in ANNUALIZED vol units (e.g. 12.0 for "
            "12%%); converted to per-period via "
            "vol_per_period = vol_annualized / sqrt(252 / 10) before "
            "passing to ModelConfig. Only consumed when "
            "--vol-regime-label-mode=absolute. Default: 12%%."
        ),
    )
    parser.add_argument(
        "--absolute-high-min",
        dest="absolute_high_min",
        type=float,
        default=None,
        help=(
            "high_min boundary in ANNUALIZED vol units (e.g. 22.0 for "
            "22%%); converted to per-period via "
            "vol_per_period = vol_annualized / sqrt(252 / 10) before "
            "passing to ModelConfig. Only consumed when "
            "--vol-regime-label-mode=absolute. Default: 22%%."
        ),
    )
    parser.set_defaults(
        use_retrieval_analogs=False,
        use_regime_conditioning=False,
        use_statement_delta=False,
        use_vote_features=False,
        use_press_conf=False,
        use_text_embeddings=True,
        use_vix_features=False,
    )
    return parser.parse_args()


def _parse_aux_horizons(value: str) -> tuple[int, ...]:
    """Parse the ``--aux-horizons`` CLI value into a validated tuple.

    Accepts a comma-separated list of integers (e.g. ``"5,20"``).
    Rejects any entry not in the supported aux set
    (``{1, 3, 5, 20, 30}``) so the misconfiguration surfaces at parse
    time rather than after the loader read.
    """

    from app.models.config import SUPPORTED_VOL_TARGET_HORIZONS

    if not value or not value.strip():
        return ()
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    horizons: list[int] = []
    allowed = tuple(h for h in SUPPORTED_VOL_TARGET_HORIZONS if h != 10)
    for chunk in parts:
        try:
            horizon = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--aux-horizons entry {chunk!r} is not an integer"
            ) from exc
        if horizon not in allowed:
            raise argparse.ArgumentTypeError(
                f"--aux-horizons entry {horizon} is not in the allowed "
                f"set {allowed} (10 is the primary and excluded)"
            )
        horizons.append(horizon)
    # Preserve operator-supplied ordering so the column order on the
    # aux target tensor is reproducible across runs.
    return tuple(horizons)


def _resolve_fold_ids(training_package_id: str, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    from app.training.loaders import _read_fold_manifest, _resolve_training_package_dir

    package_dir = _resolve_training_package_dir(training_package_id)
    manifest = _read_fold_manifest(package_dir)
    if not manifest:
        raise RuntimeError(
            "fold_manifest_expanding_walk_forward.json is empty / missing "
            f"for training_package_id={training_package_id!r}; provide "
            "--folds explicitly."
        )
    return sorted(manifest.keys())


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dual_head_comparison.json"


def _trial_metrics(summary: Any) -> dict[str, Any]:
    """Pull headline numbers + the per-fold classification breakdown
    out of a ``TrainingRunSummary``. The classification breakdown
    carries the 3x3 confusion matrix and the per-class P/R/F1 + support
    counts that the downstream defense analyses (#496 ordinal confusion,
    #500 per-fold baselines) read.

    ``classification_breakdown`` is included whenever the test partition
    ran a classification head; on regression-only arms it lands as
    ``None`` and consumers degrade cleanly.
    """

    test = getattr(summary, "test_metrics", None) or getattr(summary, "metrics", None)
    if test is None:
        return {}
    breakdown = getattr(test, "classification_breakdown", None)
    breakdown_payload: dict[str, Any] | None = None
    if breakdown is not None:
        # ``classification_breakdown`` on EvaluationMetrics is already a
        # dict at this stage (loop.py assigns ``breakdown.to_dict()``).
        if isinstance(breakdown, dict):
            breakdown_payload = breakdown
        elif hasattr(breakdown, "to_dict"):
            breakdown_payload = breakdown.to_dict()
    return {
        "regime_f1_macro": getattr(test, "regime_f1_macro", None),
        "regime_accuracy": getattr(test, "regime_accuracy", None),
        "regime_loss": getattr(test, "regime_loss", None),
        "regression_rmse_log_rv": getattr(test, "regression_rmse_log_rv", None),
        "regression_mae_log_rv": getattr(test, "regression_mae_log_rv", None),
        "regression_loss": getattr(test, "regression_loss", None),
        "classification_breakdown": breakdown_payload,
    }


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "n": len(finite),
    }


def _resolve_auto_rates_heads(
    rates_target_mode: str,
    rates_heads: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Auto-activate the canonical rates-head set under FOMC-attributable.

    ``--rates-target-mode`` only steers the rates heads' supervised
    target; it has no observable effect unless at least one rates head
    is mounted. The canonical-comparison sweep does not expose a
    ``--rates-heads`` flag (it sweeps head_modes, not rates heads), so
    when the operator opts the runner into ``fomc_attributable`` we
    mount the same canonical set (``2y``, ``5y``, ``terminal``) the
    ``make rates-heads-sweep`` target uses. ``raw`` (default) keeps
    ``rates_heads=()`` so the pre-#401 canonical sweep stays
    byte-identical.
    """

    from app.models.rates_heads import RATES_HEAD_NAMES

    if rates_heads:
        return tuple(rates_heads)
    if rates_target_mode != "raw":
        return tuple(RATES_HEAD_NAMES)
    return ()


def _run_one_cell(  # noqa: PLR0913 -- canonical sweep needs every knob inline
    head_mode: str,
    seed: int,
    *,
    training_package_id: str,
    fold_ids: list[str],
    epochs: int,
    regression_alpha: float,
    hidden_size: int,
    rates_target_mode: str = "raw",
    vol_target_mode: str = "raw",
    vol_target_horizon: int = 10,
    sequence_length: int | None = None,
    use_retrieval_analogs: bool = False,
    use_regime_conditioning: bool = False,
    use_statement_delta: bool = False,
    use_vote_features: bool = False,
    use_press_conf: bool = False,
    use_vix_features: bool = False,
    regime_loss: str = "ce",
    focal_gamma: float = 2.0,
    class_balanced_beta: float = 0.999,
    text_encoder: str | None = None,
    use_text_embeddings: bool = True,
    vol_regime_label_mode: str = "per_fold_quantile",
    absolute_vol_thresholds: tuple[float, float] | None = None,
    use_mp_surprise: bool = True,
    aux_horizons: tuple[int, ...] = (),
    aux_horizon_alpha: float = 0.3,
) -> dict[str, Any]:
    # Imports happen here so the script is importable without a torch
    # install (useful for doc-only environments).
    from app.models.config import (
        DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
        RICH_FEATURE_SIZE,
        SEQUENCE_LENGTH,
        ModelConfig,
    )
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    active_sequence_length = int(sequence_length) if sequence_length else SEQUENCE_LENGTH
    rates_heads = _resolve_auto_rates_heads(rates_target_mode, rates_heads=None)
    resolved_absolute_thresholds: tuple[float, float] = (
        absolute_vol_thresholds
        if absolute_vol_thresholds is not None
        else DEFAULT_ABSOLUTE_VOL_THRESHOLDS
    )
    # ``input_size`` stays at the base RICH_FEATURE_SIZE. Every opt-in
    # tail (regime / sep / press_conf / statement_delta / vote / vix) is
    # widened inside ``ForecasterBase.__init__`` via the per-tail
    # ``*_tail_dim`` accumulators; widening here would double-count.
    # Text-channel resolution. The loader already loads the encoder
    # and emits per-event pooled embeddings when both ``text_encoder``
    # and ``use_text_embeddings`` are truthy, but the model only
    # consumes the channel when BOTH ``text_embedding_dim`` and
    # ``text_adapter_dim`` are positive on ModelConfig. Before #546
    # this runner threaded neither, so every text-encoder arm trained
    # against a 0-width text channel (i.e. the no-text baseline). The
    # block below resolves the encoder's native hidden_size off the
    # registry-pinned config and sets both dims so the model actually
    # consumes the embeddings. The 128-dim adapter target mirrors the
    # forecaster_credibility default.
    resolved_text_embedding_dim = 0
    resolved_text_adapter_dim = 0
    resolved_text_channel = "scalar"
    if text_encoder and use_text_embeddings:
        from app.models.registry import encoder_ref

        ref = encoder_ref(text_encoder)
        if ref is None:
            raise ValueError(
                f"text_encoder={text_encoder!r} did not resolve via "
                "app.models.registry.encoder_ref. Add it to "
                "backend/app/models/registry.yaml or pass a valid alias."
            )
        from transformers import AutoConfig

        encoder_config = AutoConfig.from_pretrained(
            ref.repo, revision=ref.revision or None
        )
        resolved_text_embedding_dim = int(
            getattr(encoder_config, "hidden_size", 0) or 0
        )
        if resolved_text_embedding_dim <= 0:
            raise ValueError(
                f"text_encoder={text_encoder!r} resolved a config but "
                f"hidden_size was {resolved_text_embedding_dim!r}. The "
                "encoder cannot drive the text channel without a positive "
                "hidden_size on its transformer config."
            )
        resolved_text_adapter_dim = 128
        resolved_text_channel = "embeddings"

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        head_mode=head_mode,
        regression_alpha=regression_alpha,
        n_classes=3,
        hidden_size=hidden_size,
        rates_heads=rates_heads,
        rates_target_mode=rates_target_mode,
        vol_target_mode=vol_target_mode,
        vol_target_horizon=vol_target_horizon,
        sequence_length=active_sequence_length,
        use_regime_conditioning=use_regime_conditioning,
        use_press_conf=use_press_conf,
        use_statement_delta=use_statement_delta,
        use_vote_features=use_vote_features,
        use_vix_features=use_vix_features,
        regime_loss_mode=regime_loss,
        focal_gamma=float(focal_gamma),
        class_balanced_beta=float(class_balanced_beta),
        vol_regime_label_mode=vol_regime_label_mode,
        absolute_vol_thresholds=resolved_absolute_thresholds,
        aux_horizons=tuple(aux_horizons),
        aux_horizon_alpha=float(aux_horizon_alpha),
        text_channel=resolved_text_channel,
        text_embedding_dim=resolved_text_embedding_dim,
        text_adapter_dim=resolved_text_adapter_dim,
    )

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=training_package_id,
            fold_id=fold_id,
            rich_features=True,
            use_retrieval_analogs=use_retrieval_analogs,
            use_regime_conditioning=use_regime_conditioning,
            use_statement_delta=use_statement_delta,
            use_vote_features=use_vote_features,
            use_press_conf=use_press_conf,
            text_encoder=text_encoder,
            use_text_embeddings=use_text_embeddings,
            use_vix_features=use_vix_features,
            vol_target_horizon=vol_target_horizon,
            sequence_length=active_sequence_length,
            use_mp_surprise=use_mp_surprise,
        )
        result = train_model(
            model_config=config,
            train_sequence_groups=split.train,
            val_sequence_groups=split.val,
            test_sequence_groups=split.test,
            fold_id=split.fold_id,
            protocol=split.protocol,
            epochs=epochs,
            seed=seed,
            save_checkpoint=False,
        )
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "head_mode": head_mode,
        "seed": seed,
        "regression_alpha": regression_alpha,
        "training_package_id": training_package_id,
        "folds": per_fold,
    }


def _resolve_absolute_thresholds(
    label_mode: str,
    calm_max_annualized: float | None,
    high_min_annualized: float | None,
) -> tuple[float, float] | None:
    """Convert CLI annualized vol percentages to per-period thresholds.

    Returns ``None`` when ``label_mode != 'absolute'`` so the default
    ModelConfig threshold pair is used; when ``absolute`` is requested
    but the operator did not supply explicit values, returns ``None``
    too so the canonical defaults (12% / 22% annualized) flow through.
    The CLI accepts percentages either as a fraction (``0.12``) or as
    a percent (``12.0``); values >= 1.0 are treated as percent so the
    operator can type the integer they read in the docstring.
    """

    from app.models.config import ANNUALIZATION_SQRT_10D

    if label_mode != "absolute":
        return None
    if calm_max_annualized is None and high_min_annualized is None:
        return None

    def _normalize(v: float | None, fallback: float) -> float:
        if v is None:
            return fallback
        if v >= 1.0:
            return float(v) / 100.0
        return float(v)

    calm_max = _normalize(calm_max_annualized, 0.12)
    high_min = _normalize(high_min_annualized, 0.22)
    return calm_max / ANNUALIZATION_SQRT_10D, high_min / ANNUALIZATION_SQRT_10D


def main() -> int:
    ensure_compile_safe()
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[dual_head_comparison] writing -> {output_path}")

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[dual_head_comparison] folds={fold_ids}")
    if str(args.rates_target_mode) != "raw":
        print(
            "[dual_head_comparison] auto-activating rates heads for "
            f"rates_target_mode={args.rates_target_mode}"
        )
    from app.models.config import DEFAULT_ABSOLUTE_VOL_THRESHOLDS

    absolute_thresholds = _resolve_absolute_thresholds(
        str(args.vol_regime_label_mode),
        args.absolute_calm_max,
        args.absolute_high_min,
    )

    trials: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.head_modes}
    for head_mode in args.head_modes:
        for seed in args.seeds:
            print(
                f"[dual_head_comparison] head_mode={head_mode} seed={seed} "
                f"epochs={args.epochs}",
                flush=True,
            )
            trials[head_mode].append(
                _run_one_cell(
                    head_mode,
                    seed,
                    training_package_id=args.training_package_id,
                    fold_ids=fold_ids,
                    epochs=args.epochs,
                    regression_alpha=args.regression_alpha,
                    hidden_size=args.hidden_size,
                    rates_target_mode=str(args.rates_target_mode),
                    vol_target_mode=str(args.vol_target_mode),
                    vol_target_horizon=int(args.target_horizon),
                    sequence_length=int(args.sequence_length),
                    use_retrieval_analogs=bool(args.use_retrieval_analogs),
                    use_regime_conditioning=bool(args.use_regime_conditioning),
                    use_statement_delta=bool(args.use_statement_delta),
                    use_vote_features=bool(args.use_vote_features),
                    use_press_conf=bool(args.use_press_conf),
                    use_vix_features=bool(args.use_vix_features),
                    regime_loss=str(args.regime_loss),
                    focal_gamma=float(args.focal_gamma),
                    class_balanced_beta=float(args.class_balanced_beta),
                    text_encoder=(
                        str(args.text_encoder) if args.text_encoder else None
                    ),
                    use_text_embeddings=bool(args.use_text_embeddings),
                    vol_regime_label_mode=str(args.vol_regime_label_mode),
                    absolute_vol_thresholds=absolute_thresholds,
                    use_mp_surprise=bool(args.use_mp_surprise),
                    aux_horizons=tuple(args.aux_horizons),
                    aux_horizon_alpha=float(args.aux_horizon_alpha),
                )
            )

    summary: dict[str, Any] = {}
    for head_mode, trial_list in trials.items():
        per_fold_f1: list[float] = []
        per_fold_rmse: list[float] = []
        for trial in trial_list:
            for fold in trial["folds"]:
                metrics = fold.get("metrics", {}) or {}
                f1 = metrics.get("regime_f1_macro")
                rmse = metrics.get("regression_rmse_log_rv")
                if f1 is not None:
                    per_fold_f1.append(float(f1))
                if rmse is not None:
                    per_fold_rmse.append(float(rmse))
        summary[head_mode] = {
            "regime_f1_macro": _summary_stats(per_fold_f1),
            "regression_rmse_log_rv": _summary_stats(per_fold_rmse),
        }

    payload = {
        "head_modes": args.head_modes,
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": args.epochs,
        "regression_alpha": args.regression_alpha,
        "training_package_id": args.training_package_id,
        "rates_target_mode": str(args.rates_target_mode),
        "rates_heads": list(
            _resolve_auto_rates_heads(str(args.rates_target_mode), rates_heads=None)
        ),
        "vol_target_mode": str(args.vol_target_mode),
        "vol_target_horizon": int(args.target_horizon),
        "sequence_length": int(args.sequence_length),
        "use_retrieval_analogs": bool(args.use_retrieval_analogs),
        "use_regime_conditioning": bool(args.use_regime_conditioning),
        "use_statement_delta": bool(args.use_statement_delta),
        "use_vote_features": bool(args.use_vote_features),
        "use_press_conf": bool(args.use_press_conf),
        "use_vix_features": bool(args.use_vix_features),
        "regime_loss": str(args.regime_loss),
        "focal_gamma": float(args.focal_gamma),
        "class_balanced_beta": float(args.class_balanced_beta),
        "text_encoder": (
            str(args.text_encoder) if args.text_encoder else None
        ),
        "use_text_embeddings": bool(args.use_text_embeddings),
        "vol_regime_label_mode": str(args.vol_regime_label_mode),
        # When absolute mode is on, persist the EFFECTIVE thresholds
        # (CLI values when set, defaults otherwise) so the artefact
        # unambiguously records what trained -- avoids the "did
        # absolute mode use defaults or quantile mode" ambiguity that
        # would otherwise need source-code reading to resolve.
        "absolute_vol_thresholds": (
            list(absolute_thresholds)
            if absolute_thresholds is not None
            else (
                list(DEFAULT_ABSOLUTE_VOL_THRESHOLDS)
                if str(args.vol_regime_label_mode) == "absolute"
                else None
            )
        ),
        "absolute_calm_max_annualized": args.absolute_calm_max,
        "absolute_high_min_annualized": args.absolute_high_min,
        "use_mp_surprise": bool(args.use_mp_surprise),
        "aux_horizons": list(args.aux_horizons),
        "aux_horizon_alpha": float(args.aux_horizon_alpha),
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[dual_head_comparison] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
