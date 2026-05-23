from __future__ import annotations

import argparse
import sys
import warnings
import concurrent.futures
import csv
import itertools
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from app.models import FORECASTER_ARCHITECTURES
from app.models.config import (
    DEFAULT_TEXT_ADAPTER_DIM,
    DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
    FEATURE_SIZE,
    RICH_FEATURE_SIZE,
    TEXT_ADAPTER_DIM_CHOICES,
    rich_feature_size_with_text,
)
from app.services.forecaster import (
    BEST_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VALIDATION_SPLIT,
    FeatureVector,
    ModelConfig,
    TrainingRunSummary,
    SEQUENCE_LENGTH,
    inspect_training_data_sources,
    train_model,
)
from app.training.batched_sweep import (
    BATCHING_MODES,
    BucketKey,
    format_bucket_log_line,
    group_candidates_into_buckets,
    route_bucket,
    run_bucket_streams,
)
from app.training.loaders import (
    WalkForwardSplit,
    load_training_sequences_from_package,
    load_walk_forward_split,
)

# Official seed set for the multi-architecture sweep (mirrors the NLP bake-off
# protocol in ``docs/benchmark-policy.md``).
DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 97)

DEFAULT_REPORT_PATH = BEST_MODEL_PATH.parent / "forecaster_sweep_results.json"

# Random-search defaults. The sampler draws ``DEFAULT_RANDOM_SEARCH_SAMPLES``
# HP combos uniformly without replacement from the full HP cross-product;
# Bergstra & Bengio (JMLR 2012) show this matches grid search at a fraction
# of the cost when only a handful of axes dominate the loss surface.
DEFAULT_RANDOM_SEARCH_SAMPLES = 50
DEFAULT_RANDOM_SEARCH_SEED = 42

# VRAM-saturation warning threshold for the parallel-worker pool. The
# RTX 4080 carries 16 GB and the largest registered architecture
# (transformer, hidden=128, layers=3) holds roughly 1 GB per cell, so
# eight concurrent cells leave headroom for the CUDA allocator's
# fragmentation pool. Higher worker counts log a warning so a typo on
# the make target does not silently OOM the GPU mid-sweep.
PARALLEL_WORKERS_VRAM_WARN_THRESHOLD = 8

_LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the quantitative forecaster from prepared local datasets."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing JSON, JSONL, or CSV training datasets.",
    )
    parser.add_argument(
        "--training-package-id",
        default=None,
        help=(
            "Phase 8 training-package id under data/processed/. When set, the "
            "trainer consumes events.parquet's prior_bars_json column instead "
            "of scanning --data-dir for raw market-record JSON/JSONL/CSV files. "
            "Takes precedence over --data-dir when both are supplied."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(BEST_MODEL_PATH),
        help="Where to save the best-performing checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for optimizer steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    # The walk-forward validation slice is a *chronological* prefix of the
    # windows we hold out from training, not a sklearn-style random split.
    # The canonical flag is now ``--validation-fraction``; ``--validation-split``
    # stays available as a deprecated alias so existing run scripts keep
    # working until the next major version of the CLI lands.
    parser.add_argument(
        "--validation-fraction",
        "--validation-split",
        dest="validation_fraction",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help=(
            "Fraction of windows reserved for the walk-forward validation "
            "slice (chronological prefix, not a shuffle). The deprecated "
            "--validation-split alias is accepted for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Number of non-improving epochs allowed before early stop.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="LSTM hidden size for a single training run.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="Number of stacked LSTM layers for a single training run.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout applied to the LSTM stack and forecast head.",
    )
    parser.add_argument(
        "--head-hidden-size",
        type=int,
        default=DEFAULT_HEAD_HIDDEN_SIZE,
        help="Hidden layer width for the projection head.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--architecture",
        choices=list(FORECASTER_ARCHITECTURES),
        default="lstm",
        help="Forecaster architecture for a single training run.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=list(FORECASTER_ARCHITECTURES),
        help="Sweep over the listed architectures (combine with --sweep).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for a single training run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Sweep over the listed seeds. Defaults to the official seed set when "
        "combined with --architectures.",
    )
    parser.add_argument(
        "--credibility-features",
        action="store_true",
        help="Enable the 4-axis credibility feature path on the forecaster. Default off "
        "preserves the byte-identical regression-test contract for architecture=lstm.",
    )
    # Rich-feature input space. Default on so a make forecaster-sweep
    # against a training package consumes the four-family per-bar
    # vector. Explicit --no-rich-features reproduces the pre-PR-#173
    # 6-dim path for back-compat smoke checks.
    rich_group = parser.add_mutually_exclusive_group()
    rich_group.add_argument(
        "--rich-features",
        dest="rich_features",
        action="store_true",
        help=(
            "Enable the 35-dim rich-feature input space on the "
            "training-package loader. The credibility / linguistic / "
            "mp-surprise / multi-axis families are joined onto each "
            "event and broadcast to every bar of the 20-day prior "
            "window. Default on; ignored when --training-package-id "
            "is unset."
        ),
    )
    rich_group.add_argument(
        "--no-rich-features",
        dest="rich_features",
        action="store_false",
        help=(
            "Disable the rich-feature input space. Reproduces the "
            "pre-PR-#173 6-feature path for back-compat smoke runs."
        ),
    )
    parser.set_defaults(rich_features=True)
    # Per-family ablation flags. When a family is off, its slice in
    # the per-bar feature vector is zeroed but the feature size stays
    # at RICH_FEATURE_SIZE, so a single sweep can measure per-family
    # lift without retraining the model architecture.
    parser.add_argument(
        "--no-credibility",
        dest="use_credibility",
        action="store_false",
        help=(
            "Zero the credibility 4-vector slice in the rich-feature "
            "input. Default on. Ignored when --no-rich-features is set."
        ),
    )
    parser.add_argument(
        "--no-linguistic",
        dest="use_linguistic",
        action="store_false",
        help=(
            "Zero the 15-dim linguistic slice in the rich-feature "
            "input. Default on. Ignored when --no-rich-features is set."
        ),
    )
    parser.add_argument(
        "--no-mp-surprise",
        dest="use_mp_surprise",
        action="store_false",
        help=(
            "Zero the MP-surprise 4-vector slice in the rich-feature "
            "input. Default on. Ignored when --no-rich-features is set."
        ),
    )
    parser.add_argument(
        "--no-multi-axis",
        dest="use_multi_axis",
        action="store_false",
        help=(
            "Zero the 6-dim multi-axis slice in the rich-feature "
            "input. Default on. Ignored when --no-rich-features is set."
        ),
    )
    # B1 (#212) LLM-as-features. Default off so the existing
    # Tier 1/2/3 sweep baselines stay byte-identical with cached
    # extractions present on disk; --use-llm-features flips the per-event
    # one-hot block + missing flag on for the Tier 4 comparison.
    parser.add_argument(
        "--use-llm-features",
        dest="use_llm_features",
        action="store_true",
        help=(
            "Attach the cached LLM-extracted categorical feature block "
            "(35-dim one-hot + 1-dim missing flag) to every event. "
            "Requires the catalogue extractor to have been run for the "
            "training package. Default off."
        ),
    )
    parser.add_argument(
        "--no-llm-features",
        dest="use_llm_features",
        action="store_false",
        help="Disable the LLM-features block (zeros + missing=1.0).",
    )
    parser.set_defaults(
        use_credibility=True,
        use_linguistic=True,
        use_mp_surprise=True,
        use_multi_axis=True,
        use_llm_features=False,
    )
    # Phase 9 V2 (#195) classification dispatch. Default stays
    # ``regression`` so the existing ablation grid + determinism
    # regression test reproduce the same close / vol RMSE numbers.
    # ``--output-mode classification`` swaps the (close, vol) regression
    # target for a per-fold ``vol_regime_10d`` class index; the per-fold
    # quantile cutoffs are fitted on the train slice only and persist
    # onto the saved checkpoint via ``ModelConfig.vol_regime_quantiles``.
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=("regression", "classification"),
        default="regression",
        help=(
            "Forecaster head dispatch. ``regression`` keeps the legacy "
            "(close, vol) SmoothL1 path; ``classification`` swaps in a "
            "CrossEntropy head over the ``forward_realized_vol_10d`` "
            "target with per-fold quantile cutoffs."
        ),
    )
    parser.add_argument(
        "--vol-regime-classes",
        type=int,
        default=3,
        help=(
            "Number of vol-regime classes (default 3: calm / normal / "
            "high). Cutoffs are interior quantiles fitted on the train "
            "slice of each walk-forward fold."
        ),
    )
    # Text-embedding path. ``--text-encoder=none`` keeps the no-text
    # path. ``--no-text-embeddings`` is the symmetric per-family flag
    # that mirrors PR #173's per-family ablation pattern (model input
    # shape stays constant; the embedding slot zeros out).
    _TEXT_ENCODER_CHOICES = (
        "none",
        "finbert",
        "finbert_fomc",
        "finbert_fed_adjacent",
        "finbert_fed_adjacent_xbank",
        "bert_base_fed_adjacent",
        "bge_large_en_v15",
        "nomic_embed_text_v15",
        "voyage_finance_2",
    )
    parser.add_argument(
        "--text-encoder",
        choices=_TEXT_ENCODER_CHOICES,
        default="none",
        help=(
            "Encoder alias whose pooled FOMC statement embeddings the "
            "forecaster consumes as a 5th feature family. The loader "
            "pulls the per-statement embeddings from "
            "data/raw/embeddings/<encoder>_<rev>.parquet and applies a "
            "softmax(-Delta t / lambda) weighted mean over the four "
            "most recent prior statements per event. Default 'none' "
            "keeps the rich-features-only path byte-identical."
        ),
    )
    parser.add_argument(
        "--text-adapter-dim",
        type=int,
        choices=list(TEXT_ADAPTER_DIM_CHOICES),
        default=DEFAULT_TEXT_ADAPTER_DIM,
        help=(
            "Projection target for the encoder-agnostic text-embedding "
            "adapter. The sweep iterates over {32, 64, 128} so the "
            "diminishing-returns curve across adapter widths is visible "
            "in the aggregator table."
        ),
    )
    parser.add_argument(
        "--text-pool-lambda-inv-days",
        type=float,
        default=DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
        help=(
            "Time-decay window for the prior-4 statement pool. Smaller "
            "values concentrate the weight on the most recent statement, "
            "larger values spread the weight across all four."
        ),
    )
    parser.add_argument(
        "--no-text-embeddings",
        dest="use_text_embeddings",
        action="store_false",
        help=(
            "Zero the text-embedding slice while keeping the model "
            "input shape constant. Mirrors the per-family ablation "
            "pattern from PR #173 -- the model architecture is fixed "
            "across the with/without rows so a single sweep can "
            "measure text-embedding lift without retraining."
        ),
    )
    parser.add_argument(
        "--shuffle-targets-control",
        action="store_true",
        help=(
            "Permute the target column per fold (seed-fixed) before "
            "training. macro-RMSE on the shuffled-targets run should "
            "sit near the constant-mean predictor; a real-targets run "
            "whose RMSE is close to its shuffled counterpart is "
            "memorising, not learning."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help=(
            "AdamW weight decay. Default 1e-4 preserves the pre-PR-#176 "
            "regularisation; the sweep searches over {0, 1e-4, 1e-3}."
        ),
    )
    parser.set_defaults(use_text_embeddings=True)
    parser.add_argument(
        "--no-time-decay",
        dest="use_time_decay",
        action="store_false",
        help=(
            "Disable the elapsed-time decay path "
            "(``TimeDecayAttention``). Default off-flag is on, so the "
            "decay multiplies the sentiment channel by "
            "``exp(-lambda * |elapsed|)``. Round 4 (#243) ablation "
            "flips this to measure whether the mechanism still earns "
            "its complexity on the post-embargo baseline."
        ),
    )
    parser.set_defaults(use_time_decay=True)
    parser.add_argument(
        "--encoder-lora",
        dest="encoder_lora",
        action="store_true",
        help=(
            "Round 5 (#244) ceiling probe: pull the configured "
            "``--text-encoder`` checkpoint into the training loop, "
            "wrap with PEFT LoRA (r=8, alpha=16, dropout=0.1, target "
            "modules {query, value}), and run the forward per batch "
            "so the regime loss flows gradients into the encoder. "
            "Default off keeps the parquet-cached embedding path. "
            "Only supported on the walk-forward path with a registered "
            "encoder alias (revision must be pinned)."
        ),
    )
    parser.set_defaults(encoder_lora=False)
    # Phase B (#227) LR schedule + sequence-length knobs.
    parser.add_argument(
        "--lr-schedule",
        choices=("plateau", "cosine_warmup"),
        default="plateau",
        help=(
            "LR schedule. ``plateau`` (default) is the legacy ReduceLROnPlateau "
            "path locked by the determinism regression. ``cosine_warmup`` "
            "builds a OneCycleLR (warmup -> cosine -> tail) over the epoch "
            "budget."
        ),
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help=(
            "Sliding-window length per training row. ``0`` (default) means "
            "use the module-level ``SEQUENCE_LENGTH`` constant (20). Override "
            "to 40, 60, ... for the capacity push at longer sequences."
        ),
    )
    parser.add_argument(
        "--lr-schedules",
        nargs="+",
        choices=("plateau", "cosine_warmup"),
        help="Sweep-mode grid for LR schedule.",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        help="Sweep-mode grid for sequence length. Overrides --sequence-length.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a grid search and select the best checkpoint by validation RMSE.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        help="Grid-search values for hidden size.",
    )
    parser.add_argument(
        "--num-layers-grid",
        nargs="+",
        type=int,
        help="Grid-search values for number of LSTM layers.",
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        help="Grid-search values for dropout.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Grid-search values for optimizer learning rate.",
    )
    parser.add_argument(
        "--epochs-grid",
        nargs="+",
        type=int,
        help="Grid-search values for epochs.",
    )
    parser.add_argument(
        "--weight-decays",
        nargs="+",
        type=float,
        help="Grid-search values for AdamW weight decay.",
    )
    parser.add_argument(
        "--text-adapter-dims",
        nargs="+",
        type=int,
        choices=list(TEXT_ADAPTER_DIM_CHOICES),
        help=(
            "Grid-search values for the text-embedding adapter "
            "projection dim. Ignored when --text-encoder=none."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the hyperparameter search report JSON.",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        help=(
            "Sample HP combos uniformly from the full cross-product instead "
            "of iterating every cell exhaustively. The architecture and seed "
            "axes are still enumerated in full -- only the HP grid "
            "(hidden_size / num_layers / dropout / learning_rate / "
            "epochs / weight_decay / text_adapter_dim) is subsampled. "
            "Default off keeps the back-compat exhaustive path."
        ),
    )
    parser.add_argument(
        "--random-search-samples",
        type=int,
        default=DEFAULT_RANDOM_SEARCH_SAMPLES,
        help=(
            "Number of HP combos to draw when --random-search is on. "
            "Clamps to the grid size when M exceeds the cross-product. "
            "Ignored when --random-search is off."
        ),
    )
    parser.add_argument(
        "--random-search-seed",
        type=int,
        default=DEFAULT_RANDOM_SEARCH_SEED,
        help=(
            "RNG seed for the random-search sampler. Separate from per-cell "
            "training seeds so re-running with the same value samples the "
            "same HP subset regardless of the seeds axis."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of cells to train concurrently on the same GPU. Each "
            "worker is a spawn-mode subprocess with its own CUDA context. "
            "Default 1 (sequential) preserves the existing back-compat "
            "behaviour. Recommended N=8 on an RTX 4080; higher values log "
            "a VRAM-saturation warning. Ignored when --batching-mode is "
            "anything other than 'off' (the bucketed runner schedules "
            "concurrency inside one Python process)."
        ),
    )
    parser.add_argument(
        "--batching-mode",
        choices=("auto", "stacked", "streams", "off"),
        default="off",
        help=(
            "Bucketing strategy for hyperparameter cells that share the "
            "same model topology and data feed. 'auto' (recommended) "
            "consults the per-arch table -- dlinear routes to stacked, "
            "every other architecture routes to streams. 'stacked' "
            "stacks per-cell parameters along a synthetic batch axis "
            "and runs one matmul per bucket; falls back to streams "
            "automatically on architectures whose forward is not yet "
            "vmap-friendly. 'streams' overlaps per-cell kernel launches "
            "across CUDA streams + threads inside one CUDA context. "
            "'off' preserves the legacy ProcessPoolExecutor path "
            "verbatim for the byte-identity regression contract. Default "
            "'off' keeps the existing CLI surface untouched; opt in by "
            "passing --batching-mode=auto on a real sweep."
        ),
    )
    parser.add_argument(
        "--max-bucket-size",
        type=int,
        default=None,
        help=(
            "Override the per-architecture bucket-size cap. Default "
            "unset, so the per-arch table picks 64 for dlinear, 32 for "
            "lstm/gru/tcn, 16 for lstm_attn, 8 for transformer, 4 for "
            "informer/tft. Lower values trade throughput for VRAM "
            "headroom on smaller GPUs."
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Walk-forward fold ids from "
            "fold_manifest_expanding_walk_forward.json (e.g. wf_fold_1 "
            "wf_fold_2 wf_fold_3 wf_fold_4). When set, each cell trains "
            "once per (architecture, seed, hp_combo, fold). The per-fold "
            "splits come from the manifest's expanding-window date "
            "ranges; the val + test partitions are honoured separately "
            "by the training loop. Ignored when "
            "--training-package-id is unset."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=("auto", "single-fold", "walk-forward"),
        default="auto",
        help=(
            "Force the split protocol. 'auto' (default) routes to "
            "walk-forward when --folds is supplied and to single-fold "
            "(package's splits_train_val_test.parquet) otherwise. "
            "'single-fold' / 'walk-forward' override the auto choice. "
            "Ignored when --training-package-id is unset."
        ),
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help=(
            "Purge buffer between adjacent walk-forward partitions "
            "(López de Prado, Advances in Financial ML, ch. 7). Drops "
            "val rows whose event date sits within this many calendar "
            "days of the fold's train_end, and test rows within this "
            "many days of val_end. The default 20 covers a 10-day "
            "forward target while leaving headroom; the strict no-overlap "
            "threshold for SEQUENCE_LENGTH=20 + 10d horizon is 30 days. "
            "Pass 0 to opt out (reproduces the pre-2026-05-23 leaky "
            "baseline). Honoured on the walk-forward path only."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("event_study", "realized_return"),
        default="event_study",
        help=(
            "Target-frame derivation for the training-package loader. "
            "'event_study' (default) uses abnormal_return + volatility_shift "
            "so the synthesised target removes the broad-market component "
            "from the close and reconstructs the post-event 10d realised "
            "vol from the prior vol + shift. 'realized_return' reproduces "
            "the pre-fix behaviour (close * (1 + realized_return) and a "
            "literal copy of prior vol_5d) and is preserved for back-compat "
            "smoke tests only. Ignored when --training-package-id is unset."
        ),
    )
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="Only inspect discovered datasets and exit without training.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help=(
            "Per-step gradient-norm clip applied via "
            "``nn.utils.clip_grad_norm_``. Default 0.0 disables the clip "
            "(the per-step host sync the clip forced was a measurable "
            "drag on small-model GPU saturation). Pass a positive value "
            "to opt into clipping at that norm. The previous always-on "
            "1.0 clip is reproducible by ``--grad-clip-norm 1.0``."
        ),
    )
    parser.add_argument(
        "--no-compile",
        dest="use_compile",
        action="store_false",
        help=(
            "Skip ``torch.compile(model, mode='reduce-overhead')`` on "
            "the forecaster forward path. Default off-flag is on, so "
            "compile fires whenever the architecture is in the "
            "compatible table and the device is CUDA. CPU and "
            "incompatible architectures auto-fall back to eager."
        ),
    )
    parser.set_defaults(use_compile=True)
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help=(
            "Skip ``torch.cuda.amp.autocast`` + ``GradScaler`` on the "
            "train step. Default off-flag is on, so autocast fires on "
            "CUDA for the architectures in the compatible table; CPU "
            "and incompatible architectures fall back to fp32."
        ),
    )
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        help=(
            "Skip the A1 (#206) per-fold inverse-frequency class "
            "weighting in classification mode. Default off-flag is on, "
            "so ``CrossEntropyLoss(weight=…)`` fires with weights fit "
            "on the train slice. Round 2c (#234) ablation toggles this "
            "to measure whether the weighting actually shifts macro-F1 "
            "now that PR #233 fixed the val-loss arithmetic."
        ),
    )
    parser.set_defaults(use_class_weights=True)
    args = parser.parse_args()
    # Emit DeprecationWarning when the legacy --validation-split alias is
    # used. argparse silently maps it onto validation_fraction, so callers
    # would otherwise have no signal that the name is on its way out.
    if any(arg == "--validation-split" or arg.startswith("--validation-split=") for arg in sys.argv[1:]):
        warnings.warn(
            "--validation-split is deprecated; use --validation-fraction. "
            "The walk-forward validation slice is a chronological prefix, "
            "not a sklearn-style random split, and the new flag name reflects that.",
            DeprecationWarning,
            stacklevel=2,
        )
    return args


def _text_path_active(args: argparse.Namespace) -> bool:
    """Return True when the text-embedding path is wired for this run."""

    encoder = str(getattr(args, "text_encoder", "none") or "none")
    use_text = bool(getattr(args, "use_text_embeddings", True))
    use_package_path = bool(getattr(args, "training_package_id", None))
    return encoder != "none" and use_text and use_package_path


def _resolved_input_size(args: argparse.Namespace) -> int:
    """Return the per-bar scalar input size implied by the rich-feature flag.

    The rich-feature input only takes effect when the loader actually
    populates the rich payload (``--training-package-id`` set AND
    ``--rich-features`` on). On the legacy ``--data-dir`` JSON / JSONL
    path the per-bar size stays at ``FEATURE_SIZE`` regardless of the
    flag, so the regression-test contract on the
    ``--data-dir`` code path is unaffected. The text-embedding adapter
    slot is widened separately inside ``ForecasterModel`` via
    ``text_adapter_dim`` so this scalar size stays at 35 even when
    text embeddings are on.
    """

    rich_on = bool(getattr(args, "rich_features", False))
    use_package_path = bool(getattr(args, "training_package_id", None))
    if rich_on and use_package_path:
        return RICH_FEATURE_SIZE
    return FEATURE_SIZE


def _resolved_text_adapter_dim(args: argparse.Namespace, override: int | None = None) -> int:
    """Return the adapter dim for the current run (0 disables the path)."""

    if not _text_path_active(args):
        return 0
    if override is not None:
        return int(override)
    return int(getattr(args, "text_adapter_dim", DEFAULT_TEXT_ADAPTER_DIM))


def _resolve_text_embedding_dim(args: argparse.Namespace) -> int:
    """Resolve the encoder-native pooled embedding dim for the current encoder.

    The model needs the encoder-native ``in_dim`` to materialise the
    adapter; the loader emits embeddings of that width per row. The
    helper reads the first non-empty pooled row off the loader's
    output via a peek-style import, but the simpler approach used
    here is to pin the dim from a small static table that matches
    the registry. The forecaster sweep CLI does not have the parquet
    open at this point so we fall back to a registry table; a
    mismatch surfaces at the adapter's first forward pass.
    """

    if not _text_path_active(args):
        return 0
    table = {
        "finbert": 768,
        "finbert_fomc": 768,
        "finbert_fed_adjacent": 768,
        "finbert_fed_adjacent_xbank": 768,
        "bert_base_fed_adjacent": 768,
        "bge_large_en_v15": 1024,
        "nomic_embed_text_v15": 768,
        "voyage_finance_2": 1024,
    }
    encoder = str(getattr(args, "text_encoder", "none") or "none")
    if encoder == "none":
        return 0
    return int(table.get(encoder, 768))


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    output_mode = str(getattr(args, "output_mode", "regression") or "regression")
    n_classes = int(getattr(args, "vol_regime_classes", 3) or 3)
    return ModelConfig(
        input_size=_resolved_input_size(args),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_size=args.head_hidden_size,
        architecture=args.architecture,
        credibility_features=bool(args.credibility_features),
        text_embedding_dim=_resolve_text_embedding_dim(args),
        text_adapter_dim=_resolved_text_adapter_dim(args),
        output_mode=output_mode,
        n_classes=n_classes,
        lr_schedule=str(getattr(args, "lr_schedule", "plateau") or "plateau"),
        sequence_length=int(getattr(args, "sequence_length", 0) or 0),
        use_time_decay=bool(getattr(args, "use_time_decay", True)),
        encoder_lora=bool(getattr(args, "encoder_lora", False)),
    )


def _print_data_inventory(
    data_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    summaries: Sequence[Any],
    *,
    sequence_count: int,
    observation_count: int,
    window_count: int,
) -> None:
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Existing checkpoint: {'yes' if checkpoint_path.exists() else 'no'}")
    print(f"Sequence groups discovered: {sequence_count}")
    print(f"Observations discovered: {observation_count}")
    print(f"Training windows available: {window_count}")
    if summaries:
        print("Discovered data sources:")
        for summary in summaries:
            relative_path = summary.path.relative_to(data_dir) if summary.path.is_relative_to(data_dir) else summary.path
            print(
                f"  - {relative_path} [{summary.status}] "
                f"groups={summary.record_groups}, records={summary.records}, "
                f"vectors={summary.vectors}, usable={summary.usable_sequences} :: {summary.message}"
            )
    else:
        print("Discovered data sources: none")


def _metrics_rank(summary: TrainingRunSummary) -> tuple[float, float]:
    if summary.metrics is None:
        return float("inf"), float("inf")
    return summary.metrics.combined_rmse, summary.metrics.loss


def select_best_summary(summaries: Sequence[TrainingRunSummary]) -> TrainingRunSummary | None:
    ranked = [summary for summary in summaries if summary.metrics is not None]
    if not ranked:
        return None
    return min(ranked, key=_metrics_rank)


def _build_hp_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the cross-product of the HP axes (no architectures or seeds).

    The HP grid is the seven hyperparameter axes the sweep tunes per
    architecture-seed pair: hidden_size, num_layers, dropout,
    learning_rate, epochs, weight_decay, text_adapter_dim. The outer
    architecture and seed enumeration happens in
    ``build_sweep_candidates`` so the same HP combo can be evaluated
    across the bake-off architecture roster and the official seed set.
    """

    hidden_sizes = args.hidden_sizes or [args.hidden_size]
    num_layers_options = args.num_layers_grid or [args.num_layers]
    dropouts = args.dropouts or [args.dropout]
    learning_rates = args.learning_rates or [args.learning_rate]
    epochs_options = args.epochs_grid or [args.epochs]
    # ``getattr`` fallbacks below keep the existing
    # ``test_build_sweep_candidates_creates_cartesian_product`` namespace
    # (which predates the weight-decay / text-adapter axes) intact while
    # the production CLI runs always pass the new flags through.
    weight_decays = (
        getattr(args, "weight_decays", None)
        or [getattr(args, "weight_decay", 1e-4)]
    )
    # Text-adapter-dim axis. Iterated only when the text-embedding path
    # is wired; on the no-text path the iteration collapses to a single
    # dummy value so the loop product stays one-deep.
    if _text_path_active(args):
        text_adapter_dims = (
            getattr(args, "text_adapter_dims", None)
            or [getattr(args, "text_adapter_dim", DEFAULT_TEXT_ADAPTER_DIM)]
        )
    else:
        text_adapter_dims = [0]
    # Phase B (#227): new sweep axes for LR-schedule and sequence-length.
    # Defaults collapse to a single value so legacy callers that do not
    # pass --lr-schedules / --sequence-lengths reproduce the pre-PR grid
    # byte-identical.
    lr_schedules = (
        getattr(args, "lr_schedules", None)
        or [getattr(args, "lr_schedule", "plateau")]
    )
    sequence_lengths = (
        getattr(args, "sequence_lengths", None)
        or [getattr(args, "sequence_length", 0)]
    )
    hp_grid: list[dict[str, Any]] = []
    for (
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        epochs,
        weight_decay,
        text_adapter_dim,
        lr_schedule,
        sequence_length,
    ) in itertools.product(
        hidden_sizes,
        num_layers_options,
        dropouts,
        learning_rates,
        epochs_options,
        weight_decays,
        text_adapter_dims,
        lr_schedules,
        sequence_lengths,
    ):
        hp_grid.append(
            {
                "hidden_size": int(hidden_size),
                "num_layers": int(num_layers),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "weight_decay": float(weight_decay),
                "text_adapter_dim": int(text_adapter_dim),
                "lr_schedule": str(lr_schedule),
                "sequence_length": int(sequence_length),
            }
        )
    return hp_grid


def sample_random_search_subset(
    hp_grid: Sequence[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Draw ``samples`` HP combos uniformly without replacement.

    The returned tuples carry the original grid index as the first
    element so the caller can persist a stable ``hp_combo_id`` per
    sampled combo. ``samples`` clamps to ``len(hp_grid)`` -- asking
    for more combos than the grid contains returns every combo, not
    an error. The sampler RNG is isolated from per-cell training
    seeds: re-running with the same ``seed`` reproduces the same
    subset of combos.
    """

    grid_size = len(hp_grid)
    if grid_size == 0:
        return []
    draw_count = min(int(samples), grid_size)
    if draw_count < 1:
        draw_count = grid_size
    rng = np.random.RandomState(int(seed))
    indices = rng.choice(grid_size, size=draw_count, replace=False)
    # ``np.choice`` returns ``np.int64`` entries; cast to ``int`` so the
    # downstream ``hp_combo_id`` field round-trips through JSON without
    # extra serialiser shims.
    return [(int(idx), dict(hp_grid[int(idx)])) for idx in indices]


def _resolved_fold_ids(args: argparse.Namespace) -> list[str | None]:
    """Return the fold ids the candidate enumeration iterates over.

    The list always has at least one entry. On the single-fold path the
    entry is ``None`` (the trainer reads
    ``splits_train_val_test.parquet`` for the package's default
    partition); on the walk-forward path the list mirrors the
    ``--folds`` argument verbatim, so each candidate cell expands to
    ``len(folds)`` trials.
    """

    folds = getattr(args, "folds", None) or []
    folds = [str(f).strip() for f in folds if str(f).strip()]
    if not folds:
        return [None]
    return list(folds)


def build_sweep_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    architectures = args.architectures or [args.architecture]
    # When the caller asks for an architecture sweep but doesn't list seeds we
    # fall back to the official five-seed set; this matches the bake-off
    # aggregator and avoids accidentally publishing a single-seed table.
    if args.seeds:
        seeds: list[int | None] = [int(s) for s in args.seeds]
    elif args.architectures:
        seeds = list(DEFAULT_SWEEP_SEEDS)
    else:
        seeds = [args.seed]

    fold_ids = _resolved_fold_ids(args)

    use_random_search = bool(getattr(args, "random_search", False))
    text_embedding_dim = _resolve_text_embedding_dim(args)

    if use_random_search:
        # Random-search path: subsample the HP grid before the
        # architecture-by-seed outer enumeration. Each sampled combo
        # keeps its index into the full grid as ``hp_combo_id`` so the
        # aggregator can group by-combo for ablation.
        hp_grid = _build_hp_grid(args)
        sampled_hp = sample_random_search_subset(
            hp_grid,
            int(getattr(args, "random_search_samples", DEFAULT_RANDOM_SEARCH_SAMPLES)),
            int(getattr(args, "random_search_seed", DEFAULT_RANDOM_SEARCH_SEED)),
        )
        candidates: list[dict[str, Any]] = []
        for architecture, seed in itertools.product(architectures, seeds):
            for hp_combo_id, hp in sampled_hp:
                text_adapter_dim = int(hp["text_adapter_dim"])
                for fold_id in fold_ids:
                    candidate = {
                        "model_config": ModelConfig(
                            input_size=_resolved_input_size(args),
                            hidden_size=hp["hidden_size"],
                            num_layers=hp["num_layers"],
                            dropout=hp["dropout"],
                            head_hidden_size=args.head_hidden_size,
                            architecture=str(architecture),
                            credibility_features=bool(args.credibility_features),
                            text_embedding_dim=int(text_embedding_dim) if text_adapter_dim > 0 else 0,
                            text_adapter_dim=text_adapter_dim,
                            output_mode=str(getattr(args, "output_mode", "regression") or "regression"),
                            n_classes=int(getattr(args, "vol_regime_classes", 3) or 3),
                            use_time_decay=bool(getattr(args, "use_time_decay", True)),
                            encoder_lora=bool(getattr(args, "encoder_lora", False)),
                        ),
                        "learning_rate": float(hp["learning_rate"]),
                        "epochs": int(hp["epochs"]),
                        "weight_decay": float(hp["weight_decay"]),
                        "text_adapter_dim": text_adapter_dim,
                        "seed": int(seed) if seed is not None else None,
                        "hp_combo_id": int(hp_combo_id),
                    }
                    if fold_id is not None:
                        candidate["fold_id"] = fold_id
                    candidates.append(candidate)
        return candidates

    # Exhaustive path: byte-identical to the pre-PR enumeration order so
    # the regression-test contract on the legacy sweep output is preserved.
    hidden_sizes = args.hidden_sizes or [args.hidden_size]
    num_layers_options = args.num_layers_grid or [args.num_layers]
    dropouts = args.dropouts or [args.dropout]
    learning_rates = args.learning_rates or [args.learning_rate]
    epochs_options = args.epochs_grid or [args.epochs]
    weight_decays = (
        getattr(args, "weight_decays", None)
        or [getattr(args, "weight_decay", 1e-4)]
    )
    if _text_path_active(args):
        text_adapter_dims = (
            getattr(args, "text_adapter_dims", None)
            or [getattr(args, "text_adapter_dim", DEFAULT_TEXT_ADAPTER_DIM)]
        )
    else:
        text_adapter_dims = [0]
    # Phase B (#227): cross the LR-schedule + sequence-length grids
    # alongside the existing axes. Both default to a single value so
    # legacy callers that don't pass the new grids reproduce the
    # pre-PR cartesian product byte-identical.
    lr_schedules = (
        getattr(args, "lr_schedules", None)
        or [getattr(args, "lr_schedule", "plateau")]
    )
    sequence_lengths = (
        getattr(args, "sequence_lengths", None)
        or [getattr(args, "sequence_length", 0)]
    )
    candidates = []
    for (
        architecture,
        hidden_size,
        num_layers,
        dropout,
        learning_rate,
        epochs,
        weight_decay,
        text_adapter_dim,
        lr_schedule,
        sequence_length,
        seed,
    ) in itertools.product(
        architectures,
        hidden_sizes,
        num_layers_options,
        dropouts,
        learning_rates,
        epochs_options,
        weight_decays,
        text_adapter_dims,
        lr_schedules,
        sequence_lengths,
        seeds,
    ):
        for fold_id in fold_ids:
            candidate = {
                "model_config": ModelConfig(
                    input_size=_resolved_input_size(args),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    head_hidden_size=args.head_hidden_size,
                    architecture=str(architecture),
                    credibility_features=bool(args.credibility_features),
                    text_embedding_dim=int(text_embedding_dim) if int(text_adapter_dim) > 0 else 0,
                    text_adapter_dim=int(text_adapter_dim),
                    output_mode=str(getattr(args, "output_mode", "regression") or "regression"),
                    n_classes=int(getattr(args, "vol_regime_classes", 3) or 3),
                    lr_schedule=str(lr_schedule),
                    sequence_length=int(sequence_length),
                    use_time_decay=bool(getattr(args, "use_time_decay", True)),
                    encoder_lora=bool(getattr(args, "encoder_lora", False)),
                ),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "weight_decay": float(weight_decay),
                "text_adapter_dim": int(text_adapter_dim),
                "seed": int(seed) if seed is not None else None,
            }
            if fold_id is not None:
                candidate["fold_id"] = fold_id
            candidates.append(candidate)
    return candidates


def _flatten_trial_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    metrics = summary.get("metrics") or {}
    train_metrics = summary.get("train_metrics") or {}
    val_metrics = summary.get("val_metrics") or {}
    test_metrics = summary.get("test_metrics") or {}
    model_config = summary.get("model_config") or {}
    flattened: dict[str, Any] = {
        "trial_index": record["trial_index"],
        "selected": record.get("selected", False),
        "architecture": model_config.get("architecture") or record.get("architecture"),
        "seed": record.get("seed"),
        "credibility_features": model_config.get("credibility_features"),
        "hidden_size": model_config.get("hidden_size"),
        "num_layers": model_config.get("num_layers"),
        "dropout": model_config.get("dropout"),
        "head_hidden_size": model_config.get("head_hidden_size"),
        "epochs_requested": summary.get("epochs_requested"),
        "epochs_completed": summary.get("epochs_completed"),
        "learning_rate": summary.get("learning_rate"),
        "batch_size": summary.get("batch_size"),
        "validation_split": summary.get("validation_split"),
        "best_epoch": summary.get("best_epoch"),
        "weight_decay": summary.get("weight_decay"),
        "target_mode": summary.get("target_mode"),
        "text_encoder": summary.get("text_encoder"),
        "text_adapter_dim": summary.get("text_adapter_dim") or model_config.get("text_adapter_dim"),
        "text_embedding_dim": model_config.get("text_embedding_dim"),
        "text_pool_lambda_inv_days": summary.get("text_pool_lambda_inv_days"),
        "combined_rmse": metrics.get("combined_rmse"),
        "loss": metrics.get("loss"),
        "close_rmse": metrics.get("close_rmse"),
        "volatility_rmse": metrics.get("volatility_rmse"),
        "train_combined_rmse": train_metrics.get("combined_rmse"),
        "train_close_rmse": train_metrics.get("close_rmse"),
        "train_volatility_rmse": train_metrics.get("volatility_rmse"),
        "train_loss": train_metrics.get("loss"),
        # Walk-forward partition metrics. On the legacy single-tensor
        # path ``val_metrics`` collapses to ``metrics`` and
        # ``test_metrics`` is absent; emit the columns when present so
        # the CSV reflects the new train/val/test contract on the
        # walk-forward path without breaking the legacy column set.
        "val_combined_rmse": val_metrics.get("combined_rmse"),
        "val_close_rmse": val_metrics.get("close_rmse"),
        "val_volatility_rmse": val_metrics.get("volatility_rmse"),
        "test_combined_rmse": test_metrics.get("combined_rmse"),
        "test_close_rmse": test_metrics.get("close_rmse"),
        "test_volatility_rmse": test_metrics.get("volatility_rmse"),
        "fold_id": summary.get("fold_id"),
        "protocol": summary.get("protocol"),
        "checkpoint_saved": summary.get("checkpoint_saved"),
        "checkpoint_path": summary.get("checkpoint_path"),
    }
    # ``hp_combo_id`` is only present on random-search runs; the
    # exhaustive path never carries it, so the column set stays
    # byte-identical to the pre-PR CSV when --random-search is off.
    if "hp_combo_id" in record:
        flattened["hp_combo_id"] = record["hp_combo_id"]
    return flattened


def _write_sweep_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = report_path.with_suffix(".csv")
    trial_rows = [_flatten_trial_record(trial) for trial in payload.get("trials", [])]
    if not trial_rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)


def _run_single_training(
    *,
    data_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    early_stopping_patience: int,
    model_config: ModelConfig,
    save_checkpoint: bool,
    seed: int | None = None,
    sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    walk_forward_split: WalkForwardSplit | None = None,
    weight_decay: float = 1e-4,
    shuffle_targets_control: bool = False,
    text_encoder: str | None = None,
    text_pool_lambda_inv_days: float = 0.0,
    grad_clip_norm: float = 0.0,
    use_compile: bool = True,
    use_amp: bool = True,
    use_class_weights: bool = True,
) -> TrainingRunSummary:
    # Three input paths, in precedence order:
    #
    # - ``walk_forward_split`` set: pre-split train/val/test partitions
    #   from ``load_walk_forward_split``; the training loop honours
    #   them separately and reports the real held-out test_rmse.
    # - ``sequence_groups`` set: single-list training-package path
    #   (pre-walk-forward back-compat); the training loop falls back
    #   to the legacy 80/20 internal split. Kept callable so the
    #   regression-test fixture path keeps the byte-identity contract.
    # - neither set: data-dir JSON / JSONL / CSV scan, also legacy.
    # Phase B (#227): the schedule choice rides on the model config so a
    # resumed checkpoint reuses the same schedule the original run used.
    lr_schedule_choice = str(getattr(model_config, "lr_schedule", "plateau") or "plateau")
    if walk_forward_split is not None:
        result = train_model(
            train_sequence_groups=walk_forward_split.train,
            val_sequence_groups=walk_forward_split.val,
            test_sequence_groups=walk_forward_split.test,
            fold_id=walk_forward_split.fold_id,
            protocol=walk_forward_split.protocol,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            device=device,
            model_config=model_config,
            seed=seed,
            weight_decay=weight_decay,
            shuffle_targets_control=shuffle_targets_control,
            text_encoder=text_encoder,
            text_pool_lambda_inv_days=text_pool_lambda_inv_days,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            use_amp=use_amp,
            lr_schedule=lr_schedule_choice,
            use_class_weights=use_class_weights,
        )
    elif sequence_groups:
        result = _train_model_with_groups(
            sequence_groups=sequence_groups,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            device=device,
            model_config=model_config,
            seed=seed,
            weight_decay=weight_decay,
            shuffle_targets_control=shuffle_targets_control,
            text_encoder=text_encoder,
            text_pool_lambda_inv_days=text_pool_lambda_inv_days,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            use_amp=use_amp,
            lr_schedule=lr_schedule_choice,
            use_class_weights=use_class_weights,
        )
    else:
        result = train_model(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path,
            save_checkpoint=save_checkpoint,
            device=device,
            model_config=model_config,
            seed=seed,
            weight_decay=weight_decay,
            shuffle_targets_control=shuffle_targets_control,
            text_encoder=text_encoder,
            text_pool_lambda_inv_days=text_pool_lambda_inv_days,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            use_amp=use_amp,
            lr_schedule=lr_schedule_choice,
            use_class_weights=use_class_weights,
        )
    return result.summary


def _train_model_with_groups(
    *,
    sequence_groups: Sequence[Sequence[FeatureVector]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    early_stopping_patience: int,
    checkpoint_path: Path,
    save_checkpoint: bool,
    device: torch.device,
    model_config: ModelConfig,
    seed: int | None,
    weight_decay: float = 1e-4,
    shuffle_targets_control: bool = False,
    text_encoder: str | None = None,
    text_pool_lambda_inv_days: float = 0.0,
    grad_clip_norm: float = 0.0,
    use_compile: bool = True,
    use_amp: bool = True,
    lr_schedule: str = "plateau",
    use_class_weights: bool = True,
) -> Any:
    """Invoke ``train_model`` against pre-loaded sequence groups.

    Routes through the ``sequence_groups`` kwarg on ``train_model`` so
    the legacy ``data_dir`` scan is bypassed entirely. Each inner list
    becomes one group consumed by ``_build_training_tensors`` -- the
    slicer treats each group independently, so prior bars from one
    FOMC event never leak into another event's training window.
    """

    materialised: list[list[FeatureVector]] = [list(group) for group in sequence_groups]
    return train_model(
        sequence_groups=materialised,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_fraction,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_checkpoint=save_checkpoint,
        device=device,
        model_config=model_config,
        seed=seed,
        weight_decay=weight_decay,
        shuffle_targets_control=shuffle_targets_control,
        text_encoder=text_encoder,
        text_pool_lambda_inv_days=text_pool_lambda_inv_days,
        grad_clip_norm=grad_clip_norm,
        use_compile=use_compile,
        use_amp=use_amp,
        lr_schedule=lr_schedule,
        use_class_weights=use_class_weights,
    )


def _worker_run_cell(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a single sweep cell inside a spawn-mode subprocess.

    The payload carries every argument the worker needs to recreate
    the call without sharing memory with the parent: a pickled
    ``ModelConfig``, the per-cell HP values, the per-fold
    ``WalkForwardSplit`` (or the legacy flat sequence-groups payload),
    and the trial-index metadata. The worker re-imports torch (the
    spawn context guarantees a fresh CUDA context) and routes through
    the same ``_run_single_training`` helper the sequential path uses,
    so the per-cell training code is shared across both schedulers.
    """

    summary = _run_single_training(
        data_dir=payload["data_dir"],
        checkpoint_path=payload["checkpoint_path"],
        device=torch.device(payload["device"]),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        validation_fraction=float(payload["validation_fraction"]),
        early_stopping_patience=int(payload["early_stopping_patience"]),
        model_config=payload["model_config"],
        save_checkpoint=False,
        seed=payload["seed"],
        sequence_groups=payload.get("sequence_groups"),
        walk_forward_split=payload.get("walk_forward_split"),
        weight_decay=float(payload["weight_decay"]),
        shuffle_targets_control=bool(payload["shuffle_targets_control"]),
        text_encoder=payload["text_encoder"],
        text_pool_lambda_inv_days=float(payload["text_pool_lambda_inv_days"]),
        grad_clip_norm=float(payload.get("grad_clip_norm", 0.0)),
        use_compile=bool(payload.get("use_compile", True)),
        use_amp=bool(payload.get("use_amp", True)),
        use_class_weights=bool(payload.get("use_class_weights", True)),
    )
    return {
        "trial_index": int(payload["trial_index"]),
        "architecture": str(payload["model_config"].architecture),
        "seed": payload["seed"],
        "hp_combo_id": payload.get("hp_combo_id"),
        "fold_id": payload.get("fold_id"),
        "summary": summary,
    }


def _build_worker_payload(
    *,
    candidate: dict[str, Any],
    trial_index: int,
    args: argparse.Namespace,
    data_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    sequence_groups: Sequence[Sequence[FeatureVector]] | None,
    walk_forward_split: WalkForwardSplit | None,
    text_encoder_arg: str | None,
    text_pool_lambda: float,
) -> dict[str, Any]:
    """Build a pickleable dict the worker process can consume."""

    return {
        "trial_index": int(trial_index),
        "model_config": candidate["model_config"],
        "learning_rate": candidate["learning_rate"],
        "epochs": candidate["epochs"],
        "weight_decay": candidate.get("weight_decay", args.weight_decay),
        "seed": candidate.get("seed"),
        "hp_combo_id": candidate.get("hp_combo_id"),
        "fold_id": candidate.get("fold_id"),
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "validation_fraction": float(args.validation_fraction),
        "early_stopping_patience": int(args.early_stopping_patience),
        "sequence_groups": sequence_groups,
        "walk_forward_split": walk_forward_split,
        "shuffle_targets_control": bool(args.shuffle_targets_control),
        "text_encoder": text_encoder_arg,
        "text_pool_lambda_inv_days": float(text_pool_lambda),
        "grad_clip_norm": float(getattr(args, "grad_clip_norm", 0.0)),
        "use_compile": bool(getattr(args, "use_compile", True)),
        "use_amp": bool(getattr(args, "use_amp", True)),
        "use_class_weights": bool(getattr(args, "use_class_weights", True)),
    }


def _format_cell_log(
    *,
    trial_index: int,
    total: int,
    candidate: dict[str, Any],
    summary: TrainingRunSummary,
) -> str:
    model_config = candidate["model_config"]
    metrics = summary.metrics
    metrics_label = (
        f"combined_rmse={metrics.combined_rmse:.6f}, loss={metrics.loss:.6f}"
        if metrics is not None
        else "no-metrics"
    )
    return (
        f"[trial {trial_index}/{total}] "
        f"arch={model_config.architecture}, seed={candidate.get('seed')}, "
        f"hidden={model_config.hidden_size}, layers={model_config.num_layers}, "
        f"dropout={model_config.dropout:.3f}, lr={candidate['learning_rate']:.6g}, "
        f"epochs={candidate['epochs']} -> {metrics_label}"
    )


def _sort_trial_records(trial_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort trials by (architecture, seed, hp_combo_id) deterministically.

    The sort key is independent of worker-scheduling order so the
    parallel and sequential paths emit byte-identical JSON / CSV at
    the same input. ``hp_combo_id`` is missing on the exhaustive path
    -- the fallback to ``trial_index`` preserves the legacy ordering
    in that case.
    """

    def _key(record: dict[str, Any]) -> tuple[str, int, int, int]:
        seed = record.get("seed")
        hp_combo_id = record.get("hp_combo_id")
        return (
            str(record.get("architecture") or ""),
            int(seed) if seed is not None else -1,
            int(hp_combo_id) if hp_combo_id is not None else -1,
            int(record.get("trial_index") or 0),
        )

    return sorted(trial_records, key=_key)


class _NullCudaStreamContext:
    """No-op stand-in for ``torch.cuda.stream`` on the CPU device path."""

    def __enter__(self) -> None:  # pragma: no cover -- trivial
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover -- trivial
        return None


def _run_sweep(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    device: torch.device,
    sequence_groups: Sequence[Sequence[FeatureVector]] | None = None,
    walk_forward_splits: dict[str, WalkForwardSplit] | None = None,
    training_package_id: str | None = None,
) -> int:
    candidates = build_sweep_candidates(args)
    if not candidates:
        print("No sweep candidates generated.")
        return 1

    parallel_workers = max(1, int(getattr(args, "parallel_workers", 1)))
    batching_mode = str(getattr(args, "batching_mode", "off") or "off")
    if batching_mode not in BATCHING_MODES:
        raise ValueError(
            f"--batching-mode={batching_mode!r} is not one of {BATCHING_MODES}"
        )
    max_bucket_size_override = getattr(args, "max_bucket_size", None)
    # When the bucketed runner is active the legacy
    # ProcessPoolExecutor path is bypassed; parallel_workers becomes
    # an inert knob so a user who passes both --batching-mode=auto
    # and --parallel-workers=8 does not end up with 8 processes each
    # spawning their own buckets. The legacy path stays selectable
    # via --batching-mode=off, which is the regression-test contract.
    if batching_mode == "off" and parallel_workers > PARALLEL_WORKERS_VRAM_WARN_THRESHOLD:
        _LOGGER.warning(
            "parallel_workers=%d exceeds the RTX 4080 VRAM-saturation threshold "
            "(%d). The CUDA allocator may OOM on the larger architectures "
            "(transformer hidden=128, layers=3). Recommend N<=%d.",
            parallel_workers,
            PARALLEL_WORKERS_VRAM_WARN_THRESHOLD,
            PARALLEL_WORKERS_VRAM_WARN_THRESHOLD,
        )

    use_random_search = bool(getattr(args, "random_search", False))
    if use_random_search:
        print(
            "Random-search HP subset: "
            f"M={getattr(args, 'random_search_samples', DEFAULT_RANDOM_SEARCH_SAMPLES)}, "
            f"seed={getattr(args, 'random_search_seed', DEFAULT_RANDOM_SEARCH_SEED)}"
        )
    print(
        f"Starting hyperparameter sweep with {len(candidates)} trial(s) "
        f"(parallel_workers={parallel_workers}, batching_mode={batching_mode})..."
    )
    trial_records: list[dict[str, Any]] = []
    summaries: list[TrainingRunSummary] = []
    text_encoder_arg = (
        None if str(getattr(args, "text_encoder", "none")) == "none" else str(args.text_encoder)
    )
    text_pool_lambda = float(getattr(args, "text_pool_lambda_inv_days", 0.0))

    if batching_mode != "off":
        # Bucketed-HP path: group cells with the same model topology +
        # data feed and dispatch each bucket as one concurrent unit.
        # The streams variant overlaps kernel launches across CUDA
        # streams inside one process / one CUDA context; the stacked
        # variant routes to streams as a transparent fallback on
        # architectures that the per-arch table does not yet flag as
        # vmap-friendly.
        target_mode = str(getattr(args, "target_mode", "event_study") or "event_study")
        buckets = group_candidates_into_buckets(
            candidates,
            text_encoder=text_encoder_arg,
            target_mode=target_mode,
            max_bucket_size=max_bucket_size_override,
        )

        def _split_for_candidate(c: dict[str, Any]) -> WalkForwardSplit | None:
            if walk_forward_splits is None:
                return None
            cand_fold = c.get("fold_id") or next(iter(walk_forward_splits))
            return walk_forward_splits.get(cand_fold)

        def _train_one_cell_for_bucket(
            trial_index: int,
            candidate: dict[str, Any],
            stream: "torch.cuda.Stream | None",
        ) -> dict[str, Any]:
            # The streams runner shares the parent CUDA context, so
            # the per-cell training reuses the same _run_single_training
            # entry point the sequential path uses. The stream context
            # manager pipelines the cell's kernels behind the bucket's
            # other cells; on CPU the stream is None and the cell runs
            # inline.
            model_config = candidate["model_config"]
            learning_rate = candidate["learning_rate"]
            epochs = candidate["epochs"]
            weight_decay = candidate.get("weight_decay", args.weight_decay)
            seed = candidate.get("seed")
            fold_id = candidate.get("fold_id")
            wf_split: WalkForwardSplit | None = None
            cell_sequence_groups: Sequence[Sequence[FeatureVector]] | None = sequence_groups
            if walk_forward_splits is not None:
                wf_split = _split_for_candidate(candidate)
                cell_sequence_groups = None
            ctx = (
                torch.cuda.stream(stream)
                if stream is not None
                else _NullCudaStreamContext()
            )
            with ctx:
                summary = _run_single_training(
                    data_dir=data_dir,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    validation_fraction=args.validation_fraction,
                    early_stopping_patience=args.early_stopping_patience,
                    model_config=model_config,
                    save_checkpoint=False,
                    seed=seed,
                    sequence_groups=cell_sequence_groups,
                    walk_forward_split=wf_split,
                    weight_decay=float(weight_decay),
                    shuffle_targets_control=bool(args.shuffle_targets_control),
                    text_encoder=text_encoder_arg,
                    text_pool_lambda_inv_days=text_pool_lambda,
                    grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
                    use_compile=bool(getattr(args, "use_compile", True)),
                    use_amp=bool(getattr(args, "use_amp", True)),
                    use_class_weights=bool(getattr(args, "use_class_weights", True)),
                )
            record: dict[str, Any] = {
                "trial_index": int(trial_index),
                "architecture": str(model_config.architecture),
                "seed": seed,
                "summary": summary,
                "candidate": candidate,
            }
            if "hp_combo_id" in candidate:
                record["hp_combo_id"] = candidate["hp_combo_id"]
            if fold_id is not None:
                record["fold_id"] = fold_id
            return record

        for bucket_key, bucket_cells in buckets:
            routed = route_bucket(bucket_key.architecture, mode=batching_mode)
            print(
                format_bucket_log_line(
                    bucket_key, bucket_cells, routed_mode=routed
                )
            )
            # ``stacked`` and ``streams`` share the same per-cell entry
            # point for now -- the stacked path inside StackedDLinear
            # is reserved for the dlinear forward audit in a follow-up;
            # the current PR ships the streams scheduler as the
            # workhorse, with stacked-mode primitives in place for the
            # next promotion step.
            cell_results = run_bucket_streams(
                bucket_cells,
                train_one_cell=_train_one_cell_for_bucket,
                device=device,
            )
            for result in cell_results:
                summary_obj: TrainingRunSummary = result["summary"]
                record = {
                    "trial_index": result["trial_index"],
                    "architecture": result["architecture"],
                    "seed": result["seed"],
                    "summary": summary_obj.to_dict(),
                }
                if "hp_combo_id" in result:
                    record["hp_combo_id"] = result["hp_combo_id"]
                if "fold_id" in result:
                    record["fold_id"] = result["fold_id"]
                trial_records.append(record)
                summaries.append(summary_obj)
                print(
                    _format_cell_log(
                        trial_index=result["trial_index"],
                        total=len(candidates),
                        candidate=result["candidate"],
                        summary=summary_obj,
                    )
                )

        # Deterministic ordering across bucket boundaries. The streams
        # runner returns cells in their submission order, but a
        # downstream consumer should see (architecture, seed, hp_combo_id,
        # trial_index) ordering regardless of bucket layout.
        index_to_summary = {
            record["trial_index"]: summary
            for record, summary in zip(trial_records, summaries, strict=True)
        }
        trial_records = _sort_trial_records(trial_records)
        summaries = [index_to_summary[record["trial_index"]] for record in trial_records]
    elif parallel_workers > 1:
        # ProcessPoolExecutor with spawn context: each worker re-imports
        # torch and acquires its own CUDA context. Results come back in
        # completion order so the trial_records list is sorted
        # deterministically at the end of the loop.
        spawn_ctx = multiprocessing.get_context("spawn")
        index_to_candidate = dict(enumerate(candidates, start=1))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=parallel_workers,
            mp_context=spawn_ctx,
        ) as pool:
            def _split_for_candidate(c: dict[str, Any]) -> WalkForwardSplit | None:
                if walk_forward_splits is None:
                    return None
                cand_fold = c.get("fold_id") or next(iter(walk_forward_splits))
                return walk_forward_splits.get(cand_fold)

            future_to_index = {
                pool.submit(
                    _worker_run_cell,
                    _build_worker_payload(
                        candidate=candidate,
                        trial_index=index,
                        args=args,
                        data_dir=data_dir,
                        checkpoint_path=checkpoint_path,
                        device=device,
                        sequence_groups=None if walk_forward_splits is not None else sequence_groups,
                        walk_forward_split=_split_for_candidate(candidate),
                        text_encoder_arg=text_encoder_arg,
                        text_pool_lambda=text_pool_lambda,
                    ),
                ): index
                for index, candidate in index_to_candidate.items()
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                candidate = index_to_candidate[index]
                result = future.result()
                summary = result["summary"]
                record = {
                    "trial_index": index,
                    "architecture": result["architecture"],
                    "seed": result["seed"],
                    "summary": summary.to_dict(),
                }
                if result.get("hp_combo_id") is not None:
                    record["hp_combo_id"] = result["hp_combo_id"]
                if result.get("fold_id") is not None:
                    record["fold_id"] = result["fold_id"]
                trial_records.append(record)
                summaries.append(summary)
                print(
                    _format_cell_log(
                        trial_index=index,
                        total=len(candidates),
                        candidate=candidate,
                        summary=summary,
                    )
                )
    else:
        for index, candidate in enumerate(candidates, start=1):
            model_config = candidate["model_config"]
            learning_rate = candidate["learning_rate"]
            epochs = candidate["epochs"]
            weight_decay = candidate.get("weight_decay", args.weight_decay)
            seed = candidate.get("seed")
            fold_id = candidate.get("fold_id")
            wf_split: WalkForwardSplit | None = None
            cell_sequence_groups: Sequence[Sequence[FeatureVector]] | None = sequence_groups
            if walk_forward_splits is not None:
                # On the walk-forward path the per-fold split carries
                # train/val/test partitions; the trainer skips the
                # legacy single-list code path entirely.
                split_key = fold_id if fold_id is not None else next(iter(walk_forward_splits))
                wf_split = walk_forward_splits[split_key]
                cell_sequence_groups = None
            summary = _run_single_training(
                data_dir=data_dir,
                checkpoint_path=checkpoint_path,
                device=device,
                epochs=epochs,
                batch_size=args.batch_size,
                learning_rate=learning_rate,
                validation_fraction=args.validation_fraction,
                early_stopping_patience=args.early_stopping_patience,
                model_config=model_config,
                save_checkpoint=False,
                seed=seed,
                sequence_groups=cell_sequence_groups,
                walk_forward_split=wf_split,
                weight_decay=float(weight_decay),
                shuffle_targets_control=bool(args.shuffle_targets_control),
                text_encoder=text_encoder_arg,
                text_pool_lambda_inv_days=text_pool_lambda,
                grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
                use_compile=bool(getattr(args, "use_compile", True)),
                use_amp=bool(getattr(args, "use_amp", True)),
                use_class_weights=bool(getattr(args, "use_class_weights", True)),
            )
            summaries.append(summary)
            record = {
                "trial_index": index,
                "architecture": model_config.architecture,
                "seed": seed,
                "summary": summary.to_dict(),
            }
            if "hp_combo_id" in candidate:
                record["hp_combo_id"] = candidate["hp_combo_id"]
            if fold_id is not None:
                record["fold_id"] = fold_id
            trial_records.append(record)
            print(
                _format_cell_log(
                    trial_index=index,
                    total=len(candidates),
                    candidate=candidate,
                    summary=summary,
                )
            )

    # Deterministic ordering. The parallel branch collects in
    # completion order; sort by (architecture, seed, hp_combo_id,
    # trial_index) so the JSON / CSV emitter is independent of worker
    # scheduling. ``summaries`` is re-aligned to the post-sort
    # trial_records so the best-selection downstream picks the right
    # record. The sequential branch is already in candidate order;
    # trial_index then dominates the tiebreak and the sort is a
    # stable no-op against the legacy exhaustive layout.
    if parallel_workers > 1:
        index_to_summary = {
            record["trial_index"]: summary
            for record, summary in zip(trial_records, summaries, strict=True)
        }
        trial_records = _sort_trial_records(trial_records)
        summaries = [index_to_summary[record["trial_index"]] for record in trial_records]

    best_summary = select_best_summary(summaries)
    if best_summary is None or best_summary.metrics is None:
        print("Sweep completed, but no valid validation metrics were produced.")
        return 1

    # ``best_summary_position`` is the 1-based slot in the
    # post-sort summaries list; ``best_record["trial_index"]`` carries
    # the original candidate's 1-based index, which is the key into
    # the ``candidates`` list regardless of completion order.
    best_summary_position = next(
        index
        for index, summary in enumerate(summaries, start=1)
        if summary == best_summary
    )
    best_record = trial_records[best_summary_position - 1]
    best_trial_index = int(best_record["trial_index"])
    best_model_config = best_summary.model_config
    best_candidate = candidates[best_trial_index - 1]
    best_seed = best_candidate.get("seed")
    print(
        "Re-training best configuration for final checkpoint: "
        f"arch={best_model_config.architecture}, seed={best_seed}, "
        f"hidden={best_model_config.hidden_size}, layers={best_model_config.num_layers}, "
        f"dropout={best_model_config.dropout:.3f}, lr={best_summary.learning_rate:.6g}, "
        f"epochs={best_summary.epochs_requested}"
    )
    best_weight_decay = best_candidate.get("weight_decay", args.weight_decay)
    best_fold_id = best_candidate.get("fold_id")
    best_wf_split: WalkForwardSplit | None = None
    final_sequence_groups: Sequence[Sequence[FeatureVector]] | None = sequence_groups
    if walk_forward_splits is not None:
        key = best_fold_id if best_fold_id is not None else next(iter(walk_forward_splits))
        best_wf_split = walk_forward_splits[key]
        final_sequence_groups = None
    final_summary = _run_single_training(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=best_summary.epochs_requested,
        batch_size=best_summary.batch_size,
        learning_rate=best_summary.learning_rate,
        # ``best_summary.validation_split`` is the persisted summary
        # field (frozen for back-compat with the existing
        # TrainingRunSummary dataclass); the kwarg routes under the
        # current ``validation_fraction`` name.
        validation_fraction=best_summary.validation_split,
        early_stopping_patience=best_summary.early_stopping_patience,
        model_config=best_model_config,
        save_checkpoint=True,
        seed=best_seed,
        sequence_groups=final_sequence_groups,
        walk_forward_split=best_wf_split,
        weight_decay=float(best_weight_decay),
        shuffle_targets_control=bool(args.shuffle_targets_control),
        text_encoder=text_encoder_arg,
        text_pool_lambda_inv_days=text_pool_lambda,
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
        use_compile=bool(getattr(args, "use_compile", True)),
        use_amp=bool(getattr(args, "use_amp", True)),
        use_class_weights=bool(getattr(args, "use_class_weights", True)),
    )
    for trial in trial_records:
        trial["selected"] = trial["trial_index"] == best_trial_index

    report_payload = {
        "mode": "sweep",
        "selection_metric": "combined_rmse",
        "device": str(device),
        "data_dir": str(data_dir),
        "training_package_id": training_package_id,
        "checkpoint_path": str(checkpoint_path),
        "credibility_features": bool(args.credibility_features),
        "rich_features": bool(args.rich_features) and bool(training_package_id),
        "rich_feature_families": {
            "credibility": bool(args.use_credibility),
            "linguistic": bool(args.use_linguistic),
            "mp_surprise": bool(args.use_mp_surprise),
            "multi_axis": bool(args.use_multi_axis),
            "llm_features": bool(args.use_llm_features),
        },
        "text_embeddings": {
            "encoder": text_encoder_arg,
            "use_text_embeddings": bool(args.use_text_embeddings),
            "pool_lambda_inv_days": text_pool_lambda,
        },
        "shuffle_targets_control": bool(args.shuffle_targets_control),
        "protocol": (
            "walk-forward" if walk_forward_splits is not None else "single-fold"
        ),
        "folds": sorted({str(trial.get("fold_id")) for trial in trial_records if trial.get("fold_id")}),
        "architectures": sorted({trial["architecture"] for trial in trial_records}),
        "seeds": sorted({trial["seed"] for trial in trial_records if trial["seed"] is not None}),
        "trial_count": len(trial_records),
        "best_trial_index": best_trial_index,
        "best_trial": best_record,
        "selected_checkpoint": final_summary.to_dict(),
        "trials": trial_records,
    }
    if use_random_search:
        report_payload["random_search"] = {
            "samples": int(getattr(args, "random_search_samples", DEFAULT_RANDOM_SEARCH_SAMPLES)),
            "seed": int(getattr(args, "random_search_seed", DEFAULT_RANDOM_SEARCH_SEED)),
        }
        report_payload["parallel_workers"] = parallel_workers
    elif parallel_workers > 1:
        report_payload["parallel_workers"] = parallel_workers
    # ``batching_mode`` is omitted from the payload on the legacy
    # ``off`` path so the byte-identity regression contract on the
    # sweep-report JSON stays green; any non-off mode emits the field
    # so downstream consumers can confirm the bucketed runner ran.
    if batching_mode != "off":
        report_payload["batching_mode"] = batching_mode
        if max_bucket_size_override is not None:
            report_payload["max_bucket_size"] = int(max_bucket_size_override)
    _write_sweep_report(report_path, report_payload)
    print(
        "Sweep complete. "
        f"Best combined RMSE={best_summary.metrics.combined_rmse:.6f}. "
        f"Final checkpoint saved to {checkpoint_path}"
    )
    print(f"Sweep report written to {report_path} and {report_path.with_suffix('.csv')}")
    return 0


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    report_path = Path(args.report_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ``--training-package-id`` takes precedence over ``--data-dir`` so an
    # exported DATA_DIR override never silently falls back to the legacy
    # JSON/JSONL scan. The override path is logged so precedence is
    # auditable from the run log.
    use_package_path = bool(args.training_package_id)
    if use_package_path and args.data_dir != str(DEFAULT_DATA_DIR):
        print(
            f"--training-package-id is set; ignoring --data-dir={args.data_dir} "
            "in favour of the Phase 8 events.parquet path."
        )

    package_sequences: list[list[FeatureVector]] | None = None
    walk_forward_splits: dict[str, WalkForwardSplit] | None = None
    if use_package_path:
        print(f"Training-package id: {args.training_package_id}")
        print(f"Target mode: {args.target_mode}")
        print(
            f"Rich features: {'on' if args.rich_features else 'off'} "
            f"(credibility={args.use_credibility}, linguistic={args.use_linguistic}, "
            f"mp_surprise={args.use_mp_surprise}, multi_axis={args.use_multi_axis}, "
            f"llm_features={args.use_llm_features})"
        )
        loader_text_encoder = (
            None if str(args.text_encoder) == "none" else str(args.text_encoder)
        )

        protocol_choice = str(getattr(args, "protocol", "auto") or "auto")
        cli_folds: list[str] = [str(f).strip() for f in (args.folds or []) if str(f).strip()]
        if protocol_choice == "auto":
            resolved_protocol = "walk-forward" if cli_folds else "single-fold"
        else:
            resolved_protocol = protocol_choice
        if resolved_protocol == "walk-forward" and not cli_folds:
            raise SystemExit(
                "--protocol walk-forward requires --folds <fold_id> [<fold_id> ...]; "
                "got an empty fold list"
            )
        if str(getattr(args, "validation_fraction", 0.0)) and (cli_folds or resolved_protocol != "single-fold"):
            # The walk-forward and single-fold paths both honour the
            # package's pre-built partitions; the random/chronological
            # validation_fraction knob is inert on those paths.
            _LOGGER.info(
                "--validation-fraction=%s ignored on protocol=%s; the "
                "training partition is honoured from the package.",
                args.validation_fraction,
                resolved_protocol,
            )

        print(f"Protocol: {resolved_protocol}")

        if resolved_protocol == "walk-forward":
            walk_forward_splits = {}
            for fold_id in cli_folds:
                split = load_walk_forward_split(
                    args.training_package_id,
                    fold_id=fold_id,
                    target_mode=args.target_mode,
                    rich_features=bool(args.rich_features),
                    use_credibility=bool(args.use_credibility),
                    use_linguistic=bool(args.use_linguistic),
                    use_mp_surprise=bool(args.use_mp_surprise),
                    use_multi_axis=bool(args.use_multi_axis),
                    use_llm_features=bool(args.use_llm_features),
                    text_encoder=loader_text_encoder,
                    text_adapter_dim=int(args.text_adapter_dim),
                    text_pool_lambda_inv_days=float(args.text_pool_lambda_inv_days),
                    use_text_embeddings=bool(args.use_text_embeddings),
                    embargo_days=int(args.embargo_days),
                )
                walk_forward_splits[fold_id] = split
                print(
                    f"  {fold_id}: train={len(split.train)} val={len(split.val)} "
                    f"test={len(split.test)}"
                )
            sequence_count = sum(
                len(s.train) + len(s.val) + len(s.test) for s in walk_forward_splits.values()
            )
            observation_count = sum(
                sum(len(seq) for seq in (s.train + s.val + s.test))
                for s in walk_forward_splits.values()
            )
            window_count = sum(
                sum(max(0, len(seq) - SEQUENCE_LENGTH) for seq in (s.train + s.val + s.test))
                for s in walk_forward_splits.values()
            )
        else:
            # Single-fold path: read the package's
            # splits_train_val_test.parquet partition. The trainer
            # honours the val and test lists as the early-stopping
            # signal and the held-out evaluation set.
            split = load_walk_forward_split(
                args.training_package_id,
                fold_id=None,
                target_mode=args.target_mode,
                rich_features=bool(args.rich_features),
                use_credibility=bool(args.use_credibility),
                use_linguistic=bool(args.use_linguistic),
                use_mp_surprise=bool(args.use_mp_surprise),
                use_multi_axis=bool(args.use_multi_axis),
                use_llm_features=bool(args.use_llm_features),
                text_encoder=loader_text_encoder,
                text_adapter_dim=int(args.text_adapter_dim),
                text_pool_lambda_inv_days=float(args.text_pool_lambda_inv_days),
                use_text_embeddings=bool(args.use_text_embeddings),
                # Single-fold uses split_tag, not manifest dates -- embargo
                # passes through but the loader will no-op on this path.
                embargo_days=int(args.embargo_days),
            )
            walk_forward_splits = {"_single_fold": split}
            print(
                f"  single-fold: train={len(split.train)} val={len(split.val)} "
                f"test={len(split.test)}"
            )
            sequence_count = len(split.train) + len(split.val) + len(split.test)
            observation_count = sum(
                len(seq) for seq in (split.train + split.val + split.test)
            )
            window_count = sum(
                max(0, len(seq) - SEQUENCE_LENGTH)
                for seq in (split.train + split.val + split.test)
            )
        print(f"Device: {device}")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Existing checkpoint: {'yes' if checkpoint_path.exists() else 'no'}")
        print(f"Sequence groups discovered: {sequence_count}")
        print(f"Observations discovered: {observation_count}")
        print(f"Training windows available: {window_count}")
        if args.list_data:
            return 0 if sequence_count else 1
        if not sequence_count or not window_count:
            print(
                "No sufficient training data found in training package "
                f"{args.training_package_id}. Verify events.parquet carries "
                "prior_bars_json with the full 20-bar prior window."
            )
            return 1
    else:
        sequences, summaries = inspect_training_data_sources(data_dir)
        sequence_count = len(sequences)
        observation_count = sum(len(sequence) for sequence in sequences)
        window_count = sum(max(0, len(sequence) - SEQUENCE_LENGTH) for sequence in sequences)

        _print_data_inventory(
            data_dir,
            checkpoint_path,
            device,
            summaries,
            sequence_count=sequence_count,
            observation_count=observation_count,
            window_count=window_count,
        )

        if args.list_data:
            return 0 if summaries else 1

        if not sequence_count or not window_count:
            print(
                "No sufficient training data found. Add prepared market series files under the data directory "
                "with fields like date, close, volatility_5d, and optional sentiment_score."
            )
            return 1

    sweep_mode = args.sweep or any(
        option is not None
        for option in (
            args.hidden_sizes,
            args.num_layers_grid,
            args.dropouts,
            args.learning_rates,
            args.epochs_grid,
            args.architectures,
            args.seeds,
            args.weight_decays,
            args.text_adapter_dims,
            args.folds,
        )
    )
    if sweep_mode:
        return _run_sweep(
            args=args,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            device=device,
            sequence_groups=package_sequences,
            walk_forward_splits=walk_forward_splits,
            training_package_id=args.training_package_id,
        )

    print(
        "Starting professional forecaster training "
        f"(architecture={args.architecture}, credibility_features={bool(args.credibility_features)})..."
    )
    single_run_split: WalkForwardSplit | None = None
    single_run_sequence_groups: Sequence[Sequence[FeatureVector]] | None = package_sequences
    if walk_forward_splits is not None:
        # On the package path the single-run also honours the
        # walk-forward partition the loader resolved; the legacy
        # ``--data-dir`` path keeps the flat sequence list.
        single_run_split = next(iter(walk_forward_splits.values()))
        single_run_sequence_groups = None
    summary = _run_single_training(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        model_config=_build_model_config(args),
        save_checkpoint=True,
        seed=args.seed,
        sequence_groups=single_run_sequence_groups,
        walk_forward_split=single_run_split,
        weight_decay=float(args.weight_decay),
        shuffle_targets_control=bool(args.shuffle_targets_control),
        text_encoder=None if str(args.text_encoder) == "none" else str(args.text_encoder),
        text_pool_lambda_inv_days=float(args.text_pool_lambda_inv_days),
        grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
        use_compile=bool(getattr(args, "use_compile", True)),
        use_amp=bool(getattr(args, "use_amp", True)),
        use_class_weights=bool(getattr(args, "use_class_weights", True)),
    )
    metrics = summary.metrics
    if metrics is not None:
        print(
            "Validation metrics: "
            f"loss={metrics.loss:.6f}, combined_rmse={metrics.combined_rmse:.6f}, "
            f"close_rmse={metrics.close_rmse:.6f}, volatility_rmse={metrics.volatility_rmse:.6f}"
        )
    print(f"Training complete. Best checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
