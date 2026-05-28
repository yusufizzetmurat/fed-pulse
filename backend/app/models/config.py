from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import DATA_DIR, MODEL_CHECKPOINT_DIR

# v2 reference: 20-day lookback over daily bars. v1 used 5 (sub-week)
# which was too short for the recurrent core to learn temporal structure.
# Tests/regression callers reference this constant; on-disk checkpoints
# persist their training-time value via training/checkpoint.py.
SEQUENCE_LENGTH = 20
FEATURE_SIZE = 6  # [sentiment_score, market_close, market_volatility, close_change_pct, volatility_change, elapsed_time]
SENTIMENT_FEATURE_INDEX = 0
ELAPSED_TIME_FEATURE_INDEX = 5

# Rich-feature input space (PR-#173 onward). The training-package loader
# joins the four feature families produced under Phase 8 onto the
# per-bar feature vector:
#
# - 4 credibility fields (drift_score, realized_vs_stated_gap,
#   market_implied_gap, months_since_reversal) -- direct off the
#   events.parquet row.
# - 15 linguistic features (8 LDA topic shares + 6 hand-crafted
#   densities + pivot_distance) -- joined on text_hash from
#   linguistic_features.parquet.
# - 4 MP-surprise fields (mp_surprise_level, mp_surprise_path_factor,
#   fed_info_factor, mp_is_intermeeting) -- joined on event_date from
#   mp_surprises.parquet. ``mp_is_intermeeting`` is the boolean
#   ``is_intermeeting`` field encoded as 0.0 / 1.0.
# - 6 multi-axis fields (axis_factor, axis_certainty, axis_time, each
#   with a paired *_missing flag) -- direct off the events.parquet row.
#
# The event-level signal is broadcast to every bar of the 20-day prior
# window plus the appended event-day target frame, so every bar in a
# supervised window carries the same rich-feature row.
#
# Per-bar slice ordering (deterministic; documented on
# ``FeatureVector`` below):
#
#   [0:6]    market features (existing FEATURE_SIZE slice)
#   [6:10]   credibility 4-vector
#   [10:25]  linguistic 15-vector
#   [25:29]  MP-surprise 4-vector
#   [29:35]  multi-axis 6-vector (3 values + 3 missing flags)
#
# ``RICH_FEATURE_SIZE`` is the constant downstream model factories /
# CLI use to widen the input projection when ``rich_features=True``.
RICH_CREDIBILITY_DIM = 4
RICH_LINGUISTIC_DIM = 15
RICH_MP_SURPRISE_DIM = 4
RICH_MULTI_AXIS_DIM = 6
# A2 (#207): additional realised-vol horizons (20d, 60d). The existing
# market slice already carries vol_5d as ``market_volatility``; this
# extra slice extends the vol-autocorrelation surface so the model can
# anchor against longer realised-vol windows, which the literature
# consistently identifies as the strongest single predictor of forward
# vol regime.
RICH_REALIZED_VOL_DIM = 2
# A3 (#208): cross-asset daily close levels (VIX, USD index, 10Y yield,
# gold). Each is a single float per bar. VIX is the market's own
# forward-vol forecast; the others capture risk-on/risk-off (DXY),
# the rates-pricing layer (TNX), and flight-to-safety (gold).
# Path B Chunk 1 (#239 follow-up): widened from 4 (VIX, DXY, TNX, gold)
# to 8 by adding the 3M VIX term-structure series, the 13-week T-bill
# yield, plus two derived stationary slopes (VIX term slope, 10Y-3M
# yield-curve slope). These are the macro features the vol-forecasting
# literature consistently relies on; the raw levels alone were not
# sufficient. Bumps RICH_FEATURE_SIZE by 4 (the four new dims); old
# checkpoints carrying the pre-widen size become incompatible and the
# next sweep refits both the model and the per-fold RobustScaler.
RICH_CROSS_ASSET_DIM = 8
# B1 (#212) LLM-as-features one-hot block. 10 catalogue features with
# {4, 4, 5, 4, 4, 4, 2, 2, 3, 3} levels = 35 one-hot dimensions plus
# one ``llm_features_missing`` flag for events where the extraction
# failed or the document was too short to assess.
RICH_LLM_FEATURE_DIM = 35
RICH_LLM_FEATURE_MISSING_DIM = 1
# #306 retrieval-augmented features. Per-event summary stats over the
# top-K analog hits from the on-disk retrieval index (#294). 5 contextual
# scalars + 1 missing flag. The scalars are contextual (similarity
# scores + stance agreement, both of which are T-snapshot text-level
# signals) — the analog's post-event vol-regime is NOT a feature here
# because admitting it would be a label leak via similarity. See ADR
# 0028 and the per-feature row in docs/feature-provenance-audit.md.
RICH_RETRIEVAL_ANALOG_DIM = 5
RICH_RETRIEVAL_ANALOG_MISSING_DIM = 1
# #307 macro-regime conditioning block. Three signed scalars in {-1, 0, +1}
# (policy-cycle phase, VIX-level regime, term-spread sign) plus a paired
# missing flag. Opt-in via ``--use-regime-conditioning``; the block is
# appended past ``RICH_FEATURE_SIZE`` by ``as_rich_list`` only when the
# loader populates ``macro_regime_features``, so the legacy default path
# keeps the byte-identical pre-#307 per-bar feature size. See ADR 0029.
RICH_MACRO_REGIME_DIM = 3
RICH_MACRO_REGIME_MISSING_DIM = 1
# #215 Summary of Economic Projections (SEP) dot-plot block. Five scalars
# (current-year / next-year / longer-run median FFR projections + the
# current-year central-tendency range + a release flag) plus a paired
# missing flag. Opt-in via ``--use-sep``; the block is appended past
# ``RICH_FEATURE_SIZE`` (and past the regime block when both are on) by
# ``as_rich_list`` only when the loader populates ``sep_features``, so
# the legacy default path keeps the byte-identical pre-#215 per-bar
# feature size. See ADR 0030.
RICH_SEP_DIM = 4
RICH_SEP_MISSING_DIM = 1
<<<<<<< HEAD
# #214 FOMC press conference Q&A block. One scalar carries the
# ``has_press_conf`` covariate-shift flag (1.0 on FOMC events whose Q&A
# transcript landed in the press-conf lookup, 0.0 on pre-2011 events and
# every other event_kind where the joint corpus does not apply). The flag
# IS the missingness signal — zero-imputation is the canonical handling
# of the pre-2011 era under route 1 of #214 (a separate fold split was
# rejected because it would fragment the walk-forward protocol). The
# block sits past the SEP tail in ``as_rich_list`` and is appended only
# when the loader populates ``press_conf_features`` under
# ``--use-press-conf``, so the default flag-off path stays byte-identical
# to pre-#214. See ADR 0037.
RICH_PRESS_CONF_DIM = 1
=======
# #443 statement-delta (redline) mean-pooled embedding block. Width
# matches the encoder hidden size on the canonical FinBERT-Fed-Adjacent
# checkpoint (768). Opt-in via ``--use-statement-delta``; the block is
# appended past the SEP tail by ``as_rich_list`` only when the loader
# populates the slot, so the default per-bar feature size stays
# byte-identical to pre-#443. See ADR 0038.
RICH_STATEMENT_DELTA_DIM = 768
RICH_STATEMENT_DELTA_MISSING_DIM = 1
# #444 vote tally + dissent block. Four scalars per event (votes_for /
# votes_against / is_unanimous / signed dissent_direction). The
# dissent_direction column comes off events.parquet as a string
# ("hawkish_dissent" / "dovish_dissent" / None) and the loader maps it
# to +1.0 / -1.0 / 0.0 so the model consumes a signed scalar in the
# same band as ``mp_surprise_level``. Opt-in via ``--use-vote-features``;
# the block is appended past the delta tail by ``as_rich_list`` only
# when the loader populates the slot, so the default per-bar feature
# size stays byte-identical to pre-#444. See ADR 0038.
RICH_VOTE_FEATURES_DIM = 4
RICH_VOTE_FEATURES_MISSING_DIM = 1
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
RICH_EXTRA_FEATURE_SIZE = (
    RICH_CREDIBILITY_DIM
    + RICH_LINGUISTIC_DIM
    + RICH_MP_SURPRISE_DIM
    + RICH_MULTI_AXIS_DIM
    + RICH_REALIZED_VOL_DIM
    + RICH_CROSS_ASSET_DIM
    + RICH_LLM_FEATURE_DIM
    + RICH_LLM_FEATURE_MISSING_DIM
    + RICH_RETRIEVAL_ANALOG_DIM
    + RICH_RETRIEVAL_ANALOG_MISSING_DIM
)
RICH_FEATURE_SIZE = FEATURE_SIZE + RICH_EXTRA_FEATURE_SIZE

# Slice offsets inside the rich vector. Used by the per-family
# ablation path on the loader to zero an individual family without
# changing the per-bar feature size; a downstream sweep can then
# measure per-family lift while keeping the model input shape
# constant.
RICH_MARKET_SLICE = slice(0, FEATURE_SIZE)
RICH_CREDIBILITY_SLICE = slice(
    FEATURE_SIZE, FEATURE_SIZE + RICH_CREDIBILITY_DIM
)
RICH_LINGUISTIC_SLICE = slice(
    RICH_CREDIBILITY_SLICE.stop,
    RICH_CREDIBILITY_SLICE.stop + RICH_LINGUISTIC_DIM,
)
RICH_MP_SURPRISE_SLICE = slice(
    RICH_LINGUISTIC_SLICE.stop,
    RICH_LINGUISTIC_SLICE.stop + RICH_MP_SURPRISE_DIM,
)
RICH_MULTI_AXIS_SLICE = slice(
    RICH_MP_SURPRISE_SLICE.stop,
    RICH_MP_SURPRISE_SLICE.stop + RICH_MULTI_AXIS_DIM,
)
# A2 (#207) realised-vol horizons slice (positions [35:37]).
RICH_REALIZED_VOL_SLICE = slice(
    RICH_MULTI_AXIS_SLICE.stop,
    RICH_MULTI_AXIS_SLICE.stop + RICH_REALIZED_VOL_DIM,
)
# A3 (#208) cross-asset slice (positions [37:45] after Path B Chunk 1:
# VIX, DXY, TNX, gold, VIX3M, IRX, vix_term_slope, yield_curve_slope_10y_3m).
RICH_CROSS_ASSET_SLICE = slice(
    RICH_REALIZED_VOL_SLICE.stop,
    RICH_REALIZED_VOL_SLICE.stop + RICH_CROSS_ASSET_DIM,
)
# B1 (#212) LLM-as-features slice (positions [45:80] one-hot + 80 flag
# after Path B Chunk 1 widened cross-asset by 4).
RICH_LLM_FEATURE_SLICE = slice(
    RICH_CROSS_ASSET_SLICE.stop,
    RICH_CROSS_ASSET_SLICE.stop + RICH_LLM_FEATURE_DIM,
)
RICH_LLM_FEATURE_MISSING_SLICE = slice(
    RICH_LLM_FEATURE_SLICE.stop,
    RICH_LLM_FEATURE_SLICE.stop + RICH_LLM_FEATURE_MISSING_DIM,
)
# #306 retrieval-augmented analog summary block. 5 contextual scalars
# + 1 missing flag. Position appended after the LLM-features block so
# the pre-#306 slice offsets are byte-identical.
RICH_RETRIEVAL_ANALOG_SLICE = slice(
    RICH_LLM_FEATURE_MISSING_SLICE.stop,
    RICH_LLM_FEATURE_MISSING_SLICE.stop + RICH_RETRIEVAL_ANALOG_DIM,
)
RICH_RETRIEVAL_ANALOG_MISSING_SLICE = slice(
    RICH_RETRIEVAL_ANALOG_SLICE.stop,
    RICH_RETRIEVAL_ANALOG_SLICE.stop + RICH_RETRIEVAL_ANALOG_MISSING_DIM,
)
# #307 macro-regime block. Sits past ``RICH_FEATURE_SIZE`` and is only
# emitted when ``FeatureVector.macro_regime_features`` is populated;
# the default per-bar feature size therefore stays at the legacy
# ``RICH_FEATURE_SIZE`` width and a downstream caller iterating slices
# inside ``[0:RICH_FEATURE_SIZE]`` never sees the new block.
RICH_MACRO_REGIME_SLICE = slice(
    RICH_RETRIEVAL_ANALOG_MISSING_SLICE.stop,
    RICH_RETRIEVAL_ANALOG_MISSING_SLICE.stop + RICH_MACRO_REGIME_DIM,
)
RICH_MACRO_REGIME_MISSING_SLICE = slice(
    RICH_MACRO_REGIME_SLICE.stop,
    RICH_MACRO_REGIME_SLICE.stop + RICH_MACRO_REGIME_MISSING_DIM,
)
# #215 SEP block slice positions. The constants describe where the SEP
# block lands when BOTH the regime and SEP blocks are populated -- i.e.
# past the regime tail in the both-on path. ``as_rich_list`` keeps the
# documented order: market | rich | regime? | sep?. When only SEP is
# populated (regime off), the block sits at
# ``[RICH_FEATURE_SIZE : RICH_FEATURE_SIZE + RICH_SEP_DIM]`` instead;
# callers iterating the SEP block on the only-SEP-on path should slice
# at the dynamic offset.
RICH_SEP_SLICE = slice(
    RICH_MACRO_REGIME_MISSING_SLICE.stop,
    RICH_MACRO_REGIME_MISSING_SLICE.stop + RICH_SEP_DIM,
)
RICH_SEP_MISSING_SLICE = slice(
    RICH_SEP_SLICE.stop,
    RICH_SEP_SLICE.stop + RICH_SEP_MISSING_DIM,
)
# #214 press-conf block slice (one scalar, no missing flag — the flag is
# itself the missingness signal). When the block is populated it sits
# past the regime + SEP tails per the documented append order
# (market | rich | regime? | sep? | press_conf?). When the
# regime / SEP flags are off the press-conf block lands at the dynamic
# offset their absence opens up; callers iterating the press-conf block
# should compute the offset off the active flag tuple.
RICH_PRESS_CONF_SLICE = slice(
    RICH_SEP_MISSING_SLICE.stop,
    RICH_SEP_MISSING_SLICE.stop + RICH_PRESS_CONF_DIM,
)

# Multi-task head (#78) axis cardinalities and canonical label maps.
# The multi-task head emits four branches; the cardinalities are pinned
# here so the loader, the model factory, and the inference path agree on
# the shape. Adding a topic label would require bumping
# MULTI_TASK_TOPIC_LABELS in lockstep with the loader's topic-string
# normaliser; the four buckets below cover the only topic-string
# families that show up on gtfintechlab + scraped Fed rows.
MULTI_TASK_STANCE_CLASSES = 3
MULTI_TASK_CERTAINTY_CLASSES = 3
MULTI_TASK_TOPIC_CLASSES = 4
MULTI_TASK_STANCE_LABELS: tuple[str, ...] = ("hawkish", "dovish", "neutral")
MULTI_TASK_CERTAINTY_LABELS: tuple[str, ...] = ("certain", "uncertain", "neutral")
MULTI_TASK_TOPIC_LABELS: tuple[str, ...] = (
    "macro",
    "forward_guidance",
    "market_reaction",
    "other",
)

# Text-embedding adapter dim search axis. The forecaster sweep iterates
# over these values so the diminishing-returns curve across {32, 64, 128}
# shows up in the aggregator table. The default mirrors the small-data
# regime: 64 dims is enough capacity to register the encoder signal
# without overfitting the ~895 training sequences.
DEFAULT_TEXT_ADAPTER_DIM = 64
TEXT_ADAPTER_DIM_CHOICES: tuple[int, ...] = (32, 64, 128)

# Default time-decay window for the prior-4 statement pooling
# (softmax(-Delta t / lambda_inv_days)). 30 days places roughly half the
# weight on the most recent statement when the prior FOMC release was
# 45 days ago; the sweep can override this through the CLI.
DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS = 30.0


def rich_feature_size_with_text(text_adapter_dim: int) -> int:
    """Return the per-bar input size when the text-embedding path is on.

    The scalar slice stays at ``RICH_FEATURE_SIZE`` (35). The adapter
    projection contributes another ``text_adapter_dim`` dims that the
    model broadcasts to every bar of the prior window plus the
    event-day target frame. The trailing ``+1`` is the missing flag the
    loader emits when fewer than one prior statement is available.
    """

    if text_adapter_dim <= 0:
        raise ValueError(
            f"text_adapter_dim must be a positive integer; got {text_adapter_dim}"
        )
    return RICH_FEATURE_SIZE + int(text_adapter_dim) + 1


def rich_feature_size_with_regime(use_regime: bool) -> int:
    """Return the per-bar rich-feature size with the #307 regime block.

    ``use_regime=False`` returns the legacy ``RICH_FEATURE_SIZE`` so the
    pre-#307 path is byte-identical. ``use_regime=True`` adds
    ``RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM`` for the
    macro-regime block appended at the end of ``as_rich_list``.

    The model factory reads this through ``ModelConfig.use_regime_conditioning``
    so the input projection widens in lockstep with the loader's
    decision to attach a populated block per event.
    """

    if not bool(use_regime):
        return RICH_FEATURE_SIZE
    return RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM


def rich_feature_size_with_sep(use_sep: bool) -> int:
    """Return the per-bar rich-feature size with only the #215 SEP block.

    ``use_sep=False`` returns the legacy ``RICH_FEATURE_SIZE`` so the
    pre-#215 path is byte-identical. ``use_sep=True`` adds
    ``RICH_SEP_DIM + RICH_SEP_MISSING_DIM`` for the SEP block appended
    at the end of ``as_rich_list``.
    """

    if not bool(use_sep):
        return RICH_FEATURE_SIZE
    return RICH_FEATURE_SIZE + RICH_SEP_DIM + RICH_SEP_MISSING_DIM


def rich_feature_size_with_blocks(
    *,
    use_regime: bool,
    use_sep: bool,
<<<<<<< HEAD
    use_press_conf: bool = False,
) -> int:
    """Combined helper: the per-bar size with regime, SEP, and press-conf block flags.

    The blocks are independent — every subset can be active. ``as_rich_list``
    appends them in a fixed order (regime, then SEP, then press_conf) so
    a downstream caller iterating slices knows where each block sits
    without ambiguity. Adding new optional blocks to this helper without
    bumping the legacy ``RICH_FEATURE_SIZE`` is the structural lock that
    keeps default-off paths byte-identical across feature additions.
=======
    use_statement_delta: bool = False,
    use_vote_features: bool = False,
) -> int:
    """Combined helper: the per-bar size with every opt-in tail block.

    All four blocks are independent — any combination can be on. The
    append order on ``as_rich_list`` is fixed: regime, SEP, statement-
    delta, vote-features. A downstream caller iterating slices knows
    where each block sits without ambiguity given the four flags.
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
    """

    size = RICH_FEATURE_SIZE
    if bool(use_regime):
        size += RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
    if bool(use_sep):
        size += RICH_SEP_DIM + RICH_SEP_MISSING_DIM
<<<<<<< HEAD
    if bool(use_press_conf):
        size += RICH_PRESS_CONF_DIM
=======
    if bool(use_statement_delta):
        size += RICH_STATEMENT_DELTA_DIM + RICH_STATEMENT_DELTA_MISSING_DIM
    if bool(use_vote_features):
        size += RICH_VOTE_FEATURES_DIM + RICH_VOTE_FEATURES_MISSING_DIM
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
    return size


FORECAST_CONFIDENCE_LEVEL = 0.8
CONFIDENCE_Z_SCORE = 1.2816  # Approximate central 80% interval
DEFAULT_CLOSE_SCALE = 10000.0
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.15
DEFAULT_HEAD_HIDDEN_SIZE = 32
DEFAULT_INITIAL_DECAY_RATE = 1.5
DEFAULT_CHUNK_DECAY_RATE = 1.0 / 30.0
DEFAULT_CHUNK_EMBEDDING_SIZE = 768
DEFAULT_CHUNK_PROJECTION_DIM = 8
CREDIBILITY_FEATURE_DIM = 4

DEFAULT_DATA_DIR = DATA_DIR
MODELS_DIR = MODEL_CHECKPOINT_DIR
BEST_MODEL_PATH = MODELS_DIR / "forecaster_best.pt"


FORECASTER_ARCHITECTURES: tuple[str, ...] = (
    "lstm",
    "lstm_attn",
    "gru",
    "tcn",
    "transformer",
    "dlinear",
    "informer",
    "tft",
    # #327 Arm B. ``flat_mlp`` drops the sequence wrap on the text path
    # entirely: the recurrent core is replaced by a flat MLP head that
    # consumes ``[pooled_market_window || pooled_text_adapter || rich]``
    # so the broadcast-static framing of the text path has an honest
    # comparator. Wires through :class:`app.models.flat_mlp.ForecasterFlatMLP`.
    "flat_mlp",
)

# Architectures excluded from the canonical sweep targets. See ADR 0020
# (``docs/adr/0020-tft-architecture-comparison-exclusion.md``) and the
# footnote on ``fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.7`` for
# the rationale. TFT's published recipe routes predictions through its
# native quantile output + Variable Selection Network; the in-repo
# encoder pools to a generic classifier head, which strips the
# inductive bias the architecture is designed around. The 0.3803 row
# from §6.6 is retained as historical record only. A faithful
# quantile-head reimplementation is filed as a STRETCH follow-up.
#
# The architecture identifier stays in ``FORECASTER_ARCHITECTURES`` so
# existing checkpoints that recorded ``architecture='tft'`` continue to
# load and the ``TFTEncoder`` module remains importable; new sweeps
# should iterate ``CANONICAL_SWEEP_ARCHITECTURES`` instead.
TFT_EXCLUSION_REASON: str = (
    "TFT excluded from canonical architecture sweep per ADR 0020 "
    "(generic classifier head strips the native quantile-output + "
    "Variable Selection Network inductive bias). Faithful "
    "quantile-head reimplementation is a STRETCH-tier follow-up."
)
CANONICAL_SWEEP_ARCHITECTURES: tuple[str, ...] = tuple(
    arch for arch in FORECASTER_ARCHITECTURES if arch != "tft"
)


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = FEATURE_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE
    initial_decay_rate: float = DEFAULT_INITIAL_DECAY_RATE
    text_channel: str = "scalar"
    embedding_adapter_dim: int = 128
    credibility_features: bool = False
    architecture: str = "lstm"
    # Pooled text-embedding path (PR #176 onward). ``text_embedding_dim``
    # is the encoder-native dim of the pooled vector the loader emits
    # (FinBERT 768, voyage-finance-2 1024, ...); ``text_adapter_dim``
    # is the projection target the recurrent core actually sees. Both
    # default to ``0`` so any pre-existing checkpoint deserialises into
    # the byte-identical no-text path.
    text_embedding_dim: int = 0
    text_adapter_dim: int = 0
    # Phase 9 V2 (#195) classification mode. ``"regression"`` keeps
    # the existing 2-output (close, vol) head and SmoothL1 loss path
    # byte-identical for every legacy caller. ``"classification"``
    # routes the forward pass through a new ``Linear(hidden, n_classes)``
    # head and switches the training loss to CrossEntropy. The
    # per-fold quantile cutoffs that turn the continuous target into
    # class indices are fitted on the train slice and persisted into
    # the checkpoint payload alongside ``close_scale`` and
    # ``rich_feature_scaler``.
    output_mode: str = "regression"
    n_classes: int = 3
    vol_regime_quantiles: tuple[float, ...] = ()
    vol_regime_target: str = "forward_realized_vol_10d"
    # Phase B (#227) LR-schedule selector. ``plateau`` is the legacy
    # ReduceLROnPlateau path (locked by the determinism regression).
    # ``cosine_warmup`` builds a OneCycleLR over the configured epoch
    # budget (warmup -> cosine -> tail). Persisted on the checkpoint so
    # resume reuses the same schedule the original run trained under.
    lr_schedule: str = "plateau"
    # Sequence length the loader emits per training row. ``0`` means
    # "use the module-level ``SEQUENCE_LENGTH`` default" so legacy
    # checkpoints deserialise into the byte-identical 20-bar window.
    # The CLI surfaces this as ``--sequence-length`` for the capacity
    # push at hidden=512 / seq=60.
    sequence_length: int = 0
    # Round 4 (#243) elapsed-time decay toggle. ``True`` keeps the
    # ``TimeDecayAttention`` path (the advisor-mandated mechanism that
    # multiplies the sentiment channel by ``exp(-lambda * |elapsed|)``);
    # ``False`` swaps it for a no-op pass-through so the ablation can
    # measure whether the mechanism actually earns its complexity on
    # the post-embargo baseline. Default ``True`` preserves the legacy
    # forward path byte-identical.
    use_time_decay: bool = True
    # Round 5 (#244) LoRA ceiling probe. ``False`` (default) reads
    # pooled text embeddings from the parquet cache -- the static path
    # the rest of the pipeline depends on. ``True`` pulls the encoder
    # named by ``text_encoder`` into ``train_model``, wraps it with
    # PEFT LoRA, and runs the forward per batch so the regime loss
    # flows gradients into the encoder. Per-event raw text is loaded
    # from events.parquet at load time, tokenised once, then re-encoded
    # per batch through the LoRA-wrapped tower. Scoped to a single
    # arch x seed cell -- not a default replacement.
    encoder_lora: bool = False
    # Bundle B LoRA freeze curriculum. When ``None`` (default) the LoRA
    # adapter stays trainable for the full epoch budget -- byte-identical
    # to the pre-Bundle-B path. When set to a non-negative integer, the
    # training loop freezes every LoRA matrix at the start of that epoch
    # (0-indexed) so subsequent epochs only update the classification
    # head. Used by the stage-1-train-then-freeze schedule that lets the
    # adapter absorb cross-bank-shared linguistic structure under FOMC
    # supervision in stage 1, then specialises the head in stage 2
    # without further encoder drift.
    lora_curriculum_freeze_epoch: int | None = None
    # Multi-modal fusion (#235). ``concat`` (default) keeps the legacy
    # ForecasterModel path where the text adapter output is
    # broadcast-concatenated to every LSTM timestep. ``gated_infonce``
    # routes the build through :class:`MultiModalForecasterModel`: the
    # market features stream through the recurrent core untouched, the
    # text embedding feeds the gated fusion directly, and the training
    # loop adds an InfoNCE alignment loss on the two modality
    # projections.
    fusion_mode: str = "concat"
    infonce_lambda: float = 0.1
    infonce_temperature: float = 0.07
    infonce_latent_dim: int = 64
    # #273 follow-up to the multi-task head (#272). When True, the
    # training loop swaps the single-axis CrossEntropy for
    # :class:`app.training.loss.MultiTaskLoss`, which folds per-axis
    # CE / SmoothL1 terms onto stance / factor / certainty / topic
    # with per-axis class weights and a per-row availability mask.
    # Default False keeps the byte-identity regression contract on
    # every existing classification run (stance-only training).
    multi_task_loss: bool = False
    multi_task_lambda_stance: float = 1.0
    multi_task_lambda_factor: float = 0.3
    multi_task_lambda_certainty: float = 0.3
    multi_task_lambda_topic: float = 0.3
    # Steepens inverse-frequency class weights via ``raw[c] = 1 / (n_c + 1) ** power``.
    # ``1.0`` (default) is the legacy formula and preserves byte-identity with
    # pre-2026-05-25 sweep numbers; higher values force the gradient onto the
    # rare classes, mitigating the class-1 collapse the 3-class vol-regime
    # head exhibits on single-seed runs.
    class_weight_power: float = 1.0
    # #304 dual-head methodology, recanonicalised under ADR 0015 (#322),
    # with the empirical refinement per the dual-head three-way
    # comparison (`artifacts/experiments/dual_head_comparison_canonical.json`,
    # 2026-05-27). ``dual`` (default) trains the joint loss
    # ``(1 - regression_alpha) * CE + regression_alpha * MSE`` so the
    # 3-class CE head and the ``log(forward_realized_vol_10d)`` MSE head
    # share a backbone; the sweep showed dual matches classification
    # macro-F1 (0.419 vs 0.417) while shipping the regression band the
    # canonical surface needs. ``regression`` trains the log-RV MSE head
    # only — kept for the comparison sweep and for ablation, but on this
    # corpus loses ~20pp macro-F1 vs classification on the UI-bucketed
    # label space. ``classification`` keeps the legacy 3-class
    # CrossEntropy head as the sole supervised signal and is retained
    # for back-compat with pre-#322 checkpoints and the cross-objective
    # ablation. The regression head is only meaningful when
    # ``output_mode == "classification"`` because that branch carries
    # the ``forward_realized_vol_10d`` target; regression-output mode
    # (close, vol) ignores ``head_mode`` entirely.
    head_mode: str = "dual"
    regression_alpha: float = 0.5
    # #309 derived-text-features ablation. ``True`` (default) keeps the
    # FeatureVector's per-bar ``sentiment_score`` slot and the multi-axis
    # stance / certainty / topic slots wired into the forecaster head
    # exactly as the pre-#309 path does, so back-compat is byte-identical.
    # ``False`` zeros those slots after loader fan-out but before
    # tensorisation, leaving the document-level encoder text path as the
    # only text-derived signal flowing into the recurrent core. Used by
    # the three-way comparison (baseline / ablation / replacement-with-
    # pre-meeting) in ``scripts/run_derived_features_ablation.py``.
    use_derived_text_features: bool = True
    # #292 rates-complex heads. Tuple of head short-names (``"2y"`` /
    # ``"5y"`` / ``"terminal"``) the training run should mount alongside
    # the existing vol-regime head. Default ``()`` keeps the pre-#292
    # path byte-identical: no rates heads mount, no rates loss
    # contribution, no rates output on the inference path. Resolved from
    # the ``--rates-heads`` CLI flag via :func:`resolve_rates_heads`.
    rates_heads: tuple[str, ...] = ()
    # Per-head training mode. ``regression`` (default) drives the head
    # off MSE on the raw bps target only -- the aux classification
    # surface still emits at inference time for the API response, but
    # contributes no gradient. ``classification`` is the inverse:
    # cross-entropy on the per-fold tertile labels only. ``dual`` mixes
    # both terms via ``rates_alpha * MSE + (1 - rates_alpha) * CE_aux``.
    # The mode applies uniformly to every mounted rates head; per-head
    # mode-mixing was deferred so the CLI stays one knob deep. The
    # field is a plain string (not enum) so checkpoint round-tripping
    # via ``ModelConfig.from_model`` reads it back cleanly off the
    # stashed module attribute.
    rates_head_mode: str = "regression"
    # #292 auxiliary 3-class direction surface opt-in. ``False`` (default)
    # mounts only the regression heads on every active rates target; the
    # response surface emits the regression card with a ``None``
    # directional_bucket / bucket_probabilities. ``True`` mounts the
    # paired easing/neutral/tightening classifier alongside each
    # regression head and wires the CE term into the joint loss when
    # ``rates_head_mode != "regression"``. The flag stays orthogonal to
    # ``rates_head_mode``: a run with ``rates_head_mode="dual"`` and
    # ``rates_aux_classification=False`` is rejected at the factory
    # because the CE term has no head to land on.
    rates_aux_classification: bool = False
    # Weight on the regression term in the rates dual-head joint loss.
    # ``1.0`` collapses ``dual`` to regression-only at the loss level;
    # ``0.0`` collapses it to classification-only. The CLI default 0.5
    # picks an equal split so the comparison sweep starts on a balanced
    # base; the ablation sweep can drive the boundary cases for the
    # parity test.
    rates_alpha: float = 0.5
    # #305 rates-head target derivation. ``raw`` (default, byte-identical
    # to the pre-#305 path) predicts the observed
    # ``yield_<tenor>_change_5d`` move in bps. ``fomc_attributable``
    # predicts the 1-D projection of the observed move onto the
    # strict-prior policy-surprise direction ``sign(mp_surprise_level)``.
    # The mode applies uniformly to every mounted rates head; per-head
    # mode-mixing was deferred so the CLI stays one knob deep. See ADR
    # 0027 and :mod:`app.training.rates_targets`.
    rates_target_mode: str = "raw"
    # #435 forward-vol-target derivation. ``raw`` (default, byte-identical
    # to the pre-#236 path) feeds the regression head the standardised
    # ``log(forward_realized_vol_10d)`` scalar. ``garch_residual`` swaps
    # in ``forward_realized_vol_10d_garch_residual`` (raw minus the
    # GARCH(1,1) baseline; signed, no log) so the supervised target is
    # the unanticipated component the classical conditional-variance
    # model leaves on the table. Rows whose residual is ``None``
    # (insufficient fit history per ``MIN_FIT_RETURNS`` or QMLE
    # convergence failure) fall back to the raw target so row alignment
    # with ``y`` is preserved. See #434 for the data side and ADR 0034.
    vol_target_mode: str = "raw"
    # #307 macro-regime conditioning toggle. ``False`` (default) keeps
    # the pre-#307 path byte-identical: the loader leaves
    # ``FeatureVector.macro_regime_features`` at ``None`` and
    # ``as_rich_list`` does not append the regime block. ``True`` wires
    # the strict-prior 3-scalar block onto every supervised sequence and
    # mounts a multiplicative gating layer that modulates the text-
    # derived rich-feature slices on the input side of the recurrent
    # core. The gate is initialised so its output is identically 1.0 at
    # start of training, which keeps the OFF behaviour byte-identical
    # when the flag is later flipped without a re-init. See ADR 0029.
    use_regime_conditioning: bool = False
    # #215 SEP dot-plot opt-in. Default ``False`` keeps the per-bar
    # feature size byte-identical to pre-#215. When ``True`` the model
    # factory widens the recurrent core's input projection by
    # ``RICH_SEP_DIM + RICH_SEP_MISSING_DIM`` to absorb the SEP tail
    # the loader appends past ``RICH_FEATURE_SIZE`` (and past the
    # regime block when both flags are on). See ADR 0030.
    use_sep: bool = False
<<<<<<< HEAD
    # #214 FOMC press conference Q&A opt-in. Default ``False`` keeps the
    # per-bar feature size byte-identical to pre-#214. When ``True`` the
    # loader joins the press-conf lookup onto every supervised statement
    # event: the ``has_press_conf`` scalar fires on FOMC events with a
    # locatable Q&A transcript (post-2011 era), and the LoRA path
    # concatenates the Q&A text onto the statement's ``raw_text`` so the
    # encoder sees a joint statement-plus-Q&A document under route 1 of
    # the #214 scope brief. Pre-2011 events get a zero-imputed flag —
    # the covariate-shift handling rejected fragmenting the walk-forward
    # fold protocol for an era-specific subset. See ADR 0037.
    use_press_conf: bool = False
=======
    # #443 statement-delta (redline) opt-in. Default ``False`` keeps the
    # per-bar feature size byte-identical to pre-#443. When ``True`` the
    # loader populates ``FeatureVector.statement_delta_embedding`` from
    # the events.parquet column and ``as_rich_list`` appends the
    # ``RICH_STATEMENT_DELTA_DIM + 1`` tail. See ADR 0038.
    use_statement_delta: bool = False
    # #444 vote-tally + dissent opt-in. Default ``False`` keeps the
    # per-bar feature size byte-identical to pre-#444. When ``True`` the
    # loader populates ``FeatureVector.vote_features`` from the
    # events.parquet vote columns and ``as_rich_list`` appends the
    # ``RICH_VOTE_FEATURES_DIM + 1`` tail. See ADR 0038.
    use_vote_features: bool = False
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))

    @classmethod
    def from_model(cls, model: "Any") -> "ModelConfig":
        architecture = getattr(model, "model_type", None) or "lstm"
        if architecture not in FORECASTER_ARCHITECTURES:
            architecture = "lstm"
        return cls(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout,
            head_hidden_size=model.head_hidden_size,
            initial_decay_rate=model.initial_decay_rate,
            text_channel=getattr(model, "text_channel", "scalar"),
            embedding_adapter_dim=getattr(model, "chunk_projection_dim", 128) or 128,
            credibility_features=bool(getattr(model, "credibility_features", False)),
            architecture=str(architecture),
            text_embedding_dim=int(getattr(model, "text_embedding_dim", 0) or 0),
            text_adapter_dim=int(getattr(model, "text_adapter_dim", 0) or 0),
            output_mode=str(getattr(model, "output_mode", "regression") or "regression"),
            n_classes=int(getattr(model, "n_classes", 3) or 3),
            vol_regime_quantiles=tuple(
                float(v) for v in getattr(model, "vol_regime_quantiles", ()) or ()
            ),
            vol_regime_target=str(
                getattr(model, "vol_regime_target", "forward_realized_vol_10d")
                or "forward_realized_vol_10d"
            ),
            lr_schedule=str(getattr(model, "lr_schedule", "plateau") or "plateau"),
            sequence_length=int(getattr(model, "sequence_length", 0) or 0),
            use_time_decay=bool(getattr(model, "use_time_decay", True)),
            encoder_lora=bool(getattr(model, "encoder_lora", False)),
            lora_curriculum_freeze_epoch=(
                int(model.lora_curriculum_freeze_epoch)
                if getattr(model, "lora_curriculum_freeze_epoch", None) is not None
                else None
            ),
            fusion_mode=str(getattr(model, "fusion_mode", "concat") or "concat"),
            infonce_lambda=float(getattr(model, "infonce_lambda", 0.1)),
            infonce_temperature=float(getattr(model, "infonce_temperature", 0.07)),
            infonce_latent_dim=int(getattr(model, "infonce_latent_dim", 64)),
            multi_task_loss=bool(getattr(model, "multi_task_loss", False)),
            multi_task_lambda_stance=float(getattr(model, "multi_task_lambda_stance", 1.0)),
            multi_task_lambda_factor=float(getattr(model, "multi_task_lambda_factor", 0.3)),
            multi_task_lambda_certainty=float(getattr(model, "multi_task_lambda_certainty", 0.3)),
            multi_task_lambda_topic=float(getattr(model, "multi_task_lambda_topic", 0.3)),
            class_weight_power=float(getattr(model, "class_weight_power", 1.0)),
            head_mode=str(getattr(model, "head_mode", "dual") or "dual"),
            regression_alpha=float(getattr(model, "regression_alpha", 0.5)),
            use_derived_text_features=bool(
                getattr(model, "use_derived_text_features", True)
            ),
            rates_heads=tuple(
                str(v) for v in getattr(model, "rates_heads", ()) or ()
            ),
            rates_head_mode=str(
                getattr(model, "rates_head_mode", "regression") or "regression"
            ),
            rates_aux_classification=bool(
                getattr(model, "rates_aux_classification", False)
            ),
            rates_alpha=float(getattr(model, "rates_alpha", 0.5)),
            rates_target_mode=str(
                getattr(model, "rates_target_mode", "raw") or "raw"
            ),
            vol_target_mode=str(
                getattr(model, "vol_target_mode", "raw") or "raw"
            ),
            use_regime_conditioning=bool(
                getattr(model, "use_regime_conditioning", False)
            ),
            use_sep=bool(getattr(model, "use_sep", False)),
<<<<<<< HEAD
            use_press_conf=bool(getattr(model, "use_press_conf", False)),
=======
            use_statement_delta=bool(
                getattr(model, "use_statement_delta", False)
            ),
            use_vote_features=bool(
                getattr(model, "use_vote_features", False)
            ),
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RichFeatureScalerParams:
    """Median + IQR fitted on the rich-feature block [FEATURE_SIZE:RICH_FEATURE_SIZE].

    Fitted on the training slice only via
    ``app.training.loaders.fit_rich_feature_scaler_tensor`` and persisted
    into the checkpoint payload alongside ``close_scale`` so resume + the
    inference path apply the same transform deterministically.

    The scaler is a robust z-score ``(x - median) / iqr``. Constant
    columns (IQR < ``epsilon`` on the train slice) get their IQR coerced
    to ``1.0`` so the transform reduces to a pure centering step --
    safe against the placeholder ``credibility_market_implied_gap``
    (always 0.0 by contract today) and against any per-family ablation
    that zeros a slot before the scaler sees it.
    """

    medians: tuple[float, ...]
    iqrs: tuple[float, ...]
    epsilon: float = 1e-6
    fitted_at_utc: str = ""
    n_train_observations: int = 0

    def __post_init__(self) -> None:
        if len(self.medians) != RICH_EXTRA_FEATURE_SIZE:
            raise ValueError(
                "RichFeatureScalerParams.medians must have length "
                f"{RICH_EXTRA_FEATURE_SIZE}; got {len(self.medians)}"
            )
        if len(self.iqrs) != RICH_EXTRA_FEATURE_SIZE:
            raise ValueError(
                "RichFeatureScalerParams.iqrs must have length "
                f"{RICH_EXTRA_FEATURE_SIZE}; got {len(self.iqrs)}"
            )
        for i, iqr in enumerate(self.iqrs):
            if iqr <= 0.0:
                raise ValueError(
                    f"RichFeatureScalerParams.iqrs[{i}] must be positive "
                    "(constant columns are floored to 1.0 at fit time); "
                    f"got {iqr}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "medians": list(self.medians),
            "iqrs": list(self.iqrs),
            "epsilon": float(self.epsilon),
            "fitted_at_utc": str(self.fitted_at_utc),
            "n_train_observations": int(self.n_train_observations),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RichFeatureScalerParams | None":
        """Rehydrate from a checkpoint payload entry.

        Returns ``None`` on any malformed input so legacy checkpoints
        without the scaler key load cleanly under the unscaled-train
        regression contract.
        """

        if not isinstance(data, dict) or not data:
            return None
        try:
            return cls(
                medians=tuple(float(x) for x in data["medians"]),
                iqrs=tuple(float(x) for x in data["iqrs"]),
                epsilon=float(data.get("epsilon", 1e-6)),
                fitted_at_utc=str(data.get("fitted_at_utc", "")),
                n_train_observations=int(data.get("n_train_observations", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class FeatureVector:
    """Per-bar feature row consumed by the forecaster.

    The 6 market fields (``sentiment_score`` through ``elapsed_time``)
    are the legacy ``FEATURE_SIZE`` input; ``as_list`` emits exactly
    that slice and is back-compat with every pre-PR-#173 inference and
    training path.

    The trailing fields carry the rich-feature input added in PR #173
    (and extended subsequently). They are populated by
    ``app.training.loaders.load_training_sequences_from_package`` when
    ``rich_features=True``; on the legacy path they stay at their
    documented defaults so ``as_list`` and ``as_rich_list`` agree on
    the 6 market positions. ``as_rich_list`` emits the full 35-dim
    layout in the order documented at the module-level slice
    constants:

    - positions ``[0:6]`` -- market features.
    - positions ``[6:10]`` -- credibility 4-vector
      (``credibility_drift_score`` / ``credibility_realized_vs_stated_gap``
      / ``credibility_market_implied_gap`` /
      ``credibility_months_since_reversal``).
    - positions ``[10:25]`` -- 15-dim linguistic vector
      (8 LDA topic shares + 6 hand-crafted densities +
      ``pivot_distance``), in the same field order as
      ``app.features.linguistic.LinguisticVector``.
    - positions ``[25:29]`` -- MP-surprise 4-vector
      (``mp_surprise_level`` / ``mp_surprise_path_factor`` /
      ``fed_info_factor`` / ``mp_is_intermeeting``).
    - positions ``[29:35]`` -- 6-dim Option-A multi-axis slot:
      ``stance_hawk`` / ``stance_dove`` / ``stance_neutral`` (one-hot
      from ``axis_stance`` on events.parquet) plus
      ``time_label_forward`` / ``certain_label_certain`` (binary
      indicators lifted off ``multi_axis_extras`` for gtfintechlab
      cross-bank rows) plus ``stance_missing`` (1.0 when the stance
      label is absent so the model can tell "unknown" apart from a
      genuine neutral). Replaces the pre-2026-05-17 numeric axes
      (``axis_factor`` / ``axis_certainty`` / ``axis_time``) that were
      0% populated upstream. Slot size unchanged so existing checkpoints
      load without state_dict reshape.
    """

    date: str
    sentiment_score: float
    market_close: float
    market_volatility: float
    close_change_pct: float = 0.0
    volatility_change: float = 0.0
    elapsed_time: float = 0.0
    text_embedding: list[float] | None = None
    # Rich-feature payload (PR #173). Default values match
    # "all-zero / no-signal" so a FeatureVector built via the legacy
    # constructors round-trips ``as_rich_list`` to the existing
    # ``as_list`` plus zero-padding. The loader sets ``rich_payload``
    # to ``True`` after populating the trailing fields; the tensor
    # builder dispatches on that flag.
    credibility_drift_score: float = 0.0
    credibility_realized_vs_stated_gap: float = 0.0
    credibility_market_implied_gap: float = 0.0
    credibility_months_since_reversal: float = 0.0
    linguistic_features: list[float] | None = None
    mp_surprise_level: float = 0.0
    mp_surprise_path_factor: float = 0.0
    fed_info_factor: float = 0.0
    mp_is_intermeeting: float = 0.0
    # Option-A multi-axis slot (PR after 2026-05-17). Slot positions
    # [29:35] in ``as_rich_list``; field-order in this dataclass is
    # cosmetic. ``stance_missing`` defaults to 1.0 ("unknown" prior)
    # so a default-constructed FeatureVector behaves as "no stance
    # signal" rather than "stance=hawkish/dovish/neutral with weight 0".
    stance_hawk: float = 0.0
    stance_dove: float = 0.0
    stance_neutral: float = 0.0
    time_label_forward: float = 0.0
    certain_label_certain: float = 0.0
    stance_missing: float = 1.0
    # A2 (#207) realised-vol autoregressive horizons. Default 0.0 so
    # FeatureVectors built without rich-payload flow stay byte-identical
    # on the as_list (6-feature) path. Populated by the events.parquet
    # loader on the rich-feature path from per-bar prior_bars_json.
    realized_vol_20d: float = 0.0
    realized_vol_60d: float = 0.0
    # A3 (#208) cross-asset close levels per bar. Joined from
    # independent yfinance caches; a series with no observation on the
    # bar's date emits 0.0 rather than blocking the whole bar. The
    # per-fold RobustScaler handles the cross-symbol scale mismatch
    # downstream.
    vix_close: float = 0.0
    dxy_close: float = 0.0
    tnx_close: float = 0.0
    gold_close: float = 0.0
    # Path B Chunk 1: VIX term structure + short-end yield + the two
    # derived stationary slopes. Same back-compat contract as the
    # original four — missing fields on pre-widen events.parquet
    # default to 0.0 in the loader.
    vix3m_close: float = 0.0
    irx_close: float = 0.0
    vix_term_slope: float = 0.0
    yield_curve_slope_10y_3m: float = 0.0
    # B1 (#212) LLM-as-features one-hot block. 35-dim list mirroring
    # the catalogue order; each per-feature slot is a one-hot over the
    # feature's allowed levels. Default ``None`` keeps the regression /
    # legacy paths byte-identical (``as_rich_list`` emits an all-zeros
    # block + a missing flag of 1.0).
    llm_features: list[float] | None = None
    llm_features_missing: float = 1.0
    # #306 retrieval-augmented summary stats over the top-K analog hits
    # from the on-disk retrieval index (#294). Five contextual scalars
    # — similarity moments + stance-agreement + above-floor count — plus
    # a paired missing flag. Default ``None`` keeps the regression /
    # legacy paths byte-identical: ``as_rich_list`` emits an all-zeros
    # block + ``analog_features_missing=1.0`` when this slot is empty,
    # which is also the contract when the retrieval bundle is absent on
    # disk (graceful degrade for ops without the retrieval bundle). The
    # analog's post-event observed move is NOT in this block — only
    # contextual (similarity + stance-agreement) summary stats. See ADR
    # 0028 and the per-feature row in ``docs/feature-provenance-audit.md``.
    analog_features: list[float] | None = None
    analog_features_missing: float = 1.0
    # #307 macro-regime conditioning block. Three signed scalars
    # (``policy_cycle_phase_score`` / ``vix_level_regime_score`` /
    # ``term_spread_sign``) plus a paired missing flag. Default
    # ``None`` keeps the regression / legacy paths byte-identical:
    # ``as_rich_list`` does NOT append the block when this slot is
    # empty, so the per-bar feature size stays at the legacy
    # ``RICH_FEATURE_SIZE`` width. The loader sets the slot only when
    # ``--use-regime-conditioning`` is on. See ADR 0029 and the
    # per-feature row in ``docs/feature-provenance-audit.md``.
    macro_regime_features: list[float] | None = None
    macro_regime_features_missing: float = 1.0
    # #215 Summary of Economic Projections (SEP) dot-plot block. Five
    # scalars (current-year / next-year / longer-run median FFR
    # projections + the current-year central-tendency range + a release
    # flag distinguishing fresh SEP meetings from forward-filled rows)
    # plus a paired missing flag. Default ``None`` keeps the regression /
    # legacy paths byte-identical: ``as_rich_list`` does NOT append the
    # block when this slot is empty, so the per-bar feature size stays
    # at the legacy ``RICH_FEATURE_SIZE`` width (or the regime-widened
    # width when ``--use-regime-conditioning`` is on). The loader sets
    # the slot only when ``--use-sep`` is on. See ADR 0030 and the
    # per-feature row in ``docs/feature-provenance-audit.md``.
    sep_features: list[float] | None = None
    sep_features_missing: float = 1.0
<<<<<<< HEAD
    # #214 FOMC press conference Q&A block. One scalar (``has_press_conf``)
    # that fires on FOMC events with a locatable Q&A transcript and
    # zero-imputes for pre-2011 events / every other event_kind. The
    # block doubles as its own missingness flag — the covariate shift
    # between the pre-2011 (no scheduled press conf) and post-2011 eras
    # is the entire signal the scalar carries, and a separate
    # ``*_missing`` would just be its complement. Default ``None`` keeps
    # the regression / legacy paths byte-identical: ``as_rich_list`` does
    # NOT append the block when this slot is empty, so the per-bar
    # feature size stays at the legacy ``RICH_FEATURE_SIZE`` width (or
    # the regime / SEP-widened width when those flags are on). The
    # loader sets the slot only when ``--use-press-conf`` is on. See
    # ADR 0037 and the per-feature row in
    # ``docs/feature-provenance-audit.md``.
    press_conf_features: list[float] | None = None
=======
    # #443 statement-delta mean-pooled embedding block. Default ``None``
    # keeps the regression / legacy paths byte-identical: ``as_rich_list``
    # does NOT append the block when this slot is empty, so the per-bar
    # feature size stays at the legacy ``RICH_FEATURE_SIZE`` (or the
    # widened width when prior opt-in blocks are on). The loader sets
    # the slot only when ``--use-statement-delta`` is on and the
    # events.parquet row carries a non-null embedding (cold-start events
    # and non-statement kinds keep ``None`` so the missing flag fires).
    # See ADR 0038 and the per-feature row in
    # ``docs/feature-provenance-audit.md``.
    statement_delta_embedding: list[float] | None = None
    statement_delta_embedding_missing: float = 1.0
    # #444 vote-tally signed feature block. The loader composes a
    # 4-vector off the events.parquet vote columns:
    # ``[votes_for_norm, votes_against_norm, is_unanimous_float,
    # dissent_direction_signed]`` where ``*_norm`` is the raw count
    # divided by 12 (the canonical FOMC voting-member cap) and
    # ``dissent_direction_signed`` maps hawkish_dissent → +1.0,
    # dovish_dissent → -1.0, unanimous / unparseable → 0.0. Default
    # ``None`` keeps the regression / legacy paths byte-identical.
    # See ADR 0038.
    vote_features: list[float] | None = None
    vote_features_missing: float = 1.0
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
    rich_payload: bool = False
    # Phase 9 V2 (#195) classification target. The forward 10-trading-day
    # realised volatility lives on the target row (the last vector in
    # a sequence). For non-target lookback bars the loader leaves it as
    # ``None``; the per-fold quantile-cutoff fitter and class-index
    # mapper consume the target-row value only. Default ``None`` so
    # regression-only callers stay byte-identical.
    forward_realized_vol_10d: float | None = None
    # #236 GARCH(1,1)-residual decomposition of the same target. The
    # baseline is the GARCH(1,1) 10-day-ahead 1-day-equivalent vol
    # forecast (fitted on strict-prior log returns); the residual is
    # ``forward_realized_vol_10d - baseline``. Both ride on the target
    # row alongside the raw forward-vol target; lookback bars carry
    # ``None`` so the per-fold builder can filter the leading target
    # the same way the raw vol-regime helper does. ``None`` on every
    # legacy / non-vol path keeps the dataclass shape round-trip clean
    # against the determinism regression contract. See ADR 0034 and
    # ``app.data.garch_residual.compute_for_event``.
    forward_realized_vol_10d_garch_baseline: float | None = None
    forward_realized_vol_10d_garch_residual: float | None = None
    # #292 rates-complex targets. Strict-forward 5-day yield change in
    # basis points (raw bps; the loader emits the value the
    # events.parquet column already carries). Populated by the
    # training-package loader on the target row of each supervised
    # sequence; the lookback bars stay at ``None`` so the per-fold
    # ``_build_partition_rates_target`` helper can filter against the
    # leading-target row the same way the vol-regime helper does.
    # ``None`` on every legacy / non-rates path so the dataclass shape
    # round-trips clean through the determinism regression contract.
    target_yield_2y_change_5d: float | None = None
    target_yield_5y_change_5d: float | None = None
    target_terminal_rate_change_5d: float | None = None
    # #305 FOMC-attributable rates targets. Strict-forward 5-day yield
    # change projected onto the strict-prior policy-surprise direction
    # ``sign(mp_surprise_level)``. ``None`` on no-change meetings (where
    # the surprise magnitude is below ``SURPRISE_DIRECTION_EPSILON_BPS``
    # and the direction is ill-defined); ``None`` on every legacy /
    # non-rates path so the dataclass shape round-trips clean through
    # the determinism regression contract. Populated by the
    # training-package loader on the target row alongside the raw
    # ``target_yield_*_change_5d`` columns; the per-fold target builder
    # reads one or the other based on ``ModelConfig.rates_target_mode``.
    target_yield_2y_change_5d_fomc_attributable: float | None = None
    target_yield_5y_change_5d_fomc_attributable: float | None = None
    target_terminal_rate_change_5d_fomc_attributable: float | None = None
    # Pooled text-embedding payload (PR #176 onward). Carries the
    # variable-length encoder-output vector (FinBERT 768, voyage-finance-2
    # 1024, BGE 1024, ...) materialised by the loader's softmax(-Delta t /
    # lambda) weighted mean over the four most recent prior statements.
    # The list stays empty (and ``text_embedding_missing`` stays at 1.0)
    # on the legacy 6-feature path; the model factory only widens the
    # recurrent input when the loader explicitly attaches a pooled
    # vector. ``as_rich_list`` does NOT include this field — the
    # projection happens inside the model's forward pass through the
    # ``TextEmbeddingAdapter`` so the scalar slice stays at 35 dims.
    text_embedding_pooled: list[float] = field(default_factory=list)
    text_embedding_missing: float = 1.0
    # #327 Arm A. Per-bar pooled-text payload (``seq_len`` rows each of
    # encoder-native width). Default ``None`` keeps the broadcast-static
    # path (``text_channel='scalar'`` / ``'embeddings'``) byte-identical:
    # the loader only populates this slot when ``text_channel='per_bar'``
    # is wired, in which case each entry carries a per-day pool over the
    # prior-N FOMC documents aligned to that bar's calendar date. The
    # length is enforced to match the lookback the loader emits the
    # sequence under; ``None`` collapses the per-bar slot at model
    # forward time to the same broadcast-zero path the embedding adapter
    # walks when the missing flag fires.
    text_per_bar: list[list[float]] | None = None
    # Round 5 (#244) LoRA path raw text. Populated by the loader on the
    # target-row bar of each sequence only when ``encoder_lora=True`` is
    # threaded into the package-loading call. The other 20 lookback
    # bars in the sequence carry the default empty string; the
    # tokeniser in the LoRA training step reads from
    # ``sequence[-1].raw_text`` so memory does not duplicate the text
    # across the prior-window. Stays empty on every static-cache path
    # so the legacy embedding pipeline is byte-identical.
    raw_text: str = ""
    # Multi-task head (#78) per-axis training targets. Populated by the
    # loader on the target-row bar (last index of each supervised
    # sequence); the lookback bars carry the defaults. ``target_*_present``
    # is the per-axis mask the loss reads to decide whether the row
    # contributes to that axis's loss. Indices use the canonical
    # mappings: stance {hawkish: 0, dovish: 1, neutral: 2}, certainty
    # {certain: 0, uncertain: 1, neutral: 2}, topic {macro: 0,
    # forward_guidance: 1, market_reaction: 2, other: 3}. Factor is a
    # signed scalar in [-1, 1] (no idx). When a label is absent the
    # target field stays at its default and the mask is False, so the
    # masked loss contributes zero for that axis on that row.
    target_stance_idx: int = -1
    target_stance_present: bool = False
    target_factor: float = 0.0
    target_factor_present: bool = False
    target_certainty_idx: int = -1
    target_certainty_present: bool = False
    target_topic_idx: int = -1
    target_topic_present: bool = False

    @classmethod
    def from_market_state(
        cls,
        *,
        date: str,
        sentiment_score: float,
        market_close: float,
        market_volatility: float,
        previous_close: float | None = None,
        previous_volatility: float | None = None,
        elapsed_time: float = 0.0,
        text_embedding: list[float] | None = None,
    ) -> "FeatureVector":
        close_change_pct = 0.0
        if previous_close is not None and abs(previous_close) > 1e-12:
            close_change_pct = (float(market_close) - float(previous_close)) / float(previous_close)

        volatility_change = 0.0
        if previous_volatility is not None:
            volatility_change = float(market_volatility) - float(previous_volatility)

        return cls(
            date=date,
            sentiment_score=float(sentiment_score),
            market_close=float(market_close),
            market_volatility=float(market_volatility),
            close_change_pct=float(close_change_pct),
            volatility_change=float(volatility_change),
            elapsed_time=float(elapsed_time),
            text_embedding=list(text_embedding) if text_embedding is not None else None,
        )

    def as_list(self, close_scale: float = DEFAULT_CLOSE_SCALE) -> list[float]:
        return [
            float(self.sentiment_score),
            float(self.market_close) / close_scale,
            float(self.market_volatility),
            max(min(float(self.close_change_pct), 1.0), -1.0),
            max(min(float(self.volatility_change), 1.0), -1.0),
            float(self.elapsed_time) / 30.0,
        ]

    def as_rich_list(self, close_scale: float = DEFAULT_CLOSE_SCALE) -> list[float]:  # noqa: C901
        """Emit the full 35-dim per-bar feature vector.

        Layout matches the slice constants at the top of this module
        and the docstring on :class:`FeatureVector`. The first six
        positions are byte-identical to :meth:`as_list` so models
        widened to ``RICH_FEATURE_SIZE`` still see the legacy market
        signal in positions ``[0:6]``.
        """

        market = self.as_list(close_scale=close_scale)
        credibility = [
            float(self.credibility_drift_score),
            float(self.credibility_realized_vs_stated_gap),
            float(self.credibility_market_implied_gap),
            float(self.credibility_months_since_reversal),
        ]
        linguistic_source = self.linguistic_features or []
        linguistic = [float(v) for v in linguistic_source[:RICH_LINGUISTIC_DIM]]
        if len(linguistic) < RICH_LINGUISTIC_DIM:
            linguistic = linguistic + [0.0] * (RICH_LINGUISTIC_DIM - len(linguistic))
        mp_surprise = [
            float(self.mp_surprise_level),
            float(self.mp_surprise_path_factor),
            float(self.fed_info_factor),
            float(self.mp_is_intermeeting),
        ]
        multi_axis = [
            float(self.stance_hawk),
            float(self.stance_dove),
            float(self.stance_neutral),
            float(self.time_label_forward),
            float(self.certain_label_certain),
            float(self.stance_missing),
        ]
        realized_vol = [
            float(self.realized_vol_20d),
            float(self.realized_vol_60d),
        ]
        cross_asset = [
            float(self.vix_close),
            float(self.dxy_close),
            float(self.tnx_close),
            float(self.gold_close),
            float(self.vix3m_close),
            float(self.irx_close),
            float(self.vix_term_slope),
            float(self.yield_curve_slope_10y_3m),
        ]
        # B1 (#212) LLM-as-features block. When ``llm_features`` is
        # ``None`` (legacy path or extraction not yet attached) the
        # whole 35-dim slot collapses to zeros and the missing flag
        # stays at its 1.0 default. The loader sets the flag to 0.0
        # only on rows that received a successful extraction.
        if self.llm_features is None:
            llm_block = [0.0] * RICH_LLM_FEATURE_DIM
        else:
            llm_block = [float(v) for v in self.llm_features[:RICH_LLM_FEATURE_DIM]]
            if len(llm_block) < RICH_LLM_FEATURE_DIM:
                llm_block = llm_block + [0.0] * (RICH_LLM_FEATURE_DIM - len(llm_block))
        llm_missing = [float(self.llm_features_missing)]
        # #306 retrieval-augmented analog summary block. When
        # ``analog_features`` is ``None`` (legacy path, opt-out, or
        # retrieval bundle absent on disk) the whole slot collapses to
        # zeros and the missing flag stays at its 1.0 default. The
        # loader sets the flag to 0.0 only on rows that received a
        # populated top-K retrieval result.
        if self.analog_features is None:
            analog_block = [0.0] * RICH_RETRIEVAL_ANALOG_DIM
        else:
            analog_block = [
                float(v) for v in self.analog_features[:RICH_RETRIEVAL_ANALOG_DIM]
            ]
            if len(analog_block) < RICH_RETRIEVAL_ANALOG_DIM:
                analog_block = analog_block + [0.0] * (
                    RICH_RETRIEVAL_ANALOG_DIM - len(analog_block)
                )
        analog_missing = [float(self.analog_features_missing)]
        out = (
            market
            + credibility
            + linguistic
            + mp_surprise
            + multi_axis
            + realized_vol
            + cross_asset
            + llm_block
            + llm_missing
            + analog_block
            + analog_missing
        )
        # #307 macro-regime conditioning block. Appended only when the
        # loader populated ``macro_regime_features``; otherwise the
        # per-bar feature size stays at the legacy ``RICH_FEATURE_SIZE``
        # width, byte-identical to pre-#307. The conditional append is
        # the structural lock that keeps the default ``--no-regime-conditioning``
        # path identical to existing callers iterating slices inside
        # ``[0:RICH_FEATURE_SIZE]``. See ADR 0029.
        if self.macro_regime_features is not None:
            regime_block = [
                float(v) for v in self.macro_regime_features[:RICH_MACRO_REGIME_DIM]
            ]
            if len(regime_block) < RICH_MACRO_REGIME_DIM:
                regime_block = regime_block + [0.0] * (
                    RICH_MACRO_REGIME_DIM - len(regime_block)
                )
            out = out + regime_block + [float(self.macro_regime_features_missing)]
        # #215 SEP dot-plot block. Appended after the regime block so
        # the two are independent: a checkpoint trained with one flag
        # on still loads cleanly when the other is off, because the
        # block widths past ``RICH_FEATURE_SIZE`` are additive in the
        # documented order (regime, then SEP). The conditional append
        # is the structural lock that keeps the default ``--no-sep``
        # path identical to existing callers iterating slices inside
        # ``[0:RICH_FEATURE_SIZE]`` (or inside the regime-widened width
        # when ``--use-regime-conditioning`` is on). See ADR 0030.
        if self.sep_features is not None:
            sep_block = [
                float(v) for v in self.sep_features[:RICH_SEP_DIM]
            ]
            if len(sep_block) < RICH_SEP_DIM:
                sep_block = sep_block + [0.0] * (
                    RICH_SEP_DIM - len(sep_block)
                )
            out = out + sep_block + [float(self.sep_features_missing)]
<<<<<<< HEAD
        # #214 FOMC press conference Q&A block. Appended past the SEP
        # tail under the documented append order (regime, then SEP, then
        # press_conf). Conditional append is the structural lock that
        # keeps the default ``--no-press-conf`` path byte-identical to
        # pre-#214; the single ``has_press_conf`` scalar is the entire
        # block — the missingness flag is folded into the same slot
        # because the covariate-shift signal is the scalar itself.
        if self.press_conf_features is not None:
            press_conf_block = [
                float(v) for v in self.press_conf_features[:RICH_PRESS_CONF_DIM]
            ]
            if len(press_conf_block) < RICH_PRESS_CONF_DIM:
                press_conf_block = press_conf_block + [0.0] * (
                    RICH_PRESS_CONF_DIM - len(press_conf_block)
                )
            out = out + press_conf_block
=======
        # #443 statement-delta (redline) tail. Same conditional-emission
        # contract as the regime / SEP blocks: appended only when the
        # loader populated ``statement_delta_embedding``; otherwise the
        # per-bar feature size stays byte-identical to the pre-#443
        # state. The block sits after the SEP tail when both are on so
        # downstream slice arithmetic is deterministic given the four
        # opt-in flags (regime, SEP, statement-delta, vote-features).
        if self.statement_delta_embedding is not None:
            delta_block = [
                float(v) for v in self.statement_delta_embedding[:RICH_STATEMENT_DELTA_DIM]
            ]
            if len(delta_block) < RICH_STATEMENT_DELTA_DIM:
                delta_block = delta_block + [0.0] * (
                    RICH_STATEMENT_DELTA_DIM - len(delta_block)
                )
            out = out + delta_block + [float(self.statement_delta_embedding_missing)]
        # #444 vote-tally tail. Appended last so the four opt-in blocks
        # land in a fixed order: regime, SEP, statement-delta, vote.
        if self.vote_features is not None:
            vote_block = [
                float(v) for v in self.vote_features[:RICH_VOTE_FEATURES_DIM]
            ]
            if len(vote_block) < RICH_VOTE_FEATURES_DIM:
                vote_block = vote_block + [0.0] * (
                    RICH_VOTE_FEATURES_DIM - len(vote_block)
                )
            out = out + vote_block + [float(self.vote_features_missing)]
>>>>>>> 671c784 (add statement-delta + vote-tally structured signal channels (#443, #444))
        return out


def build_lookback_sequence(vectors: Iterable[FeatureVector], length: int = SEQUENCE_LENGTH) -> list[FeatureVector]:
    """Pad-front (with the oldest vector) or truncate to a fixed lookback window."""

    items = list(vectors)
    if not items:
        items = [FeatureVector(date="", sentiment_score=0.0, market_close=0.0, market_volatility=0.0)]

    if len(items) >= length:
        return items[-length:]

    pad = [items[0] for _ in range(length - len(items))]
    return pad + items
