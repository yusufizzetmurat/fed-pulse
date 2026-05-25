"""LoRA-wrapped encoder helpers for the Round 5 (#244) ceiling probe.

The static text-embedding path (PR #176 onward) reads pre-computed
pooled vectors from a parquet cache. The encoder weights stay frozen
throughout training, so the regime-loss gradient never flows back into
the language model. Round 5 hypothesises that joint training -- letting
the regime loss update a small LoRA adapter on top of FinBERT (or
whichever encoder is configured) -- lifts the macro-F1 ceiling. This
module wires the encoder + LoRA wrapper + per-batch forward path that
``app.training.loop.train_model`` calls into when
``ModelConfig.encoder_lora`` is on.

Scope: one architecture x one seed x four folds x one trial. Not a
default replacement for the static cache.

Dependencies (``transformers``, ``peft``) are imported lazily inside
the helpers so the module imports cleanly on environments without the
LoRA stack -- the static-cache path stays runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

if TYPE_CHECKING:
    from app.models.config import FeatureVector


# Token-window cap for the LoRA-wrapped encoder. FinBERT / BERT-base /
# DeBERTa-v3-base all share the same 512-position limit; 256 covers a
# typical FOMC statement (≤ 250 tokens after WordPiece) while keeping
# the per-batch forward memory inside the RTX 4080 budget. Override at
# call time if a longer-context encoder lands.
DEFAULT_LORA_MAX_TOKENS = 256

# LoRA hyperparameters fixed for the ceiling probe. Chosen to mirror
# the PEFT defaults for BERT-family encoders (Hu et al. 2021):
# ``r=8`` keeps the adapter small (~0.3% of FinBERT's 110M params),
# ``alpha=16`` is the textbook 2x rank ratio, and the dropout matches
# the standard ``LoraConfig`` recommendation. ``target_modules`` picks
# the query + value projections in every attention layer -- the LoRA
# paper's lowest-cost variant that still recovers most of the lift
# over a frozen encoder.
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("query", "value")


@dataclass(frozen=True)
class LoraEncoderBundle:
    """Container for the wrapped encoder + its tokenizer.

    The training loop holds one instance for the lifetime of a single
    sweep cell; the per-batch forward path uses ``encoder`` and
    ``tokenizer`` directly. ``out_dim`` is the encoder's hidden size
    (768 for FinBERT, 1024 for BGE-large, etc.) so the downstream
    text-adapter projection picks the correct ``text_embedding_dim``.
    """

    encoder: nn.Module
    tokenizer: Any
    out_dim: int
    encoder_alias: str


def build_lora_encoder(
    encoder_alias: str,
    *,
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
) -> LoraEncoderBundle:
    """Load the encoder named by ``encoder_alias`` and wrap it with PEFT LoRA.

    Reads the checkpoint path + revision from
    ``app.models.registry``. The base encoder is frozen end-to-end;
    only the LoRA adapter parameters (``r * (in_dim + out_dim)`` per
    target module) train. Returns the wrapped model + the matching
    tokenizer + the encoder's hidden size so callers can size the
    downstream adapter dim correctly.

    Raises ``ValueError`` when the encoder alias is not registered or
    its revision is empty (unpinned-local placeholder). Raises
    ``ImportError`` (with a clear remediation hint) when ``transformers``
    or ``peft`` is missing.
    """

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "transformers is required for the encoder_lora path; "
            "install via the backend pyproject.toml dependency block."
        ) from exc
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "peft is required for the encoder_lora path; "
            "install ``peft>=0.10`` via the backend pyproject.toml."
        ) from exc

    from app.models.registry import encoder_ref

    ref = encoder_ref(encoder_alias)
    if ref is None:
        raise ValueError(
            f"encoder alias {encoder_alias!r} is not registered in "
            "models/registry.yaml"
        )
    if not ref.revision:
        raise ValueError(
            f"encoder alias {encoder_alias!r} is unpinned (empty revision); "
            "the LoRA path refuses to train against an unpinned checkpoint "
            "so the ceiling-probe number can be traced back to a specific "
            "encoder build"
        )

    tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
    base_encoder = AutoModel.from_pretrained(ref.repo, revision=ref.revision)
    base_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    wrapped = get_peft_model(base_encoder, lora_config)
    out_dim = int(getattr(base_encoder.config, "hidden_size", 0))
    if out_dim <= 0:
        raise RuntimeError(
            f"encoder {encoder_alias!r} did not expose ``hidden_size`` on "
            "its config -- cannot size the downstream text-adapter dim"
        )
    return LoraEncoderBundle(
        encoder=wrapped,
        tokenizer=tokenizer,
        out_dim=out_dim,
        encoder_alias=encoder_alias,
    )


def tokenize_sequence_texts(
    sequences: Sequence[Sequence["FeatureVector"]],
    tokenizer: Any,
    *,
    max_tokens: int = DEFAULT_LORA_MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise each sequence's target-row text into a (N, T) tensor pair.

    For each sequence group, reads ``sequence[-1].raw_text`` (the
    target-row bar populated by
    ``app.training.loaders._load_package_sequences_with_metadata``
    when ``encoder_lora=True`` is threaded through). Sequences whose
    target-row carries an empty string fall back to the empty-string
    encoding so the encoder forward still produces a deterministic
    zero-context vector for them; the resulting attention_mask is
    all-zero, which mean-pooling collapses to a zero output, matching
    the "missing flag" semantics of the static-cache path.

    Returns ``(input_ids, attention_mask)`` torch tensors with dtype
    ``int64`` and shape ``(len(sequences), max_tokens)``.
    """

    texts: list[str] = []
    for sequence in sequences:
        if not sequence:
            texts.append("")
            continue
        target_text = str(getattr(sequence[-1], "raw_text", "") or "").strip()
        texts.append(target_text)

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"].to(dtype=torch.long)
    attention_mask = encoded["attention_mask"].to(dtype=torch.long)
    # Zero out attention mask on rows whose text is empty so the
    # downstream mean-pool collapses to a zero vector.
    for i, text in enumerate(texts):
        if not text:
            attention_mask[i].zero_()
    return input_ids, attention_mask


def freeze_adapter(model_or_module: nn.Module) -> int:
    """Freeze every LoRA adapter parameter on ``model_or_module``.

    Iterates over ``named_parameters()`` and sets ``requires_grad=False``
    on each tensor whose name carries the canonical PEFT ``lora_`` token
    (``lora_A`` / ``lora_B`` / ``lora_embedding_A`` / ``lora_embedding_B``).
    When PEFT is installed and exposes :class:`peft.tuners.lora.LoraLayer`,
    every parameter that lives inside such a layer is also frozen as a
    belt-and-braces fallback against future renames.

    Non-LoRA parameters (the frozen base encoder, layer norms, biases on
    non-LoRA modules, etc.) are left untouched. The helper is idempotent
    -- calling it on an already-frozen adapter is a no-op that returns
    the same count.

    Used by the stage-1-train-then-freeze curriculum
    (``ModelConfig.lora_curriculum_freeze_epoch``): stage 1 trains the
    LoRA adapter on the combined supervision pool; the loop calls this
    helper at the configured epoch boundary so stage 2 only updates the
    classification head while the encoder representation stays fixed.

    Returns the number of LoRA parameter tensors that were marked
    non-trainable after the call (matches the count of LoRA tensors on
    the module, since the helper is idempotent).
    """

    lora_layer_cls: type | None = None
    try:
        from peft.tuners.lora import LoraLayer

        lora_layer_cls = LoraLayer
    except ImportError:  # pragma: no cover - defensive
        lora_layer_cls = None
    except AttributeError:  # pragma: no cover - older peft layouts
        lora_layer_cls = None

    # Collect LoRA parameter ids first by walking sub-modules (so we
    # catch every tensor PEFT inserted regardless of the exact field
    # name) and union with a name-pattern pass (covers PEFT releases
    # that ship without ``LoraLayer`` at the documented import path).
    lora_param_ids: set[int] = set()
    if lora_layer_cls is not None:
        for module in model_or_module.modules():
            if isinstance(module, lora_layer_cls):
                for param in module.parameters(recurse=True):
                    lora_param_ids.add(id(param))

    frozen = 0
    for name, param in model_or_module.named_parameters():
        is_lora = id(param) in lora_param_ids or ("lora_" in name)
        if not is_lora:
            continue
        param.requires_grad_(False)
        frozen += 1
    return frozen


def should_freeze_lora_at_epoch(
    freeze_epoch: int | None,
    current_epoch: int,
    *,
    already_frozen: bool,
) -> bool:
    """Return True when the loop should freeze the LoRA adapter this epoch.

    Encapsulates the boundary policy of the stage-1-train-then-freeze
    curriculum so the training loop and the unit tests share the same
    decision logic:

    - ``freeze_epoch is None`` → never freeze (current default).
    - ``already_frozen`` → no-op; the adapter is frozen for the rest of
      the run.
    - ``current_epoch >= freeze_epoch`` → freeze this epoch.

    The check is 0-indexed: ``freeze_epoch=2`` means "freeze at the start
    of the 3rd epoch", which leaves epochs 0 and 1 training the adapter.
    """

    if freeze_epoch is None:
        return False
    if already_frozen:
        return False
    return int(current_epoch) >= int(freeze_epoch)


def encode_batch_pooled(
    bundle: LoraEncoderBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LoRA-wrapped encoder over one batch and mean-pool tokens.

    Returns ``(pooled, missing_flag)`` where ``pooled`` has shape
    ``(B, out_dim)`` and ``missing_flag`` has shape ``(B, 1)`` and is
    ``1.0`` for rows whose attention mask is all-zero (the empty-text
    sentinel from ``tokenize_sequence_texts``). The pair matches the
    ``(text_embedding, text_embedding_missing)`` contract that
    ``ForecasterModel.forward`` consumes, so the LoRA branch in
    ``train_model`` can substitute it directly for the static-cache
    tensors emitted by ``_build_text_embedding_tensors``.

    Gradients flow through ``pooled`` back into the LoRA adapter
    parameters (the base encoder weights are frozen by
    ``build_lora_encoder``) so the regime loss updates the adapter
    end-to-end. ``missing_flag`` carries no gradient by construction
    -- it is a derived signal, not a learnable parameter.
    """

    encoder_outputs = bundle.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden_states = encoder_outputs.last_hidden_state
    mask_expanded = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    masked_hidden = hidden_states * mask_expanded
    token_counts = mask_expanded.sum(dim=1).clamp_min(1.0)
    pooled = masked_hidden.sum(dim=1) / token_counts
    # Missing flag: 1.0 when no tokens attended (empty-text sentinel),
    # 0.0 otherwise. Detached so no spurious gradient flows.
    row_token_sum = attention_mask.sum(dim=1)
    missing_flag = (row_token_sum == 0).to(dtype=pooled.dtype).unsqueeze(-1).detach()
    return pooled, missing_flag
