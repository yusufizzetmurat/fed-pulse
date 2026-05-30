"""B2 end-to-end fine-tune harness for vol-regime classification (#213).

Standalone harness that fine-tunes
``AutoModelForSequenceClassification`` directly on the FOMC document text
against the per-fold ``vol_regime_10d`` 3-class label (calm / normal /
high). Bypasses the LSTM-on-frozen-embeddings stack the §6 tier baselines
ship with; the regime-classification gradient reaches the encoder.

Pipeline per (seed, fold):

1. Read ``registry_normalized.jsonl`` for the FOMC document text +
   ``event_date`` per supervised row.
2. Read ``events.parquet`` for ``forward_realized_vol_10d`` per
   ``event_date``.
3. Apply the walk-forward split from
   ``fold_manifest_expanding_walk_forward.json``.
4. Fit tertile cutoffs on the TRAIN slice only via
   :func:`app.training.loaders.fit_vol_regime_quantiles`; assign the
   3-class label to every row in the fold using
   :func:`app.training.loaders.vol_regime_class_for`.
5. Tokenise, run an AdamW fine-tune for ``--epochs`` epochs, evaluate
   on the test slice, write per-(seed, fold) metrics into the sweep
   artefact at ``backend/artifacts/experiments/finetune_pilot_b2.json``
   under the canonical-comparison-style schema.

The encoder defaults to the canonical classifier role per ADR 0019,
falling back to ``finbert_fed_adjacent`` when the role tag is missing.
Operator can override via ``--encoder-alias``.

The full sweep (5 seeds x 4 folds x 5 epochs at AdamW 2e-5 on a 110M-
param backbone) is a Runpod follow-up; CI does not run it. The CI smoke
path exercises one epoch on a synthetic fixture under 60 s on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import BACKEND_ROOT, DATA_DIR as DEFAULT_DATA_DIR
from app.evaluation.classification_breakdown import compute_classification_breakdown
from app.models.registry import encoder_ref, resolve_by_role, revision_for

DEFAULT_OUTPUT_PATH = (
    BACKEND_ROOT.parent / "artifacts" / "experiments" / "finetune_pilot_b2.json"
)
DEFAULT_ARTIFACT_ROOT = DEFAULT_DATA_DIR / "artifacts" / "finetune_pilot_b2"

# Class order is pinned by the existing vol-regime classifier convention
# (see ``app.services.forecaster.VOL_REGIME_CLASS_LABELS``): index 0 is
# the lowest tertile (calm), index 1 is the middle tertile (normal),
# index 2 is the highest tertile (high). The per-fold tertile cutoffs
# fitted on the TRAIN slice produce these class indices via
# :func:`app.training.loaders.vol_regime_class_for`.
VOL_REGIME_LABELS: tuple[str, ...] = ("calm", "normal", "high")
N_CLASSES = len(VOL_REGIME_LABELS)
ID2LABEL: dict[int, str] = dict(enumerate(VOL_REGIME_LABELS))
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(VOL_REGIME_LABELS)}

# Default seed set + default encoder alias mirror the rest of the
# canonical surface. The seed tuple matches docs/benchmark-policy.md;
# the encoder fallback matches the ``train_text_multi_axis_classifier``
# DEFAULT_ENCODER_ALIAS fallback so this harness inherits the same
# unpinned-local guard surface.
DEFAULT_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 97)
DEFAULT_ENCODER_FALLBACK = "finbert_fed_adjacent"


@dataclass
class FomcRow:
    """One FOMC document with its forward-vol target attached."""

    record_id: str
    text: str
    event_date: str
    forward_vol: float
    source: str = ""


@dataclass
class FoldCell:
    """Per-(seed, fold) result row in the sweep artefact."""

    seed: int
    fold_id: str
    train_rows: int
    test_rows: int
    tertile_cutoffs: tuple[float, ...]
    train_class_counts: list[int]
    test_class_counts: list[int]
    train_loss: float
    macro_f1: float
    accuracy: float
    weighted_f1: float
    classification_breakdown: dict[str, Any]
    train_runtime_s: float
    eval_runtime_s: float
    phrasebank_aux_train_loss: float | None = None
    phrasebank_aux_lambda: float = 0.0
    phrasebank_aux_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fold_id": self.fold_id,
            "metrics": {
                "regime_f1_macro": self.macro_f1,
                "regime_accuracy": self.accuracy,
                "regime_weighted_f1": self.weighted_f1,
                "train_loss": self.train_loss,
            },
            "tertile_cutoffs": list(self.tertile_cutoffs),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_class_counts": list(self.train_class_counts),
            "test_class_counts": list(self.test_class_counts),
            "classification_breakdown": self.classification_breakdown,
            "train_runtime_s": self.train_runtime_s,
            "eval_runtime_s": self.eval_runtime_s,
        }
        if self.phrasebank_aux_lambda > 0.0:
            payload["phrasebank_aux"] = {
                "train_loss": self.phrasebank_aux_train_loss,
                "aux_lambda": self.phrasebank_aux_lambda,
                "n_rows": self.phrasebank_aux_rows,
            }
        return payload


@dataclass
class SeedTrial:
    """Per-seed bundle holding every fold cell."""

    seed: int
    folds: list[FoldCell] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "folds": [cell.to_dict() for cell in self.folds],
        }


def _resolve_default_encoder_alias() -> str:
    """Default to the classifier-role encoder per ADR 0019.

    Falls back to ``finbert_fed_adjacent`` when the registry has no
    classifier-role tag (mirrors the fallback in
    ``train_text_multi_axis_classifier.py``).
    """

    try:
        return resolve_by_role("classifier")
    except KeyError:
        return DEFAULT_ENCODER_FALLBACK


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def _resolve_training_package_dir(training_package_id: str) -> Path:
    package_dir = DEFAULT_DATA_DIR / "processed" / training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")
    return package_dir


def load_fomc_rows(package_dir: Path) -> list[FomcRow]:
    """Read FOMC document text + forward-vol per ``event_date``.

    Reads ``registry_normalized.jsonl`` for the supervised document text
    and ``events.parquet`` for the ``forward_realized_vol_10d`` target.
    Drops rows whose forward-vol is missing / non-finite so downstream
    callers never see a NaN label.
    """

    registry_path = package_dir / "registry_normalized.jsonl"
    events_path = package_dir / "events.parquet"
    if not registry_path.exists():
        raise SystemExit(f"Missing registry: {registry_path}")
    if not events_path.exists():
        raise SystemExit(f"Missing events.parquet: {events_path}")

    import pandas as pd

    events = pd.read_parquet(events_path)
    if "event_date" not in events.columns or "forward_realized_vol_10d" not in events.columns:
        raise SystemExit(
            f"events.parquet at {events_path} missing required columns; "
            "need event_date + forward_realized_vol_10d."
        )
    vol_lookup: dict[str, float] = {}
    for record in events[["event_date", "forward_realized_vol_10d"]].to_dict("records"):
        ed = str(record.get("event_date") or "")
        v = record.get("forward_realized_vol_10d")
        if not ed or v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vf):
            continue
        vol_lookup[ed] = vf

    rows: list[FomcRow] = []
    seen_record_ids: set[str] = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        event_date = str(payload.get("event_date", "")).strip()
        record_id = str(payload.get("record_id", "")).strip()
        source = str(payload.get("source", "")).strip()
        try:
            sample_weight = float(payload.get("sample_weight", 1.0))
        except (TypeError, ValueError):
            sample_weight = 1.0
        # Cross-bank rows enter the registry at sample_weight=0.0; this
        # harness fine-tunes against FOMC text only, so we drop them
        # explicitly. The encoder-alias override is the lever for
        # cross-bank experimentation in a follow-up.
        if sample_weight == 0.0:
            continue
        if not text or not event_date:
            continue
        if record_id and record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        vol = vol_lookup.get(event_date)
        if vol is None:
            continue
        rows.append(
            FomcRow(
                record_id=record_id,
                text=text,
                event_date=event_date,
                forward_vol=vol,
                source=source,
            )
        )
    return rows


def _load_fold(package_dir: Path, fold_id: str) -> dict[str, Any]:
    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    if not path.exists():
        raise SystemExit(f"Fold manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fold in payload.get("folds", []):
        if fold.get("fold_id") == fold_id:
            return dict(fold)
    raise SystemExit(f"Fold {fold_id} not found in {path}")


def _all_fold_ids(package_dir: Path) -> list[str]:
    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    if not path.exists():
        raise SystemExit(f"Fold manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(f.get("fold_id")) for f in payload.get("folds", []) if f.get("fold_id")]


def _split_by_fold(
    rows: Sequence[FomcRow], fold: dict[str, Any]
) -> tuple[list[FomcRow], list[FomcRow]]:
    train_end = str(fold["train_end"])
    test_start = str(fold["test_start"])
    test_end = str(fold["test_end"])
    train_rows = [r for r in rows if r.event_date <= train_end]
    test_rows = [r for r in rows if test_start <= r.event_date <= test_end]
    return train_rows, test_rows


def build_partition_classification_targets(
    rows: Sequence[FomcRow],
    *,
    train_rows: Sequence[FomcRow],
) -> tuple[list[int], tuple[float, ...], list[int]]:
    """Fit tertile cutoffs on the train slice and label every row.

    Returns ``(labels_for_rows, cutoffs, kept_indices)`` where
    ``kept_indices`` lists the indices of ``rows`` whose forward-vol
    landed in a defined class (i.e. ``vol_regime_class_for`` did not
    return ``-1``). The cutoffs come from the TRAIN slice only --
    leakage-protective contract for the classifier.
    """

    from app.training.loaders import fit_vol_regime_quantiles, vol_regime_class_for

    cutoffs = fit_vol_regime_quantiles(
        [r.forward_vol for r in train_rows], n_classes=N_CLASSES
    )
    labels: list[int] = []
    kept: list[int] = []
    for idx, row in enumerate(rows):
        cls = vol_regime_class_for(row.forward_vol, cutoffs)
        if cls < 0:
            continue
        labels.append(cls)
        kept.append(idx)
    return labels, cutoffs, kept


def _class_counts(labels: Sequence[int]) -> list[int]:
    counts = [0] * N_CLASSES
    for v in labels:
        if 0 <= v < N_CLASSES:
            counts[v] += 1
    return counts


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


def _block_bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    block_size: int = 4,
    rng_seed: int = 11,
) -> dict[str, float] | None:
    """Block-bootstrap CI on a flat list of (seed, fold) macro-F1 cells.

    Mirrors the canonical-comparison CI convention: pair adjacent fold
    cells inside each seed so the resample preserves intra-seed
    correlation. ``block_size`` defaults to 4 (one walk-forward fold
    panel per seed); the caller can override when the fold count
    differs.
    """

    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(rng_seed)
    n = len(finite)
    arr = np.asarray(finite, dtype=np.float64)
    n_blocks = max(1, math.ceil(n / max(block_size, 1)))
    boot_means: list[float] = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        idxs: list[int] = []
        for start in starts:
            for offset in range(block_size):
                idxs.append(int((start + offset) % n))
            if len(idxs) >= n:
                break
        sample = arr[np.asarray(idxs[:n], dtype=np.int64)]
        boot_means.append(float(sample.mean()))
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return {"lower": lo, "upper": hi, "confidence": confidence, "n_resamples": n_resamples}


def _train_and_eval_one_cell(  # noqa: PLR0913 — per-cell knobs surface as named kwargs by design
    *,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    encoder_alias: str,
    seed: int,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    phrasebank_rows: list[Any] | None = None,
    phrasebank_aux_lambda: float = 0.0,
) -> dict[str, Any]:
    """Run one fine-tune cell end-to-end and return its metrics dict.

    When ``phrasebank_rows`` is supplied and ``phrasebank_aux_lambda``
    is strictly positive, the FOMC stance CE is augmented by an
    auxiliary PhraseBank 3-way sentiment CE on a small linear head
    over the encoder's pooled output (#33 Path B). The aux head reads
    its rows from a separate DataLoader that round-robins one
    PhraseBank batch per FOMC batch; the auxiliary loss is added to
    the main loss as ``lambda * aux_ce`` so the aux gradient flows
    through the same encoder as the main task. When the aux is off
    the path stays byte-identical to pre-#33 B2.
    """

    # Imports happen here so module import stays cheap on environments
    # without torch (the dataclass + helper surface is importable
    # standalone for the unit-test smoke).
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _set_all_seeds(seed)

    enable_aux = bool(
        phrasebank_rows
        and phrasebank_aux_lambda > 0.0
        and len(phrasebank_rows) > 0
    )

    hf_token = _hf_token()
    revision = revision_for(encoder_alias)
    # ``encoder_alias`` is the registry alias (e.g. ``finbert_fed_adjacent``);
    # ``from_pretrained`` needs the underlying repo id (HF slug like
    # ``yusufizzetmurat/finbert-fed-adjacent`` after #464, or a local
    # path for unpinned-local entries). Look it up via ``encoder_ref``.
    ref = encoder_ref(encoder_alias)
    encoder_repo = ref.repo if ref is not None else encoder_alias
    tokenizer_kwargs: dict[str, Any] = {"token": hf_token} if hf_token else {}
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(encoder_repo, **tokenizer_kwargs)

    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_repo,
        num_labels=N_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        token=hf_token,
        revision=revision,
    )

    # Auxiliary 3-class linear head over the encoder's pooled output.
    # Only constructed when aux is on so the default-off path stays
    # byte-identical to pre-#33 B2 — same module graph, same parameter
    # set, same optimiser state.
    aux_head: nn.Linear | None = None
    if enable_aux:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        if hidden_size <= 0:
            raise RuntimeError(
                "Encoder model.config.hidden_size missing / non-positive; "
                "auxiliary head needs a pooled-output dimension."
            )
        aux_head = nn.Linear(hidden_size, 3)

    class _TextDataset(Dataset[dict[str, "torch.Tensor"]]):
        def __init__(self, texts: list[str], labels: list[int]) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, "torch.Tensor"]:
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }

    train_ds = _TextDataset(train_texts, train_labels)
    test_ds = _TextDataset(test_texts, test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if aux_head is not None:
        aux_head.to(device)

    optimizer_params: list[torch.nn.Parameter] = list(model.parameters())
    if aux_head is not None:
        optimizer_params.extend(list(aux_head.parameters()))
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
    )
    eval_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    # Auxiliary PhraseBank stream. A second DataLoader cycles over the
    # PhraseBank rows independently of the FOMC fold split; aux batches
    # are zipped one-for-one with FOMC batches inside the train step
    # via ``itertools.cycle`` so the aux pool drives no extra epochs.
    aux_loader: DataLoader | None = None
    aux_iter: Any = None
    if aux_head is not None and phrasebank_rows is not None:
        aux_texts = [str(r.sentence) for r in phrasebank_rows]
        aux_labels = [int(r.label_idx) for r in phrasebank_rows]
        aux_ds = _TextDataset(aux_texts, aux_labels)
        aux_generator = torch.Generator()
        aux_generator.manual_seed(seed + 1)
        aux_loader = DataLoader(
            aux_ds,
            batch_size=train_batch_size,
            shuffle=True,
            generator=aux_generator,
        )

    def _cycle(loader: DataLoader) -> Any:
        while True:
            yield from loader

    train_t0 = time.perf_counter()
    losses: list[float] = []
    aux_losses: list[float] = []
    model.train()
    if aux_head is not None:
        aux_head.train()
    if aux_loader is not None:
        aux_iter = _cycle(aux_loader)
    for _ in range(epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            optimizer.zero_grad()
            if aux_head is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_t,
                )
                loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_t,
                )
                main_loss = outputs.loss

                aux_batch = next(aux_iter)
                aux_input_ids = aux_batch["input_ids"].to(device)
                aux_attention_mask = aux_batch["attention_mask"].to(device)
                aux_labels_t = aux_batch["labels"].to(device)
                aux_outputs = model.base_model(
                    input_ids=aux_input_ids,
                    attention_mask=aux_attention_mask,
                )
                aux_pooled = _pooled_from_base_model_output(
                    aux_outputs, aux_attention_mask
                )
                aux_logits = aux_head(aux_pooled)
                aux_loss = nn.functional.cross_entropy(aux_logits, aux_labels_t)
                aux_losses.append(float(aux_loss.detach().item()))
                loss = main_loss + phrasebank_aux_lambda * aux_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
    train_runtime = time.perf_counter() - train_t0
    mean_train_loss = float(statistics.fmean(losses)) if losses else 0.0
    mean_aux_loss = float(statistics.fmean(aux_losses)) if aux_losses else None

    eval_t0 = time.perf_counter()
    model.eval()
    preds: list[int] = []
    truth: list[int] = []
    softmaxes: list[list[float]] = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1).tolist()
            preds.extend(int(p) for p in logits.argmax(dim=-1).tolist())
            truth.extend(int(t) for t in labels_t.tolist())
            softmaxes.extend(probs)
    eval_runtime = time.perf_counter() - eval_t0

    breakdown = compute_classification_breakdown(
        preds, truth, n_classes=N_CLASSES, class_scores=softmaxes
    )

    accuracy = (
        sum(1 for p, t in zip(preds, truth) if p == t) / len(truth)
        if truth
        else 0.0
    )

    return {
        "train_loss": mean_train_loss,
        "macro_f1": breakdown.macro_f1,
        "accuracy": float(accuracy),
        "weighted_f1": breakdown.weighted_f1,
        "classification_breakdown": breakdown.to_dict(),
        "train_runtime_s": train_runtime,
        "eval_runtime_s": eval_runtime,
        "phrasebank_aux_train_loss": mean_aux_loss,
        "phrasebank_aux_lambda": (
            phrasebank_aux_lambda if aux_head is not None else 0.0
        ),
        "phrasebank_aux_rows": (
            len(phrasebank_rows) if (aux_head is not None and phrasebank_rows) else 0
        ),
    }


def _pooled_from_base_model_output(
    outputs: Any, attention_mask: "Any"
) -> "Any":
    """Extract a pooled vector from a HF base-model output.

    BERT-family backbones expose ``pooler_output`` directly; backbones
    without a pooler (e.g. RoBERTa configured without one, DistilBERT)
    fall back to a masked-mean over ``last_hidden_state``. Mean-pool
    keeps the gradient flowing through the encoder for the aux task
    even when the pooler is absent.
    """

    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full (seed x fold) sweep and return the artefact payload."""

    encoder_alias = args.encoder_alias or _resolve_default_encoder_alias()
    package_dir = _resolve_training_package_dir(args.training_package_id)
    rows = load_fomc_rows(package_dir)
    if not rows:
        raise SystemExit("Empty FOMC row set after registry + events join.")

    fold_ids = list(args.folds) if args.folds else _all_fold_ids(package_dir)
    if not fold_ids:
        raise SystemExit(
            f"No folds resolved from {package_dir}; supply --folds explicitly."
        )

    # PhraseBank auxiliary-task rows (#33 Path B). Loaded once and
    # shared across every (seed, fold) cell so the aux pool is constant
    # across the sweep; only the FOMC fold split varies. Cells never
    # use PhraseBank text as their fine-tune validation slice — the
    # aux loader is wholly separate from the FOMC fold's train / test
    # slices, no row indexing crossover.
    phrasebank_rows: list[Any] | None = None
    phrasebank_meta: dict[str, Any] = {"enabled": False}
    aux_lambda = float(getattr(args, "phrasebank_aux_lambda", 0.0))
    if getattr(args, "enable_phrasebank_aux", False):
        if aux_lambda <= 0.0:
            print(
                f"[finetune-pilot-b2] WARN: --enable-phrasebank-aux is set "
                f"but --phrasebank-aux-lambda={aux_lambda} <= 0; treating "
                f"the run as aux-disabled.",
                flush=True,
            )
        else:
            from app.data.phrasebank import (
                class_counts as _pb_class_counts,
                load_phrasebank_rows,
            )

            subset = (
                getattr(args, "phrasebank_subset", None)
                or "sentences_allagree"
            )
            local_jsonl = getattr(args, "phrasebank_jsonl", None)
            cache_root = getattr(args, "phrasebank_cache_root", None)
            phrasebank_rows = load_phrasebank_rows(
                subset=subset,
                local_jsonl=Path(local_jsonl) if local_jsonl else None,
                cache_root=Path(cache_root) if cache_root else None,
            )
            if not phrasebank_rows:
                raise SystemExit(
                    "PhraseBank loader returned no rows; aux flag is on but "
                    "pool is empty."
                )
            phrasebank_meta = {
                "enabled": True,
                "subset": subset,
                "n_rows": len(phrasebank_rows),
                "class_counts": _pb_class_counts(phrasebank_rows),
                "aux_lambda": aux_lambda,
            }

    seed_trials: list[SeedTrial] = []
    all_macro_f1: list[float] = []
    for seed in args.seeds:
        trial = SeedTrial(seed=seed)
        for fold_id in fold_ids:
            fold = _load_fold(package_dir, fold_id)
            train_rows, test_rows = _split_by_fold(rows, fold)
            if not train_rows or not test_rows:
                print(
                    f"[finetune_pilot_b2] seed={seed} fold={fold_id} "
                    f"skipped: train={len(train_rows)} test={len(test_rows)}"
                )
                continue

            train_labels, cutoffs, train_kept = build_partition_classification_targets(
                train_rows, train_rows=train_rows
            )
            test_labels, _, test_kept = build_partition_classification_targets(
                test_rows, train_rows=train_rows
            )
            if not train_labels or not test_labels:
                print(
                    f"[finetune_pilot_b2] seed={seed} fold={fold_id} "
                    "skipped: empty class labels after tertile fit"
                )
                continue

            train_texts = [train_rows[i].text for i in train_kept]
            test_texts = [test_rows[i].text for i in test_kept]

            print(
                f"[finetune_pilot_b2] seed={seed} fold={fold_id} "
                f"train_rows={len(train_texts)} test_rows={len(test_texts)} "
                f"cutoffs={cutoffs} encoder={encoder_alias}",
                flush=True,
            )
            cell_metrics = _train_and_eval_one_cell(
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                encoder_alias=encoder_alias,
                seed=seed,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_length=args.max_length,
                phrasebank_rows=phrasebank_rows,
                phrasebank_aux_lambda=float(
                    getattr(args, "phrasebank_aux_lambda", 0.0)
                ),
            )
            cell = FoldCell(
                seed=seed,
                fold_id=fold_id,
                train_rows=len(train_texts),
                test_rows=len(test_texts),
                tertile_cutoffs=cutoffs,
                train_class_counts=_class_counts(train_labels),
                test_class_counts=_class_counts(test_labels),
                **cell_metrics,
            )
            trial.folds.append(cell)
            all_macro_f1.append(cell.macro_f1)
            print(
                f"[finetune_pilot_b2] seed={seed} fold={fold_id} "
                f"macro_f1={cell.macro_f1:.4f} acc={cell.accuracy:.4f} "
                f"train_runtime={cell.train_runtime_s:.1f}s"
            )
        seed_trials.append(trial)

    summary = {
        "regime_f1_macro": _summary_stats(all_macro_f1),
        "regime_f1_macro_ci": _block_bootstrap_ci(
            all_macro_f1,
            block_size=max(1, len(fold_ids)),
        ),
    }

    payload = {
        "pipeline": "finetune_pilot_b2",
        "training_package_id": args.training_package_id,
        "encoder_alias": encoder_alias,
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "n_classes": N_CLASSES,
        "labels": list(VOL_REGIME_LABELS),
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "phrasebank_aux": phrasebank_meta,
        "trials": [trial.to_dict() for trial in seed_trials],
        "summary": summary,
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package ID under data/processed/<id>.",
    )
    parser.add_argument(
        "--encoder-alias",
        default=None,
        help=(
            "Encoder alias from registry.yaml. Defaults to the "
            "classifier-role encoder per ADR 0019, falling back to "
            f"{DEFAULT_ENCODER_FALLBACK!r}."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Official seed set. Default mirrors docs/benchmark-policy.md.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Subset of walk-forward fold IDs. Defaults to every fold "
            "in fold_manifest_expanding_walk_forward.json."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Fine-tune epochs per cell (default 5).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Train batch size (default 16, fits a 24GB GPU).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate (default 2e-5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (default 0.01).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokeniser max length (default 256).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "artifacts/experiments/finetune_pilot_b2.json."
        ),
    )
    # PhraseBank auxiliary-task knobs (#33 Path B). Default off so the
    # CLI without these flags reproduces pre-#33 B2 byte-identically.
    parser.add_argument(
        "--enable-phrasebank-aux",
        action="store_true",
        help=(
            "Enable the PhraseBank auxiliary 3-way sentiment CE on top "
            "of the vol-regime CE during fine-tune (#33 Path B). Off by "
            "default so the harness reproduces the existing B2 numerics."
        ),
    )
    parser.add_argument(
        "--phrasebank-aux-lambda",
        type=float,
        default=0.3,
        help=(
            "Auxiliary-loss weight applied to the PhraseBank CE term "
            "when --enable-phrasebank-aux is on. Default 0.3 mirrors "
            "the multi-task LSTM-stage default lambdas in "
            "MultiTaskLoss (lambda_factor / lambda_certainty)."
        ),
    )
    parser.add_argument(
        "--phrasebank-subset",
        default="sentences_allagree",
        help=(
            "PhraseBank subset name. Defaults to the strict 100%%-"
            "agreement subset (2 264 rows); operator can override with "
            "'sentences_50agree' for the full 4 840-row pool."
        ),
    )
    parser.add_argument(
        "--phrasebank-cache-root",
        type=Path,
        default=None,
        help=(
            "Override for the on-disk PhraseBank cache root. Defaults "
            "to data/external/phrasebank/."
        ),
    )
    parser.add_argument(
        "--phrasebank-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional local JSONL fixture path for PhraseBank rows. "
            "When supplied the HF read path is skipped — used by tests "
            "and air-gapped reproductions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    # Avoid TorchDynamo on the pod -- the canonical sweep machinery
    # disables it because of a triton mismatch; this harness piggybacks
    # on the same guard.
    from app.training.runtime_compat import ensure_compile_safe

    ensure_compile_safe()
    args = _parse_args()
    payload = run_sweep(args)
    output_path = args.output if args.output is not None else DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[finetune_pilot_b2] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
