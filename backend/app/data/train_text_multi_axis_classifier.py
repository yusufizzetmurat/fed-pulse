"""Train the text-only multi-axis classifier (#78 follow-up).

Pulls the supervised rows out of events.parquet, derives per-axis
targets + masks using the same axis-label normalisation the training
loader uses, then fine-tunes a transformer encoder + MultiTaskHead
end-to-end against the per-axis weighted + masked loss.

Output is a single checkpoint at
``backend/models/text_multi_axis_best.pt`` (the path the inference
service singleton at ``app.services.multi_axis_classifier`` reads on
cold start). The checkpoint envelope records the encoder alias +
revision, the head config, the per-axis class weights fitted on the
train slice, and the best-epoch val metrics so a future run can
inspect provenance without re-running the trainer.

Usage::

    python -m app.data.train_text_multi_axis_classifier \\
        --training-package-id tp_v3_macro_aug_2026_05_23_sentiment_market_core_v1.1_epv1_v1.0 \\
        --encoder-alias finbert_fed_adjacent \\
        --epochs 4 --seed 97
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.config import DATA_DIR, MODEL_CHECKPOINT_DIR
from app.models.config import (
    MULTI_TASK_CERTAINTY_CLASSES,
    MULTI_TASK_CERTAINTY_LABELS,
    MULTI_TASK_STANCE_CLASSES,
    MULTI_TASK_STANCE_LABELS,
    MULTI_TASK_TOPIC_CLASSES,
    MULTI_TASK_TOPIC_LABELS,
)
from app.models.text_multi_axis_classifier import TextMultiAxisClassifier
from app.training.loss import MultiTaskLoss

_logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR / "text_multi_axis_best.pt"
DEFAULT_ENCODER_ALIAS = "finbert_fed_adjacent"


@dataclass
class _AxisRow:
    """One supervised row passed to the classifier.

    ``targets`` and ``masks`` are dicts keyed by axis name so the
    DataLoader collate path can stack them into batched tensors
    without per-axis branching.
    """

    text: str
    targets: dict[str, float | int] = field(default_factory=dict)
    masks: dict[str, bool] = field(default_factory=dict)


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _stance_target(row: dict[str, Any]) -> tuple[int, bool]:
    raw = row.get("axis_stance")
    if isinstance(raw, str) and raw.strip().lower() in MULTI_TASK_STANCE_LABELS:
        return MULTI_TASK_STANCE_LABELS.index(raw.strip().lower()), True
    return 0, False


def _factor_target(row: dict[str, Any]) -> tuple[float, bool]:
    value = _coerce_float(row.get("axis_factor"))
    if value is None:
        return 0.0, False
    return max(min(value, 1.0), -1.0), True


def _certainty_target(row: dict[str, Any]) -> tuple[int, bool]:
    raw = row.get("axis_certain_label")
    if isinstance(raw, str) and raw.strip().lower() in MULTI_TASK_CERTAINTY_LABELS:
        return MULTI_TASK_CERTAINTY_LABELS.index(raw.strip().lower()), True
    float_val = _coerce_float(row.get("axis_certainty"))
    if float_val is None:
        return 0, False
    if float_val >= 0.66:
        return MULTI_TASK_CERTAINTY_LABELS.index("certain"), True
    if float_val <= 0.33:
        return MULTI_TASK_CERTAINTY_LABELS.index("uncertain"), True
    return MULTI_TASK_CERTAINTY_LABELS.index("neutral"), True


# Explicit aliases for upstream topic values that do not contain a
# canonical substring. "economic_indicator" comes off the macro-release
# augmentation (CPI / NFP rows) and is the macro topic by construction.
_TOPIC_ALIASES: dict[str, str] = {
    "economic_indicator": "macro",
    "rate_decision": "forward_guidance",
}


def _topic_target(row: dict[str, Any]) -> tuple[int, bool]:
    raw = row.get("axis_topic")
    if not isinstance(raw, str) or not raw.strip():
        return 0, False
    topic_str = raw.strip().lower()
    if topic_str in _TOPIC_ALIASES:
        return MULTI_TASK_TOPIC_LABELS.index(_TOPIC_ALIASES[topic_str]), True
    for canonical in MULTI_TASK_TOPIC_LABELS[:-1]:
        if canonical in topic_str:
            return MULTI_TASK_TOPIC_LABELS.index(canonical), True
    return MULTI_TASK_TOPIC_LABELS.index("other"), True


def _row_targets(row: dict[str, Any]) -> tuple[dict[str, float | int], dict[str, bool]]:
    """Map one events.parquet row to per-axis targets + masks.

    Mirrors the extraction in
    ``app.training.loaders._attach_rich_features`` so the classifier
    consumes the same canonical label mappings the forecaster does.
    """

    stance_idx, stance_present = _stance_target(row)
    factor_value, factor_present = _factor_target(row)
    certainty_idx, certainty_present = _certainty_target(row)
    topic_idx, topic_present = _topic_target(row)
    return (
        {
            "stance": stance_idx,
            "factor": factor_value,
            "certainty": certainty_idx,
            "topic": topic_idx,
        },
        {
            "stance": stance_present,
            "factor": factor_present,
            "certainty": certainty_present,
            "topic": topic_present,
        },
    )


def _load_supervised_rows(events_parquet: Path) -> list[_AxisRow]:
    """Read events.parquet and keep rows with ≥1 axis label populated."""

    import pandas as pd

    if not events_parquet.exists():
        raise SystemExit(f"events.parquet not found at {events_parquet}")
    frame = pd.read_parquet(events_parquet)
    rows: list[_AxisRow] = []
    for record in frame.to_dict(orient="records"):
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        targets, masks = _row_targets(record)
        if not any(masks.values()):
            continue
        rows.append(_AxisRow(text=text, targets=targets, masks=masks))
    return rows


# Dataset IDs are pinned here in iteration order (FOMC first so
# ``--gtfintechlab-fed-only`` short-circuits cleanly); revisions are
# read off the canonical ``_DATASET_REVISIONS`` map in
# ``app.data.ingest_sources`` at load time so the SHAs do not drift
# between ingestion and training.
_GTFINTECHLAB_DATASET_IDS: tuple[str, ...] = (
    "gtfintechlab/federal_reserve_system",
    "gtfintechlab/european_central_bank",
    "gtfintechlab/bank_of_japan",
    "gtfintechlab/bank_of_england",
    "gtfintechlab/bank_of_canada",
    "gtfintechlab/reserve_bank_of_australia",
)


def _gtfintechlab_row_to_axis_row(item: dict[str, Any]) -> _AxisRow | None:
    """Map a single gtfintechlab sentence-level row to ``_AxisRow``.

    The gtfintechlab schema is uniform across all six bank datasets:
    ``{sentences, stance_label, time_label, certain_label, year}``.
    The trainer currently maps stance and certainty into the
    classifier's heads; ``time_label`` is not yet plumbed (#235
    follow-up). Rows whose stance is ``irrelevant`` (or any non-
    canonical value) are kept but their stance mask stays False so
    the loss does not train on them — they still contribute
    certainty supervision when that axis is populated.
    """

    text = str(item.get("sentences") or "").strip()
    if not text:
        return None

    targets: dict[str, float | int] = {
        "stance": 0,
        "factor": 0.0,
        "certainty": 0,
        "topic": 0,
    }
    masks: dict[str, bool] = {
        "stance": False,
        "factor": False,
        "certainty": False,
        "topic": False,
    }

    stance_raw = str(item.get("stance_label") or "").strip().lower()
    if stance_raw in MULTI_TASK_STANCE_LABELS:
        targets["stance"] = MULTI_TASK_STANCE_LABELS.index(stance_raw)
        masks["stance"] = True

    certain_raw = str(item.get("certain_label") or "").strip().lower()
    if certain_raw in MULTI_TASK_CERTAINTY_LABELS:
        targets["certainty"] = MULTI_TASK_CERTAINTY_LABELS.index(certain_raw)
        masks["certainty"] = True

    # The gtfintechlab corpus does not carry a topic taxonomy and the
    # factor axis is exclusive to the gss_factor source. The classifier
    # learns those branches only from rows that DO carry the label
    # (mask=True); on this dataset both stay at False so the masked
    # loss contributes nothing for them. The branches still emit
    # predictions at inference, which the frontend can choose to
    # render as low-confidence.
    if not any(masks.values()):
        return None
    return _AxisRow(text=text, targets=targets, masks=masks)


def _gtfintechlab_datasets_module() -> Any:
    """Import the ``datasets`` package lazily so the trainer module
    stays cheap to import on hosts without the HF stack installed."""

    try:
        from datasets import (  # type: ignore
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
        )
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "datasets package is required for the gtfintechlab loader. "
            "Install dependencies first."
        ) from exc
    return type(
        "_HFDatasetsModule",
        (),
        {
            "get_dataset_config_names": staticmethod(get_dataset_config_names),
            "get_dataset_split_names": staticmethod(get_dataset_split_names),
            "load_dataset": staticmethod(load_dataset),
        },
    )


def _gtfintechlab_split_rows(
    hf_mod: Any,
    *,
    dataset_id: str,
    config: str,
    split: str,
    revision: str,
) -> list[_AxisRow]:
    """Materialise one ``(dataset, config, split)`` triple's rows.

    Any HF / network failure is caught and logged; the caller treats
    a returned empty list as "skip this slice" so a single bad split
    cannot abort the rest of the 18-slice walk.
    """

    try:
        ds = hf_mod.load_dataset(dataset_id, config, split=split, revision=revision)
    except Exception:
        _logger.exception(
            "gtfintechlab_split_load_failed dataset=%s config=%s split=%s",
            dataset_id,
            config,
            split,
        )
        return []
    out: list[_AxisRow] = []
    for record in ds:
        row = _gtfintechlab_row_to_axis_row(dict(record))
        if row is not None:
            out.append(row)
    return out


def _gtfintechlab_dataset_rows(
    hf_mod: Any, *, dataset_id: str, revision: str
) -> list[_AxisRow]:
    """Walk every (config, split) under one gtfintechlab dataset."""

    try:
        configs = list(hf_mod.get_dataset_config_names(dataset_id, revision=revision))
    except Exception:
        _logger.exception("gtfintechlab_configs_failed dataset=%s", dataset_id)
        return []
    rows: list[_AxisRow] = []
    for config in configs:
        try:
            splits = list(
                hf_mod.get_dataset_split_names(dataset_id, config, revision=revision)
            )
        except Exception:
            _logger.exception(
                "gtfintechlab_splits_failed dataset=%s config=%s",
                dataset_id,
                config,
            )
            continue
        for split in splits:
            rows.extend(
                _gtfintechlab_split_rows(
                    hf_mod,
                    dataset_id=dataset_id,
                    config=config,
                    split=split,
                    revision=revision,
                )
            )
    return rows


def _load_gtfintechlab_rows(*, fed_only: bool = False) -> list[_AxisRow]:
    """Pull supervised sentence-level rows from the gtfintechlab HF datasets.

    Iterates every (config, split) under each dataset to materialise
    the full 18 000-row corpus (1 000 sentences × 3 configs × 6 banks).
    With ``fed_only=True`` the loader restricts to the
    ``federal_reserve_system`` subset (~3 000 rows) for FOMC-specific
    fine-tuning at the cost of a smaller, more imbalanced training pool.
    """

    hf_mod = _gtfintechlab_datasets_module()

    # Pull the pinned revisions from the canonical map in
    # ``app.data.ingest_sources``; that file is the single source of
    # truth for every dataset SHA the project consumes, so the
    # training side stays aligned with the ingestion side.
    from app.data.ingest_sources import _dataset_revision

    dataset_ids = _GTFINTECHLAB_DATASET_IDS[:1] if fed_only else _GTFINTECHLAB_DATASET_IDS
    rows: list[_AxisRow] = []
    for dataset_id in dataset_ids:
        revision = _dataset_revision(dataset_id)
        if revision is None:
            _logger.warning(
                "gtfintechlab_revision_missing dataset=%s (skipping)", dataset_id
            )
            continue
        rows.extend(
            _gtfintechlab_dataset_rows(
                hf_mod, dataset_id=dataset_id, revision=revision
            )
        )
    _logger.info(
        "loaded_gtfintechlab_rows total=%d fed_only=%s", len(rows), fed_only
    )
    return rows


def _fit_class_weights(
    indices: list[int],
    n_classes: int,
    *,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency weights normalised to sum to ``n_classes``.

    Mirrors :func:`app.training.loaders.fit_class_weights`; the classifier
    fits one weight tensor per axis on its masked rows.
    """

    counts = [0] * n_classes
    for idx in indices:
        if 0 <= idx < n_classes:
            counts[idx] += 1
    if sum(counts) == 0:
        return torch.ones(n_classes, dtype=torch.float32)
    raw = [1.0 / (c + smoothing) for c in counts]
    total = sum(raw)
    return torch.tensor(
        [(w / total) * n_classes for w in raw], dtype=torch.float32
    )


def _build_axis_class_weights(rows: list[_AxisRow]) -> dict[str, torch.Tensor]:
    """Per-axis class weights fitted only on rows where the axis is masked True."""

    stance_indices = [int(r.targets["stance"]) for r in rows if r.masks["stance"]]
    certainty_indices = [int(r.targets["certainty"]) for r in rows if r.masks["certainty"]]
    topic_indices = [int(r.targets["topic"]) for r in rows if r.masks["topic"]]
    return {
        "stance": _fit_class_weights(stance_indices, MULTI_TASK_STANCE_CLASSES),
        "certainty": _fit_class_weights(certainty_indices, MULTI_TASK_CERTAINTY_CLASSES),
        "topic": _fit_class_weights(topic_indices, MULTI_TASK_TOPIC_CLASSES),
    }


class _MultiAxisDataset(Dataset):
    def __init__(self, rows: list[_AxisRow], tokenizer: Any, max_length: int = 256) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_stance": int(row.targets["stance"]),
            "target_factor": float(row.targets["factor"]),
            "target_certainty": int(row.targets["certainty"]),
            "target_topic": int(row.targets["topic"]),
            "mask_stance": bool(row.masks["stance"]),
            "mask_factor": bool(row.masks["factor"]),
            "mask_certainty": bool(row.masks["certainty"]),
            "mask_topic": bool(row.masks["topic"]),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    out["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    out["target_stance"] = torch.tensor(
        [item["target_stance"] for item in batch], dtype=torch.long
    )
    out["target_factor"] = torch.tensor(
        [item["target_factor"] for item in batch], dtype=torch.float32
    )
    out["target_certainty"] = torch.tensor(
        [item["target_certainty"] for item in batch], dtype=torch.long
    )
    out["target_topic"] = torch.tensor(
        [item["target_topic"] for item in batch], dtype=torch.long
    )
    out["mask_stance"] = torch.tensor(
        [item["mask_stance"] for item in batch], dtype=torch.bool
    )
    out["mask_factor"] = torch.tensor(
        [item["mask_factor"] for item in batch], dtype=torch.bool
    )
    out["mask_certainty"] = torch.tensor(
        [item["mask_certainty"] for item in batch], dtype=torch.bool
    )
    out["mask_topic"] = torch.tensor(
        [item["mask_topic"] for item in batch], dtype=torch.bool
    )
    return out


def _train_one_epoch(
    model: TextMultiAxisClassifier,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sum_total = 0.0
    sum_axis = {"stance": 0.0, "factor": 0.0, "certainty": 0.0, "topic": 0.0}
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn)
        targets = {
            "stance": batch["target_stance"].to(device),
            "factor": batch["target_factor"].to(device),
            "certainty": batch["target_certainty"].to(device),
            "topic": batch["target_topic"].to(device),
        }
        masks = {
            "stance_mask": batch["mask_stance"].to(device),
            "factor_mask": batch["mask_factor"].to(device),
            "certainty_mask": batch["mask_certainty"].to(device),
            "topic_mask": batch["mask_topic"].to(device),
        }
        total, breakdown = loss_fn(logits, targets, masks)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        sum_total += float(total.detach().item())
        for axis_name in sum_axis:
            sum_axis[axis_name] += float(breakdown[axis_name].item())
        n_batches += 1
    if n_batches == 0:
        return {"loss": 0.0}
    out = {"loss": sum_total / n_batches}
    for axis_name, total in sum_axis.items():
        out[f"loss_{axis_name}"] = total / n_batches
    return out


@torch.no_grad()
def _evaluate(
    model: TextMultiAxisClassifier,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sum_total = 0.0
    sum_axis = {"stance": 0.0, "factor": 0.0, "certainty": 0.0, "topic": 0.0}
    correct_axis = {"stance": 0, "certainty": 0, "topic": 0}
    seen_axis = {"stance": 0, "certainty": 0, "topic": 0}
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn)
        targets = {
            "stance": batch["target_stance"].to(device),
            "factor": batch["target_factor"].to(device),
            "certainty": batch["target_certainty"].to(device),
            "topic": batch["target_topic"].to(device),
        }
        masks = {
            "stance_mask": batch["mask_stance"].to(device),
            "factor_mask": batch["mask_factor"].to(device),
            "certainty_mask": batch["mask_certainty"].to(device),
            "topic_mask": batch["mask_topic"].to(device),
        }
        total, breakdown = loss_fn(logits, targets, masks)
        sum_total += float(total.item())
        for axis_name in sum_axis:
            sum_axis[axis_name] += float(breakdown[axis_name].item())
        n_batches += 1
        for axis_name in ("stance", "certainty", "topic"):
            mask = masks[f"{axis_name}_mask"]
            if mask.any():
                pred = logits[axis_name][mask].argmax(dim=-1)
                tgt = targets[axis_name][mask]
                correct_axis[axis_name] += int((pred == tgt).sum().item())
                seen_axis[axis_name] += int(mask.sum().item())
    if n_batches == 0:
        return {"loss": 0.0}
    out = {"loss": sum_total / n_batches}
    for axis_name, total in sum_axis.items():
        out[f"loss_{axis_name}"] = total / n_batches
    for axis_name in ("stance", "certainty", "topic"):
        if seen_axis[axis_name] > 0:
            out[f"acc_{axis_name}"] = correct_axis[axis_name] / seen_axis[axis_name]
    return out


def _save_checkpoint(
    model: TextMultiAxisClassifier,
    *,
    path: Path,
    metrics: dict[str, float],
    args: argparse.Namespace,
    class_weights: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": model.metadata(),
        "metrics": metrics,
        "class_weights": {k: v.tolist() for k, v in class_weights.items()},
        "training_args": {
            "training_package_id": args.training_package_id,
            "encoder_alias": args.encoder_alias,
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "val_fraction": args.val_fraction,
            "max_length": args.max_length,
            # Data-selection flags so the checkpoint payload records
            # which corpus the model was trained on. Without these,
            # a gtfintechlab-trained checkpoint looks identical on
            # disk to an events_parquet-trained one, blocking
            # reproducibility audits.
            "data_source": getattr(args, "data_source", "events_parquet"),
            "gtfintechlab_fed_only": bool(
                getattr(args, "gtfintechlab_fed_only", False)
            ),
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)
    _logger.info("checkpoint_written path=%s", path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    _set_all_seeds(args.seed)

    if args.data_source == "gtfintechlab_hf":
        rows = _load_gtfintechlab_rows(fed_only=args.gtfintechlab_fed_only)
        if not rows:
            raise SystemExit(
                "gtfintechlab loader returned zero rows; check network access + "
                "datasets package install."
            )
    else:
        if not args.training_package_id:
            raise SystemExit(
                "--training-package-id is required when --data-source=events_parquet"
            )
        package_dir = DATA_DIR / "processed" / args.training_package_id
        events_path = package_dir / "events.parquet"
        rows = _load_supervised_rows(events_path)
        if not rows:
            raise SystemExit(f"No supervised rows with axis labels in {events_path}")
    _logger.info("loaded_supervised_rows count=%d", len(rows))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * args.val_fraction))
    train_rows = rows[val_size:]
    val_rows = rows[:val_size]
    _logger.info(
        "split rows total=%d train=%d val=%d",
        len(rows),
        len(train_rows),
        len(val_rows),
    )

    class_weights = _build_axis_class_weights(train_rows)

    from transformers import AutoTokenizer

    from app.models.registry import encoder_ref

    ref = encoder_ref(args.encoder_alias)
    if ref is None or not ref.revision:
        raise SystemExit(
            f"Encoder alias {args.encoder_alias!r} is unpinned in registry.yaml; "
            "add a revision before training."
        )
    tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextMultiAxisClassifier.from_encoder_alias(
        encoder_alias=args.encoder_alias,
        head_hidden_size=args.head_hidden_size,
        dropout=args.dropout,
    )
    model.to(device)

    loss_fn = MultiTaskLoss(
        stance_weight=class_weights["stance"].to(device),
        certainty_weight=class_weights["certainty"].to(device),
        topic_weight=class_weights["topic"].to(device),
        lambda_stance=args.lambda_stance,
        lambda_factor=args.lambda_factor,
        lambda_certainty=args.lambda_certainty,
        lambda_topic=args.lambda_topic,
    ).to(device)

    train_ds = _MultiAxisDataset(train_rows, tokenizer, max_length=args.max_length)
    val_ds = _MultiAxisDataset(val_rows, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    no_decay = {"bias", "LayerNorm.weight"}
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        train_metrics = _train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = _evaluate(model, val_loader, loss_fn, device)
        _logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc_stance=%.3f val_acc_certainty=%.3f val_acc_topic=%.3f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics.get("acc_stance", float("nan")),
            val_metrics.get("acc_certainty", float("nan")),
            val_metrics.get("acc_topic", float("nan")),
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            _save_checkpoint(
                model,
                path=Path(args.output_checkpoint),
                metrics=best_metrics,
                args=args,
                class_weights=class_weights,
            )

    _logger.info("training_complete best_val_loss=%.4f", best_val_loss)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=("events_parquet", "gtfintechlab_hf"),
        default="gtfintechlab_hf",
        help=(
            "Source of the supervised rows. ``gtfintechlab_hf`` (default) pulls "
            "balanced sentence-level rows from the 6 gtfintechlab central-bank "
            "datasets on HuggingFace (~18 000 rows, ~5 k each stance class). "
            "``events_parquet`` reads from the supplied training package's "
            "events.parquet — useful for FOMC-specific fine-tuning but suffers "
            "from heavy class imbalance (92 % neutral on the current pool)."
        ),
    )
    parser.add_argument(
        "--gtfintechlab-fed-only",
        action="store_true",
        help=(
            "Restrict the gtfintechlab loader to the federal_reserve_system "
            "subset (~3 000 rows) instead of all 6 banks."
        ),
    )
    parser.add_argument(
        "--training-package-id",
        default="",
        help=(
            "Required when --data-source=events_parquet; ignored otherwise."
        ),
    )
    parser.add_argument("--encoder-alias", default=DEFAULT_ENCODER_ALIAS)
    parser.add_argument(
        "--output-checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Destination .pt path for the best-epoch checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--head-hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-stance", type=float, default=1.0)
    parser.add_argument("--lambda-factor", type=float, default=0.3)
    parser.add_argument("--lambda-certainty", type=float, default=0.3)
    parser.add_argument("--lambda-topic", type=float, default=0.3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
