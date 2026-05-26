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
DEFAULT_ARTIFACT_ROOT = DATA_DIR / "artifacts" / "text_multi_axis"


@dataclass
class _AxisRow:
    """One supervised row passed to the classifier.

    ``targets`` and ``masks`` are dicts keyed by axis name so the
    DataLoader collate path can stack them into batched tensors
    without per-axis branching. ``source`` carries the originating
    corpus (e.g. ``"gtfintechlab/federal_reserve_system"``,
    ``"events_parquet"``) so the eval pass can slice per-bank metrics
    (D from the 2026-05-24 plan). ``provenance`` mirrors the
    registry's provenance bucket (e.g. ``"peer_reviewed"``,
    ``"peer_reviewed_cross_bank"``) so the cross-bank supervision flag
    can scope its mask + weight rewrites without re-deriving the
    bucket from ``source``. ``stance_sample_weight`` is a per-row
    multiplier on the stance branch's loss; it stays 1.0 for every
    FOMC row and is scaled down for cross-bank rows under the
    ``weighted`` arm of ``--cross-bank-supervision``.
    """

    text: str
    targets: dict[str, float | int] = field(default_factory=dict)
    masks: dict[str, bool] = field(default_factory=dict)
    source: str = ""
    provenance: str = ""
    stance_sample_weight: float = 1.0


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
        source = str(record.get("source") or "events_parquet")
        # ``events.parquet`` does not carry ``provenance`` directly —
        # the event builder collapses by (source, event_date, kind),
        # dropping the per-row provenance column. Cross-bank rows do
        # not reach this path today, so a constant "" is correct: the
        # cross-bank supervision flag is a no-op for events_parquet
        # by design and the per-axis sanity log will record zero
        # cross-bank rows on this corpus.
        rows.append(
            _AxisRow(
                text=text,
                targets=targets,
                masks=masks,
                source=source,
                provenance="",
                stance_sample_weight=1.0,
            )
        )
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


def _gtfintechlab_row_to_axis_row(
    item: dict[str, Any],
    *,
    source: str = "gtfintechlab",
    provenance: str = "",
    cross_bank_mode: str = "off",
    cross_bank_stance_weight: float = 1.0,
) -> _AxisRow | None:
    """Map a single gtfintechlab sentence-level row to ``_AxisRow``.

    The gtfintechlab schema is uniform across all six bank datasets:
    ``{sentences, stance_label, time_label, certain_label, year}``.
    The trainer currently maps stance and certainty into the
    classifier's heads; ``time_label`` is not yet plumbed (#235
    follow-up). Rows whose stance is ``irrelevant`` (or any non-
    canonical value) are kept but their stance mask stays False so
    the loss does not train on them — they still contribute
    certainty supervision when that axis is populated.

    ``cross_bank_mode`` rewrites the per-row stance mask + weight for
    rows whose ``provenance == "peer_reviewed_cross_bank"`` (see
    ``--cross-bank-supervision`` on the CLI). The rewrite happens
    here, at row materialisation, so every downstream layer reads the
    corrected mask without further special-casing.
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
    stance_sample_weight = 1.0
    if provenance == "peer_reviewed_cross_bank":
        if cross_bank_mode == "stance_masked":
            # Substitute-not-complement guard: drop the stance label
            # so the head does not fit the cross-bank stance
            # distribution. Other axes keep their natural per-row
            # masks so the encoder still trains on the auxiliary
            # signal.
            masks["stance"] = False
        elif cross_bank_mode == "weighted":
            # Diagnostic A/B: keep the stance mask so the head sees
            # the cross-bank labels, but scale the row's contribution
            # to the stance loss down so it does not dominate the
            # FOMC distribution.
            stance_sample_weight = float(cross_bank_stance_weight)
    if not any(masks.values()):
        return None
    return _AxisRow(
        text=text,
        targets=targets,
        masks=masks,
        source=source,
        provenance=provenance,
        stance_sample_weight=stance_sample_weight,
    )


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


def _gtfintechlab_split_rows(  # noqa: PLR0913 — keyword-only HF load coords plus cross-bank context; grouping would obscure call sites.
    hf_mod: Any,
    *,
    dataset_id: str,
    config: str,
    split: str,
    revision: str,
    provenance: str = "",
    cross_bank_mode: str = "off",
    cross_bank_stance_weight: float = 1.0,
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
        row = _gtfintechlab_row_to_axis_row(
            dict(record),
            source=dataset_id,
            provenance=provenance,
            cross_bank_mode=cross_bank_mode,
            cross_bank_stance_weight=cross_bank_stance_weight,
        )
        if row is not None:
            out.append(row)
    return out


def _gtfintechlab_dataset_rows(  # noqa: PLR0913 — same cross-bank context as _gtfintechlab_split_rows; grouping into a dataclass would obscure call sites.
    hf_mod: Any,
    *,
    dataset_id: str,
    revision: str,
    provenance: str = "",
    cross_bank_mode: str = "off",
    cross_bank_stance_weight: float = 1.0,
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
                    provenance=provenance,
                    cross_bank_mode=cross_bank_mode,
                    cross_bank_stance_weight=cross_bank_stance_weight,
                )
            )
    return rows


def _provenance_for_gtfintechlab_dataset(dataset_id: str) -> str:
    """Bucket each gtfintechlab dataset into the registry's provenance vocab.

    The FOMC dataset is part of the supervised pool (``peer_reviewed``,
    sample_weight 1.0). The other five central-bank datasets enter
    the cross-bank generalisation pool (``peer_reviewed_cross_bank``,
    sample_weight 0.0). This mirrors the same bucketing
    ``ingest_sources._iter_gtfintechlab_cross_bank_records`` writes
    onto the registry so the trainer's view of provenance stays in
    sync with the registry view.
    """

    if dataset_id == _GTFINTECHLAB_DATASET_IDS[0]:
        return "peer_reviewed"
    return "peer_reviewed_cross_bank"


def _load_gtfintechlab_rows(
    *,
    fed_only: bool = False,
    cross_bank_mode: str = "off",
    cross_bank_stance_weight: float = 1.0,
) -> list[_AxisRow]:
    """Pull supervised sentence-level rows from the gtfintechlab HF datasets.

    Iterates every (config, split) under each dataset to materialise
    the full 18 000-row corpus (1 000 sentences × 3 configs × 6 banks).
    With ``fed_only=True`` the loader restricts to the
    ``federal_reserve_system`` subset (~3 000 rows) for FOMC-specific
    fine-tuning at the cost of a smaller, more imbalanced training pool.

    ``cross_bank_mode`` controls how rows from the five non-FOMC
    central-bank datasets (provenance ``peer_reviewed_cross_bank``)
    enter the supervised pool. ``off`` (default) drops them entirely
    — the loader walks the FOMC dataset only, reproducing the
    strict-FOMC training pool byte-identically. ``stance_masked``
    admits them with their stance mask forced to False so the head
    only fits the FOMC stance distribution while the encoder still
    sees the cross-bank text. ``weighted`` admits them with the
    natural stance label and downscales their stance loss
    contribution by ``cross_bank_stance_weight``.
    """

    hf_mod = _gtfintechlab_datasets_module()

    # Pull the pinned revisions from the canonical map in
    # ``app.data.ingest_sources``; that file is the single source of
    # truth for every dataset SHA the project consumes, so the
    # training side stays aligned with the ingestion side.
    from app.data.ingest_sources import _dataset_revision

    if fed_only or cross_bank_mode == "off":
        dataset_ids = _GTFINTECHLAB_DATASET_IDS[:1]
    else:
        dataset_ids = _GTFINTECHLAB_DATASET_IDS
    rows: list[_AxisRow] = []
    for dataset_id in dataset_ids:
        revision = _dataset_revision(dataset_id)
        if revision is None:
            _logger.warning(
                "gtfintechlab_revision_missing dataset=%s (skipping)", dataset_id
            )
            continue
        provenance = _provenance_for_gtfintechlab_dataset(dataset_id)
        rows.extend(
            _gtfintechlab_dataset_rows(
                hf_mod,
                dataset_id=dataset_id,
                revision=revision,
                provenance=provenance,
                cross_bank_mode=cross_bank_mode,
                cross_bank_stance_weight=cross_bank_stance_weight,
            )
        )
    _logger.info(
        "loaded_gtfintechlab_rows total=%d fed_only=%s cross_bank_mode=%s",
        len(rows),
        fed_only,
        cross_bank_mode,
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
            # Source bank label rides on every batch row so the eval
            # pass can compute per-bank macro-F1 without re-loading
            # the original rows. Collated as a list of strings.
            "source": row.source,
            # Provenance bucket from the registry (e.g.
            # ``peer_reviewed``, ``peer_reviewed_cross_bank``). Used
            # by the first-epoch sanity log to split per-axis row
            # counts by corpus origin.
            "provenance": row.provenance,
            # Per-row multiplier on the stance branch's loss. Stays
            # 1.0 for FOMC rows; the ``weighted`` cross-bank arm
            # scales cross-bank rows down so they do not dominate
            # the FOMC stance distribution.
            "stance_sample_weight": float(row.stance_sample_weight),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
    # ``source`` and ``provenance`` are per-row strings (the per-bank
    # eval and the first-epoch sanity log read them); the rest of the
    # batch is torch.Tensor. The mixed-type return signature is
    # documented on _collate's annotation so the type hint matches
    # the actual payload. ``stance_sample_weight`` is a per-row
    # float32 vector consumed by the training loop when the
    # ``weighted`` cross-bank arm is active; FOMC rows stay at 1.0.
    out["source"] = [str(item.get("source", "")) for item in batch]
    out["provenance"] = [str(item.get("provenance", "")) for item in batch]
    out["stance_sample_weight"] = torch.tensor(
        [float(item.get("stance_sample_weight", 1.0)) for item in batch],
        dtype=torch.float32,
    )
    return out


def _log_per_axis_provenance_breakdown(rows: list[_AxisRow]) -> None:
    """One-shot sanity log of per-axis training-row counts by provenance.

    Emitted once at the start of training so any regression that
    leaks cross-bank rows into the stance head shows up as a
    non-zero ``from_cross_bank`` column on the stance line. The log
    matches the project's existing ``key=value key=value`` info-log
    style so downstream parsers (and the wiki's training-summary
    extractor) stay uniform.

    The non-cross-bank bucket is named ``from_other`` (not
    ``from_FOMC``) so a future provenance (e.g. ``scraped_cross_bank``
    or an events_parquet row carrying provenance="") that is neither
    FOMC nor the recognised cross-bank tag does not silently inflate
    the FOMC counter. Today the supervised pool is dominated by FOMC
    rows under the ``peer_reviewed`` tag, but the bucket name should
    not encode that as a permanent assumption.
    """

    axis_names: tuple[str, ...] = ("stance", "factor", "certainty", "topic")
    for axis_name in axis_names:
        from_other = 0
        from_cross_bank = 0
        for row in rows:
            if not row.masks.get(axis_name, False):
                continue
            if row.provenance == "peer_reviewed_cross_bank":
                from_cross_bank += 1
            else:
                from_other += 1
        total = from_other + from_cross_bank
        _logger.info(
            "axis=%s rows_total=%d from_other=%d from_cross_bank=%d",
            axis_name,
            total,
            from_other,
            from_cross_bank,
        )


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
        stance_weight = batch.get("stance_sample_weight")
        if stance_weight is not None:
            stance_weight = stance_weight.to(device)
        total, breakdown = _compute_weighted_total_loss(
            loss_fn=loss_fn,
            logits=logits,
            targets=targets,
            masks=masks,
            stance_sample_weight=stance_weight,
        )
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


def _compute_weighted_total_loss(
    *,
    loss_fn: MultiTaskLoss,
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    stance_sample_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward through ``MultiTaskLoss`` with the per-row stance weight.

    ``MultiTaskLoss.forward`` takes an optional ``stance_sample_weight``
    kwarg that turns the stance branch into a weighted-mean CE; the
    factor / certainty / topic branches are unchanged. Passing the
    vector straight through keeps every gradient path on the loss
    side, so the helper here is a thin call-site adapter: when the
    batch carries no weights, or every weight is exactly 1.0 (the
    only state the FOMC-only pool ever produces), we call the loss
    without the kwarg so the numerics reproduce the prior path
    byte-identically.

    Note on correctness: an earlier draft of this helper computed the
    weighted stance loss outside the loss and then subtracted
    ``breakdown["stance"]`` (a detached scalar) from the original
    ``total``. That left the original stance gradient path through
    ``logits["stance"]`` accumulated alongside the new weighted path
    — a silent double-count. The current path delegates the swap to
    ``MultiTaskLoss`` itself so there is exactly one stance loss term
    in the computation graph.
    """

    if stance_sample_weight is None:
        return loss_fn(logits, targets, masks)
    if torch.all(stance_sample_weight == 1.0):
        return loss_fn(logits, targets, masks)
    return loss_fn(
        logits, targets, masks, stance_sample_weight=stance_sample_weight
    )


@torch.no_grad()
def _macro_f1_from_arrays(
    predictions: list[int], targets: list[int], n_classes: int
) -> float:
    """Plain unweighted macro-F1 over the supplied row-level arrays.

    Avoids a heavy sklearn dependency on what is otherwise a small
    inline computation. Empty inputs return ``0.0``. Per-class F1 is
    ``0.0`` when both precision and recall vanish (a class never
    appears in either predictions or targets).
    """

    if not predictions or not targets or len(predictions) != len(targets):
        return 0.0
    f1_sum = 0.0
    for cls in range(n_classes):
        tp = sum(1 for p, t in zip(predictions, targets) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(predictions, targets) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(predictions, targets) if p != cls and t == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        f1_sum += f1
    return f1_sum / max(n_classes, 1)


@torch.no_grad()
def _evaluate_per_bank(
    model: TextMultiAxisClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-source per-axis macro-F1 breakdown on the eval partition (D, #209).

    Returns ``{source: {axis: {macro_f1, n}}}`` where ``source`` is the
    originating corpus tag (e.g. ``"gtfintechlab/federal_reserve_system"``,
    ``"events_parquet"``) and ``axis`` ∈ ``{stance, certainty, topic}``.
    Factor is regression-typed and is omitted here; the parent
    ``_evaluate`` already reports its SmoothL1 loss.

    Rows whose axis mask is False on a given (source, axis) are
    dropped before computing F1 — keeps the macro-F1 honest on the
    sparse axes (factor / certainty / topic) where most rows from
    most banks have nothing to score.
    """

    model.eval()
    per_source_axis: dict[str, dict[str, list[tuple[int, int]]]] = {}
    classification_axes: tuple[str, ...] = ("stance", "certainty", "topic")
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn)
        sources = batch["source"]
        for axis_name in classification_axes:
            mask = batch[f"mask_{axis_name}"]
            targets = batch[f"target_{axis_name}"]
            preds = logits[axis_name].argmax(dim=-1).detach().to("cpu")
            for i in range(len(sources)):
                if not bool(mask[i].item()):
                    continue
                source = str(sources[i])
                bucket = per_source_axis.setdefault(source, {})
                axis_bucket = bucket.setdefault(axis_name, [])
                axis_bucket.append((int(preds[i].item()), int(targets[i].item())))

    axis_n_classes = {
        "stance": MULTI_TASK_STANCE_CLASSES,
        "certainty": MULTI_TASK_CERTAINTY_CLASSES,
        "topic": MULTI_TASK_TOPIC_CLASSES,
    }
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for source, axis_bucket in per_source_axis.items():
        out[source] = {}
        for axis_name, rows in axis_bucket.items():
            preds = [p for p, _t in rows]
            tgts = [t for _p, t in rows]
            out[source][axis_name] = {
                "macro_f1": _macro_f1_from_arrays(preds, tgts, axis_n_classes[axis_name]),
                "n": len(rows),
            }
    return out


@torch.no_grad()
def _evaluate(
    model: TextMultiAxisClassifier,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    """Eval loop that mirrors the train loop's loss formula exactly.

    The decorator suppresses autograd graph construction for the whole
    forward + loss pass (``model.eval()`` only flips dropout/BN, it does
    NOT disable autograd) — without it, the per-batch forward retained a
    computation graph that was never backpropagated, ballooning memory.

    The per-row ``stance_sample_weight`` from the batch is threaded into
    ``loss_fn`` so the val loss tracks the same objective the optimizer
    is minimizing on the train side. Otherwise (under
    ``--cross-bank-supervision=weighted``) the train path runs weighted
    CE on stance while the val path runs plain unweighted CE, and
    best-checkpoint selection by ``val_loss`` picks against the wrong
    objective.
    """

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
        stance_weight = batch.get("stance_sample_weight")
        if stance_weight is not None:
            stance_weight = stance_weight.to(device)
        total, breakdown = _compute_weighted_total_loss(
            loss_fn=loss_fn,
            logits=logits,
            targets=targets,
            masks=masks,
            stance_sample_weight=stance_weight,
        )
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


def _save_hf_encoder_directory(
    model: TextMultiAxisClassifier,
    tokenizer: Any,
    checkpoint_dir: Path,
) -> None:
    """Persist the encoder backbone + tokenizer as a HF-format directory.

    Saves ONLY the encoder backbone (``model.encoder``), not the
    ``TextMultiAxisClassifier`` wrapper — the registry consumes the
    bare HF encoder via ``AutoModel.from_pretrained`` and the
    multi-task head is irrelevant to downstream callers
    (embedding-cache builder, forecaster). The resulting directory
    follows the same convention as
    ``app.data.finetune_pilot``'s ``hf_checkpoints/`` so future
    tooling can find HF dirs the same way regardless of which trainer
    produced them.
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))


def _factor_coverage_fraction(rows: list[_AxisRow]) -> float:
    """Fraction of supervised rows that carried a populated factor label.

    Issue #328: the inference service uses this stamp to gate the
    factor card on the /analyze response. A canonical training pool
    today carries 0 % factor coverage (the gss_factor source rows are
    not joined into the events.parquet aggregation), so the regression
    head emits effectively-random values; gating the response on the
    persisted coverage is how the surface stops rendering noise as a
    prediction. An empty row list returns 0.0 so a misconfigured run
    that produces no rows still trips the gate.
    """

    if not rows:
        return 0.0
    populated = sum(1 for r in rows if bool(r.masks.get("factor", False)))
    return populated / len(rows)


def _save_checkpoint(  # noqa: PLR0913 — kw-only checkpoint envelope; collapsing the metadata kwargs into a single dataclass would obscure the persisted fields
    model: TextMultiAxisClassifier,
    *,
    path: Path,
    metrics: dict[str, float],
    args: argparse.Namespace,
    class_weights: dict[str, torch.Tensor],
    factor_coverage: float,
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
            # Bundle A.1: record the cross-bank arm + weight so a
            # checkpoint trained under ``stance_masked`` cannot be
            # confused with one trained under ``weighted`` on disk.
            "cross_bank_supervision": str(
                getattr(args, "cross_bank_supervision", "off")
            ),
            "cross_bank_stance_weight": float(
                getattr(args, "cross_bank_stance_weight", 0.25)
            ),
            # When ``--gtfintechlab-fed-only`` is combined with a
            # non-``off`` cross-bank arm, ``main()`` warns and the
            # loader restricts to FOMC — so the run did NOT actually
            # use cross-bank supervision regardless of what was
            # requested. Surface the resolved effective mode alongside
            # the raw request so the checkpoint provenance is honest:
            # the raw field traces what the operator asked for, the
            # effective field traces what the run actually did.
            "effective_cross_bank_supervision": (
                "off"
                if bool(getattr(args, "gtfintechlab_fed_only", False))
                else str(getattr(args, "cross_bank_supervision", "off"))
            ),
            # Issue #328: persist the fraction of train rows that
            # carried a populated ``axis_factor`` label so the
            # inference service can gate the factor card on it.
            # Coverage < threshold (default 0.01) drops the card
            # from the /analyze response — the factor branch has
            # trained almost exclusively on the masked-out path on
            # those checkpoints and its outputs are noise. ADR 0018
            # captures the decision.
            "factor_coverage": float(factor_coverage),
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)
    _logger.info(
        "checkpoint_written path=%s factor_coverage=%.4f", path, factor_coverage
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    _set_all_seeds(args.seed)

    if args.data_source == "gtfintechlab_hf":
        # Conflict warning: ``--gtfintechlab-fed-only`` restricts the
        # loader to the FOMC dataset, which leaves the cross-bank
        # supervision arm with zero rows to act on — the run log would
        # otherwise advertise ``cross_bank_mode=stance_masked`` /
        # ``weighted`` while silently doing nothing. Fed-only wins by
        # design (it's the explicit FOMC-restriction switch); the
        # warning lets the caller resolve the conflict on their next
        # run instead of being misled by the cross-bank flag's
        # appearance in the args.
        if args.gtfintechlab_fed_only and args.cross_bank_supervision != "off":
            _logger.warning(
                "cross_bank_supervision_noop "
                "--gtfintechlab-fed-only=True --cross-bank-supervision=%s "
                "fed-only restricts the loader to the FOMC dataset so the "
                "cross-bank arm has no rows to act on; fed-only wins.",
                args.cross_bank_supervision,
            )
        rows = _load_gtfintechlab_rows(
            fed_only=args.gtfintechlab_fed_only,
            cross_bank_mode=args.cross_bank_supervision,
            cross_bank_stance_weight=args.cross_bank_stance_weight,
        )
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
    # Compute factor coverage on the train slice (the supervision the
    # head actually trained on, not the whole corpus). Persisted onto
    # the checkpoint payload so the inference service can gate the
    # factor card on it (#328 / ADR 0018).
    factor_coverage = _factor_coverage_fraction(train_rows)
    _logger.info(
        "factor_axis_coverage train_rows=%d populated=%d coverage=%.4f",
        len(train_rows),
        sum(1 for r in train_rows if bool(r.masks.get("factor", False))),
        factor_coverage,
    )

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

    _log_per_axis_provenance_breakdown(train_rows)

    # Per-run HF artifact directory. The singleton .pt at
    # ``args.output_checkpoint`` stays the inference-service contract;
    # this directory carries the encoder backbone + tokenizer in HF
    # format so the registry, embedding-cache builder, and forecaster
    # can consume the fine-tuned encoder via ``AutoModel.from_pretrained``.
    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(args.artifact_root) / f"text_multi_axis_{run_token}"
    hf_checkpoint_dir = artifact_dir / "hf_checkpoints"

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
                factor_coverage=factor_coverage,
            )
            _save_hf_encoder_directory(model, tokenizer, hf_checkpoint_dir)
            print(f"[multi_axis] hf checkpoint saved to {hf_checkpoint_dir}")

    _logger.info("training_complete best_val_loss=%.4f", best_val_loss)

    _write_per_bank_breakdown(
        model=model,
        checkpoint_path=Path(args.output_checkpoint),
        val_loader=val_loader,
        device=device,
    )

    return 0


def _write_per_bank_breakdown(
    *,
    model: TextMultiAxisClassifier,
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    """Reload the best-epoch weights and write per-bank metrics next to the checkpoint.

    The in-memory ``model`` carries the last-epoch weights, which can differ
    from the best-epoch checkpoint when early-stopping triggered, so the
    per-bank table next to the checkpoint must come from the persisted state.
    """

    if checkpoint_path.exists():
        try:
            best_payload = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            best_state = best_payload.get("model_state_dict")
            if best_state:
                model.load_state_dict(best_state)
        except Exception:  # pragma: no cover — defensive
            _logger.warning("best_checkpoint_reload_failed", exc_info=True)
    try:
        per_bank = _evaluate_per_bank(model, val_loader, device)
    except Exception:  # pragma: no cover — defensive
        _logger.warning("per_bank_eval_failed", exc_info=True)
        return
    if not per_bank:
        return
    # ``with_name(stem + ".per_bank_metrics.json")`` works whether or not the
    # user-supplied checkpoint path carries an extension; ``with_suffix("")``
    # raises ValueError on suffix-less paths.
    per_bank_path = checkpoint_path.with_name(
        checkpoint_path.stem + ".per_bank_metrics.json"
    )
    per_bank_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json

    per_bank_path.write_text(_json.dumps(per_bank, indent=2), encoding="utf-8")
    _logger.info("per_bank_metrics_written path=%s", per_bank_path)


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
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=(
            "Root directory for per-run HF-format encoder artifacts. Each "
            "run writes to ``{artifact_root}/text_multi_axis_{run_token}/"
            "hf_checkpoints/`` so the encoder backbone + tokenizer can be "
            "consumed by ``AutoModel.from_pretrained`` (registry, embedding "
            "cache, forecaster). Independent of the singleton .pt at "
            "``--output-checkpoint`` which the inference service reads."
        ),
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
    # Bundle A.1: cross-bank auxiliary supervision flag on the
    # text-encoder fine-tune. ``off`` (default) keeps the strict
    # FOMC stance pool — cross-bank rows are excluded so the
    # current FOMC headline reproduces byte-identically.
    # ``stance_masked`` admits cross-bank rows but forces their
    # stance mask to False so the head only fits the FOMC stance
    # distribution while the encoder still trains on the
    # cross-bank text + auxiliary (certainty / topic / factor /
    # time) supervision. ``weighted`` admits them with the natural
    # stance label and downscales their stance contribution by
    # ``--cross-bank-stance-weight`` so the head can be A/B-tested
    # against the masked arm to re-litigate the Phase C
    # substitute-vs-complement prior (#231).
    parser.add_argument(
        "--cross-bank-supervision",
        choices=("off", "stance_masked", "weighted"),
        default="off",
        help=(
            "Cross-bank (peer_reviewed_cross_bank) row handling on the "
            "encoder fine-tune. ``off`` (default) excludes them; "
            "``stance_masked`` admits them with stance masked out so the "
            "head stays on the FOMC stance distribution; ``weighted`` "
            "admits them with their stance contribution scaled by "
            "``--cross-bank-stance-weight``."
        ),
    )
    parser.add_argument(
        "--cross-bank-stance-weight",
        type=float,
        default=0.25,
        help=(
            "Per-row multiplier on the stance loss for cross-bank rows "
            "under ``--cross-bank-supervision weighted``. FOMC rows stay "
            "at 1.0; cross-bank rows scale down so they do not dominate "
            "the FOMC distribution."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
