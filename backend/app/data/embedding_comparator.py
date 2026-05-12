"""Phase 4 embedding-model comparator (SRS FR-35).

Encodes train + test slices with a sentence-embedding backbone (default
all-MiniLM-L6-v2), trains a small softmax classifier head, and reports
aggregator-compatible metrics on fold 2 test slice.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from app.config import DATA_DIR
from app.models.registry import revision_for
from app.data.phase3_finetune_pilot import (
    EvalRow,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    _compute_classification_metrics,
    _hf_token,
    _latency_summary,
    _load_fold,
    _load_registry_rows,
    _set_all_seeds,
    _split_by_fold,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARTIFACT_ROOT = DATA_DIR / "artifacts" / "phase4_embedding_comparator"


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _encode_batch(
    texts: list[str],
    *,
    tokenizer,
    model,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state
        pooled = _mean_pool(last_hidden, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
    return pooled


def _encode_rows(
    rows: list[EvalRow],
    *,
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    label_ids: list[int] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        embeddings.append(
            _encode_batch(
                [r.text for r in batch],
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                device=device,
            ).cpu()
        )
        label_ids.extend(LABEL2ID[r.label] for r in batch)
    if not embeddings:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(embeddings, dim=0), torch.tensor(label_ids, dtype=torch.long)


class _LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _train_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
) -> _LinearHead:
    head = _LinearHead(train_x.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    for epoch in range(epochs):
        perm = torch.randperm(train_x.shape[0], device=device)
        running = 0.0
        for start in range(0, train_x.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = train_x[batch_idx]
            yb = train_y[batch_idx]
            optimizer.zero_grad()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.shape[0]
        avg = running / max(train_x.shape[0], 1)
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            print(f"[emb_cmp] head epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}")
    head.eval()
    return head


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 embedding-model comparator.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default="wf_fold_2")
    parser.add_argument("--embedding-checkpoint", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--head-learning-rate", type=float, default=5e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _set_all_seeds(args.seed)

    package_dir = DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")
    rows = _load_registry_rows(package_dir)
    fold = _load_fold(package_dir, args.fold_id)
    train_rows, test_rows = _split_by_fold(rows, fold)
    if not train_rows or not test_rows:
        raise SystemExit(
            f"Insufficient rows for fold {args.fold_id}: train={len(train_rows)} test={len(test_rows)}"
        )
    print(
        f"[emb_cmp] checkpoint={args.embedding_checkpoint} train_rows={len(train_rows)} "
        f"test_rows={len(test_rows)}"
    )

    hf_token = _hf_token()
    if hf_token:
        print(f"[emb_cmp] using HF token (len={len(hf_token)})")
    revision = revision_for(args.embedding_checkpoint)
    if revision:
        print(f"[emb_cmp] pinning {args.embedding_checkpoint} to revision {revision[:12]}")
    tokenizer_kwargs: dict[str, Any] = {"token": hf_token}
    model_kwargs: dict[str, Any] = {"token": hf_token}
    if revision:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_checkpoint, **tokenizer_kwargs)
    backbone = AutoModel.from_pretrained(args.embedding_checkpoint, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()

    encode_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[emb_cmp] encoding {len(train_rows)} train rows...")
    t_train_start = time.perf_counter()
    train_x, train_y = _encode_rows(
        train_rows,
        tokenizer=tokenizer,
        model=backbone,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        device=device,
    )
    train_encode_s = time.perf_counter() - t_train_start

    print(f"[emb_cmp] encoding {len(test_rows)} test rows...")
    t_test_start = time.perf_counter()
    test_x, test_y = _encode_rows(
        test_rows,
        tokenizer=tokenizer,
        model=backbone,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        device=device,
    )
    test_encode_s = time.perf_counter() - t_test_start

    print(f"[emb_cmp] training linear head ({args.head_epochs} epochs)...")
    head = _train_head(
        train_x,
        train_y,
        epochs=args.head_epochs,
        learning_rate=args.head_learning_rate,
        batch_size=args.head_batch_size,
        weight_decay=args.head_weight_decay,
        device=device,
    )

    latencies_ms: list[float] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    test_x_dev = test_x.to(device)
    with torch.no_grad():
        for idx in range(test_x_dev.shape[0]):
            t0 = time.perf_counter()
            logits = head(test_x_dev[idx : idx + 1])
            pred_idx = int(logits.argmax(dim=-1).item())
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            y_pred.append(ID2LABEL[pred_idx])
            y_true.append(ID2LABEL[int(test_y[idx].item())])

    cls_metrics = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies_ms)
    print(
        f"[emb_cmp] macro_f1={cls_metrics['macro_f1']:.4f} "
        f"acc={cls_metrics['accuracy']:.4f} weighted_f1={cls_metrics['weighted_f1']:.4f}"
    )

    artifact_dir = Path(args.artifact_root) / f"emb_cmp_{encode_token}_s{args.seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "pipeline": "phase4_embedding_comparator",
        "owner": args.owner,
        "checkpoint": args.embedding_checkpoint,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "embedding_dim": int(train_x.shape[1]),
        "train_rows": int(train_x.shape[0]),
        "test_rows": int(test_x.shape[0]),
        "train_encode_s": float(train_encode_s),
        "test_encode_s": float(test_encode_s),
        "head_epochs": args.head_epochs,
        "head_learning_rate": args.head_learning_rate,
        "head_batch_size": args.head_batch_size,
        "classification": cls_metrics,
        "latency": latency,
        "training_package_id": args.training_package_id,
        "started_at_utc": encode_token,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[emb_cmp] metrics written to {artifact_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
