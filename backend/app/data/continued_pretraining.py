from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import DATA_DIR
from app.models.registry import revision_for
from app.training.manifest import write_run_manifest

DEFAULT_BASE_CHECKPOINT = "ProsusAI/finbert"
DEFAULT_OUT_NAME = "finbert_fed_adjacent"
DEFAULT_ARTIFACT_ROOT = DATA_DIR / "artifacts" / "continued_pretraining"
DEFAULT_CORPUS_FILES = (
    "chair_speeches.json",
    "governor_speeches.json",
    "congressional_testimonies.json",
    "press_conferences.json",
    "beige_book.json",
    "regional_research.json",
)


def _iter_corpus_texts(data_dir: Path, files: Iterable[str]) -> list[str]:
    texts: list[str] = []
    for filename in files:
        path = data_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("body") or "").strip()
            if text:
                texts.append(text)
    return texts


def run_mlm(
    *,
    base_checkpoint: str,
    texts: list[str],
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    block_size: int,
    seed: int,
    hf_token: str | None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    revision = revision_for(base_checkpoint)
    tokenizer_kwargs: dict[str, Any] = {"token": hf_token}
    model_kwargs: dict[str, Any] = {"token": hf_token}
    if revision is not None:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, **tokenizer_kwargs)
    model = AutoModelForMaskedLM.from_pretrained(base_checkpoint, **model_kwargs)

    class _SimpleTextDataset(Dataset):
        def __init__(self, texts: list[str], tokenizer, block_size: int) -> None:
            self._encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=block_size,
                return_special_tokens_mask=True,
            )

        def __len__(self) -> int:
            return len(self._encodings["input_ids"])

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {key: value[idx] for key, value in self._encodings.items()}

    dataset = _SimpleTextDataset(texts, tokenizer, block_size=block_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    train_result = trainer.train()
    model.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")
    return {
        "base_checkpoint": base_checkpoint,
        "base_revision": revision,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "train_runtime_s": float(getattr(train_result, "metrics", {}).get("train_runtime", 0.0)),
        "train_loss": float(getattr(train_result, "training_loss", 0.0) or 0.0),
        "num_examples": len(dataset),
        "checkpoint_path": str(output_dir / "checkpoint"),
    }


def _hf_token() -> str | None:
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue-pretrain a FinBERT checkpoint on the unlabelled Fed-adjacent corpus."
    )
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--checkpoint-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--corpus-files",
        nargs="+",
        default=list(DEFAULT_CORPUS_FILES),
        help="JSON files under --data-dir to read text from.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap the number of training rows (0 = use all).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    texts = _iter_corpus_texts(data_dir, args.corpus_files)
    if args.max_rows and len(texts) > args.max_rows:
        texts = texts[: args.max_rows]
    if not texts:
        raise SystemExit(f"No texts loaded from {data_dir}; check --corpus-files arguments.")

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.artifact_root) / f"{args.checkpoint_name}_{run_token}"
    print(f"[mlm] base={args.base_checkpoint} texts={len(texts)} out={run_dir}")

    result = run_mlm(
        base_checkpoint=args.base_checkpoint,
        texts=texts,
        output_dir=run_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
        hf_token=_hf_token(),
    )
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_run_manifest(
        run_dir,
        run_id=f"finbert_fed_adjacent_{run_token}_s{args.seed}",
        version_ids={"model_version": args.checkpoint_name},
        seeds=[args.seed],
        hyperparameters={
            "base_checkpoint": args.base_checkpoint,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
        },
        inputs=[data_dir / name for name in args.corpus_files],
        extra={"num_examples": result["num_examples"]},
    )
    print(f"[mlm] checkpoint at {result['checkpoint_path']}")
    print(
        "[mlm] register the new SHA in backend/app/models/registry.yaml under "
        f"alias 'finbert_fed_adjacent' before using it in fine-tunes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
