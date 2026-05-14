"""Continued pretraining for FinBERT-FedAdjacent.

Primary substrate is ``samchain/BIS_speeches_97_23_MLM`` — 909,877 NSP-formatted
sentence pairs from BIS central bank speeches (1997-2023). The dataset is
pre-chunked and pre-labelled for the MLM + Next-Sentence-Prediction joint
objective, which is the recipe Devlin et al. and Araci used for FinBERT.

Why the switch from the local JSON corpus: the 44-doc local corpus produced
~5.8M effective tokens after chunking. BIS speeches at this scale produce
~365M tokens of in-domain monetary-policy language — two orders of magnitude
larger and within range of FinBERT's original pretraining substrate.

The legacy local JSON corpus is preserved as an optional auxiliary substrate
via ``--auxiliary-local-dir`` for ablation studies; pass ``--substrate local``
to use it as the sole substrate (reproduces the pre-Sprint-1B behaviour).
"""

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
DEFAULT_BIS_DATASET_ID = "samchain/BIS_speeches_97_23_MLM"
DEFAULT_BIS_DATASET_REVISION: str | None = None  # pin once first run reports the resolved sha
DEFAULT_LOCAL_CORPUS_FILES: tuple[str, ...] = (
    "chair_speeches.json",
    "governor_speeches.json",
    "congressional_testimonies.json",
    "press_conferences.json",
    "beige_book.json",
    "regional_research.json",
)
_VALID_SUBSTRATES = ("bis", "local", "both")


def _iter_local_pairs(data_dir: Path, files: Iterable[str]) -> list[dict[str, Any]]:
    """Adapt the local JSON corpus into the same {sequenceA, sequenceB, next_sentence_label} shape."""
    pairs: list[dict[str, Any]] = []
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
                # Each doc becomes a degenerate pair where B is unused but the
                # NSP label is fixed at 0 so the model treats it as non-paired.
                pairs.append({"sequenceA": text, "sequenceB": "", "next_sentence_label": 0})
    return pairs


def _bis_pair_stream(
    dataset_id: str,
    revision: str | None,
    *,
    streaming: bool,
    max_rows: int,
):
    """Yield {sequenceA, sequenceB, next_sentence_label} rows from the BIS dataset."""
    from datasets import load_dataset  # type: ignore

    kwargs: dict[str, Any] = {"split": "train", "streaming": streaming}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(dataset_id, **kwargs)

    seen = 0
    for row in ds:
        if max_rows and seen >= max_rows:
            break
        a = (row.get("sequenceA") or "").strip()
        b = (row.get("sequenceB") or "").strip()
        if not a:
            continue
        yield {"sequenceA": a, "sequenceB": b, "next_sentence_label": int(row.get("next_sentence_label") or 0)}
        seen += 1


def run_mlm(
    *,
    base_checkpoint: str,
    pair_records: list[dict[str, Any]],
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    block_size: int,
    seed: int,
    hf_token: str | None,
    objective: str,
) -> dict[str, Any]:
    """Run the joint MLM + NSP continued pretrain over pre-built sentence pairs.

    ``objective`` selects the head: ``"mlm_nsp"`` uses ``BertForPreTraining``
    (Devlin/Araci recipe); ``"mlm"`` uses ``AutoModelForMaskedLM`` and drops
    the NSP label. MLM-only is provided for ablation against the joint loss.
    """
    import numpy as np
    import torch
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        BertForPreTraining,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset

    if objective not in {"mlm", "mlm_nsp"}:
        raise ValueError(f"Unknown objective: {objective!r}; expected one of 'mlm', 'mlm_nsp'.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    revision = revision_for(base_checkpoint)
    tokenizer_kwargs: dict[str, Any] = {"token": hf_token}
    model_kwargs: dict[str, Any] = {"token": hf_token}
    if revision is not None:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, **tokenizer_kwargs)
    if objective == "mlm_nsp":
        model = BertForPreTraining.from_pretrained(base_checkpoint, **model_kwargs)
    else:
        model = AutoModelForMaskedLM.from_pretrained(base_checkpoint, **model_kwargs)

    class _PairDataset(Dataset):
        def __init__(self, pairs: list[dict[str, Any]], tokenizer, block_size: int, use_nsp: bool) -> None:
            seq_a = [p["sequenceA"] for p in pairs]
            seq_b = [p.get("sequenceB") or "" for p in pairs] if use_nsp else None
            if use_nsp:
                # text_pair tokenisation: [CLS] A [SEP] B [SEP] with token_type_ids.
                self._enc = tokenizer(
                    seq_a,
                    text_pair=seq_b,
                    truncation=True,
                    padding="max_length",
                    max_length=block_size,
                    return_special_tokens_mask=True,
                    return_token_type_ids=True,
                )
            else:
                self._enc = tokenizer(
                    seq_a,
                    truncation=True,
                    padding="max_length",
                    max_length=block_size,
                    return_special_tokens_mask=True,
                )
            self._nsp_labels = (
                [int(p.get("next_sentence_label") or 0) for p in pairs] if use_nsp else None
            )

        def __len__(self) -> int:
            return len(self._enc["input_ids"])

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {key: value[idx] for key, value in self._enc.items()}
            if self._nsp_labels is not None:
                item["next_sentence_label"] = self._nsp_labels[idx]
            return item

    use_nsp = objective == "mlm_nsp"
    dataset = _PairDataset(pair_records, tokenizer, block_size=block_size, use_nsp=use_nsp)
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    if use_nsp:
        # DataCollatorForLanguageModeling routes batches through tokenizer.pad(),
        # which keeps only tokenizer-output keys — next_sentence_label would be
        # silently dropped before reaching BertForPreTraining. Wrap to pop NSP
        # labels first, then re-attach as a torch tensor after MLM masking.
        def collator(features: list[dict[str, Any]]):  # type: ignore[no-redef]
            import torch as _torch

            nsp_labels: list[int] = []
            for feat in features:
                if isinstance(feat, dict):
                    nsp_labels.append(int(feat.pop("next_sentence_label", 0)))
                else:
                    nsp_labels.append(0)
            batch = mlm_collator(features)
            batch["next_sentence_label"] = _torch.tensor(nsp_labels, dtype=_torch.long)
            return batch
    else:
        collator = mlm_collator
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        logging_steps=200,
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
        "objective": objective,
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
        description="Continue-pretrain a FinBERT checkpoint on the BIS central bank speeches MLM+NSP corpus."
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
        "--substrate",
        choices=_VALID_SUBSTRATES,
        default="bis",
        help="bis (default; samchain/BIS_speeches_97_23_MLM), local (legacy JSON corpus), or both.",
    )
    parser.add_argument(
        "--bis-dataset-id",
        default=DEFAULT_BIS_DATASET_ID,
        help="HF dataset id for the BIS substrate.",
    )
    parser.add_argument(
        "--bis-dataset-revision",
        default=DEFAULT_BIS_DATASET_REVISION,
        help="HF dataset revision (commit SHA) for reproducibility; first run resolves and reports.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the BIS dataset rather than caching to disk (slower per-step but no local 365MB cache).",
    )
    parser.add_argument(
        "--auxiliary-local-dir",
        default=None,
        help="Override directory for the legacy local JSON corpus (used when --substrate is local or both).",
    )
    parser.add_argument(
        "--corpus-files",
        nargs="+",
        default=list(DEFAULT_LOCAL_CORPUS_FILES),
        help="JSON files under --data-dir (or --auxiliary-local-dir) for the local substrate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap the number of training pairs (0 = use all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--objective",
        choices=("mlm", "mlm_nsp"),
        default="mlm_nsp",
        help="Joint MLM+NSP (default, Devlin/Araci recipe) or MLM-only ablation.",
    )
    return parser.parse_args(argv)


def _collect_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the training-pair list under the requested substrate(s).

    --substrate both: local pairs (small, finite) load first to guarantee
    representation, then BIS fills the remainder of --max-rows. Earlier
    behaviour (BIS first) silently emptied the local slice whenever BIS
    filled the cap, making --substrate both --max-rows N indistinguishable
    from --substrate bis.
    """
    pairs: list[dict[str, Any]] = []

    if args.substrate in {"local", "both"}:
        local_dir = Path(args.auxiliary_local_dir) if args.auxiliary_local_dir else Path(args.data_dir)
        local_pairs = _iter_local_pairs(local_dir, args.corpus_files)
        if args.substrate == "local" and args.max_rows and len(local_pairs) > args.max_rows:
            local_pairs = local_pairs[: args.max_rows]
        pairs.extend(local_pairs)

    if args.substrate in {"bis", "both"}:
        if args.substrate == "both" and args.max_rows:
            remaining = max(0, args.max_rows - len(pairs))
            bis_cap = remaining
            if remaining == 0:
                import warnings as _warnings

                _warnings.warn(
                    f"--substrate both --max-rows {args.max_rows} was reached by local "
                    f"({len(pairs)} pairs); BIS substrate dropped. Raise --max-rows or "
                    "use --substrate bis to override.",
                    stacklevel=2,
                )
        else:
            bis_cap = args.max_rows
        if bis_cap != 0:
            bis_iter = _bis_pair_stream(
                args.bis_dataset_id,
                args.bis_dataset_revision,
                streaming=args.streaming,
                max_rows=bis_cap,
            )
            pairs.extend(list(bis_iter))

    return pairs


def _resolve_dataset_sha(dataset_id: str, revision: str | None) -> str | None:
    """Resolve the actual commit SHA the HF Hub serves for (dataset_id, revision).

    When the user passes --bis-dataset-revision explicitly, that's what we
    persist. When they don't, we query the Hub for the dataset's latest commit
    so the manifest still carries a concrete sha rather than null.
    """
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().dataset_info(dataset_id)
        return getattr(info, "sha", None)
    except Exception:  # pragma: no cover — manifest still records None on failure
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        pairs = _collect_pairs(args)
    except Exception as exc:  # pragma: no cover — surfaced as a CLI error
        raise SystemExit(f"Failed to collect training pairs: {exc}") from exc
    if not pairs:
        raise SystemExit(
            f"No training pairs loaded for substrate={args.substrate!r}. "
            "Check --bis-dataset-id, --data-dir, or --corpus-files."
        )

    resolved_revision = (
        _resolve_dataset_sha(args.bis_dataset_id, args.bis_dataset_revision)
        if args.substrate in {"bis", "both"}
        else None
    )

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.artifact_root) / f"{args.checkpoint_name}_{run_token}_s{args.seed}"
    print(
        f"[mlm] base={args.base_checkpoint} substrate={args.substrate} pairs={len(pairs)} "
        f"objective={args.objective} bis_revision={resolved_revision} out={run_dir}"
    )

    result = run_mlm(
        base_checkpoint=args.base_checkpoint,
        pair_records=pairs,
        output_dir=run_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
        hf_token=_hf_token(),
        objective=args.objective,
    )
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_run_manifest(
        run_dir,
        run_id=f"{args.checkpoint_name}_{run_token}_s{args.seed}",
        version_ids={"model_version": args.checkpoint_name},
        seeds=[args.seed],
        hyperparameters={
            "base_checkpoint": args.base_checkpoint,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "objective": args.objective,
            "substrate": args.substrate,
            "bis_dataset_id": args.bis_dataset_id,
            "bis_dataset_revision_requested": args.bis_dataset_revision,
            "bis_dataset_revision_resolved": resolved_revision,
        },
        inputs=[args.bis_dataset_id] if args.substrate in {"bis", "both"} else [],
        extra={"num_examples": result["num_examples"]},
    )
    print(f"[mlm] checkpoint at {result['checkpoint_path']}")
    print(
        f"[mlm] register the new SHA in backend/app/models/registry.yaml under "
        f"alias '{args.checkpoint_name}' before using it in fine-tunes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
