"""Phase 4 LLM zero-shot baseline (SRS FR-26).

Prompts a small instruct LLM with a deterministic three-class template
(hawkish | dovish | neutral) at temperature 0 and reports
aggregator-compatible classification metrics on the fold's test slice.
Mirrors :mod:`app.data.nlp_baseline_batch` CLI shape.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import DATA_DIR
from app.models.registry import revision_for
from app.data.finetune_pilot import (
    LABELS,
    _compute_classification_metrics,
    _hf_token,
    _latency_summary,
    _load_fold,
    _load_registry_rows,
    _set_all_seeds,
    _split_by_fold,
)

DEFAULT_LLM = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ARTIFACT_ROOT = DATA_DIR / "artifacts" / "phase4_llm_zero_shot"
SYSTEM_PROMPT = (
    "You are a financial sentiment classifier for FOMC communications. "
    "Read the snippet and respond with exactly one word, lowercased, "
    "from this set: hawkish, dovish, neutral. Do not explain."
)
USER_INSTRUCTION = "Classify the FOMC snippet. Respond with one word only.\n\nSnippet:\n"


def _build_prompt(text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTION + text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{USER_INSTRUCTION}{text}\n\nAnswer:"


_LABEL_RE = re.compile(r"\b(hawkish|dovish|neutral)\b", re.IGNORECASE)


def _parse_label(generated: str) -> str:
    match = _LABEL_RE.search(generated)
    if match:
        return match.group(1).lower()
    # Conservative fallback so the row still scores.
    return "neutral"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 LLM zero-shot baseline.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default="wf_fold_2")
    parser.add_argument(
        "--llm-backend",
        choices=("hf", "gemini"),
        default="hf",
        help="hf loads a local Transformers checkpoint; gemini calls the Google Gemini API.",
    )
    parser.add_argument(
        "--llm-checkpoint",
        default=DEFAULT_LLM,
        help="HF checkpoint name (used with --llm-backend hf) or Gemini model name (used with --llm-backend gemini).",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser.parse_args()


def _score_with_hf(test_rows, args) -> tuple[list[str], list[str], list[float], str]:
    hf_token = _hf_token()
    if hf_token:
        print(f"[llm_zs] using HF token (len={len(hf_token)})")
    revision = revision_for(args.llm_checkpoint)
    if revision:
        print(f"[llm_zs] pinning {args.llm_checkpoint} to revision {revision[:12]}")
    tokenizer_kwargs: dict[str, Any] = {"token": hf_token}
    if revision:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.llm_checkpoint, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs: dict[str, Any] = {
        "token": hf_token,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    if revision:
        model_kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(args.llm_checkpoint, **model_kwargs)
    model.eval()

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    for idx, row in enumerate(test_rows):
        prompt = _build_prompt(row.text, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        pred_label = _parse_label(generated)
        y_true.append(row.label)
        y_pred.append(pred_label)
        if idx < 3 or idx % 50 == 0:
            print(
                f"[llm_zs] {idx + 1}/{len(test_rows)} pred={pred_label} truth={row.label} "
                f"gen={generated.strip()!r}"
            )
    return y_true, y_pred, latencies_ms, str(model.device)


def _score_with_gemini(test_rows, args) -> tuple[list[str], list[str], list[float], str]:
    from app.services.gemini_client import load_model, score_passage

    model = load_model(args.llm_checkpoint)
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    for idx, row in enumerate(test_rows):
        t0 = time.perf_counter()
        result = score_passage(row.text, model=model)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        pred_label = str(result.get("label", "neutral")).strip().lower()
        if pred_label not in ("hawkish", "dovish", "neutral"):
            pred_label = "neutral"
        y_true.append(row.label)
        y_pred.append(pred_label)
        if idx < 3 or idx % 50 == 0:
            print(
                f"[llm_zs] {idx + 1}/{len(test_rows)} pred={pred_label} truth={row.label} "
                f"confidence={float(result.get('confidence', 0.0)):.3f}"
            )
    return y_true, y_pred, latencies_ms, "gemini-api"


def main() -> int:
    args = _parse_args()
    _set_all_seeds(args.seed)

    if args.llm_backend == "gemini" and not args.llm_checkpoint.lower().startswith("gemini"):
        raise SystemExit(
            f"--llm-backend gemini requires a Gemini model name (e.g. 'gemini-2.5-pro'); "
            f"got --llm-checkpoint={args.llm_checkpoint!r}. Pass a Gemini model or switch "
            f"--llm-backend back to hf."
        )

    package_dir = DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")
    rows = _load_registry_rows(package_dir)
    fold = _load_fold(package_dir, args.fold_id)
    _, test_rows = _split_by_fold(rows, fold)
    if not test_rows:
        raise SystemExit(f"No test rows for fold {args.fold_id}")
    print(f"[llm_zs] backend={args.llm_backend} checkpoint={args.llm_checkpoint} test_rows={len(test_rows)}")

    if args.llm_backend == "gemini":
        y_true, y_pred, latencies_ms, device_label = _score_with_gemini(test_rows, args)
    else:
        y_true, y_pred, latencies_ms, device_label = _score_with_hf(test_rows, args)

    cls_metrics = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies_ms)
    print(
        f"[llm_zs] macro_f1={cls_metrics['macro_f1']:.4f} "
        f"acc={cls_metrics['accuracy']:.4f} weighted_f1={cls_metrics['weighted_f1']:.4f}"
    )

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(args.artifact_root) / f"llm_zs_{run_token}_s{args.seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "pipeline": "phase4_llm_zero_shot",
        "owner": args.owner,
        "checkpoint": args.llm_checkpoint,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "test_rows": len(test_rows),
        "classification": cls_metrics,
        "latency": latency,
        "training_package_id": args.training_package_id,
        "started_at_utc": run_token,
        "backend": args.llm_backend,
        "device": device_label,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "cuda_available": torch.cuda.is_available(),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[llm_zs] metrics written to {artifact_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
