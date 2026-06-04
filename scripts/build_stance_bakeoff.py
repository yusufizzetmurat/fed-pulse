"""Unified stance encoder bake-off — every model on ONE held-out split.

All models are scored on the Fed-stance held-out (TEST_SOURCES =
gtfintechlab_federal_reserve_system + op_fed, n≈1112), which has zero
text-hash overlap with the TDW training pool — so no row has train leakage.
Metric per model: macro-F1 (primary), per-class F1 (hawkish/dovish/neutral),
accuracy. Confidence intervals:

  - fine-tuned encoder rows (bert-base, ProsusAI/finbert, and the three "ours"
    variants): a 3-class head + encoder fine-tuned on TDW, repeated over the
    official seeds {11,29,47,71,97} → mean ± 95% Student-t CI over seeds.
  - zero-shot rows (ZiweiChen/FinBERT-FOMC, gtfintechlab/FOMC-RoBERTa [ceiling],
    Gemini LLM @ temp 0): deterministic → 95% bootstrap CI over the test set.
  - frozen-embedding + linear head (bge-large, MiniLM): LogReg on TDW
    embeddings, eval on the held-out → 95% bootstrap CI.
  - sanity: majority-class (deterministic) and stratified-random (5 seeds).

Every row is wrapped so one failure (missing API key, offline HF, etc.) is
recorded and the rest of the leaderboard still completes.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from app.data.finetune_stance import (
    _predict,
    _ROBERTA_MAP,
    _val_carve,
    LABELS,
    load_labeled,
    train,
)
from app.determinism import enable_deterministic_mode

OFFICIAL_SEEDS = (11, 29, 47, 71, 97)
TRAIN_SOURCES = ["hf_fomc_communication"]
TEST_SOURCES = ["gtfintechlab_federal_reserve_system", "op_fed"]
_T_95 = {4: 2.776}  # two-sided 95% Student-t, df = n_seeds - 1


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {LABELS[i]: round(float(per[i]), 4) for i in range(3)},
    }


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, *, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(11)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]


def _seed_ci(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": round(mean, 4), "std": 0.0, "ci95_lo": round(mean, 4), "ci95_hi": round(mean, 4)}
    std = float(arr.std(ddof=1))
    t = _T_95.get(len(arr) - 1, 1.96)
    half = t * std / np.sqrt(len(arr))
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95_lo": round(mean - half, 4),
        "ci95_hi": round(mean + half, 4),
    }


def _hf_label_map(model: Any) -> dict[int, int]:
    """Map a HF model's output index → stance index via its id2label names."""
    id2label = getattr(model.config, "id2label", {}) or {}
    mapping: dict[int, int] = {}
    for idx, name in id2label.items():
        low = str(name).lower()
        if "hawk" in low:
            mapping[int(idx)] = 0
        elif "dov" in low:
            mapping[int(idx)] = 1
        elif "neutr" in low:
            mapping[int(idx)] = 2
    return mapping


# ---- row runners -----------------------------------------------------------

def run_baselines(train_y: np.ndarray, test_y: np.ndarray, seeds: tuple[int, ...]) -> dict[str, Any]:
    out = {}
    # majority — modal train class, constant prediction
    maj = int(np.bincount(train_y, minlength=3).argmax())
    pred = np.full_like(test_y, maj)
    out["majority"] = {**_score(test_y, pred), "training": "none", "ci": _bootstrap_ci(test_y, pred)}
    # stratified-random — sample from train class frequencies, per seed
    freq = np.bincount(train_y, minlength=3) / len(train_y)
    seed_f1 = []
    for s in seeds:
        rng = np.random.default_rng(s)
        pr = rng.choice(3, size=len(test_y), p=freq)
        seed_f1.append(f1_score(test_y, pr, average="macro", zero_division=0))
    out["stratified_random"] = {"training": "none (5 seeds)", "macro_f1_ci": _seed_ci(seed_f1)}
    return out


def run_encoder_ft(
    enc: str, train_pool: Any, test_df: Any, *, seeds: tuple[int, ...], epochs: int, lr: float,
    device: torch.device,
) -> dict[str, Any]:
    test_texts = test_df["text"].tolist()
    y_true = test_df["y"].to_numpy()
    seed_f1, last_pred = [], None
    for s in seeds:
        enable_deterministic_mode(s)
        tr, va = _val_carve(train_pool, seed=s)
        model, tok = train(tr, va, device, epochs, lr, base_encoder=enc)
        pred = _predict(model, tok, test_texts, device)
        seed_f1.append(f1_score(y_true, pred, average="macro", zero_division=0))
        last_pred = pred
        del model
        torch.cuda.empty_cache()
    detail = _score(y_true, last_pred)  # per-class from the last seed
    return {
        "training": f"full fine-tune on TDW, {len(seeds)} seeds",
        "macro_f1_ci": _seed_ci(seed_f1),
        "seed_macro_f1": [round(float(x), 4) for x in seed_f1],
        "per_class_f1_lastseed": detail["per_class_f1"],
        "accuracy_lastseed": detail["accuracy"],
    }


def run_zeroshot_hf(
    slug: str, test_df: Any, device: torch.device, *, forced_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(slug, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForSequenceClassification.from_pretrained(
        slug, token=os.environ.get("HF_TOKEN")
    ).to(device)
    raw = _predict(model, tok, test_df["text"].tolist(), device)
    id2label = {int(k): str(v) for k, v in (getattr(model.config, "id2label", {}) or {}).items()}
    mapping = forced_map or _hf_label_map(model)
    if len(mapping) < 3:
        raise ValueError(f"could not map id2label={id2label} to stance labels for {slug}")
    pred = np.array([mapping[int(p)] for p in raw])
    y_true = test_df["y"].to_numpy()
    return {
        "training": "zero-shot (native stance head)",
        "id2label": id2label,
        "label_map": {k: LABELS[v] for k, v in mapping.items()},
        **_score(y_true, pred),
        "macro_f1_bootstrap_ci": _bootstrap_ci(y_true, pred),
    }


def run_frozen_embed(model_name: str, train_pool: Any, test_df: Any) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    enc = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    xtr = enc.encode(train_pool["text"].tolist(), batch_size=64, show_progress_bar=False,
                     normalize_embeddings=True)
    xte = enc.encode(test_df["text"].tolist(), batch_size=64, show_progress_bar=False,
                     normalize_embeddings=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(xtr, train_pool["y"].to_numpy())
    pred = clf.predict(xte)
    y_true = test_df["y"].to_numpy()
    return {
        "training": "frozen embeddings + LogReg head on TDW",
        **_score(y_true, pred),
        "macro_f1_bootstrap_ci": _bootstrap_ci(y_true, pred),
    }


def run_llm_gemini(test_df: Any, *, model_name: str = "gemini-2.5-flash") -> dict[str, Any]:
    """Zero-shot Gemini @ temp 0 via the REST endpoint (no SDK dependency)."""
    import json as _json
    import time
    import urllib.error
    import urllib.request

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY / GOOGLE_API_KEY in env")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}"
        f":generateContent?key={key}"
    )
    sysmsg = (
        "You are a monetary-policy stance classifier for central-bank communications. "
        "Read the snippet and respond with exactly one word, lowercased, from this set: "
        "hawkish, dovish, neutral. Do not explain."
    )

    def classify(txt: str) -> str:
        body = {
            "system_instruction": {"parts": [{"text": sysmsg}]},
            "contents": [{"parts": [{"text": txt[:4000]}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4},
        }
        data = _json.dumps(body).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = _json.loads(resp.read())
                cand = out.get("candidates", [{}])[0]
                parts = cand.get("content", {}).get("parts", [{}])
                return (parts[0].get("text", "") or "").strip().lower()
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                return ""
            except Exception:
                if attempt < 4:
                    time.sleep(1 + attempt)
                    continue
                return ""
        return ""

    name2idx = {"hawkish": 0, "dovish": 1, "neutral": 2}
    preds, y_true, n_parsed = [], test_df["y"].to_numpy(), 0
    for i, txt in enumerate(test_df["text"].tolist()):
        resp = classify(txt)
        word = resp.split()[0] if resp else ""
        idx = name2idx.get(word)
        if idx is not None:
            n_parsed += 1
        preds.append(idx if idx is not None else 2)
        if (i + 1) % 200 == 0:
            print(f"    gemini {i + 1}/{len(y_true)}")
    pred = np.array(preds)
    return {
        "training": f"zero-shot LLM ({model_name}, temp 0)",
        "n_parsed": int(n_parsed),
        **_score(y_true, pred),
        "macro_f1_bootstrap_ci": _bootstrap_ci(y_true, pred),
    }


def run_llm_anthropic(
    test_df: Any, *, model_name: str = "claude-haiku-4-5-20251001"
) -> dict[str, Any]:
    """Zero-shot Claude @ temp 0 via the Anthropic SDK. Reproducible, no GPU."""
    import time

    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("no ANTHROPIC_API_KEY in env")
    client = anthropic.Anthropic(api_key=key)
    sysmsg = (
        "You are a monetary-policy stance classifier for central-bank communications. "
        "Read the snippet and respond with exactly one word, lowercased, from this set: "
        "hawkish, dovish, neutral. Do not explain."
    )

    def classify(txt: str) -> str:
        for attempt in range(5):
            try:
                r = client.messages.create(
                    model=model_name, max_tokens=8, temperature=0.0, system=sysmsg,
                    messages=[{"role": "user", "content": txt[:4000]}],
                )
                return (r.content[0].text if r.content else "").strip().lower()
            except Exception as e:  # noqa: BLE001 — retry transient/rate-limit
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                return ""
        return ""

    name2idx = {"hawkish": 0, "dovish": 1, "neutral": 2}
    preds, y_true, n_parsed = [], test_df["y"].to_numpy(), 0
    for i, txt in enumerate(test_df["text"].tolist()):
        resp = classify(txt)
        word = resp.split()[0] if resp else ""
        idx = name2idx.get(word)
        if idx is not None:
            n_parsed += 1
        preds.append(idx if idx is not None else 2)
        if (i + 1) % 200 == 0:
            print(f"    {model_name} {i + 1}/{len(y_true)}")
    pred = np.array(preds)
    return {
        "training": f"zero-shot LLM ({model_name}, temp 0)",
        "n_parsed": int(n_parsed),
        **_score(y_true, pred),
        "macro_f1_bootstrap_ci": _bootstrap_ci(y_true, pred),
    }


def run_llm_local(
    test_df: Any, device: torch.device, *, model_name: str = "Qwen/Qwen2.5-3B-Instruct"
) -> dict[str, Any]:
    """Zero-shot local LLM (Qwen) @ temp 0 (greedy). 4-bit if bitsandbytes is
    available, else fp16. Reproducible; no API quota. Fallback model on OOM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
    load_kw: dict[str, Any] = {"token": os.environ.get("HF_TOKEN"), "torch_dtype": torch.float16}
    try:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        load_kw["device_map"] = "auto"
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    if "device_map" not in load_kw:
        model = model.to(device)
    model.eval()
    sysmsg = (
        "You are a monetary-policy stance classifier for central-bank communications. "
        "Read the snippet and respond with exactly one word, lowercased, from this set: "
        "hawkish, dovish, neutral. Do not explain."
    )
    name2idx = {"hawkish": 0, "dovish": 1, "neutral": 2}
    preds, y_true, n_parsed = [], test_df["y"].to_numpy(), 0
    for i, txt in enumerate(test_df["text"].tolist()):
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": txt[:3000]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        word = gen.strip().lower().split()[0] if gen.strip() else ""
        idx = name2idx.get(word)
        if idx is not None:
            n_parsed += 1
        preds.append(idx if idx is not None else 2)
        if (i + 1) % 200 == 0:
            print(f"    {model_name} {i + 1}/{len(y_true)}")
    pred = np.array(preds)
    return {
        "training": f"zero-shot LLM ({model_name}, greedy/temp 0)",
        "n_parsed": int(n_parsed),
        **_score(y_true, pred),
        "macro_f1_bootstrap_ci": _bootstrap_ci(y_true, pred),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=Path(
        "/data/processed/tp_v3_full_rebuild_2026_05_30/registry_normalized.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/artifacts/stance_bakeoff"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seeds", type=int, nargs="+", default=list(OFFICIAL_SEEDS))
    p.add_argument("--skip", nargs="*", default=[], help="row keys to skip")
    p.add_argument("--only", nargs="*", default=None, help="run only these row keys")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_labeled(args.labels)
    train_pool = df[df["source"].isin(TRAIN_SOURCES)].reset_index(drop=True)
    test_df = df[df["source"].isin(TEST_SOURCES)].reset_index(drop=True)
    train_y, test_y = train_pool["y"].to_numpy(), test_df["y"].to_numpy()
    print(f"train pool (TDW): {len(train_pool)} | held-out test: {len(test_df)}")
    print(f"test by class: {dict(test_df['mapped_label'].value_counts())}")

    seeds = tuple(args.seeds)
    # (key, callable) — encoder rows marked with the HF slug to fine-tune
    ENCODERS = {
        "bert_base_uncased": "bert-base-uncased",
        "prosusai_finbert_ft": "ProsusAI/finbert",
        "ours_finbert_fed_adjacent": "yusufizzetmurat/finbert-fed-adjacent",
        "ours_finbert_fed_adjacent_xbank": "yusufizzetmurat/finbert-fed-adjacent-xbank",
        "ours_finbert_fed_adjacent_xbank_dapt": "yusufizzetmurat/finbert-fed-adjacent-xbank-dapt",
    }
    ZEROSHOT = {
        "ziweichen_finbert_fomc": "ZiweiChen/FinBERT-FOMC",
        "gtfintechlab_fomc_roberta_CEILING": "gtfintechlab/FOMC-RoBERTa",
    }
    FROZEN = {
        "frozen_bge_large_linear": "BAAI/bge-large-en-v1.5",
        "frozen_minilm_linear": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def want(key: str) -> bool:
        if args.only is not None:
            return key in args.only
        return key not in args.skip

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if want("baselines"):
        try:
            results.update(run_baselines(train_y, test_y, seeds))
        except Exception as e:  # noqa: BLE001
            errors["baselines"] = traceback.format_exc()[-800:]

    for key, slug in ENCODERS.items():
        if not want(key):
            continue
        print(f"\n[encoder-ft] {key} ({slug})")
        try:
            results[key] = run_encoder_ft(
                slug, train_pool, test_df, seeds=seeds, epochs=args.epochs, lr=args.lr, device=device
            )
        except Exception:  # noqa: BLE001
            errors[key] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors[key].splitlines()[-1]}")

    for key, slug in ZEROSHOT.items():
        if not want(key):
            continue
        print(f"\n[zero-shot] {key} ({slug})")
        try:
            if slug == "gtfintechlab/FOMC-RoBERTa":
                fm = _ROBERTA_MAP_IDX
            elif slug == "ZiweiChen/FinBERT-FOMC":
                # sentiment head {0:Neutral,1:Positive,2:Negative} → stance under
                # the standard FOMC convention: Positive=dovish, Negative=hawkish.
                fm = {0: LABELS.index("neutral"), 1: LABELS.index("dovish"), 2: LABELS.index("hawkish")}
            else:
                fm = None
            results[key] = run_zeroshot_hf(slug, test_df, device, forced_map=fm)
        except Exception:  # noqa: BLE001
            errors[key] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors[key].splitlines()[-1]}")

    for key, name in FROZEN.items():
        if not want(key):
            continue
        print(f"\n[frozen-embed] {key} ({name})")
        try:
            results[key] = run_frozen_embed(name, train_pool, test_df)
        except Exception:  # noqa: BLE001
            errors[key] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors[key].splitlines()[-1]}")

    if want("llm_gemini"):
        print("\n[llm] gemini zero-shot")
        try:
            results["llm_gemini"] = run_llm_gemini(test_df)
        except Exception:  # noqa: BLE001
            errors["llm_gemini"] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors['llm_gemini'].splitlines()[-1]}")

    if want("llm_anthropic"):
        print("\n[llm] anthropic claude zero-shot")
        try:
            results["llm_anthropic"] = run_llm_anthropic(test_df)
        except Exception:  # noqa: BLE001
            errors["llm_anthropic"] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors['llm_anthropic'].splitlines()[-1]}")

    if want("llm_local"):
        print("\n[llm] local Qwen zero-shot")
        try:
            results["llm_local"] = run_llm_local(test_df, device)
        except Exception:  # noqa: BLE001
            errors["llm_local"] = traceback.format_exc()[-800:]
            print(f"  FAILED: {errors['llm_local'].splitlines()[-1]}")

    payload = {
        "split": "gtfintechlab_federal_reserve_system + op_fed (Fed-stance held-out)",
        "n_test": int(len(test_df)),
        "n_train_tdw": int(len(train_pool)),
        "seeds": list(seeds),
        "test_class_counts": {k: int(v) for k, v in test_df["mapped_label"].value_counts().items()},
        "results": results,
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stance_bakeoff.json").write_text(json.dumps(payload, indent=2, default=str))

    # leaderboard
    def macro(row: dict[str, Any]) -> float:
        if "macro_f1_ci" in row:
            return row["macro_f1_ci"]["mean"]
        return row.get("macro_f1", float("nan"))

    print("\n===== STANCE BAKE-OFF (macro-F1 on n=%d held-out) =====" % len(test_df))
    print(f"{'model':<40}{'macroF1':>9}{'  CI':>18}{'  training':<10}")
    for k in sorted(results, key=lambda k: -(macro(results[k]) if macro(results[k]) == macro(results[k]) else -1)):
        r = results[k]
        m = macro(r)
        ci = r.get("macro_f1_ci", {})
        ci_s = (f"[{ci['ci95_lo']},{ci['ci95_hi']}]" if ci else
                f"{r.get('macro_f1_bootstrap_ci', r.get('ci', ''))}")
        print(f"{k:<40}{m:>9.4f}  {str(ci_s):>16}  {r.get('training','')[:30]}")
    if errors:
        print("\nERRORS:", list(errors))
    print(f"\nwrote {args.out_dir / 'stance_bakeoff.json'}")
    return 0


# FOMC-RoBERTa fixed index→stance map, derived from the project's _ROBERTA_MAP
# ({LABEL_0/1/2} → stance name) so we don't depend on its id2label being present.
_ROBERTA_MAP_IDX = {i: LABELS.index(_ROBERTA_MAP[f"LABEL_{i}"]) for i in range(3)}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
