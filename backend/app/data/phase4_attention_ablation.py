"""Phase 4 attention + decay ablation (SRS FR-23 / FR-24 / FR-31).

Trains four variants of the forecaster on labelled FOMC documents augmented
with market history and (optionally) chunk-attention context, and records a
combined RMSE / directional accuracy table plus learned-lambda values and
attention heatmaps for sample documents.

Variants:
- baseline: time decay off, chunk attention off
- variant_a: time decay on, chunk attention off
- variant_b: time decay off, chunk attention on
- variant_a_b: time decay on, chunk attention on
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.data.chunk_embedding_retrieval import (
    build_lookback_tensors,
    load_chunk_store,
    resolve_store_path,
)
from app.data.phase3_finetune_pilot import (
    EvalRow,
    LABELS,
    _hf_token,
    _load_fold,
    _load_registry_rows,
    _set_all_seeds,
    _split_by_fold,
)
from app.services.forecaster import (
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_CLOSE_SCALE,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
    FeatureVector,
    ForecasterModel,
    build_last5_sequence,
)
from app.services.sentiment import analyze_text


@dataclass
class TrainingTuple:
    doc_id: str
    doc_date: str
    feature_seq: list[list[float]]  # SEQUENCE_LENGTH × FEATURE_SIZE
    target_close_norm: float  # close at t+1, normalized by close_scale
    target_volatility: float  # volatility_5d at t+1
    chunk_embeddings: np.ndarray  # max_chunks × embedding_size
    chunk_elapsed: np.ndarray  # max_chunks
    chunk_mask: np.ndarray  # max_chunks


def _fetch_full_history(symbol: str = "^GSPC", start: str = "1995-01-01") -> pd.DataFrame:
    import yfinance as yf

    end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"yfinance returned empty history for {symbol}")
    df = df[["Close"]].copy()
    df.columns = ["close"]
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df["date"] = df.index.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)
    df["volatility_5d"] = df["close"].pct_change().rolling(5).std().fillna(0.0)
    df["close_change_pct"] = df["close"].pct_change().fillna(0.0)
    df["volatility_change"] = df["volatility_5d"].diff().fillna(0.0)
    return df


def _build_feature_seq(
    market_df: pd.DataFrame,
    doc_date: str,
    sentiment_score: float,
) -> tuple[list[list[float]], float, float] | None:
    """Return (5-step sequence, target_close_norm, target_volatility) or None if not enough history."""
    rows = market_df[market_df["date"] <= doc_date]
    if len(rows) < SEQUENCE_LENGTH:
        return None
    target_idx = rows.index[-1] + 1
    if target_idx >= len(market_df):
        return None
    target_row = market_df.iloc[target_idx]
    history = rows.tail(SEQUENCE_LENGTH).reset_index(drop=True)
    parsed_doc = datetime.date.fromisoformat(doc_date)
    feature_vectors: list[FeatureVector] = []
    for _, row in history.iterrows():
        elapsed = float((datetime.date.fromisoformat(row["date"]) - parsed_doc).days)
        feature_vectors.append(
            FeatureVector(
                date=str(row["date"]),
                sentiment_score=float(sentiment_score),
                market_close=float(row["close"]),
                market_volatility=float(row["volatility_5d"]),
                close_change_pct=float(row["close_change_pct"]),
                volatility_change=float(row["volatility_change"]),
                elapsed_time=elapsed,
            )
        )
    feature_vectors = build_last5_sequence(feature_vectors)
    seq = [v.as_list() for v in feature_vectors]
    target_close_norm = float(target_row["close"]) / DEFAULT_CLOSE_SCALE
    target_volatility = float(target_row["volatility_5d"])
    return seq, target_close_norm, target_volatility


def _build_dataset(
    rows: list[EvalRow],
    *,
    chunk_store: pd.DataFrame,
    market_df: pd.DataFrame,
    max_chunks: int,
    lookback_days: int,
    sentiment_cache: dict[str, float] | None = None,
) -> list[TrainingTuple]:
    cache = sentiment_cache if sentiment_cache is not None else {}
    out: list[TrainingTuple] = []
    skipped = Counter()
    for row in rows:
        if row.text not in cache:
            try:
                cache[row.text] = float(analyze_text(row.text)["score"])
            except Exception:
                skipped["sentiment_error"] += 1
                continue
        sentiment_score = cache[row.text]
        seq_target = _build_feature_seq(market_df, row.event_date, sentiment_score)
        if seq_target is None:
            skipped["no_market_history"] += 1
            continue
        feature_seq, target_close_norm, target_volatility = seq_target
        retrieval = build_lookback_tensors(
            chunk_store,
            anchor_date=row.event_date,
            lookback_days=lookback_days,
            max_chunks=max_chunks,
        )
        out.append(
            TrainingTuple(
                doc_id=row.text[:40],
                doc_date=row.event_date,
                feature_seq=feature_seq,
                target_close_norm=target_close_norm,
                target_volatility=target_volatility,
                chunk_embeddings=retrieval.embeddings.numpy(),
                chunk_elapsed=retrieval.elapsed_days.numpy(),
                chunk_mask=retrieval.mask.numpy(),
            )
        )
    if skipped:
        print(f"[ablation] skipped: {dict(skipped)}")
    return out


class _AblationDataset(Dataset):
    def __init__(self, tuples: list[TrainingTuple]):
        self.tuples = tuples

    def __len__(self) -> int:
        return len(self.tuples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.tuples[idx]
        return {
            "x": torch.tensor(item.feature_seq, dtype=torch.float32),
            "y_close": torch.tensor(item.target_close_norm, dtype=torch.float32),
            "y_vol": torch.tensor(item.target_volatility, dtype=torch.float32),
            "chunks": torch.tensor(item.chunk_embeddings, dtype=torch.float32),
            "elapsed": torch.tensor(item.chunk_elapsed, dtype=torch.float32),
            "mask": torch.tensor(item.chunk_mask, dtype=torch.float32),
        }


def _train_variant(
    train_tuples: list[TrainingTuple],
    val_tuples: list[TrainingTuple],
    *,
    use_time_decay: bool,
    use_chunk_attention: bool,
    chunk_embedding_size: int,
    chunk_projection_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _set_all_seeds(seed)
    model = ForecasterModel(
        use_time_decay=use_time_decay,
        use_chunk_attention=use_chunk_attention,
        chunk_embedding_size=chunk_embedding_size,
        chunk_projection_dim=chunk_projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(_AblationDataset(train_tuples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_AblationDataset(val_tuples), batch_size=batch_size, shuffle=False)

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[float] = []
    decay_history: list[float] = []
    chunk_lambda_history: list[float] = []
    train_start = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y_close = batch["y_close"].to(device)
            y_vol = batch["y_vol"].to(device)
            optimizer.zero_grad()
            kwargs = {}
            if use_chunk_attention:
                kwargs = {
                    "chunks": batch["chunks"].to(device),
                    "elapsed_days": batch["elapsed"].to(device),
                    "chunk_mask": batch["mask"].to(device),
                }
            pred = model(x, **kwargs)
            loss = loss_fn(pred[:, 0], y_close) + loss_fn(pred[:, 1], y_vol)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        train_loss = running / max(n, 1)

        model.eval()
        v_running = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y_close = batch["y_close"].to(device)
                y_vol = batch["y_vol"].to(device)
                kwargs = {}
                if use_chunk_attention:
                    kwargs = {
                        "chunks": batch["chunks"].to(device),
                        "elapsed_days": batch["elapsed"].to(device),
                        "chunk_mask": batch["mask"].to(device),
                    }
                pred = model(x, **kwargs)
                loss = loss_fn(pred[:, 0], y_close) + loss_fn(pred[:, 1], y_vol)
                v_running += float(loss.item()) * x.shape[0]
                v_n += x.shape[0]
        val_loss = v_running / max(v_n, 1)
        history.append(val_loss)
        decay_history.append(float(model.time_decay.decay_rate.detach().cpu().item()))
        if use_chunk_attention and model.chunk_pooler is not None:
            chunk_lambda_history.append(float(model.chunk_pooler.decay_rate.detach().cpu().item()))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            print(
                f"[ablation]  epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} time_decay={decay_history[-1]:.4f}"
                + (f" chunk_lambda={chunk_lambda_history[-1]:.4f}" if chunk_lambda_history else "")
            )
    train_elapsed = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation: per-target RMSE + directional accuracy on val set.
    close_se: list[float] = []
    vol_se: list[float] = []
    direction_hits = 0
    direction_total = 0
    last_history_close: list[float] = []
    latencies_ms: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y_close = batch["y_close"].cpu().numpy()
            y_vol = batch["y_vol"].cpu().numpy()
            kwargs = {}
            if use_chunk_attention:
                kwargs = {
                    "chunks": batch["chunks"].to(device),
                    "elapsed_days": batch["elapsed"].to(device),
                    "chunk_mask": batch["mask"].to(device),
                }
            t0 = time.perf_counter()
            pred = model(x, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_item = elapsed_ms / max(x.shape[0], 1)
            latencies_ms.extend([per_item] * x.shape[0])
            pred_close = pred[:, 0].cpu().numpy()
            pred_vol = pred[:, 1].cpu().numpy()
            last_close_norm = x[:, -1, 1].cpu().numpy()  # close index in as_list
            for c_pred, c_true, v_pred, v_true, c_last in zip(pred_close, y_close, pred_vol, y_vol, last_close_norm):
                close_se.append((float(c_pred) - float(c_true)) ** 2)
                vol_se.append((float(v_pred) - float(v_true)) ** 2)
                pred_dir = 1 if c_pred > c_last else -1 if c_pred < c_last else 0
                true_dir = 1 if c_true > c_last else -1 if c_true < c_last else 0
                if pred_dir == true_dir and pred_dir != 0:
                    direction_hits += 1
                direction_total += 1
                last_history_close.append(float(c_last))
    close_rmse = math.sqrt(statistics.mean(close_se)) if close_se else 0.0
    vol_rmse = math.sqrt(statistics.mean(vol_se)) if vol_se else 0.0
    combined_rmse = math.sqrt((close_rmse**2 + vol_rmse**2) / 2)
    directional_accuracy = direction_hits / direction_total if direction_total else 0.0
    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))] if latencies_ms else 0.0

    final_decay = float(model.time_decay.decay_rate.detach().cpu().item())
    final_chunk_lambda = (
        float(model.chunk_pooler.decay_rate.detach().cpu().item())
        if (use_chunk_attention and model.chunk_pooler is not None)
        else None
    )

    metrics = {
        "use_time_decay": use_time_decay,
        "use_chunk_attention": use_chunk_attention,
        "epochs": epochs,
        "best_val_loss": float(best_val_loss),
        "close_rmse": close_rmse,
        "volatility_rmse": vol_rmse,
        "combined_rmse": combined_rmse,
        "directional_accuracy": directional_accuracy,
        "p50_ms": p50,
        "p95_ms": p95,
        "train_elapsed_s": float(train_elapsed),
        "decay_rate_history": decay_history,
        "chunk_lambda_history": chunk_lambda_history,
        "final_decay_rate": final_decay,
        "final_chunk_lambda": final_chunk_lambda,
    }
    return metrics, model


def _heatmap_samples(
    val_tuples: list[TrainingTuple],
    *,
    pooler_state: ForecasterModel,
    device: torch.device,
    sample_count: int = 3,
) -> list[dict[str, Any]]:
    if pooler_state.chunk_pooler is None:
        return []
    pooler_state.eval()
    samples: list[dict[str, Any]] = []
    indices = sorted(
        range(len(val_tuples)),
        key=lambda i: int(val_tuples[i].chunk_mask.sum()),
        reverse=True,
    )[:sample_count]
    with torch.no_grad():
        for idx in indices:
            t = val_tuples[idx]
            chunks = torch.tensor(t.chunk_embeddings, dtype=torch.float32, device=device)
            elapsed = torch.tensor(t.chunk_elapsed, dtype=torch.float32, device=device)
            mask = torch.tensor(t.chunk_mask, dtype=torch.float32, device=device)
            _, weights, decays = pooler_state.chunk_pooler(chunks, elapsed, mask=mask)
            samples.append(
                {
                    "doc_id": t.doc_id,
                    "doc_date": t.doc_date,
                    "actual_chunks": int(mask.sum().item()),
                    "weights": weights.cpu().tolist(),
                    "decay_coeffs": decays.cpu().tolist(),
                    "elapsed_days": elapsed.cpu().tolist(),
                }
            )
    return samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 attention + decay ablation.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default="wf_fold_2")
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--symbol", default="^GSPC")
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--chunk-projection-dim", type=int, default=DEFAULT_CHUNK_PROJECTION_DIM)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--artifact-root", default="/data/artifacts/phase4_attention")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _set_all_seeds(args.seed)
    package_dir = Path(args.data_dir) / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    rows = _load_registry_rows(package_dir)
    fold = _load_fold(package_dir, args.fold_id)
    train_rows, test_rows = _split_by_fold(rows, fold)
    print(f"[ablation] fold={args.fold_id} train_rows={len(train_rows)} test_rows={len(test_rows)}")

    chunk_store_path = resolve_store_path(args.data_dir, args.training_package_id)
    if not chunk_store_path.exists():
        raise SystemExit(f"Chunk store not found: {chunk_store_path}. Run chunk_embedding_store first.")
    chunk_store = load_chunk_store(str(chunk_store_path))
    embedding_size = int(len(chunk_store.iloc[0]["embedding"]))
    print(f"[ablation] chunk_store rows={len(chunk_store)} embedding_size={embedding_size}")

    print(f"[ablation] fetching market history for {args.symbol}...")
    market_df = _fetch_full_history(args.symbol, start="1995-01-01")
    print(f"[ablation] market history rows={len(market_df)}")

    sentiment_cache: dict[str, float] = {}
    print(f"[ablation] building train tuples...")
    train_tuples = _build_dataset(
        train_rows,
        chunk_store=chunk_store,
        market_df=market_df,
        max_chunks=args.max_chunks,
        lookback_days=args.lookback_days,
        sentiment_cache=sentiment_cache,
    )
    print(f"[ablation] building val tuples...")
    val_tuples = _build_dataset(
        test_rows,
        chunk_store=chunk_store,
        market_df=market_df,
        max_chunks=args.max_chunks,
        lookback_days=args.lookback_days,
        sentiment_cache=sentiment_cache,
    )
    print(f"[ablation] train_tuples={len(train_tuples)} val_tuples={len(val_tuples)}")
    if not train_tuples or not val_tuples:
        raise SystemExit("Insufficient tuples after dataset build.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}")

    variants = [
        ("baseline", False, False),
        ("variant_a_only", True, False),
        ("variant_b_only", False, True),
        ("variant_a_plus_b", True, True),
    ]

    results: dict[str, Any] = {}
    heatmaps_by_variant: dict[str, Any] = {}
    for name, use_a, use_b in variants:
        print(f"\n[ablation] === variant: {name} (A={use_a}, B={use_b}) ===")
        result, trained_model = _train_variant(
            train_tuples,
            val_tuples,
            use_time_decay=use_a,
            use_chunk_attention=use_b,
            chunk_embedding_size=embedding_size,
            chunk_projection_dim=args.chunk_projection_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
        )
        results[name] = result
        if use_b:
            heatmaps_by_variant[name] = _heatmap_samples(val_tuples, pooler_state=trained_model, device=device)
        print(
            f"[ablation]  {name}: combined_rmse={result['combined_rmse']:.4f} "
            f"close_rmse={result['close_rmse']:.4f} vol_rmse={result['volatility_rmse']:.4f} "
            f"directional_acc={result['directional_accuracy']:.4f}"
        )

    run_token = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(args.artifact_root) / f"ablation_{run_token}_s{args.seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline": "phase4_attention_ablation",
        "owner": args.owner,
        "training_package_id": args.training_package_id,
        "fold_id": args.fold_id,
        "symbol": args.symbol,
        "max_chunks": args.max_chunks,
        "lookback_days": args.lookback_days,
        "chunk_embedding_size": embedding_size,
        "chunk_projection_dim": args.chunk_projection_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_tuples": len(train_tuples),
        "val_tuples": len(val_tuples),
        "started_at_utc": run_token,
        "device": str(device),
        "results": results,
        "heatmaps": heatmaps_by_variant,
    }
    (artifact_dir / "ablation_table.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines: list[str] = [
        "# Phase 4 Attention + Decay Ablation",
        "",
        f"- training_package_id: `{args.training_package_id}`",
        f"- fold_id: `{args.fold_id}`",
        f"- train_tuples / val_tuples: {len(train_tuples)} / {len(val_tuples)}",
        f"- lookback_days: {args.lookback_days}",
        f"- chunk_projection_dim: {args.chunk_projection_dim}",
        f"- epochs: {args.epochs}",
        f"- seed: {args.seed}",
        "",
        "| variant | A | B | combined_rmse | close_rmse | vol_rmse | directional_acc | final_decay | final_chunk_λ | p95_ms |",
        "|---------|---|---|---------------|------------|----------|-----------------|-------------|---------------|--------|",
    ]
    for name, _, _ in variants:
        r = results[name]
        chunk_l = r["final_chunk_lambda"]
        chunk_l_str = f"{chunk_l:.4f}" if chunk_l is not None else "-"
        md_lines.append(
            f"| {name} | {'on' if r['use_time_decay'] else 'off'} | "
            f"{'on' if r['use_chunk_attention'] else 'off'} | "
            f"{r['combined_rmse']:.4f} | {r['close_rmse']:.4f} | {r['volatility_rmse']:.4f} | "
            f"{r['directional_accuracy']:.4f} | {r['final_decay_rate']:.4f} | {chunk_l_str} | "
            f"{r['p95_ms']:.2f} |"
        )
    (artifact_dir / "ablation_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n[ablation] artifacts written to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
