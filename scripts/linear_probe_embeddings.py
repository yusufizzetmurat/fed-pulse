"""Linear probe over NLP embeddings against the directional target.

Isolation test: does the raw text representation carry directional
information that the TFT failed to extract, or is there nothing in
the embeddings to begin with?

For each encoder, mean-pools chunks per ``event_date`` to get one
embedding per FOMC event, joins to ``events.parquet`` (filtered to
``horizon == 1`` so ``direction_t1d`` matches the prediction window
and one row per event), and trains an L2-regularised
``LogisticRegression`` (regularisation auto-tuned via ``LogisticRegressionCV``)
on each walk-forward fold's training partition to predict
``direction_t1d > 0`` on the test partition.

Reports per-fold accuracy + ROC-AUC + the mean across folds, against
the 53.7% majority-class baseline.

Decision matrix the caller cares about:

- mean accuracy > 53.7% on any encoder -> text embeddings carry
  directional signal the TFT did not extract; architecture was the
  bottleneck; classification head + 4-layer NLP remain viable
- mean accuracy <= 53.7% across every encoder -> text definitively
  lacks daily-resolution directional signal; abandon directional
  prediction and pivot to volatility / regime

Usage::

    python -m scripts.linear_probe_embeddings \\
        --training-package-id tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0 \\
        --encoders finbert finbert_fed_adjacent voyage_finance_2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR


def _discover_cached_encoders() -> list[str]:
    """Scan ``data/raw/embeddings/`` and return one alias per cache file.

    The cache filenames follow ``<alias>_<rev>.parquet``; the alias is
    every char before the final underscore-then-revision tail. Some
    aliases (``bge_large_en_v15``, ``nomic_embed_text_v15``) carry
    underscores themselves, so the naive last-underscore split is
    incorrect. Resolve by stripping the file extension and matching
    against a curated suffix list.
    """

    cache_dir = DATA_DIR / "raw" / "embeddings"
    if not cache_dir.exists():
        return []
    seen: set[str] = set()
    for path in sorted(cache_dir.glob("*.parquet")):
        stem = path.stem
        # Curated alias list -- prefer the longest matching prefix so
        # ``bge_large_en_v15_d4aa..`` resolves to ``bge_large_en_v15``,
        # not ``bge_large_en``.
        for candidate in sorted(
            (
                "bert_base_fed_adjacent",
                "bert_base_uncased",
                "bge_large_en_v15",
                "finbert",
                "finbert_fed_adjacent",
                "finbert_fomc",
                "fomc_roberta",
                "nomic_embed_text_v15",
                "voyage_finance_2",
            ),
            key=len,
            reverse=True,
        ):
            if stem.startswith(candidate + "_"):
                seen.add(candidate)
                break
    return sorted(seen)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package id under ``data/processed/<id>``.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=None,
        help=(
            "Encoder aliases to probe. Each must resolve via glob to "
            "``data/raw/embeddings/<alias>_*.parquet``. When omitted "
            "(default), every cached encoder under "
            "``data/raw/embeddings/`` enters the bake-off."
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=("wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"),
        help="Walk-forward fold ids to evaluate against.",
    )
    parser.add_argument(
        "--baseline-accuracy",
        type=float,
        default=0.537,
        help="Majority-class baseline (default 0.537 = +1-class share).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Per-fold + aggregate results.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="LogisticRegressionCV solver max-iter. Default 2000.",
    )
    return parser.parse_args()


_KNOWN_ALIASES: tuple[str, ...] = (
    "bert_base_fed_adjacent",
    "bert_base_uncased",
    "bge_large_en_v15",
    "finbert_fed_adjacent",
    "finbert_fomc",
    "finbert",
    "fomc_roberta",
    "nomic_embed_text_v15",
    "voyage_finance_2",
)


def _canonical_alias_for_file(stem: str) -> str | None:
    """Return the LONGEST known alias the filename starts with (alias + underscore)."""
    candidates = [
        alias
        for alias in _KNOWN_ALIASES
        if stem.startswith(alias + "_")
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _resolve_embedding_parquet(alias: str) -> Path:
    """Find the parquet whose CANONICAL alias matches ``alias`` exactly.

    Naive glob ``<alias>_*.parquet`` over-matches: globbing on ``finbert``
    swallows ``finbert_fomc_*.parquet`` and ``finbert_fed_adjacent_*.parquet``
    too, producing silent encoder duplication in the leaderboard. Resolve
    by walking the cache directory, computing the LONGEST-prefix alias
    for each file, and returning only files whose canonical alias
    equals the requested one.
    """
    cache_dir = DATA_DIR / "raw" / "embeddings"
    matches: list[Path] = []
    for path in sorted(cache_dir.glob("*.parquet")):
        canonical = _canonical_alias_for_file(path.stem)
        if canonical == alias:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No embedding parquet has canonical alias={alias!r} under "
            f"{cache_dir}. Known aliases: {_KNOWN_ALIASES}"
        )
    if len(matches) > 1:
        print(
            f"  warning: multiple parquets resolve to alias={alias!r}: "
            f"{[p.name for p in matches]}; picking {matches[-1].name}",
            file=sys.stderr,
        )
    return matches[-1]


def _pool_embeddings_per_event(emb_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-pool chunks within each event_date to get one vector per event."""
    if "event_date" not in emb_df.columns or "embedding" not in emb_df.columns:
        raise ValueError(
            f"Embedding parquet missing required columns; got {list(emb_df.columns)}"
        )
    emb_df = emb_df.copy()
    emb_df["event_date"] = emb_df["event_date"].astype(str)
    emb_df["embedding"] = emb_df["embedding"].apply(
        lambda v: np.asarray(v, dtype=np.float32) if v is not None else None
    )
    grouped: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for date, vec in zip(emb_df["event_date"], emb_df["embedding"]):
        if vec is None or len(vec) == 0:
            continue
        running = grouped.get(date)
        if running is None:
            grouped[date] = vec.astype(np.float64)
            counts[date] = 1
        else:
            grouped[date] = running + vec.astype(np.float64)
            counts[date] += 1
    rows = [
        {"event_date": date, "embedding": (grouped[date] / counts[date]).astype(np.float32)}
        for date in sorted(grouped)
    ]
    return pd.DataFrame(rows)


def _load_event_targets(package_dir: Path) -> pd.DataFrame:
    """Load events.parquet filtered to horizon=1 with non-zero direction."""
    events = pd.read_parquet(package_dir / "events.parquet")
    h1 = events[events["horizon"] == 1].copy()
    h1["event_date"] = h1["event_date"].astype(str)
    h1 = h1[h1["direction_t1d"].isin([-1, 1])]   # drop the 4 zero-rows
    # One row per event_date (the per-asset multiplicity has already
    # been collapsed during the event_row build, but defensive dedupe).
    h1 = h1.drop_duplicates(subset=["event_date"], keep="first")
    return h1[["event_date", "direction_t1d"]].reset_index(drop=True)


def _load_fold_manifest(package_dir: Path) -> dict[str, dict[str, str]]:
    """Return ``fold_id -> {train_start, train_end, test_start, test_end}``."""
    manifest = json.loads(
        (package_dir / "fold_manifest_expanding_walk_forward.json").read_text()
    )
    out: dict[str, dict[str, str]] = {}
    for fold in manifest.get("folds", []):
        fid = fold.get("fold_id")
        if not fid:
            continue
        out[fid] = {
            "train_start": str(fold.get("train_start", "")),
            "train_end": str(fold.get("train_end", "")),
            "val_start": str(fold.get("val_start", "")),
            "val_end": str(fold.get("val_end", "")),
            "test_start": str(fold.get("test_start", "")),
            "test_end": str(fold.get("test_end", "")),
        }
    return out


def _evaluate_fold(
    *,
    joined: pd.DataFrame,
    fold_id: str,
    fold_dates: dict[str, str],
    max_iter: int,
) -> dict[str, Any]:
    """Fit LogisticRegression on this fold's train slice, evaluate on test."""

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    train_mask = (
        (joined["event_date"] >= fold_dates["train_start"])
        & (joined["event_date"] < fold_dates["val_start"])
    )
    test_mask = (
        (joined["event_date"] >= fold_dates["test_start"])
        & (joined["event_date"] < fold_dates.get("test_end", "9999-12-31"))
        | (joined["event_date"] == fold_dates["test_start"])
    )
    # The fold manifest uses [train_start, val_start) ; [test_start, test_end)
    # half-open windows. The OR clause above defends against
    # zero-length test sets when test_start == test_end on the
    # final fold's calendar boundary.
    train_df = joined.loc[train_mask]
    test_df = joined.loc[test_mask]

    if len(train_df) < 20 or len(test_df) < 5:
        return {
            "fold_id": fold_id,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "skipped": "partition too small for a defensible fit",
        }

    X_train = np.vstack(train_df["embedding"].values).astype(np.float32)
    y_train = (train_df["direction_t1d"] > 0).astype(np.int64).values
    X_test = np.vstack(test_df["embedding"].values).astype(np.float32)
    y_test = (test_df["direction_t1d"] > 0).astype(np.int64).values

    if len(np.unique(y_train)) < 2:
        return {
            "fold_id": fold_id,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "skipped": "train slice carries only one class",
        }

    # LogisticRegressionCV picks C via 5-fold cross-validation on the
    # training slice; defends against the high-dim / small-N
    # overfitting risk that a fixed-C LogisticRegression would hit.
    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="roc_auc",
        max_iter=max_iter,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)

    accuracy = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    auc = (
        float(roc_auc_score(y_test, proba))
        if len(np.unique(y_test)) > 1
        else None
    )
    return {
        "fold_id": fold_id,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best_C": float(clf.C_[0]),
        "accuracy": accuracy,
        "f1_macro": f1,
        "auc": auc,
    }


def main(argv: list[str] | None = None) -> int:
    # Quiet sklearn FutureWarnings -- they obscure the leaderboard
    # and the deprecations are scheduled for sklearn 1.10, not load-
    # bearing for this short-lived script.
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    args = _parse_args()
    package_dir = DATA_DIR / "processed" / args.training_package_id

    encoders = args.encoders or _discover_cached_encoders()
    if not encoders:
        print("  no encoders specified and no parquets discovered under data/raw/embeddings/")
        return 1

    print("==== linear probe over embeddings (NLP bake-off) ====")
    print(f"  package:   {args.training_package_id}")
    print(f"  encoders:  {encoders}")
    print(f"  folds:     {list(args.folds)}")
    print(f"  baseline:  {args.baseline_accuracy:.3f}")
    print()

    target_df = _load_event_targets(package_dir)
    fold_dates = _load_fold_manifest(package_dir)
    print(f"  events at horizon=1 (non-zero direction): {len(target_df)}")
    print(f"  folds in manifest:                        {list(fold_dates.keys())}")
    print()

    summary: dict[str, Any] = {}
    for alias in encoders:
        print(f"---- {alias} ----")
        try:
            emb_path = _resolve_embedding_parquet(alias)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        print(f"  parquet:   {emb_path.name}")
        raw_emb = pd.read_parquet(emb_path)
        pooled = _pool_embeddings_per_event(raw_emb)
        joined = pooled.merge(target_df, on="event_date", how="inner")
        if joined.empty:
            print("  no overlap between embedding event_dates and target events")
            continue
        emb_dim = int(joined.iloc[0]["embedding"].shape[0])
        print(
            f"  events joined: {len(joined)}  "
            f"(embedding dim={emb_dim}, dropped "
            f"{len(target_df) - len(joined)} unmatched targets)"
        )
        fold_results: list[dict[str, Any]] = []
        for fold_id in args.folds:
            if fold_id not in fold_dates:
                print(f"  fold {fold_id} not in manifest; skipping")
                continue
            result = _evaluate_fold(
                joined=joined,
                fold_id=fold_id,
                fold_dates=fold_dates[fold_id],
                max_iter=args.max_iter,
            )
            fold_results.append(result)
        summary[alias] = fold_results
        _print_encoder_table(alias, fold_results, baseline=args.baseline_accuracy)
        print()

    print("==== NLP encoder leaderboard (sorted by mean directional accuracy) ====")
    rows: list[dict[str, Any]] = []
    for alias in encoders:
        fold_rows = [
            r
            for r in summary.get(alias, [])
            if "skipped" not in r and r.get("accuracy") is not None
        ]
        if not fold_rows:
            continue
        mean_acc = float(np.mean([r["accuracy"] for r in fold_rows]))
        std_acc = float(np.std([r["accuracy"] for r in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0
        mean_f1 = float(np.mean([r["f1_macro"] for r in fold_rows]))
        aucs = [r["auc"] for r in fold_rows if r["auc"] is not None]
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        rows.append(
            {
                "encoder": alias,
                "n_folds": len(fold_rows),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1_macro": mean_f1,
                "mean_auc": mean_auc,
                "vs_baseline": mean_acc - args.baseline_accuracy,
                "beats_baseline": mean_acc > args.baseline_accuracy,
            }
        )

    rows.sort(key=lambda r: r["mean_accuracy"], reverse=True)

    header = (
        f"  {'rank':>4}  {'encoder':<26}{'folds':>7}"
        f"{'acc (mean ± std)':>24}{'f1_macro':>12}{'auc':>10}"
        f"{'vs baseline':>14}  beat?"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, r in enumerate(rows, start=1):
        beat_marker = "  yes" if r["beats_baseline"] else "   no"
        acc_str = f"{r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f}"
        print(
            f"  {idx:>4}  {r['encoder']:<26}{r['n_folds']:>7}"
            f"{acc_str:>24}{r['mean_f1_macro']:>12.3f}{r['mean_auc']:>10.3f}"
            f"{r['vs_baseline']:>+14.3f}{beat_marker}"
        )

    print()
    above_baseline = [r for r in rows if r["beats_baseline"]]
    if above_baseline:
        winner = above_baseline[0]
        print(
            f"  Winner: {winner['encoder']} (mean accuracy "
            f"{winner['mean_accuracy']:.3f}, "
            f"{winner['vs_baseline']:+.3f} vs {args.baseline_accuracy:.3f} baseline)"
        )
    else:
        best = rows[0] if rows else None
        if best:
            print(
                f"  No encoder beats the {args.baseline_accuracy:.3f} majority-class "
                f"baseline. Best so far: {best['encoder']} at "
                f"{best['mean_accuracy']:.3f} ({best['vs_baseline']:+.3f}). "
                "Text definitively lacks daily directional signal."
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  json written to {out_path}")

    return 0


def _print_encoder_table(alias: str, rows: list[dict[str, Any]], *, baseline: float) -> None:
    header = (
        f"  {'fold':<14}{'n_train':>10}{'n_test':>8}{'accuracy':>12}"
        f"{'f1_macro':>12}{'auc':>10}{'C':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if "skipped" in r:
            print(
                f"  {r['fold_id']:<12}{r['n_train']:>10}{r['n_test']:>8}  {r['skipped']}"
            )
            continue
        acc_str = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "None"
        f1_str = f"{r['f1_macro']:.3f}" if r.get("f1_macro") is not None else "None"
        auc_str = f"{r['auc']:.3f}" if r.get("auc") is not None else "None"
        c_str = f"{r['best_C']:.3g}" if r.get("best_C") is not None else "None"
        print(
            f"  {r['fold_id']:<14}{r['n_train']:>10}{r['n_test']:>8}"
            f"{acc_str:>12}{f1_str:>12}{auc_str:>10}{c_str:>10}"
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
