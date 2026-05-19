"""Two structural sanity checks on the events parquet + market data.

Check 1 -- Feature Sparsity & Variance (PCA)
    Loads the 35-dim rich-feature vector for every event, computes the
    fraction of exact-zero cells, and fits PCA after standardisation
    to report how many components explain 95% of the variance. If
    that count is tiny (< 5), the rich-feature matrix is effectively
    low-rank and the extra dimensions are noise.

Check 2 -- Event-Day Volatility Bias
    Loads the full SPX daily close series for the event-date range,
    computes daily absolute return and ``vol_post_10d - vol_pre_10d``
    for every trading day, then compares the EVENT-DAY distribution
    against the NON-EVENT-DAY distribution. Two-sample t-tests on the
    means and Kolmogorov-Smirnov tests on the full distributions
    answer "are event days actually special?".

The two checks are independent; either may be run in isolation by
commenting out the other ``main()`` step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.models.config import (
    FEATURE_SIZE,
    RICH_CREDIBILITY_SLICE,
    RICH_FEATURE_SIZE,
    RICH_LINGUISTIC_DIM,
    RICH_LINGUISTIC_SLICE,
    RICH_MP_SURPRISE_SLICE,
    RICH_MULTI_AXIS_SLICE,
)


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
        "--linguistic-parquet",
        default=None,
        help="Override path to linguistic_features.parquet. Defaults to "
        "<package_dir>/linguistic_features.parquet.",
    )
    parser.add_argument(
        "--mp-surprise-parquet",
        default=None,
        help="Override path to mp_surprises.parquet. Defaults to "
        "data/external/fred/mp_surprises.parquet.",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=10,
        help="Trading-day window for the pre/post volatility shift "
        "comparison in Check 2 (default 10 to match the project's "
        "event-study target frame in PR #172).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Diagnostic results, machine-readable.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Check 1: rich-feature sparsity + PCA
# ---------------------------------------------------------------------------


_LINGUISTIC_COLUMNS: tuple[str, ...] = (
    "topic_share_inflation",
    "topic_share_employment",
    "topic_share_financial_stability",
    "topic_share_growth",
    "topic_share_balance_sheet",
    "topic_share_misc_1",
    "topic_share_misc_2",
    "topic_share_misc_3",
    "hedge_density",
    "comparison_density",
    "forward_density",
    "concrete_ratio",
    "hawk_dove_asymmetry",
    "log_token_count",
    "pivot_distance",
)


def _stance_one_hot(value: Any) -> tuple[float, float, float, float]:
    """Return (hawk, dove, neutral, missing) one-hot for ``axis_stance``."""
    if not isinstance(value, str):
        return (0.0, 0.0, 0.0, 1.0)
    val = value.strip().lower()
    if val == "hawkish":
        return (1.0, 0.0, 0.0, 0.0)
    if val == "dovish":
        return (0.0, 1.0, 0.0, 0.0)
    if val == "neutral":
        return (0.0, 0.0, 1.0, 0.0)
    return (0.0, 0.0, 0.0, 1.0)


def _build_rich_matrix(
    package_dir: Path,
    linguistic_path: Path,
    mp_surprise_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Construct one rich-feature vector per unique event_date.

    Mirrors the layout that ``FeatureVector.as_rich_list`` emits at
    positions [6:35]; we skip the market block [0:6] because it
    varies per-bar and is not meaningful at event-level. The output
    is (n_events, 29) -- the rich-extra slice only.
    """

    events = pd.read_parquet(package_dir / "events.parquet")
    # One row per (event_date, source) collapsed view; deduplicate to
    # one row per event_date for the PCA.
    events = events.drop_duplicates(subset=["event_date"], keep="first")
    events["event_date"] = events["event_date"].astype(str)

    # Linguistic join keyed on text_hash. Pad missing rows with zeros.
    if linguistic_path.exists():
        ling = pd.read_parquet(linguistic_path)
        ling = ling.set_index("text_hash")
    else:
        ling = pd.DataFrame()

    # MP-surprise join keyed on event_date.
    if mp_surprise_path.exists():
        mp = pd.read_parquet(mp_surprise_path)
        mp["event_date"] = mp["event_date"].astype(str)
        mp = mp.set_index("event_date")
    else:
        mp = pd.DataFrame()

    columns = [
        # credibility (4)
        "credibility_drift_score",
        "credibility_realized_vs_stated_gap",
        "credibility_market_implied_gap",
        "credibility_months_since_reversal",
        # linguistic (15)
        *_LINGUISTIC_COLUMNS,
        # mp-surprise (4)
        "mp_surprise_level",
        "mp_surprise_path_factor",
        "fed_info_factor",
        "mp_is_intermeeting",
        # multi-axis (6) -- one-hot reconstruction
        "stance_hawk",
        "stance_dove",
        "stance_neutral",
        "time_label_forward",
        "certain_label_certain",
        "stance_missing",
    ]

    rows: list[list[float]] = []
    for _, ev in events.iterrows():
        row: list[float] = []
        # credibility
        for col in (
            "credibility_drift_score",
            "credibility_realized_vs_stated_gap",
            "credibility_market_implied_gap",
            "credibility_months_since_reversal",
        ):
            v = ev.get(col)
            row.append(0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
        # linguistic
        text_hash = ev.get("text_hash")
        if text_hash and not ling.empty and text_hash in ling.index:
            ling_row = ling.loc[text_hash]
            for c in _LINGUISTIC_COLUMNS:
                v = ling_row.get(c) if isinstance(ling_row, pd.Series) else None
                row.append(
                    0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
                )
        else:
            row.extend([0.0] * RICH_LINGUISTIC_DIM)
        # mp-surprise
        edate = ev.get("event_date")
        if edate and not mp.empty and edate in mp.index:
            mp_row = mp.loc[edate]
            for c in (
                "mp_surprise_level",
                "mp_surprise_path_factor",
                "fed_info_factor",
                "mp_is_intermeeting",
            ):
                v = mp_row.get(c) if isinstance(mp_row, pd.Series) else None
                if isinstance(v, bool):
                    row.append(1.0 if v else 0.0)
                else:
                    row.append(
                        0.0
                        if v is None or (isinstance(v, float) and np.isnan(v))
                        else float(v)
                    )
        else:
            row.extend([0.0] * 4)
        # multi-axis Option-A slot (six floats)
        hawk, dove, neutral, missing = _stance_one_hot(ev.get("axis_stance"))
        time_raw = ev.get("axis_time_label")
        certain_raw = ev.get("axis_certain_label")
        time_fwd = (
            1.0
            if isinstance(time_raw, str) and time_raw.strip().lower() == "forward looking"
            else 0.0
        )
        certain_cert = (
            1.0
            if isinstance(certain_raw, str) and certain_raw.strip().lower() == "certain"
            else 0.0
        )
        row.extend([hawk, dove, neutral, time_fwd, certain_cert, missing])
        rows.append(row)

    return np.array(rows, dtype=np.float64), columns


def _run_check_1(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    package_dir = DATA_DIR / "processed" / args.training_package_id
    linguistic_path = (
        Path(args.linguistic_parquet)
        if args.linguistic_parquet
        else package_dir / "linguistic_features.parquet"
    )
    mp_path = (
        Path(args.mp_surprise_parquet)
        if args.mp_surprise_parquet
        else DATA_DIR / "external" / "fred" / "mp_surprises.parquet"
    )

    X, columns = _build_rich_matrix(package_dir, linguistic_path, mp_path)
    n_events, n_features = X.shape
    print("==== Check 1: Rich-Feature Sparsity + PCA ====")
    print(f"  matrix shape:                       {X.shape}")
    print(f"  total cells:                        {X.size}")
    zero_cells = int(np.sum(X == 0.0))
    sparsity = zero_cells / X.size
    print(f"  exact-zero cells:                   {zero_cells} ({sparsity:.2%})")
    print()

    # Per-column zero rate
    col_zero_rate = (X == 0.0).mean(axis=0)
    col_std = X.std(axis=0)
    print(f"  {'feature':<42}{'zero rate':>12}{'std':>12}")
    print("  " + "-" * 66)
    for name, zr, std in zip(columns, col_zero_rate, col_std):
        marker = "  (dead)" if std < 1e-9 else ""
        print(f"  {name:<42}{zr:>12.2%}{std:>12.4f}{marker}")
    print()

    n_dead = int((col_std < 1e-9).sum())
    print(f"  dead columns (std < 1e-9):          {n_dead}/{n_features}")
    print()

    # Drop dead columns for PCA so the standard-scaler doesn't NaN.
    live_mask = col_std > 1e-9
    X_live = X[:, live_mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_live)

    pca = PCA(n_components=min(X_scaled.shape))
    pca.fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    thresholds = (0.50, 0.80, 0.90, 0.95, 0.99)
    print(f"  PCA on standardised LIVE columns ({X_live.shape[1]} dims):")
    print(f"  {'cumulative variance':<24}{'components needed':>22}")
    print("  " + "-" * 46)
    components_for = {}
    for t in thresholds:
        n = int(np.searchsorted(cum_var, t) + 1)
        n = min(n, len(cum_var))
        components_for[t] = n
        print(f"  >= {t:.2f}                {n:>22d}")
    print()

    # Print top-10 explained-variance ratios for inspection
    print(f"  top 10 components' explained-variance ratios:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10], start=1):
        print(f"    PC{i:>2}: {ratio:.4f}   (cum {cum_var[i-1]:.4f})")
    print()

    verdict = ""
    n95 = components_for[0.95]
    if n95 <= 4:
        verdict = (
            f"VERDICT: thin/sparse -- {n95} components explain 95% of variance. "
            "The 35-dim rich vector is effectively low-rank; most positions "
            "are noise."
        )
    elif n95 <= 10:
        verdict = (
            f"VERDICT: moderate dimensionality -- {n95} components explain 95%. "
            "Real signal in a handful of axes; the rest is redundant."
        )
    else:
        verdict = (
            f"VERDICT: dense -- {n95} components needed for 95%. "
            "The rich-feature space is using its dimensions."
        )
    print(f"  {verdict}")
    print()

    return {
        "matrix_shape": list(X.shape),
        "sparsity": float(sparsity),
        "n_dead_columns": n_dead,
        "components_for_thresholds": {f"{t:.2f}": int(n) for t, n in components_for.items()},
        "per_column_zero_rate": {
            columns[i]: float(col_zero_rate[i]) for i in range(n_features)
        },
        "per_column_std": {columns[i]: float(col_std[i]) for i in range(n_features)},
        "top_pcs": [float(r) for r in pca.explained_variance_ratio_[:10]],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Check 2: event-day vs non-event-day volatility bias
# ---------------------------------------------------------------------------


def _run_check_2(args: argparse.Namespace) -> dict[str, Any]:
    from scipy import stats

    package_dir = DATA_DIR / "processed" / args.training_package_id
    events = pd.read_parquet(package_dir / "events.parquet")
    events["event_date"] = events["event_date"].astype(str)
    event_dates = set(events["event_date"].unique())

    # SPX continuous bars cached under the package.
    market_path = package_dir / "_market_cache" / "GSPC.parquet"
    if not market_path.exists():
        raise SystemExit(
            f"Continuous SPX cache not found at {market_path}. "
            "Re-run event_dataset_builder first to populate the cache."
        )
    spx = pd.read_parquet(market_path)
    # Common columns: 'date', 'close', 'open', 'high', 'low', 'volume' --
    # confirm date column name dynamically.
    date_col = "date" if "date" in spx.columns else "Date"
    close_col = "close" if "close" in spx.columns else "Close"
    spx[date_col] = pd.to_datetime(spx[date_col]).dt.strftime("%Y-%m-%d")
    spx = spx.sort_values(date_col).reset_index(drop=True)

    # Restrict to the date range of the event corpus to avoid
    # comparing apples (calm 1990s) to oranges (volatile 2020).
    event_dates_sorted = sorted(event_dates)
    earliest = event_dates_sorted[0]
    latest = event_dates_sorted[-1]
    in_range = (spx[date_col] >= earliest) & (spx[date_col] <= latest)
    spx_window = spx.loc[in_range].reset_index(drop=True)

    # Daily log-return and absolute return
    spx_window["log_return"] = np.log(
        spx_window[close_col] / spx_window[close_col].shift(1)
    )
    spx_window["abs_return"] = spx_window["log_return"].abs()

    # Rolling vol windows for the volatility shift target
    window = int(args.vol_window)
    spx_window["pre_vol"] = (
        spx_window["log_return"].rolling(window=window, min_periods=window).std()
    )
    spx_window["post_vol"] = (
        spx_window["log_return"]
        .shift(-window)
        .rolling(window=window, min_periods=window)
        .std()
    )
    spx_window["vol_shift"] = spx_window["post_vol"] - spx_window["pre_vol"]

    spx_window["is_event"] = spx_window[date_col].isin(event_dates)

    valid = spx_window.dropna(subset=["abs_return", "vol_shift"]).reset_index(drop=True)
    event_mask = valid["is_event"]
    n_event = int(event_mask.sum())
    n_non = int((~event_mask).sum())

    print("==== Check 2: Event-Day vs Non-Event-Day Volatility Bias ====")
    print(f"  date range:               {earliest} -> {latest}")
    print(f"  trading days in window:   {len(valid)}")
    print(f"  event days observed:      {n_event}")
    print(f"  non-event days:           {n_non}")
    print()

    def _summary(group: pd.Series) -> dict[str, float]:
        return {
            "n": int(group.shape[0]),
            "mean": float(group.mean()),
            "median": float(group.median()),
            "std": float(group.std()),
            "abs_mean": float(group.abs().mean()),
        }

    abs_event = valid.loc[event_mask, "abs_return"]
    abs_non = valid.loc[~event_mask, "abs_return"]
    vs_event = valid.loc[event_mask, "vol_shift"]
    vs_non = valid.loc[~event_mask, "vol_shift"]

    abs_event_stats = _summary(abs_event)
    abs_non_stats = _summary(abs_non)
    vs_event_stats = _summary(vs_event)
    vs_non_stats = _summary(vs_non)

    print(f"  abs_return  event mean:   {abs_event_stats['mean']:.6f}  (std {abs_event_stats['std']:.6f})")
    print(f"  abs_return  non-event:    {abs_non_stats['mean']:.6f}  (std {abs_non_stats['std']:.6f})")
    print(f"  abs_return  ratio:        {abs_event_stats['mean'] / max(abs_non_stats['mean'], 1e-12):.3f}x")
    print()
    print(f"  vol_shift   event mean:   {vs_event_stats['mean']:.6f}  (std {vs_event_stats['std']:.6f})")
    print(f"  vol_shift   non-event:    {vs_non_stats['mean']:.6f}  (std {vs_non_stats['std']:.6f})")
    print(f"  vol_shift   |mean| ratio: {abs(vs_event_stats['mean']) / max(abs(vs_non_stats['mean']), 1e-12):.3f}x")
    print()

    # Welch t-tests (unequal variance) on means
    t_abs = stats.ttest_ind(abs_event, abs_non, equal_var=False)
    t_vs = stats.ttest_ind(vs_event, vs_non, equal_var=False)
    ks_abs = stats.ks_2samp(abs_event, abs_non)
    ks_vs = stats.ks_2samp(vs_event, vs_non)
    print("  statistical tests:")
    print(f"    abs_return  Welch t-test    t={t_abs.statistic:+.3f}  p={t_abs.pvalue:.4g}")
    print(f"    abs_return  Kolmogorov-Smirnov D={ks_abs.statistic:.3f}  p={ks_abs.pvalue:.4g}")
    print(f"    vol_shift   Welch t-test    t={t_vs.statistic:+.3f}  p={t_vs.pvalue:.4g}")
    print(f"    vol_shift   Kolmogorov-Smirnov D={ks_vs.statistic:.3f}  p={ks_vs.pvalue:.4g}")
    print()

    def _interpret(p_t: float, p_ks: float) -> str:
        if p_t < 0.01 and p_ks < 0.01:
            return "STRONG difference (event days are significantly distinct)"
        if p_t < 0.05 or p_ks < 0.05:
            return "WEAK difference (some evidence event days differ)"
        return "INDISTINGUISHABLE (event days look like generic market noise)"

    abs_verdict = _interpret(t_abs.pvalue, ks_abs.pvalue)
    vs_verdict = _interpret(t_vs.pvalue, ks_vs.pvalue)
    print(f"  verdict abs_return:       {abs_verdict}")
    print(f"  verdict vol_shift:        {vs_verdict}")
    print()

    return {
        "date_range": [earliest, latest],
        "n_event": n_event,
        "n_non_event": n_non,
        "abs_return": {
            "event": abs_event_stats,
            "non_event": abs_non_stats,
            "ratio_mean": abs_event_stats["mean"] / max(abs_non_stats["mean"], 1e-12),
            "ttest": {"t": float(t_abs.statistic), "p": float(t_abs.pvalue)},
            "ks": {"D": float(ks_abs.statistic), "p": float(ks_abs.pvalue)},
            "verdict": abs_verdict,
        },
        "vol_shift": {
            "event": vs_event_stats,
            "non_event": vs_non_stats,
            "ttest": {"t": float(t_vs.statistic), "p": float(t_vs.pvalue)},
            "ks": {"D": float(ks_vs.statistic), "p": float(ks_vs.pvalue)},
            "verdict": vs_verdict,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    out: dict[str, Any] = {}
    out["check_1_rich_pca"] = _run_check_1(args)
    out["check_2_event_vs_non"] = _run_check_2(args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=float))
        print(f"  json written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
