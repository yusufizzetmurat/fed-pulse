"""Align the typed Fed communication corpus with the intraday realized-vol series.

Produces the two leak-safe tables the gated text↔market fusion model consumes:

  1. text↔outcome pairs (communication-level) — each communication paired with
     the forward realized variance that follows it, starting the first trading
     day strictly AFTER the text date. Feeds the InfoNCE contrastive objective
     and the supervised text head.
  2. daily fusion frame (trading-day-level) — HAR/market features at the forecast
     origin, the target forward RV, and a reference to the most-recent
     communication known by that origin (with its age in trading days and type).
     Feeds the gated forecaster, whose gate can discount stale/irrelevant text.

Embargo rule (uniform across text types, so date-only speeches are safe): a
communication dated D contributes only to windows that begin on the first
trading day > D, and a forecast origin D may use only communications dated ≤ D.
The forward RV target itself spans t+1..t+h, so no target overlaps its features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from app.config import DATA_DIR
from app.data.intraday_realized import DEFAULT_RV_PARQUET
from app.data.intraday_rv_forecast import _EPS, _har_lags
from app.data.fed_comms_scrape import DEFAULT_CORPUS_PARQUET

DEFAULT_OUT_DIR = DATA_DIR / "processed" / "fed_comms_fusion"
DEFAULT_MP_SURPRISE_PARQUET = DATA_DIR / "external" / "fred" / "mp_surprises.parquet"
DEFAULT_MARKET_CACHE_DIR = (
    DATA_DIR / "processed" / "tp_v3_full_rebuild_2026_05_30" / "_market_cache"
)
DEFAULT_HORIZONS = (1, 5, 22)
_DOC_TYPES = ("statement", "minutes", "press_conference", "speech", "testimony")

# Market-derived MP-surprise columns attached per trading day via an as-of join
# on the most-recent FOMC statement. Filled with the neutral value below before
# the first statement (no surprise is known yet).
_SURPRISE_COLUMNS = ("surprise_level", "surprise_path", "surprise_info")
_SURPRISE_NEUTRAL = 0.0


def _forward_log_rv_windows(rv: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    """For each horizon h: log mean RV over t+1..t+h (NaN where it runs off)."""

    n = len(rv)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        col = np.full(n, np.nan)
        for t in range(n - h):
            col[t] = np.log(rv[t + 1 : t + 1 + h].mean() + _EPS)
        out[h] = col
    return out


# Fair-target measures: name → (raw series from the RV parquet, log-transform?).
# rv/volume/downside are positive → log; jump-share ∈ [0,1) → identity. The two
# corr_* second-moment targets are trailing cross-asset correlations ∈ [−1,1] →
# identity (same case as jump); their daily columns are merged onto rv_df in
# build() from the market cache, so a frame without them simply omits them. The
# rate_vol_* targets are trailing realized vol of daily Treasury-yield changes
# (strictly positive → log, same case as rv); likewise merged from the cache.
MEASURES = (
    "rv",
    "volume",
    "downside",
    "jump",
    "corr_tnx",
    "corr_dxy",
    "rate_vol_2y",
    "rate_vol_10y",
)

# Correlation measures live in their own daily columns (not raw RV fields). The
# trailing window guarantees the early span is NaN until it is full.
_CORR_MEASURES = ("corr_tnx", "corr_dxy")
_CORR_CLIP = 0.999  # keep targets strictly inside (−1, 1) for numerical safety

# Interest-rate realized-vol measures: trailing-window std of daily yield CHANGES
# (Δyield, not log returns — yields are levels). Like corr_*, they live in their
# own daily columns merged onto rv_df from the market cache, so a cache-less
# frame omits them. Unlike corr_*, vol is strictly positive → log lags/targets.
_RATE_VOL_MEASURES = ("rate_vol_2y", "rate_vol_10y")


def _measure_raw(rv_df: Any, measure: str) -> tuple[np.ndarray, bool]:
    """Raw daily series + whether it is log-transformed for lags/targets."""

    rv = rv_df["rv"].to_numpy(dtype=np.float64)
    if measure == "rv":
        return rv, True
    if measure == "volume":
        return rv_df["rvol"].to_numpy(dtype=np.float64), True
    if measure == "downside":
        return rv_df["rs_neg"].to_numpy(dtype=np.float64), True
    if measure == "jump":  # jump-variation share (RV−BV)₊/RV, bounded in [0,1)
        bv = rv_df["bv"].to_numpy(dtype=np.float64)
        return np.maximum(rv - bv, 0.0) / (rv + _EPS), False
    if measure in _CORR_MEASURES:  # trailing cross-asset correlation ∈ [−1,1]
        c = rv_df[measure].to_numpy(dtype=np.float64)
        return np.clip(c, -_CORR_CLIP, _CORR_CLIP), False
    if measure in _RATE_VOL_MEASURES:  # trailing realized vol of Δyield, > 0 → log
        return rv_df[measure].to_numpy(dtype=np.float64), True
    raise ValueError(f"unknown measure {measure!r}")


def _measure_present(rv_df: Any, measure: str) -> bool:
    """Whether the raw column(s) a measure needs are in rv_df.

    corr_* and rate_vol_* live in their own daily columns merged from the market
    cache in build(); a cache-less frame (e.g. the unit-test fixture) omits them.
    """

    if measure in _CORR_MEASURES or measure in _RATE_VOL_MEASURES:
        return measure in rv_df.columns
    return True


_CORR_WINDOW = 22  # trailing trading days for the realized-correlation estimate
_RATE_VOL_WINDOW = 22  # trailing trading days for the yield-change realized-vol estimate


def _trailing_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Trailing Pearson correlation of two aligned daily series.

    corr[t] uses only the `window` observations ending at t (data ≤ t), so as a
    feature it is backward-looking and leak-safe; the first window−1 entries are
    NaN until the window is full. NaN inputs inside a window propagate to NaN.
    """

    n = len(a)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        x = a[t - window + 1 : t + 1]
        y = b[t - window + 1 : t + 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        sx, sy = x.std(), y.std()
        if sx <= 0.0 or sy <= 0.0:
            continue
        out[t] = float(np.corrcoef(x, y)[0, 1])
    return out


def _correlation_columns(dates: Any, market_cache_dir: Path | str) -> dict[str, np.ndarray]:
    """Trailing 22-day cross-asset correlations aligned to the RV `dates`.

    corr_tnx[t] = corr(GSPC daily log return, TNX daily yield change) over the
    trailing window ending at t; corr_dxy[t] = corr(GSPC log return, DXY log
    return) likewise. GSPC/DXY use log returns; TNX is the 10y yield level, so we
    use its daily first difference (yield change) as the bond signal. Series are
    left-joined onto the RV dates and small gaps forward-filled before the window
    is taken. Sign convention: a positive corr_tnx means equities and yields move
    together (the post-2000 "good-news" regime) — equivalently equities and bond
    PRICES move opposite, the familiar negative stock–bond price correlation.
    """

    import pandas as pd

    from app.data.dense_daily_dataset import load_market_cache

    series = load_market_cache(market_cache_dir, symbols=("GSPC", "TNX"))
    base = pd.DataFrame({"date": pd.Series(dates).astype(str)})
    # DXY is cached under the raw provider symbol DX-Y.NYB (not in _CACHE_FILES).
    dxy_path = Path(market_cache_dir) / "DX-Y.NYB.parquet"
    frames = {"GSPC": series.get("GSPC"), "TNX": series.get("TNX")}
    if dxy_path.exists():
        frames["DXY"] = pd.read_parquet(dxy_path)
    elif series.get("GSPC") is not None:  # cache present but DXY missing → loud, not silent all-NaN
        print(
            f"[fed_comms_dataset] WARNING: DXY cache not found at {dxy_path}; corr_dxy will be NaN"
        )
    for name, df in frames.items():
        if df is None:
            base[name] = np.nan
            continue
        s = df[["date", "close"]].rename(columns={"close": name}).copy()
        s["date"] = s["date"].astype(str)
        base = base.merge(s, on="date", how="left")
    base = base.ffill()  # carry last known close into market holidays/gaps (no bfill → no leak)

    def _log_ret(col: str) -> np.ndarray:
        p = (
            base[col].to_numpy(dtype=np.float64)
            if col in base.columns
            else np.full(len(base), np.nan)
        )
        r = np.full(len(p), np.nan)
        r[1:] = np.log(p[1:] / p[:-1])
        return r

    gspc_ret = _log_ret("GSPC")
    dxy_ret = _log_ret("DXY")
    tnx = (
        base["TNX"].to_numpy(dtype=np.float64)
        if "TNX" in base.columns
        else np.full(len(base), np.nan)
    )
    tnx_chg = np.full(len(tnx), np.nan)
    tnx_chg[1:] = tnx[1:] - tnx[:-1]  # daily 10y yield change (level diff, not log)
    return {
        "corr_tnx": _trailing_corr(gspc_ret, tnx_chg, _CORR_WINDOW),
        "corr_dxy": _trailing_corr(gspc_ret, dxy_ret, _CORR_WINDOW),
    }


def _trailing_vol(chg: np.ndarray, window: int) -> np.ndarray:
    """Trailing realized volatility (std) of a daily change series.

    vol[t] uses only the ``window`` changes ending at t (data ≤ t), so as a
    feature it is backward-looking and leak-safe; the first ``window``−1 entries
    are NaN until the window is full (and chg[0] is itself NaN, the first diff).
    Population std (ddof=0). NaN inputs inside a window propagate to NaN.
    """

    n = len(chg)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        x = chg[t - window + 1 : t + 1]
        if not np.isfinite(x).all():
            continue
        v = float(x.std())
        out[t] = v if v > 0.0 else np.nan  # degenerate constant window → NaN, not log(0)
    return out


def _rate_vol_columns(dates: Any, market_cache_dir: Path | str) -> dict[str, np.ndarray]:
    """Trailing 22-day realized vol of daily Treasury-yield CHANGES, aligned to `dates`.

    rate_vol_2y[t] = std of Δ(2Y yield) over the trailing window ending at t;
    rate_vol_10y[t] = std of Δ(10Y yield) likewise. Yields are levels, so the
    daily signal is the first difference Δyield (in percentage-point units, e.g.
    a 4.01→4.00 move is −0.01), NOT a log return. The std is left raw (not
    annualized) and carries the units of a daily yield change; downstream the
    measure is log-transformed (vol > 0) like rv. Series are left-joined onto the
    RV dates and small gaps forward-filled (no bfill → no leak) before the window.

    The 2Y yield is read from the FRED DGS2 cache (the most Fed-sensitive tenor);
    the 10Y from TNX.parquet. If DGS2 cannot be loaded cleanly we fall back to the
    3M bill (IRX.parquet) and emit rate_vol_3m instead — the caller reports which.
    """

    import pandas as pd

    from app.data.dense_daily_dataset import load_market_cache

    base = pd.DataFrame({"date": pd.Series(dates).astype(str)})

    def _trailing_for(level: np.ndarray) -> np.ndarray:
        chg = np.full(len(level), np.nan)
        chg[1:] = level[1:] - level[:-1]  # daily yield change (Δyield, level diff)
        return _trailing_vol(chg, _RATE_VOL_WINDOW)

    def _merge_level(df: Any, name: str) -> np.ndarray:
        s = df[["date", "close"]].rename(columns={"close": name}).copy()
        s["date"] = s["date"].astype(str)
        merged = base.merge(s, on="date", how="left").ffill()
        return np.asarray(merged[name].to_numpy(dtype=np.float64), dtype=np.float64)

    series = load_market_cache(market_cache_dir, symbols=("TNX", "IRX"))
    out: dict[str, np.ndarray] = {}

    tnx = series.get("TNX")
    out["rate_vol_10y"] = (
        _trailing_for(_merge_level(tnx, "TNX")) if tnx is not None else np.full(len(base), np.nan)
    )

    two_year = _load_dgs2_levels(base["date"].tolist(), market_cache_dir)
    if two_year is not None:
        out["rate_vol_2y"] = _trailing_for(two_year)
    else:  # documented fallback: 3M bill (IRX) under the rate_vol_2y key so the
        # short-tenor measure is still populated when DGS2 is unloadable.
        irx = series.get("IRX")
        print(
            "[fed_comms_dataset] WARNING: DGS2 FRED cache unavailable; "
            "substituting IRX 3M bill for the 2Y under the rate_vol_2y key"
        )
        out["rate_vol_2y"] = (
            _trailing_for(_merge_level(irx, "IRX"))
            if irx is not None
            else np.full(len(base), np.nan)
        )
    return out


def _load_dgs2_levels(dates: list[str], market_cache_dir: Path | str) -> np.ndarray | None:
    """2Y yield levels (FRED DGS2) aligned + ffilled to `dates`; None if uncached.

    Reuses :func:`app.services.fred_client.fetch_fred_series`, which serves the
    on-disk ``DGS2.json`` cache without any network call or API key when present.
    FRED encodes missing observations as the literal '.', which the client maps to
    None; we drop those before the as-of merge so a holiday gap is forward-filled
    rather than poisoning a window with NaN. The cache lives under the FRED cache
    dir, NOT the market cache; we resolve it relative to DATA_DIR like mp_surprise.
    """

    import pandas as pd

    from app.services.fred_client import DEFAULT_CACHE_DIR as FRED_CACHE_DIR
    from app.services.fred_client import fetch_fred_series

    if not (FRED_CACHE_DIR / "DGS2.json").exists():
        return None
    try:
        resp = fetch_fred_series("DGS2", cache_dir=FRED_CACHE_DIR)
    except Exception as exc:  # noqa: BLE001 — any load failure → documented 3M fallback
        print(f"[fed_comms_dataset] WARNING: DGS2 load failed ({exc!r})")
        return None
    rows = [
        {"date": obs.date, "DGS2": float(obs.value)}
        for obs in resp.observations
        if obs.value is not None and obs.date
    ]
    if not rows:
        return None
    s = pd.DataFrame(rows)
    s["date"] = s["date"].astype(str)
    base = pd.DataFrame({"date": pd.Series(dates).astype(str)})
    merged = base.merge(s, on="date", how="left").ffill()
    return np.asarray(merged["DGS2"].to_numpy(dtype=np.float64), dtype=np.float64)


def _forward_target(raw: np.ndarray, h: int, *, is_log: bool) -> np.ndarray:
    """Forward mean over t+1..t+h (log-mean for positive measures); NaN past the end."""

    n = len(raw)
    out = np.full(n, np.nan)
    for t in range(n - h):
        m = float(raw[t + 1 : t + 1 + h].mean())
        out[t] = np.log(m + _EPS) if is_log else m
    return out


def _origin_after(date_iso: str, trading_days: list[str]) -> int | None:
    """Index of the first trading day strictly greater than date_iso (embargo)."""

    import bisect

    i = bisect.bisect_right(trading_days, date_iso)
    return i if i < len(trading_days) else None


def _as_of_index(date_iso: str, trading_days: list[str]) -> int | None:
    """Index of the latest trading day ≤ date_iso (most-recent-known origin)."""

    import bisect

    i = bisect.bisect_right(trading_days, date_iso) - 1
    return i if i >= 0 else None


def build_text_outcome_pairs(
    corpus: Any, rv_df: Any, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> Any:
    """Communication-level table: text + forward RV outcome (origin = day after text)."""

    import pandas as pd

    rv_df = rv_df.sort_values("date").reset_index(drop=True)
    trading_days = rv_df["date"].astype(str).tolist()
    rv = rv_df["rv"].to_numpy(dtype=np.float64)
    fwd = _forward_log_rv_windows(rv, horizons)

    rows: list[dict[str, Any]] = []
    for _, doc in corpus.sort_values("timestamp_et").iterrows():
        origin = _origin_after(str(doc["date"]), trading_days)
        if origin is None:
            continue
        out: dict[str, Any] = {
            "date": doc["date"],
            "origin_date": trading_days[origin],
            "doc_type": doc["doc_type"],
            "time_known": bool(doc["time_known"]),
            "speaker": doc.get("speaker"),
            "text": doc["text"],
        }
        valid = False
        for h in horizons:
            val = fwd[h][origin]
            out[f"rv_fwd_{h}"] = float(val) if np.isfinite(val) else np.nan
            valid = valid or np.isfinite(val)
        if valid:
            rows.append(out)
    return pd.DataFrame(rows)


def _statement_calendar(
    corpus_sorted: Any, trading_days: list[str], *, cap: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Trading-days since the last / until the next FOMC statement (capped)."""

    import bisect

    stmt = corpus_sorted[corpus_sorted["doc_type"] == "statement"]["date"].astype(str).tolist()
    pos = sorted({p for p in (_as_of_index(d, trading_days) for d in stmt) if p is not None})
    n = len(trading_days)
    since = np.full(n, float(cap))
    to = np.full(n, float(cap))
    for i in range(n):
        left = bisect.bisect_right(pos, i) - 1
        if left >= 0:
            since[i] = min(cap, i - pos[left])
        right = bisect.bisect_right(pos, i)
        if right < len(pos):
            to[i] = min(cap, pos[right] - i)
    return since, to


def _statement_surprise_columns(
    trading_days: list[str],
    surprise: Any,
) -> dict[str, np.ndarray]:
    """As-of join the most-recent FOMC statement's surprise onto each trading day.

    LEAK-SAFETY: a trading day t carries statement S's surprise only when
    statement_date(S) ≤ t. We map each statement date to its as-of trading-day
    index (latest trading day ≤ statement_date) via :func:`_as_of_index`, sort
    those origins, and for day i pick the latest origin ≤ i — the same backward
    bisect the FOMC-calendar features use. Days before the first statement get
    the neutral fill (no surprise is known yet).
    """

    import bisect

    n = len(trading_days)
    cols = {c: np.full(n, _SURPRISE_NEUTRAL, dtype=np.float64) for c in _SURPRISE_COLUMNS}
    if surprise is None or surprise.empty:
        return cols
    # The producer (mp_surprise) keys rows on `event_date`; accept either name.
    date_col = "date" if "date" in surprise.columns else "event_date"
    surprise = surprise.sort_values(date_col)  # so later-dated statement wins same origin
    by_date = {str(r[date_col]): r for _, r in surprise.iterrows()}
    # origin trading-day index → surprise row, for statements with a value.
    origin_to_row: dict[int, Any] = {}
    for d, row in by_date.items():
        pos = _as_of_index(d, trading_days)
        if pos is not None:
            origin_to_row[pos] = row  # date-sorted above → later statement wins
    origins = sorted(origin_to_row)
    if not origins:
        return cols
    src = {
        "surprise_level": "mp_surprise_level",
        "surprise_path": "mp_surprise_path_factor",
        "surprise_info": "fed_info_factor",
    }
    for i in range(n):
        left = bisect.bisect_right(origins, i) - 1
        if left < 0:
            continue
        row = origin_to_row[origins[left]]
        for out_col, in_col in src.items():
            val = row.get(in_col)
            if val is not None and val == val:  # NaN-safe; keep neutral otherwise
                cols[out_col][i] = float(val)
    return cols


def build_daily_fusion_frame(
    rv_df: Any,
    corpus: Any,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    surprise: Any = None,
) -> Any:
    """Trading-day table: per-measure HAR lags + forward targets + most-recent comm.

    For every measure in MEASURES, emits `{m}_daily/_weekly/_monthly` backward HAR
    lags and `{m}_fwd_{h}` forward targets, so the trainer can pick any target
    (rv / volume / downside / jump) with its own HAR-style floor — same leak-safe
    construction, same text linkage.
    """

    import pandas as pd

    rv_df = rv_df.sort_values("date").reset_index(drop=True)
    trading_days = rv_df["date"].astype(str).tolist()
    # precompute per-measure lags + forward targets. corr_* measures are emitted
    # only when their daily columns are present (build() merges them from the
    # market cache); a frame without them simply omits those columns.
    measures = tuple(m for m in MEASURES if _measure_present(rv_df, m))
    lags: dict[str, np.ndarray] = {}
    fwds: dict[str, dict[int, np.ndarray]] = {}
    for m in measures:
        raw, is_log = _measure_raw(rv_df, m)
        lags[m] = _har_lags(np.log(raw + _EPS) if is_log else raw)
        fwds[m] = {h: _forward_target(raw, h, is_log=is_log) for h in horizons}

    # most-recent communication known as of each trading day (any type)
    corpus_sorted = corpus.sort_values("date").reset_index(drop=True)
    comm_dates = corpus_sorted["date"].astype(str).tolist()
    last_doc_row = np.full(len(trading_days), -1, dtype=int)
    j = 0
    for i, day in enumerate(trading_days):
        while j < len(comm_dates) and comm_dates[j] <= day:
            j += 1
        last_doc_row[i] = j - 1  # index into corpus_sorted, or -1 if none yet

    # FOMC-calendar features (scheduled, public → known at t, no leakage). These go
    # into the MARKET baseline so the text contribution isolates statement CONTENT,
    # not "is it an FOMC day" — the fairness condition for the volume target.
    days_since_stmt, days_to_stmt = _statement_calendar(corpus_sorted, trading_days)

    # Most-recent FOMC statement's market-derived MP surprise, as-of joined so a
    # day t can only see a statement dated ≤ t (leak-safe). These are MARKET
    # features — they feed both the gate-off market path and the fused path.
    surprise_cols = _statement_surprise_columns(trading_days, surprise)

    rows: list[dict[str, Any]] = []
    for i, day in enumerate(trading_days):
        row: dict[str, Any] = {"date": day}
        for m in measures:
            row[f"{m}_daily"] = float(lags[m][i, 0])
            row[f"{m}_weekly"] = float(lags[m][i, 1])
            row[f"{m}_monthly"] = float(lags[m][i, 2])
            for h in horizons:
                val = fwds[m][h][i]
                row[f"{m}_fwd_{h}"] = float(val) if np.isfinite(val) else np.nan
        row["days_since_stmt"] = float(days_since_stmt[i])
        row["days_to_stmt"] = float(days_to_stmt[i])
        for c in _SURPRISE_COLUMNS:
            row[c] = float(surprise_cols[c][i])
        di = int(last_doc_row[i])
        if di >= 0:
            doc = corpus_sorted.iloc[di]
            origin = _as_of_index(str(doc["date"]), trading_days)
            row["doc_row"] = di
            row["doc_type"] = str(doc["doc_type"])
            row["doc_age_days"] = i - origin if origin is not None else -1
            row["has_text"] = True
        else:
            row["doc_row"] = -1
            row["doc_type"] = None
            row["doc_age_days"] = -1
            row["has_text"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def build(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_PARQUET,
    rv_path: Path | str = DEFAULT_RV_PARQUET,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    mp_surprise_path: Path | str | None = DEFAULT_MP_SURPRISE_PARQUET,
    market_cache_dir: Path | str | None = DEFAULT_MARKET_CACHE_DIR,
) -> tuple[Path, Path]:
    """Build + persist both tables; return (pairs_path, daily_path)."""

    import pandas as pd

    corpus = pd.read_parquet(corpus_path)
    rv_df = pd.read_parquet(rv_path)
    # Merge trailing cross-asset correlation targets onto the RV dates BEFORE the
    # fusion frame is built, so _measure_raw can read them. Skip silently if the
    # market cache is absent — the corr_* measures are then simply not emitted.
    if market_cache_dir is not None and Path(market_cache_dir).exists():
        rv_df = rv_df.sort_values("date").reset_index(drop=True)
        corr = _correlation_columns(rv_df["date"], market_cache_dir)
        for name, col in corr.items():
            rv_df[name] = col
        # Trailing realized vol of daily yield changes (2Y from FRED DGS2, 10Y
        # from TNX) — the Fed's home-turf rate-vol targets. Merged the same way.
        rate_vol = _rate_vol_columns(rv_df["date"], market_cache_dir)
        for name, col in rate_vol.items():
            rv_df[name] = col
    surprise = None
    if mp_surprise_path is not None and Path(mp_surprise_path).exists():
        surprise = pd.read_parquet(mp_surprise_path)
    pairs = build_text_outcome_pairs(corpus, rv_df, horizons=horizons)
    daily = build_daily_fusion_frame(rv_df, corpus, horizons=horizons, surprise=surprise)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "text_outcome_pairs.parquet"
    daily_path = out_dir / "daily_fusion.parquet"
    pairs.to_parquet(pairs_path, index=False)
    daily.to_parquet(daily_path, index=False)
    by_type = pairs["doc_type"].value_counts().to_dict() if not pairs.empty else {}
    cov = float(daily["has_text"].mean()) if not daily.empty else 0.0
    print(f"[fed_comms_dataset] pairs={len(pairs)} by_type={by_type}")
    print(f"[fed_comms_dataset] daily={len(daily)} has_text_frac={cov:.3f} → {out_dir}")
    if not daily.empty:
        nz = float((daily["surprise_level"] != _SURPRISE_NEUTRAL).mean())
        print(
            f"[fed_comms_dataset] surprise: present={surprise is not None} "
            f"nonneutral_level_frac={nz:.3f}"
        )
        for m in (*_CORR_MEASURES, *_RATE_VOL_MEASURES):
            fwd_col = f"{m}_fwd_{max(horizons)}"
            if fwd_col in daily.columns:
                v = daily[fwd_col].to_numpy(dtype=np.float64)
                fin = v[np.isfinite(v)]
                if fin.size:
                    print(
                        f"[fed_comms_dataset] {fwd_col}: coverage={fin.size / len(v):.3f} "
                        f"mean={fin.mean():+.3f} range=[{fin.min():+.3f},{fin.max():+.3f}]"
                    )
    return pairs_path, daily_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Align Fed comms corpus with intraday RV.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--rv-path", type=Path, default=DEFAULT_RV_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mp-surprise-path", type=Path, default=DEFAULT_MP_SURPRISE_PARQUET)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    args = parser.parse_args()
    build(
        corpus_path=args.corpus_path,
        rv_path=args.rv_path,
        out_dir=args.out_dir,
        mp_surprise_path=args.mp_surprise_path,
        market_cache_dir=args.market_cache_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
