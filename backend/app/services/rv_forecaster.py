"""Serving layer for the QLIKE-DLq RV ensemble (multi-horizon, conformal).

Wraps the artifact produced by ``app.data.intraday_rv_production.fit_production``:
per-horizon HAR coefficients, per-seed QLIKE-DLq state_dicts, feature
standardization stats, and walk-forward conformal quantiles. The serving model
is the per-seed mean in log-RV space plus symmetric conformal bands. Eval-pool
QLIKE numbers (``qlike_ens`` / ``qlike_har`` / empirical band coverage) are
read off ``production_eval.json`` so the card can render a beat-HAR badge
without re-running the walk-forward.

The artifact lives in ``yusufizzetmurat/fomc-rv-qlike-forecaster`` on HF Hub;
the loader lazily downloads the 15 ``.pt`` weights + the two JSON sidecars
into ``BACKEND_ROOT/models/rv_qlike/`` and caches the assembled predictors
for the process lifetime.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np

from app.config import BACKEND_ROOT
from app.data.intraday_rv_forecast import _EPS as _RVEPS, _LOGV_CLAMP, _har_lags

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("app.services.rv_forecaster")


HF_REPO_ID = "yusufizzetmurat/fomc-rv-qlike-forecaster"
MODEL_DIR = BACKEND_ROOT / "models" / "rv_qlike"
ARTIFACT_FILENAME = "production_artifact.json"
EVAL_FILENAME = "production_eval.json"
HORIZONS: tuple[int, ...] = (1, 5, 22)
ALPHAS: tuple[float, ...] = (0.2, 0.1)

# Symbol -> (local artifact dir, HF repo id or ``None``).
# The canonical row is ^GSPC; per-asset rows list a local directory
# plus an HF repo so the loader either serves from a pre-populated
# local checkout (training writes the artifact straight into
# ``data/processed/rv_qlike_dlq/<alias>/``) or falls back to the HF
# Hub on a fresh container with no local copy. Lookup falls back to
# the canonical row when the requested symbol is not registered, so
# FX / commodity tickers do not blow up the call site.
_DATA_ROOT = BACKEND_ROOT.parent / "data" / "processed" / "rv_qlike_dlq"
SYMBOL_ARTIFACTS: dict[str, tuple[Path, str | None]] = {
    "^GSPC": (MODEL_DIR, HF_REPO_ID),
    "^NDX": (_DATA_ROOT / "ndx", "yusufizzetmurat/fomc-rv-qlike-forecaster-ndx"),
    "^DJI": (_DATA_ROOT / "dji", "yusufizzetmurat/fomc-rv-qlike-forecaster-dji"),
}
SUPPORTED_RV_FORECASTER_SYMBOLS: tuple[str, ...] = tuple(SYMBOL_ARTIFACTS.keys())


class RvForecasterUnavailable(RuntimeError):
    """Raised when the artifact is missing and HF Hub fetch failed."""


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download_artifact(target_dir: Path, repo_id: str | None) -> dict[str, Any]:
    """Download spec + eval + all per-seed weight files into ``target_dir``.

    ``repo_id`` is the HF repo to pull from. When ``None`` (e.g. for
    per-asset artifacts that have not been published to the Hub yet)
    we raise :class:`RvForecasterUnavailable` immediately so the caller
    surfaces a clean "asset unavailable" error instead of attempting a
    repo lookup with no name.
    """

    if repo_id is None:
        raise RvForecasterUnavailable(
            f"no HF repo registered for artifact at {target_dir}; "
            "produce the artifact locally or register a repo in SYMBOL_ARTIFACTS"
        )

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    base_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
    }
    if token:
        base_kwargs["token"] = token

    def _pull(name: str) -> Path:
        return Path(hf_hub_download(filename=name, **base_kwargs))

    try:
        spec_path = _pull(ARTIFACT_FILENAME)
        try:
            _pull(EVAL_FILENAME)
        except (EntryNotFoundError, HfHubHTTPError):
            # eval sidecar is non-blocking; QLIKE gain just won't render
            pass
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for row in spec["by_horizon"].values():
            for fname in row["seed_state_dicts"]:
                _pull(fname)
    except RepositoryNotFoundError as exc:
        raise RvForecasterUnavailable(
            f"HF repo {repo_id!r} not found; set HF_TOKEN or grant access."
        ) from exc
    except (EntryNotFoundError, HfHubHTTPError) as exc:
        raise RvForecasterUnavailable(f"HF fetch failed for {repo_id!r}: {exc}") from exc
    except OSError as exc:
        # Catches the network-level failures (ConnectionError, timeouts,
        # disk full) that hf_hub_download can raise outside the HF-specific
        # exception hierarchy; surface them as the same "unavailable" 503
        # the endpoint contract promises instead of bubbling up as 500.
        raise RvForecasterUnavailable(f"HF download failed for {repo_id!r}: {exc}") from exc
    return cast(dict[str, Any], spec)


def _load_spec(model_dir: Path, repo_id: str | None) -> dict[str, Any]:
    """Return the cached spec; fetch from HF Hub on a miss when ``repo_id`` is set."""

    spec_path = model_dir / ARTIFACT_FILENAME
    if not spec_path.exists():
        return _download_artifact(model_dir, repo_id)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for row in spec["by_horizon"].values():
        for fname in row["seed_state_dicts"]:
            if not (model_dir / fname).exists():
                return _download_artifact(model_dir, repo_id)
    return cast(dict[str, Any], spec)


def _load_eval(model_dir: Path) -> dict[str, Any] | None:
    """Read ``production_eval.json`` if it landed alongside the artifact."""

    p = model_dir / EVAL_FILENAME
    if not p.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None


def _load_seed_models(spec: dict[str, Any], model_dir: Path) -> dict[str, list["torch.nn.Module"]]:
    """Materialize the per-horizon list of QLIKE-DLq heads from saved state."""

    import torch

    from app.data.dense_forecast_train import _build_model

    n_features = len(spec["feature_order"])
    out: dict[str, list[torch.nn.Module]] = {}
    for hk, row in spec["by_horizon"].items():
        seeds: list[torch.nn.Module] = []
        for fname in row["seed_state_dicts"]:
            state = torch.load(model_dir / fname, map_location="cpu", weights_only=True)
            model = _build_model(n_features, 1)
            model.load_state_dict(state)
            model.eval()
            seeds.append(model)
        out[hk] = seeds
    return out


class _RvPredictor:
    """Cached per-process serving bundle, keyed per symbol.

    Holds the parsed spec, per-horizon seed-ensembles, and the eval sidecar so
    the endpoint can answer in a single forward pass per horizon. Construction
    is wrapped under a class-level lock so concurrent callers do not race on
    the local-load / HF download. A per-symbol registry lets the dashboard
    serve multiple assets (^GSPC, ^NDX, ^DJI) off the same code path; the
    artifact directory and HF repo for each symbol live in
    :data:`SYMBOL_ARTIFACTS`.
    """

    _instances: dict[str, "_RvPredictor"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, symbol: str = "^GSPC") -> "_RvPredictor":
        """Return the predictor bound to ``symbol``.

        Falls back to the ``^GSPC`` slot when the symbol is not registered
        in :data:`SYMBOL_ARTIFACTS` so legacy call sites that do not know
        about the per-asset map keep working.
        """

        key = symbol if symbol in SYMBOL_ARTIFACTS else "^GSPC"
        cached = cls._instances.get(key)
        if cached is not None:
            return cached
        with cls._lock:
            cached = cls._instances.get(key)
            if cached is not None:
                return cached
            instance = cls(symbol=key)
            cls._instances[key] = instance
            return instance

    @classmethod
    def reset(cls, symbol: str | None = None) -> None:
        """Drop one or all cached predictors. Tests reach in here."""

        with cls._lock:
            if symbol is None:
                cls._instances.clear()
            else:
                cls._instances.pop(symbol, None)

    def __init__(self, *, symbol: str = "^GSPC") -> None:
        model_dir, repo_id = SYMBOL_ARTIFACTS.get(symbol, SYMBOL_ARTIFACTS["^GSPC"])
        self.symbol = symbol
        self.model_dir = model_dir
        self.spec = _load_spec(self.model_dir, repo_id)
        self.eval = _load_eval(self.model_dir)
        self.seed_models = _load_seed_models(self.spec, self.model_dir)
        self.revision = f"{self.spec.get('model', 'rv')}@{self.spec.get('date_last', '')}"
        # Drift guard: the serving layer's _REALIZED_FEAT_COLS is the
        # contract that intraday_features.recent_realized_measures
        # promises to fill at indices full[3:10]. If a future training
        # artifact reorders or renames any of these columns the live
        # intraday wiring would silently scramble the feature row, so
        # we hard-fail at load time instead of degrading silently.
        artifact_cols = tuple(self.spec.get("feature_order", [])[3 : 3 + len(_REALIZED_FEAT_COLS)])
        if artifact_cols and artifact_cols != _REALIZED_FEAT_COLS:
            raise RvForecasterUnavailable(
                f"realized-measure column order drift: artifact has {artifact_cols}, "
                f"serving expects {_REALIZED_FEAT_COLS}. Re-sync "
                "rv_forecaster._REALIZED_FEAT_COLS with the training pipeline."
            )


# Feature-row column order pinned at training time
# (intraday_rv_production._FEAT_COLS + log(rvol+1) last).
# ``full[0:3]`` are the HAR daily/weekly/monthly lags. ``full[3:10]``
# are the seven realized-measure columns; ``full[10]`` is log(rvol+1).
_REALIZED_FEAT_COLS = ("rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson")


def _ensemble_log_rv(
    log_rv: np.ndarray,
    row: dict[str, Any],
    seeds: list["torch.nn.Module"],
    intraday_measures: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """One-step ensemble forecast in log-RV space for the latest window.

    Returns ``(log_point, realized_features_source)``. The source flag is
    ``"live"`` when ``intraday_measures`` was supplied and used to fill
    the seven realized-measure columns plus ``log(rvol+1)``, and
    ``"training_means"`` when the QLIKE head falls back to feat_mean
    (HAR-only effective prediction). The dashboard surfaces this so the
    user knows whether the displayed forecast is the full edge or the
    HAR-grade fallback.
    """

    import torch

    har = _har_lags(log_rv)  # backward HAR regressors at every index
    last = har[-1]
    har_coef = np.asarray(row["har_coef"], dtype=np.float64)
    har_pred = float(
        har_coef[0] + har_coef[1] * last[0] + har_coef[2] * last[1] + har_coef[3] * last[2]
    )

    feat_mean = np.asarray(row["feat_mean"], dtype=np.float64)
    feat_std = np.asarray(row["feat_std"], dtype=np.float64)
    full = feat_mean.copy()
    full[0:3] = last  # HAR daily / weekly / monthly lags

    source = "training_means"
    if intraday_measures is not None and len(full) >= 11:
        try:
            for i, col in enumerate(_REALIZED_FEAT_COLS):
                full[3 + i] = float(intraday_measures[col])
            full[10] = float(np.log(float(intraday_measures["rvol"]) + 1.0))
            source = "live"
        except (KeyError, TypeError, ValueError) as exc:
            # Any missing / malformed key short-circuits to feat_mean
            # fallback. The forecast remains valid (HAR-grade); only the
            # source flag changes. Log so a contract drift between the
            # intraday fetcher and this consumer is diagnosable.
            logger.warning(
                "rv_forecaster: intraday_measures malformed (%s); falling back to training_means",
                exc,
            )
            full = feat_mean.copy()
            full[0:3] = last
            source = "training_means"

    x = (full - feat_mean) / feat_std
    xt = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
    rs = float(row["resid_std"])
    rm = float(row["resid_mean"])
    preds: list[float] = []
    for model in seeds:
        with torch.no_grad():
            r_std = float(model(xt).cpu().numpy().ravel()[0])
        log_pred = har_pred + (r_std * rs + rm)
        log_pred = float(np.clip(log_pred, -_LOGV_CLAMP, _LOGV_CLAMP))
        preds.append(log_pred)
    return float(np.mean(preds)), source


def predict_rv(
    rv_history: list[float] | np.ndarray,
    intraday_measures: dict[str, Any] | None = None,
    *,
    symbol: str = "^GSPC",
) -> dict[str, Any]:
    """Multi-horizon banded RV forecast off a recent realized-vol series.

    ``rv_history`` is a chronologically ordered series of daily realized
    variance values (NOT log). Each entry is the per-day realized variance
    that ``intraday_realized.daily_realized_measures`` would output. The
    function returns the per-horizon point forecast in RV space, 80%/90%
    conformal bands, the QLIKE / coverage diagnostics, and the
    ``realized_features_source`` flag (``"live"`` when ``intraday_measures``
    was supplied and used; ``"training_means"`` when the QLIKE head fell
    back to feat_mean and the forecast collapses to HAR-grade).
    """

    rv = np.asarray(rv_history, dtype=np.float64)
    if rv.ndim != 1 or len(rv) < 22:
        raise ValueError("rv_history must be a 1-D series of at least 22 daily RV values")
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("rv_history values must be positive finite numbers")

    log_rv = np.log(rv + _RVEPS)
    pred = _RvPredictor.get(symbol)
    horizons_out: list[dict[str, Any]] = []
    # Source flag is determined by the first horizon's pass and held
    # constant across the three. The flag is intentionally a single
    # value per request — if intraday is unavailable, every horizon
    # falls back together, not piecemeal.
    realized_features_source = "training_means"
    for h in HORIZONS:
        hk = f"h{h}"
        row = pred.spec["by_horizon"][hk]
        seeds = pred.seed_models[hk]
        log_point, source = _ensemble_log_rv(log_rv, row, seeds, intraday_measures)
        if h == HORIZONS[0]:
            realized_features_source = source
        point = float(np.exp(log_point))
        quants = row["conformal_quantiles"]
        q80 = float(quants.get("0.20", 0.0))
        q90 = float(quants.get("0.10", 0.0))
        eval_row = (pred.eval or {}).get("by_horizon", {}).get(hk, {}) if pred.eval else {}
        coverage_90 = float(
            eval_row.get("coverage", {}).get("0.90", {}).get("empirical")
            if eval_row.get("coverage", {}).get("0.90", {}).get("empirical") is not None
            else float("nan")
        )
        horizons_out.append(
            {
                "h": h,
                "point": point,
                "band_lo_80": float(np.exp(log_point - q80)),
                "band_hi_80": float(np.exp(log_point + q80)),
                "band_lo_90": float(np.exp(log_point - q90)),
                "band_hi_90": float(np.exp(log_point + q90)),
                "qlike_model": _safe_float(eval_row.get("qlike_ens")),
                "qlike_har": _safe_float(eval_row.get("qlike_har")),
                "coverage_empirical_90": coverage_90,
            }
        )
    realized_features_date: str | None = None
    if (
        realized_features_source == "live"
        and intraday_measures is not None
        and isinstance(intraday_measures.get("date"), str)
    ):
        realized_features_date = intraday_measures["date"]
    return {
        "horizons": horizons_out,
        "model_revision": pred.revision,
        "realized_features_source": realized_features_source,
        "realized_features_date": realized_features_date,
    }


_HISTORICAL_BANDS_WARMUP = 22


def predict_rv_historical_bands(
    rv_history: list[float] | np.ndarray,
    dates: list[str],
    *,
    symbol: str = "^GSPC",
) -> list[dict[str, Any]]:
    """Walk-forward h=1 conformal bands over the recent RV window.

    For each date in ``dates`` we run the h=1 ensemble against the
    leading slice ``rv_history[:i]`` to obtain the predicted log-RV, then
    expand the q80 conformal quantile around it. The realized observation
    at the same date is the actual ``rv_history[i]``. The first
    ``_HISTORICAL_BANDS_WARMUP`` rows are skipped — HAR's monthly lag
    needs ~22 days of warmup before the prediction is well-defined.

    Returns a chronologically ordered list of
    ``{date, band_lo_80, band_hi_80, realized_rv}`` rows. The realized
    sparkline can render the bands as a muted "we covered" overlay.
    """

    rv = np.asarray(rv_history, dtype=np.float64)
    if rv.ndim != 1:
        raise ValueError("rv_history must be a 1-D series")
    if len(rv) != len(dates):
        raise ValueError("rv_history and dates must be the same length")
    if len(rv) <= _HISTORICAL_BANDS_WARMUP:
        return []
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("rv_history values must be positive finite numbers")

    pred = _RvPredictor.get(symbol)
    row = pred.spec["by_horizon"]["h1"]
    seeds = pred.seed_models["h1"]
    quants = row["conformal_quantiles"]
    q80 = float(quants.get("0.20", 0.0))

    out: list[dict[str, Any]] = []
    for i in range(_HISTORICAL_BANDS_WARMUP, len(rv)):
        # Predict day i using only rv_history[:i] (no leakage). The
        # historical-band walk is deliberately HAR-grade (intraday
        # measures are not reconstructed for past days), so the second
        # return value is ignored.
        log_rv_prefix = np.log(rv[:i] + _RVEPS)
        log_point, _ = _ensemble_log_rv(log_rv_prefix, row, seeds)
        out.append(
            {
                "date": str(dates[i]),
                "band_lo_80": float(np.exp(log_point - q80)),
                "band_hi_80": float(np.exp(log_point + q80)),
                "realized_rv": float(rv[i]),
            }
        )
    return out


def _safe_float(v: Any) -> float:
    """Coerce ``v`` to a finite float; return NaN on non-numeric input."""

    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
