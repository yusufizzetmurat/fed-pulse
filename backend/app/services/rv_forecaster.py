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
import os
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from app.config import BACKEND_ROOT
from app.data.intraday_rv_forecast import _EPS as _RVEPS, _LOGV_CLAMP, _har_lags

if TYPE_CHECKING:
    import torch


HF_REPO_ID = "yusufizzetmurat/fomc-rv-qlike-forecaster"
MODEL_DIR = BACKEND_ROOT / "models" / "rv_qlike"
ARTIFACT_FILENAME = "production_artifact.json"
EVAL_FILENAME = "production_eval.json"
HORIZONS: tuple[int, ...] = (1, 5, 22)
ALPHAS: tuple[float, ...] = (0.2, 0.1)


class RvForecasterUnavailable(RuntimeError):
    """Raised when the artifact is missing and HF Hub fetch failed."""


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download_artifact(target_dir: Path) -> dict[str, Any]:
    """Download spec + eval + all per-seed weight files into ``target_dir``."""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    base_kwargs: dict[str, Any] = {
        "repo_id": HF_REPO_ID,
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
        for hk, row in spec["by_horizon"].items():
            for fname in row["seed_state_dicts"]:
                _pull(fname)
    except RepositoryNotFoundError as exc:
        raise RvForecasterUnavailable(
            f"HF repo {HF_REPO_ID!r} not found; set HF_TOKEN or grant access."
        ) from exc
    except (EntryNotFoundError, HfHubHTTPError) as exc:
        raise RvForecasterUnavailable(
            f"HF fetch failed for {HF_REPO_ID!r}: {exc}"
        ) from exc
    return spec


def _load_spec(model_dir: Path) -> dict[str, Any]:
    """Return the cached spec; fetch from HF Hub on a miss."""

    spec_path = model_dir / ARTIFACT_FILENAME
    if not spec_path.exists():
        return _download_artifact(model_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for hk, row in spec["by_horizon"].items():
        for fname in row["seed_state_dicts"]:
            if not (model_dir / fname).exists():
                return _download_artifact(model_dir)
    return spec


def _load_eval(model_dir: Path) -> dict[str, Any] | None:
    """Read ``production_eval.json`` if it landed alongside the artifact."""

    p = model_dir / EVAL_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_seed_models(
    spec: dict[str, Any], model_dir: Path
) -> dict[str, list["torch.nn.Module"]]:
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
    """Cached per-process serving bundle.

    Holds the parsed spec, per-horizon seed-ensembles, and the eval sidecar so
    the endpoint can answer in a single forward pass per horizon. Construction
    is wrapped under a class-level lock so concurrent callers do not race on
    the HF download.
    """

    _instance: "_RvPredictor | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "_RvPredictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached instance; used by tests."""

        with cls._lock:
            cls._instance = None

    def __init__(self) -> None:
        self.model_dir = MODEL_DIR
        self.spec = _load_spec(self.model_dir)
        self.eval = _load_eval(self.model_dir)
        self.seed_models = _load_seed_models(self.spec, self.model_dir)
        self.revision = f"{self.spec.get('model', 'rv')}@{self.spec.get('date_last', '')}"


def _ensemble_log_rv(
    log_rv: np.ndarray,
    row: dict[str, Any],
    seeds: list["torch.nn.Module"],
) -> float:
    """One-step ensemble forecast in log-RV space for the latest window."""

    import torch

    har = _har_lags(log_rv)  # backward HAR regressors at every index
    # Reproduce the training-time feature layout for the LAST observation only.
    # The QLIKE-DLq head takes the standardized full feature row and emits a
    # standardized residual that is un-standardized and added to the OLS HAR
    # prediction (intercept + d/w/m coef).
    last = har[-1]
    har_coef = np.asarray(row["har_coef"], dtype=np.float64)
    har_pred = float(har_coef[0] + har_coef[1] * last[0] + har_coef[2] * last[1] + har_coef[3] * last[2])

    feat_mean = np.asarray(row["feat_mean"], dtype=np.float64)
    feat_std = np.asarray(row["feat_std"], dtype=np.float64)
    # The non-HAR realized-measure columns are not available from a bare RV
    # history; we substitute their training-set means (standardized → zeros)
    # so the head reduces to a HAR-only point forecast plus the learned
    # bias on the realized lags. This keeps the predictor usable from a
    # plain RV series; richer features can be plumbed when intraday bars
    # are reachable.
    full = feat_mean.copy()
    full[0:3] = last  # the HAR daily/weekly/monthly lags
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
    return float(np.mean(preds))


def predict_rv(rv_history: list[float] | np.ndarray) -> dict[str, Any]:
    """Multi-horizon banded RV forecast off a recent realized-vol series.

    ``rv_history`` is a chronologically ordered series of daily realized
    variance values (NOT log). Each entry is the per-day realized variance
    that ``intraday_realized.daily_realized_measures`` would output. The
    function returns the per-horizon point forecast in RV space, 80%/90%
    conformal bands, and the QLIKE / coverage diagnostics for the card.
    """

    rv = np.asarray(rv_history, dtype=np.float64)
    if rv.ndim != 1 or len(rv) < 22:
        raise ValueError(
            "rv_history must be a 1-D series of at least 22 daily RV values"
        )
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("rv_history values must be positive finite numbers")

    log_rv = np.log(rv + _RVEPS)
    pred = _RvPredictor.get()
    horizons_out: list[dict[str, Any]] = []
    for h in HORIZONS:
        hk = f"h{h}"
        row = pred.spec["by_horizon"][hk]
        seeds = pred.seed_models[hk]
        log_point = _ensemble_log_rv(log_rv, row, seeds)
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
    return {
        "horizons": horizons_out,
        "model_revision": pred.revision,
    }


def _safe_float(v: Any) -> float:
    """Coerce ``v`` to a finite float; return NaN on non-numeric input."""

    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
