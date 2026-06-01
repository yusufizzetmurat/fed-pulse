"""Serving for the volume head — forward daily volume forecast.

Loads the published HAR volume-head artifact (per-horizon Corsi-lag coefficients
on log-volume) and emits a multi-horizon "expected volume" forecast. Mirrors
``rv_forecaster``: the artifact is pulled from the Hub when no local copy exists
(public + ungated), so the forecast is available in every environment. Volume is
persistence-dominated (lag-1 autocorr ~0.99) so HAR is the deployed model; see
wiki page 25.

Self-contained on purpose (its own response models live here, not in schemas.py)
so wiring it adds only one endpoint and touches no shared module.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from app.config import MODEL_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

HF_REPO_ID = "yusufizzetmurat/fomc-rv-qlike-forecaster"
_ARTIFACT_IN_REPO = "volume_head/volume_head_artifact.json"
_LOCAL_ARTIFACT = MODEL_CHECKPOINT_DIR / "volume_head_artifact.json"
_HORIZON_DAYS = {"h1": 1, "h5": 5, "h22": 22}

_spec_cache: dict[str, Any] | None = None


class VolumeForecasterUnavailable(RuntimeError):
    """Raised when the volume-head artifact cannot be loaded (offline / missing)."""


class VolumeHorizonForecast(BaseModel):
    horizon_days: int = Field(..., description="Forecast horizon in trading days.")
    expected_log_volume: float = Field(..., description="Forecast mean log-volume over the horizon.")
    expected_volume: float = Field(..., description="exp(expected_log_volume) — volume level.")


class VolumeForecastResponse(BaseModel):
    symbol: str
    horizons: list[VolumeHorizonForecast]
    model: str = Field("volume-head-HAR", description="Deployed model (HAR on log-volume lags).")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download_artifact() -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=_ARTIFACT_IN_REPO, token=_hf_token()
        )
    except Exception as exc:  # noqa: BLE001 - surface as a typed unavailable error
        raise VolumeForecasterUnavailable(
            f"could not fetch volume-head artifact from {HF_REPO_ID!r}: {exc}"
        ) from exc
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def _load_spec() -> dict[str, Any]:
    """Local artifact if present, else pull from the Hub (memoized)."""
    global _spec_cache
    if _spec_cache is not None:
        return _spec_cache
    if _LOCAL_ARTIFACT.exists():
        _spec_cache = cast(dict[str, Any], json.loads(_LOCAL_ARTIFACT.read_text(encoding="utf-8")))
    else:
        _spec_cache = _download_artifact()
    return _spec_cache


def _har_features(log_volume: list[float]) -> list[float]:
    """Corsi HAR features: [lag1, mean over last 5, mean over last 22]."""
    if len(log_volume) < 22:
        raise ValueError(f"need >=22 days of volume history, got {len(log_volume)}")
    return [
        log_volume[-1],
        sum(log_volume[-5:]) / 5,
        sum(log_volume[-22:]) / 22,
    ]


def predict_volume(log_volume_history: list[float], symbol: str = "^GSPC") -> VolumeForecastResponse:
    """Forecast forward mean log-volume at each horizon from recent daily log-volume.

    ``log_volume_history`` is the trailing series of daily log(volume) (most recent
    last, >= 22 points). Applies the saved per-horizon standardization + HAR
    coefficients from the artifact.
    """
    spec = _load_spec()
    feats = _har_features(log_volume_history)
    horizons: list[VolumeHorizonForecast] = []
    for key, days in _HORIZON_DAYS.items():
        h = spec["by_horizon"].get(key)
        if h is None:
            continue
        std = [(feats[i] - h["feat_mean"][i]) / (h["feat_std"][i] or 1.0) for i in range(len(feats))]
        log_vol = float(sum(std[i] * h["coef"][i] for i in range(len(std))) + h["intercept"])
        horizons.append(
            VolumeHorizonForecast(
                horizon_days=days,
                expected_log_volume=round(log_vol, 6),
                expected_volume=round(math.exp(log_vol), 2),
            )
        )
    if not horizons:
        raise VolumeForecasterUnavailable("artifact has no usable horizons")
    return VolumeForecastResponse(symbol=symbol, horizons=horizons, model="volume-head-HAR")


def load_recent_log_volume(symbol: str = "^GSPC", lookback_days: int = 60) -> list[float]:
    """Recent daily log-volume for the symbol (yfinance), most-recent last."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=f"{lookback_days}d", auto_adjust=False)
    if hist is None or hist.empty or "Volume" not in hist.columns:
        raise VolumeForecasterUnavailable(f"no volume history for {symbol!r}")
    vols = [float(v) for v in hist["Volume"].tolist() if v and v > 0]
    if len(vols) < 22:
        raise VolumeForecasterUnavailable(f"insufficient volume history for {symbol!r} ({len(vols)})")
    return [math.log(v) for v in vols]


def forecast_volume(symbol: str = "^GSPC") -> VolumeForecastResponse:
    """End-to-end: fetch recent volume and forecast forward volume for the symbol."""
    return predict_volume(load_recent_log_volume(symbol), symbol=symbol)
