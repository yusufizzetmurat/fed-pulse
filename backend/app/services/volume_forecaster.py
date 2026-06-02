"""Serving layer for the HAR-based expected-volume forecaster.

Mirrors :mod:`app.services.rv_forecaster` but speaks log-volume instead of
log-realized-variance. The artifact is hand-built from the per-fold HAR +
calendar regression coefficients fit by
:mod:`app.data.late_fusion_volume`; see
``backend/models/volume_har/README.md`` (to be written when the artifact
is published) for the assembly procedure. Per-horizon contents: HAR
coefficients on Corsi log-volume lags ``[intercept, daily, weekly,
monthly]``, an optional weekday + month-end / quarter-end seasonality
block (``calendar_dummy_names`` / ``calendar_dummy_coef``), conformal
residual quantiles at the 80% / 90% bands, and the offline pooled
walk-forward R^2 the calibration chip surfaces.

Note the contract is enforced at load time — :func:`predict_abnormal_volume`
raises :class:`VolumeForecasterUnavailable` if ``by_horizon['h{h}']['har_coef']``
is missing or shorter than 4 entries. The evaluation-only entrypoint at
:mod:`app.data.late_fusion_volume` emits only R^2 stats (``r2_har`` /
``r2_rich_linear`` / ``r2_dl`` / ``*_minus_har_ci90``) — it does NOT
produce ``har_coef`` / ``conformal_quantiles`` / calendar block on its
own, so that JSON is not a valid serving artifact. The hand-assembled
artifact is the source of truth for this service.

The card is intentionally market-data only: it consumes a recent daily
log-volume history and emits the per-horizon expected log-residual against
the HAR baseline along with a back-transformed percent-vs-baseline read.
Text features never enter this surface.

The artifact lives in ``yusufizzetmurat/fomc-volume-har`` on HF Hub;
:func:`predict_abnormal_volume` lazily downloads it on first call and
caches the parsed spec for the process lifetime.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.config import BACKEND_ROOT

logger = logging.getLogger(__name__)


# No exact match for an existing volume artifact in code (rv_forecaster
# uses ``yusufizzetmurat/fomc-rv-qlike-forecaster``); the volume head
# is its own artifact. The fallback ID below is the canonical name; if
# the repo is missing or private without an HF_TOKEN, the endpoint
# returns a structured 503 via ``VolumeForecasterUnavailable``.
HF_REPO_ID = "yusufizzetmurat/fomc-volume-har"
MODEL_DIR = BACKEND_ROOT / "models" / "volume_har"
ARTIFACT_FILENAME = "volume_har_artifact.json"
HORIZONS: tuple[int, ...] = (1, 5, 22)
# Conformal residual widths are stored under the same q-tail keys the
# RV forecaster uses (q[1-cov] = q[alpha]).
_ALPHA_80 = "0.20"
_ALPHA_90 = "0.10"
# Minimum log-volume history required to populate the HAR Corsi lags
# (lag1 + 5-day mean + 22-day mean). Mirrors ``_MAX_LAG`` in
# :mod:`app.data.late_fusion_volume`.
_MIN_LOG_VOL_HISTORY = 22


class VolumeForecasterUnavailable(RuntimeError):
    """Raised when the artifact is missing and HF Hub fetch failed."""


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download_artifact(target_dir: Path) -> dict[str, Any]:
    """Download the HAR-volume spec JSON into ``target_dir``.

    Reads the pinned revision from ``registry.yaml`` when present so a
    corrupt-cache or cold-start fallback through this path stays bound
    to the same sha the boot-time eager-pull resolves. Falls back to
    HEAD (``main``) only when the registry entry is missing or unpinned,
    matching the conservative behaviour of the legacy implementation.
    """

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    from app.models.registry import artefact_ref

    target_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    kwargs: dict[str, Any] = {
        "repo_id": HF_REPO_ID,
        "filename": ARTIFACT_FILENAME,
        "local_dir": str(target_dir),
    }
    if token:
        kwargs["token"] = token
    try:
        ref = artefact_ref("volume_har_canonical")
    except Exception:
        ref = None
    if ref is not None and ref.revision:
        kwargs["revision"] = ref.revision

    try:
        spec_path = Path(hf_hub_download(**kwargs))
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except RepositoryNotFoundError as exc:
        raise VolumeForecasterUnavailable(
            f"HF repo {HF_REPO_ID!r} not found; set HF_TOKEN or grant access."
        ) from exc
    except (EntryNotFoundError, HfHubHTTPError) as exc:
        raise VolumeForecasterUnavailable(f"HF fetch failed for {HF_REPO_ID!r}: {exc}") from exc
    except OSError as exc:
        # Network-level surface (connection reset, timeout, disk full)
        # that ``hf_hub_download`` raises outside the HF exception
        # hierarchy. Surface as the same 503-mapped failure rather than
        # bubbling up as a 500.
        raise VolumeForecasterUnavailable(f"HF download failed for {HF_REPO_ID!r}: {exc}") from exc
    return cast(dict[str, Any], spec)


# Cold-start fit defaults. Mirrors ``_VOLUME_HISTORY_DAYS`` in main.py:
# 180 calendar days yields ~126 trading bars, which clears the
# ``walk_forward_splits`` floor of ``n_folds * embargo = 5 * 22 = 110``.
_COLD_START_PERIOD = "180d"
_COLD_START_SYMBOL = "^GSPC"


def _cold_start_fit(model_dir: Path) -> dict[str, Any]:
    """Fit a serving artifact from live yfinance volume on first boot.

    Pulls ~180 calendar days of daily volume for ``^GSPC``, writes a
    temporary parquet, and calls
    :func:`app.data.late_fusion_volume.fit_production_artifact` to build
    the per-horizon HAR + seasonality + conformal-quantile spec. The
    output is written to ``model_dir / ARTIFACT_FILENAME`` so subsequent
    process restarts skip the fit. Re-raises
    :class:`VolumeForecasterUnavailable` on any failure so the existing
    503 path is preserved when yfinance is also unreachable.
    """

    import pandas as pd
    import yfinance as yf

    from app.data.late_fusion_volume import fit_production_artifact

    try:
        ticker = yf.Ticker(_COLD_START_SYMBOL)
        frame = ticker.history(period=_COLD_START_PERIOD, auto_adjust=True)
        if frame is None or frame.empty:
            raise RuntimeError(f"no market history available for {_COLD_START_SYMBOL}")
        vol = frame["Volume"].astype(float).dropna()
        if hasattr(vol, "columns"):
            vol = vol.iloc[:, 0].dropna()
        vol = vol[vol > 0]
        if vol.empty:
            raise RuntimeError(f"no positive volume rows for {_COLD_START_SYMBOL}")
        dates = pd.to_datetime(vol.index).tz_localize(None)
        bars = pd.DataFrame({"date": dates, "volume": vol.to_numpy(dtype=float)}).reset_index(
            drop=True
        )

        out_path = model_dir / ARTIFACT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            vol_path = Path(tmp) / "volume_cold_start.parquet"
            bars.to_parquet(vol_path)
            spec = fit_production_artifact(vol_path, out_path)
        logger.info(
            "volume_har_cold_start_fit symbol=%s rows=%d artifact=%s",
            _COLD_START_SYMBOL,
            len(bars),
            out_path,
        )
        return cast(dict[str, Any], spec)
    except VolumeForecasterUnavailable:
        raise
    except Exception as exc:
        raise VolumeForecasterUnavailable(
            f"cold-start fit failed for {_COLD_START_SYMBOL!r}: {exc}"
        ) from exc


def _load_spec(model_dir: Path) -> dict[str, Any]:
    """Return the cached spec; fetch from HF Hub on a miss.

    Falls back to an in-process cold-start fit off live yfinance volume
    when the HF Hub repo is missing or unreachable, so a fresh checkout
    without a pre-trained artifact still serves the Expected Volume
    card. The cold-start fit is the genuine fallback — the HF download
    stays the primary path when an artifact is published.
    """

    spec_path = model_dir / ARTIFACT_FILENAME
    if not spec_path.exists():
        try:
            return _download_artifact(model_dir)
        except VolumeForecasterUnavailable as exc:
            logger.info(
                "volume_har_hf_unavailable falling_back_to_cold_start error=%s",
                exc,
            )
            return _cold_start_fit(model_dir)
    try:
        return cast(dict[str, Any], json.loads(spec_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        # Corrupt local cache; force a clean re-download.
        try:
            return _download_artifact(model_dir)
        except VolumeForecasterUnavailable as exc:
            logger.info(
                "volume_har_hf_unavailable falling_back_to_cold_start error=%s",
                exc,
            )
            return _cold_start_fit(model_dir)


class _VolumePredictor:
    """Cached per-process serving bundle for the HAR-volume head.

    Holds the parsed spec under a class-level lock so concurrent
    callers do not race on the HF download. The instance carries only
    the JSON spec — there are no torch weights on the volume head, so
    the predictor is cheap to construct once.
    """

    _instance: "_VolumePredictor | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "_VolumePredictor":
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
        self.revision = f"{self.spec.get('model', 'volume_har')}@{self.spec.get('date_last', '')}"


def _har_lag_row(log_vol: np.ndarray) -> tuple[float, float, float]:
    """Return the last HAR Corsi triple ``(lag1, mean5, mean22)``."""

    last = log_vol[-1]
    mean5 = float(log_vol[-5:].mean())
    mean22 = float(log_vol[-22:].mean())
    return float(last), mean5, mean22


def _calendar_features_row(forecast_date: datetime, dummy_names: list[str]) -> list[float]:
    """Build the calendar-feature row matching the artifact ``dummy_names``.

    Mirrors :func:`app.data.late_fusion_volume._calendar_features`:
    Mon..Thu indicator (Fri baseline), month-end (day >= 25), quarter-end
    (month-end in Mar/Jun/Sep/Dec). Any name the artifact does not
    declare degrades to 0.0 so an artifact with a different seasonality
    block still serializes cleanly.
    """

    dow = forecast_date.weekday()
    dom = forecast_date.day
    month = forecast_date.month
    is_month_end = dom >= 25
    is_quarter_end = is_month_end and month in (3, 6, 9, 12)
    row: list[float] = []
    for name in dummy_names:
        if name.startswith("dow_"):
            try:
                k = int(name.split("_", 1)[1])
            except ValueError:
                row.append(0.0)
                continue
            row.append(1.0 if dow == k else 0.0)
        elif name == "month_end":
            row.append(1.0 if is_month_end else 0.0)
        elif name == "quarter_end":
            row.append(1.0 if is_quarter_end else 0.0)
        else:
            row.append(0.0)
    return row


def _point_log_residual(
    log_vol: np.ndarray,
    row: dict[str, Any],
    forecast_date: datetime,
) -> tuple[float, bool]:
    """Compute the per-horizon HAR-volume point in log-residual space.

    ``log_residual`` here is ``log_pred - baseline`` where ``baseline``
    is the rolling 22-day mean of the supplied history. The chart reads
    the same residual as ``point_pct_vs_baseline = (exp(residual)-1)*100``.
    Returns the residual plus a ``calendar_adjusted`` flag indicating
    whether the artifact carried a seasonality block applied here.
    """

    lag1, mean5, mean22 = _har_lag_row(log_vol)
    har_coef = np.asarray(row["har_coef"], dtype=np.float64)
    # Convention: [intercept, daily, weekly, monthly] — matches
    # rv_forecaster's HAR triple.
    if har_coef.size < 4:
        raise VolumeForecasterUnavailable(
            "HAR coefficient vector must carry 4 entries (intercept, d, w, m)"
        )
    log_pred = float(har_coef[0] + har_coef[1] * lag1 + har_coef[2] * mean5 + har_coef[3] * mean22)
    calendar_adjusted = False
    dummy_names = row.get("calendar_dummy_names")
    dummy_coef = row.get("calendar_dummy_coef")
    if (
        isinstance(dummy_names, list)
        and isinstance(dummy_coef, list)
        and len(dummy_names) == len(dummy_coef)
        and dummy_names
    ):
        cal_row = _calendar_features_row(forecast_date, dummy_names)
        adjustment = float(np.dot(np.asarray(dummy_coef, dtype=np.float64), cal_row))
        log_pred += adjustment
        # Only flip the chip when the calendar block actually moves the
        # forecast. An artifact that declares only unrecognized names —
        # or that supplies all-zero coefficients, or whose recognized
        # dummies happen to all evaluate to zero on this forecast date —
        # dot-products into a zero adjustment; the math collapses to the
        # no-calendar branch, so the UX signal must as well.
        calendar_adjusted = adjustment != 0.0
    baseline = mean22
    return log_pred - baseline, calendar_adjusted


def _safe_float(v: Any) -> float | None:
    """Coerce ``v`` to a finite float; return None on non-numeric input."""

    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _parse_forecast_date(forecast_date: str | datetime | None) -> datetime:
    if isinstance(forecast_date, datetime):
        return forecast_date
    if isinstance(forecast_date, str) and forecast_date:
        try:
            return datetime.fromisoformat(forecast_date)
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def predict_abnormal_volume(
    volume_history: list[float] | np.ndarray,
    symbol: str = "^GSPC",
    forecast_date: str | datetime | None = None,
) -> dict[str, Any]:
    """Multi-horizon banded expected-volume forecast off recent daily volumes.

    ``volume_history`` is a chronologically ordered series of strictly
    positive daily share volumes (NOT log). The function transforms to
    log-volume, applies the per-horizon HAR Corsi triple from the
    artifact, layers in the seasonality dummies when present, and
    returns per-horizon point log-residual + 80% / 90% conformal bands.

    ``point_pct_vs_baseline = (exp(point_log_residual) - 1) * 100`` is
    the headline percent-vs-baseline number the card renders. Bands
    follow the same back-transform.

    Raises:
        ValueError: history is too short / non-positive.
        VolumeForecasterUnavailable: artifact cannot be loaded.
    """

    vol = np.asarray(volume_history, dtype=np.float64)
    if vol.ndim != 1 or len(vol) < _MIN_LOG_VOL_HISTORY:
        raise ValueError(
            "volume_history must be a 1-D series of at least "
            f"{_MIN_LOG_VOL_HISTORY} daily volume values"
        )
    if np.any(vol <= 0) or not np.all(np.isfinite(vol)):
        raise ValueError("volume_history values must be positive finite numbers")

    log_vol = np.log(vol)
    pred = _VolumePredictor.get()
    fdate = _parse_forecast_date(forecast_date)

    horizons_out: list[dict[str, Any]] = []
    for h in HORIZONS:
        hk = f"h{h}"
        row = pred.spec.get("by_horizon", {}).get(hk)
        if not isinstance(row, dict):
            raise VolumeForecasterUnavailable(
                f"HAR-volume artifact missing per-horizon block {hk!r}"
            )
        log_residual, calendar_adjusted = _point_log_residual(log_vol, row, fdate)
        quants = row.get("conformal_quantiles") or {}
        q80 = float(quants.get(_ALPHA_80, 0.0) or 0.0)
        q90 = float(quants.get(_ALPHA_90, 0.0) or 0.0)
        # Back-transform residual + bands into multiplicative percent
        # space. ``exp(r) - 1`` puts ``+0.10`` log-residual at +10.5%.
        point_pct = (math.exp(log_residual) - 1.0) * 100.0
        band_lo_80 = (math.exp(log_residual - q80) - 1.0) * 100.0
        band_hi_80 = (math.exp(log_residual + q80) - 1.0) * 100.0
        band_lo_90 = (math.exp(log_residual - q90) - 1.0) * 100.0
        band_hi_90 = (math.exp(log_residual + q90) - 1.0) * 100.0
        r2 = _safe_float(row.get("r2_har"))
        horizons_out.append(
            {
                "h": h,
                "point_log_residual": float(log_residual),
                "point_pct_vs_baseline": float(point_pct),
                "band_lo_80": float(band_lo_80),
                "band_hi_80": float(band_hi_80),
                "band_lo_90": float(band_lo_90),
                "band_hi_90": float(band_hi_90),
                "r2_har": r2,
                "calendar_adjusted": bool(calendar_adjusted),
            }
        )
    return {
        "symbol": symbol,
        "horizons": horizons_out,
        "model_revision": pred.revision,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
