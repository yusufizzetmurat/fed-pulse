"""Replay-mode (time-machine) fold resolution.

The /analyze flow can run "as of" a historical date X. To honour the
walk-forward invariant (nothing from after X leaks into the request),
the forecaster + trajectory services must load the checkpoint of the
fold whose ``train_end < X``. This module resolves that fold from a
manifest on disk and points the caller at the per-fold checkpoint
path.

Per-fold checkpoints are NOT shipped in the repo today -- they live
on the training runner. When the manifest is missing or no checkpoint
file exists, :func:`resolve_fold_for_date` returns a
:class:`FoldRef.unavailable(...)` instance so the API can emit a
clean 422 instead of silently falling back to the post-X checkpoint
sitting on disk.

Manifest shape (JSON, anchored at
``data/processed/canonical/fold_manifest_expanding_walk_forward.json``)::

    {
      "training_package_id": "canonical",
      "folds": [
        {
          "fold_id": "wf_fold_1",
          "train_end": "2018-12-31",
          "test_start": "2019-01-02",
          "test_end": "2019-09-30",
          "checkpoint_dir": "backend/models/folds/wf_fold_1"
        },
        ...
      ]
    }

``checkpoint_dir`` is interpreted relative to the repo root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# Resolved at import time so callers stay cheap; the file system is only
# touched at resolve-time.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent

_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / "data" / "processed" / "canonical" / "fold_manifest_expanding_walk_forward.json"
)


@dataclass(frozen=True)
class FoldRef:
    """A resolved (fold, checkpoint) reference, or a structured reason
    why one could not be served.

    ``available`` is the boolean callers branch on. When False the
    other fields except ``reason`` may be ``None``; the API surfaces
    ``reason`` verbatim in the 422 detail so the user knows whether the
    manifest is missing, the date is out of range, etc.
    """

    available: bool
    fold_id: str | None = None
    train_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    forecaster_checkpoint: Path | None = None
    trajectory_bundle: Path | None = None
    reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "FoldRef":
        return cls(available=False, reason=reason)


def _parse_iso(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]] | None:
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    folds = raw.get("folds") if isinstance(raw, dict) else None
    if not isinstance(folds, list):
        return None
    return folds


def _resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return candidate


def resolve_fold_for_date(  # noqa: C901 - branching is defensive parsing
    as_of: date,
    *,
    manifest_path: Path | None = None,
) -> FoldRef:
    """Return the fold whose ``train_end < as_of`` is the largest.

    The selected fold is the last one whose training window closed
    strictly before ``as_of`` -- so the model never saw an event on or
    after ``as_of`` during training. Tie-break: the latest ``train_end``
    wins.

    If no manifest exists on disk, the per-fold checkpoint scheme is
    not deployed on this host -- replay cannot serve a real prediction
    so we return :meth:`FoldRef.unavailable`. The API maps that to a
    422 and the frontend renders the graceful empty state.
    """

    if not isinstance(as_of, date):
        return FoldRef.unavailable("invalid_as_of_date")

    path = manifest_path or _DEFAULT_MANIFEST_PATH
    folds = _load_manifest(path)
    if folds is None:
        return FoldRef.unavailable("fold_manifest_missing")

    candidate: dict[str, Any] | None = None
    candidate_train_end: date | None = None
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        train_end = _parse_iso(fold.get("train_end"))
        if train_end is None or train_end >= as_of:
            continue
        if candidate_train_end is None or train_end > candidate_train_end:
            candidate = fold
            candidate_train_end = train_end

    if candidate is None or candidate_train_end is None:
        return FoldRef.unavailable("no_fold_before_as_of")

    fold_id = candidate.get("fold_id")
    if not isinstance(fold_id, str) or not fold_id:
        return FoldRef.unavailable("fold_id_missing")

    checkpoint_dir = _resolve_path(candidate.get("checkpoint_dir"))
    forecaster_ckpt: Path | None = None
    trajectory_bundle: Path | None = None
    if checkpoint_dir is not None:
        forecaster_candidate = checkpoint_dir / "forecaster_best.pt"
        if forecaster_candidate.exists():
            forecaster_ckpt = forecaster_candidate
        trajectory_candidate = checkpoint_dir / "trajectory"
        if trajectory_candidate.exists():
            trajectory_bundle = trajectory_candidate

    if forecaster_ckpt is None:
        return FoldRef.unavailable("fold_checkpoint_missing")

    return FoldRef(
        available=True,
        fold_id=fold_id,
        train_end=candidate_train_end,
        test_start=_parse_iso(candidate.get("test_start")),
        test_end=_parse_iso(candidate.get("test_end")),
        forecaster_checkpoint=forecaster_ckpt,
        trajectory_bundle=trajectory_bundle,
        reason=None,
    )


def realised_outcome(
    as_of: date,
    *,
    symbol: str = "^GSPC",
) -> dict[str, Any]:
    """Compute realised vol_h and log-return for the 1/5/10 trading days
    after ``as_of``.

    Realised log-return at horizon h is ``ln(close_{t+h} / close_{t})``
    where ``close_t`` is the last close on or before ``as_of`` and
    ``close_{t+h}`` is the h-th trading bar after ``as_of``. Realised
    vol_h is the rolling-5d stdev of daily returns measured at bar
    ``t+h`` (i.e. the same series the forecaster targets via the
    ``volatility_5d`` field).

    Each per-horizon read is ``None`` when the underlying market
    history does not extend that far -- a near-current date may have
    no t+10 bar yet.
    """

    # Imported lazily so the module stays cheap to import inside the
    # test environment when yfinance is not on the path.
    from app.services.market_data import fetch_event_study_window

    iso = as_of.isoformat()
    bars: list[dict[str, Any]]
    try:
        bars = list(
            fetch_event_study_window(
                event_date=iso,
                symbol=symbol,
                steps=10,
                window_days=30,
            )
        )
    except Exception:  # noqa: BLE001 -- defensive: never break /analyze
        bars = []

    by_step: dict[int, dict[str, Any]] = {}
    for idx, bar in enumerate(bars, start=1):
        by_step[idx] = bar

    realised: dict[str, Any] = {
        "as_of_date": iso,
        "symbol": symbol,
        "horizons": [],
    }
    # log_return on the bar dict already references the anchor close on
    # event_date so a simple cumulative-sum across the path gives
    # ln(close_{t+h} / close_t) -- but the bar's own log_return is
    # bar-to-bar. Sum them to recover the cumulative read.
    cum = 0.0
    for step in (1, 5, 10):
        if step not in by_step:
            realised["horizons"].append(
                {
                    "horizon": step,
                    "log_return": None,
                    "realised_volatility_5d": None,
                    "close": None,
                    "date": None,
                }
            )
            continue
        # Walk forward summing log_returns up to ``step``.
        path_sum = 0.0
        for s in range(1, step + 1):
            bar_at_step: dict[str, Any] | None = by_step.get(s)
            if bar_at_step is None:
                path_sum = float("nan")
                break
            path_sum += float(bar_at_step.get("log_return") or 0.0)
        cum = path_sum
        bar_h = by_step[step]
        # Volatility: a rolling 5d stdev measured at the bar. We do not
        # have the daily series here, so compute it as the stdev of the
        # bar-to-bar log-returns over the trailing 5 bars (or whatever
        # is available).
        window: list[float] = []
        for s in range(max(1, step - 4), step + 1):
            bar_s = by_step.get(s)
            if bar_s is None:
                continue
            window.append(float(bar_s.get("log_return") or 0.0))
        vol: float | None
        if len(window) >= 2:
            mean = sum(window) / len(window)
            var = sum((v - mean) ** 2 for v in window) / (len(window) - 1)
            vol = var**0.5
        else:
            vol = None
        close_val = bar_h.get("close")
        realised["horizons"].append(
            {
                "horizon": step,
                "log_return": (
                    float(cum)
                    if isinstance(cum, float) and cum == cum  # filter NaN
                    else None
                ),
                "realised_volatility_5d": vol,
                "close": float(close_val) if close_val is not None else None,
                "date": str(bar_h.get("date") or ""),
            }
        )
    return realised


__all__ = ["FoldRef", "resolve_fold_for_date", "realised_outcome"]
