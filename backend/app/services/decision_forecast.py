"""Read-only loader for the next-FOMC-decision artifacts.

The forecaster at :mod:`app.forecasting.next_fomc_decision` writes
``results.json``, ``metrics.json``, and ``feature_attribution.md`` into
``data/artifacts/next_fomc/`` after a CLI run. This module reads those
files for the /decisions dashboard.

When the directory is absent the loader returns an ``available: False``
shape rather than raising -- the dashboard surfaces a documented
empty-state.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.forecasting.next_fomc_decision import ORDINAL_CLASSES
from app.services.fomc_calendar import list_all_meetings

LOGGER = logging.getLogger(__name__)

# Cold-start fallback: like the volume / RV / multi-axis cards, pull the
# generated artifacts from the HF Hub when they are absent locally so a fresh
# deploy hydrates without a manual ``make next-fomc`` run. Set the env to "" to
# disable (e.g. in tests of the genuine empty-state).
_NEXT_FOMC_HF_REPO = os.environ.get(
    "FED_PULSE_NEXT_FOMC_HF_REPO", "yusufizzetmurat/fomc-next-decision"
)
_ARTIFACT_FILES: tuple[str, ...] = (
    "results.json",
    "metrics.json",
    "feature_attribution.md",
)
_hub_hydration_attempted = False


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _maybe_hydrate_from_hub(artifacts_dir: Path) -> None:
    """Best-effort: fetch the next-FOMC artifacts from the Hub into
    ``artifacts_dir`` when ``results.json`` is missing locally. Mirrors the
    volume/RV cold-start fallbacks. Fail-safe and attempted once per process:
    any error (no repo, no network, no access) leaves the directory untouched
    and the loader returns its documented unavailable shape.
    """

    global _hub_hydration_attempted
    if (artifacts_dir / "results.json").exists():
        return
    if _hub_hydration_attempted or not _NEXT_FOMC_HF_REPO:
        return
    _hub_hydration_attempted = True
    try:
        from huggingface_hub import hf_hub_download

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        token = _hf_token()
        for filename in _ARTIFACT_FILES:
            kwargs: dict[str, Any] = {
                "repo_id": _NEXT_FOMC_HF_REPO,
                "filename": filename,
                "local_dir": str(artifacts_dir),
            }
            if token:
                kwargs["token"] = token
            try:
                hf_hub_download(**kwargs)
            except Exception as exc:  # per-file: a missing optional file is fine
                LOGGER.info(
                    "decision_forecast: Hub file %s unavailable on %s: %s",
                    filename,
                    _NEXT_FOMC_HF_REPO,
                    exc,
                )
        LOGGER.info(
            "decision_forecast: hydrated next-FOMC artifacts from Hub %s",
            _NEXT_FOMC_HF_REPO,
        )
    except Exception as exc:  # fail-safe — never break the endpoint on a cold start
        LOGGER.info(
            "decision_forecast: could not hydrate from Hub %s: %s",
            _NEXT_FOMC_HF_REPO,
            exc,
        )


@dataclass(frozen=True)
class _PredictedDecision:
    target_event_date: str
    target_as_of_ts: str
    target_class: str | None
    n_train_rows: int
    probabilities: dict[str, dict[str, float]]


def _argmax_class(probs: dict[str, float]) -> str:
    """Return the class with the highest probability."""

    if not probs:
        return ""
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _predicted_class_per_model(
    probabilities: dict[str, dict[str, float]],
) -> dict[str, str]:
    return {model: _argmax_class(probs) for model, probs in probabilities.items()}


def _next_scheduled_after(reference: date) -> dict[str, Any] | None:
    """Pick the next scheduled meeting strictly after ``reference``.

    Returns a dict shaped for :class:`NextFomcUpcomingMeeting` or
    ``None`` when no future meeting is on the calendar.
    """

    for meeting in list_all_meetings():
        if meeting.meeting_date >= reference:
            return {
                "meeting_date": meeting.meeting_date.isoformat(),
                "meeting_type": meeting.meeting_type,
                "statement_release_date": (
                    meeting.statement_release_date.isoformat()
                    if meeting.statement_release_date
                    else None
                ),
                "days_until": (meeting.meeting_date - reference).days,
            }
    return None


_MARKDOWN_TABLE_HEADER_RE = re.compile(r"^\|\s*Subset\s*\|", re.IGNORECASE)


def _parse_feature_attribution_markdown(text: str) -> list[dict[str, Any]]:
    """Parse the ablation table emitted by ``_format_attribution_md``.

    Returns one dict per table row with the same keys as
    :class:`NextFomcAttributionRow`.
    """

    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    table_started = False
    headers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if table_started:
                # Table ended; bail out.
                break
            continue
        if _MARKDOWN_TABLE_HEADER_RE.match(stripped):
            headers = [c.strip().lower() for c in _split_pipe_row(stripped)]
            table_started = True
            continue
        if not table_started:
            continue
        if set(stripped) <= set("|- "):
            # Divider row.
            continue
        cells = _split_pipe_row(stripped)
        if len(cells) != len(headers):
            continue
        record = dict(zip(headers, cells))
        rows.append(
            {
                "subset": record.get("subset", ""),
                "families": [
                    f.strip() for f in str(record.get("families", "")).split(",") if f.strip()
                ],
                "n_features": _maybe_int(record.get("#features")),
                "n": _maybe_int(record.get("n")),
                "brier": _maybe_float(record.get("brier")),
                "log_loss": _maybe_float(record.get("logloss")),
                "top1_accuracy": _maybe_float(record.get("top1acc")),
                "macro_f1": _maybe_float(record.get("macrof1")),
            }
        )
    return rows


def _split_pipe_row(line: str) -> list[str]:
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_next_fomc_artifacts(
    artifacts_dir: Path, *, reference_date: date | None = None
) -> dict[str, Any]:
    """Load + assemble the next-FOMC decision dashboard payload.

    Reads ``results.json`` / ``metrics.json`` / ``feature_attribution.md``
    from ``artifacts_dir``. Returns a dict matching
    :class:`NextFomcForecastResponse`. The dashboard renders an
    empty-state when ``available`` is False.
    """

    reference = reference_date or date.today()
    upcoming = _next_scheduled_after(reference)

    # Cold-start: pull artifacts from the Hub if they are not on disk yet.
    _maybe_hydrate_from_hub(artifacts_dir)

    base_response = {
        "available": False,
        "artifacts_dir": str(artifacts_dir),
        "ordinal_classes": list(ORDINAL_CLASSES),
        "model_names": [],
        "upcoming_meeting": upcoming,
        "headline": None,
        "history": [],
        "metrics_full_window": {},
        "metrics_ex_pandemic": {},
        "feature_attribution": [],
        "summary": {},
    }

    if not artifacts_dir.is_dir():
        return base_response

    results_path = artifacts_dir / "results.json"
    metrics_path = artifacts_dir / "metrics.json"
    attribution_path = artifacts_dir / "feature_attribution.md"

    if not results_path.exists():
        return base_response

    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("decision_forecast: failed to parse %s: %s", results_path, exc)
        return base_response

    predictions_raw: list[dict[str, Any]] = list(results.get("predictions") or [])
    summary = {
        k: int(v) for k, v in (results.get("summary") or {}).items() if isinstance(v, int | float)
    }

    # Decode each prediction; tolerate missing fields.
    decoded: list[_PredictedDecision] = []
    for entry in predictions_raw:
        if not isinstance(entry, dict):
            continue
        try:
            decoded.append(
                _PredictedDecision(
                    target_event_date=str(entry["target_event_date"]),
                    target_as_of_ts=str(entry["target_as_of_ts"]),
                    target_class=(
                        str(entry["target_class"])
                        if entry.get("target_class") is not None
                        else None
                    ),
                    n_train_rows=int(entry.get("n_train_rows", 0)),
                    probabilities={
                        str(model): {str(cls): float(p) for cls, p in (probs or {}).items()}
                        for model, probs in (entry.get("probabilities") or {}).items()
                    },
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    decoded.sort(key=lambda d: d.target_event_date)
    history = [
        {
            "target_event_date": d.target_event_date,
            "target_as_of_ts": d.target_as_of_ts,
            "target_class": d.target_class,
            "n_train_rows": d.n_train_rows,
            "probabilities": d.probabilities,
            "predicted_class": _predicted_class_per_model(d.probabilities),
        }
        for d in decoded
    ]

    headline_payload: dict[str, Any] | None = None
    if history:
        # Prefer the prediction whose target_event_date matches the next
        # upcoming meeting; otherwise fall back to the latest entry.
        upcoming_iso = upcoming["meeting_date"] if upcoming else None
        matched = next(
            (h for h in history if h["target_event_date"] == upcoming_iso),
            None,
        )
        headline_payload = matched if matched is not None else history[-1]

    metrics_full: dict[str, Any] = {}
    metrics_ex_pandemic: dict[str, Any] = {}
    model_names: list[str] = []
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("decision_forecast: failed to parse %s: %s", metrics_path, exc)
            metrics_payload = {}
        model_names = list(metrics_payload.get("model_names") or [])
        metrics_full = _coerce_metrics(metrics_payload.get("full_window") or {})
        metrics_ex_pandemic = _coerce_metrics(metrics_payload.get("ex_pandemic_window") or {})

    feature_attribution: list[dict[str, Any]] = []
    if attribution_path.exists():
        try:
            feature_attribution = _parse_feature_attribution_markdown(
                attribution_path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            LOGGER.warning("decision_forecast: failed to read %s: %s", attribution_path, exc)

    return {
        "available": True,
        "artifacts_dir": str(artifacts_dir),
        "ordinal_classes": list(ORDINAL_CLASSES),
        "model_names": model_names,
        "upcoming_meeting": upcoming,
        "headline": headline_payload,
        "history": history,
        "metrics_full_window": metrics_full,
        "metrics_ex_pandemic": metrics_ex_pandemic,
        "feature_attribution": feature_attribution,
        "summary": summary,
    }


def _coerce_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name, value in payload.items():
        if not isinstance(value, dict):
            continue
        out[str(model_name)] = {
            "n": _maybe_int(value.get("n")) or 0,
            "brier": _maybe_float(value.get("brier")),
            "log_loss": _maybe_float(value.get("log_loss")),
            "top1_accuracy": _maybe_float(value.get("top1_accuracy")),
            "macro_f1": _maybe_float(value.get("macro_f1")),
            "confusion_matrix": _coerce_confusion_matrix(value.get("confusion_matrix")),
        }
    return out


def _coerce_confusion_matrix(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for truth_class, row in value.items():
        if not isinstance(row, dict):
            continue
        out[str(truth_class)] = {
            str(pred_class): _maybe_int(count) or 0 for pred_class, count in row.items()
        }
    return out
