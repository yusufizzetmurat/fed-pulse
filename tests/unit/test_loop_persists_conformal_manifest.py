"""The training loop must write a conformal sidecar after
classification-mode val eval at the best epoch (#216).

The helper ``_maybe_write_classification_conformal_manifest`` reads
``class_scores`` + ``targets`` off ``best_val_metrics`` (already
collected by ``_evaluate_model`` in classification mode) and persists
the APS threshold to ``<checkpoint_stem>.conformal.json``. Regression-
only runs leave the helper as a no-op so the legacy contract is
byte-identical.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.evaluation.metrics import EvaluationMetrics
from app.training.loop import _maybe_write_classification_conformal_manifest


def _build_metrics(
    *,
    class_scores: list[list[float]] | None,
    targets: list[int] | None,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regime_accuracy=0.6,
        regime_f1_macro=0.5,
        regime_loss=0.5,
        class_scores=class_scores,
        targets=targets,
    )


def test_writes_sidecar_when_class_scores_and_targets_present() -> None:
    """A classification-mode best_val_metrics with both arrays must
    produce a sidecar containing ``softmax_quantile``."""

    n = 200
    class_scores: list[list[float]] = []
    targets: list[int] = []
    for i in range(n):
        y = i % 3
        targets.append(y)
        row = [0.1, 0.1, 0.1]
        row[y] = 0.8
        s = sum(row)
        class_scores.append([v / s for v in row])
    metrics = _build_metrics(class_scores=class_scores, targets=targets)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")  # placeholder file
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        assert sidecar.exists(), "expected conformal sidecar next to checkpoint"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload.get("softmax_quantile") is not None
    assert payload["softmax_quantile"] > 0.0
    assert payload["calibration_n"] == n


def test_no_op_when_metrics_is_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(None, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        assert not sidecar.exists()


def test_no_op_when_class_scores_missing() -> None:
    """Regression-only runs leave ``class_scores`` and ``targets`` at
    None; the helper must skip the write so the legacy regression
    sidecar (when it exists, written by the calibrate_split_conformal
    path) is not clobbered."""

    metrics = _build_metrics(class_scores=None, targets=None)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        assert not sidecar.exists()


def test_no_op_when_arrays_are_length_mismatched() -> None:
    metrics = _build_metrics(
        class_scores=[[0.3, 0.4, 0.3]] * 10,
        targets=[0] * 5,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "forecaster_best.pt"
        ckpt.write_bytes(b"")
        _maybe_write_classification_conformal_manifest(metrics, ckpt)
        sidecar = Path(td) / "forecaster_best.conformal.json"
        assert not sidecar.exists()
