from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.models.config import ModelConfig


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    close_rmse: float
    volatility_rmse: float
    combined_rmse: float
    # Classification view of the same predictions (Phase 9). Derived in
    # ``app.evaluation.directional_metrics.compute_directional_metrics``
    # by comparing the sign of the predicted close-delta against
    # ``direction_t1d`` from the events parquet. Optional so legacy
    # eval paths (the regression-only tests, pre-Phase-9 checkpoints
    # round-tripped through ``from_dict``) keep validating; populated
    # by :func:`app.training.loop._evaluate_model` on every new run.
    direction_accuracy: float | None = None
    f1_macro: float | None = None
    direction_auc: float | None = None
    # Phase 9 V2 (#195) vol-regime classification view. Populated when
    # the model was trained with ``output_mode="classification"``;
    # ``None`` on regression-only runs so the legacy contract holds.
    # ``regime_accuracy`` is plain top-1 over the (n_classes) head;
    # ``regime_f1_macro`` is the unweighted macro F1; ``regime_loss``
    # is the cross-entropy averaged across the partition.
    regime_accuracy: float | None = None
    regime_f1_macro: float | None = None
    regime_loss: float | None = None
    # Per-class breakdown for the regime classifier (#199). The
    # dataclass lives in ``app.evaluation.classification_breakdown`` but
    # serialises to a dict here so the existing JSON contract on
    # ``EvaluationMetrics.to_dict()`` keeps round-tripping cleanly.
    # ``None`` on regression-only runs so the legacy contract holds.
    classification_breakdown: dict[str, Any] | None = None
    # Phase A (#226) row-level test-partition surface. Populated only on
    # the test partition (val/train stay None so the per-trial JSON does
    # not balloon with the train-window predictions). Lets the pooled-
    # fold aggregator pool across folds and the ensemble aggregator
    # average logits / softmax probabilities across architectures.
    # ``None`` everywhere except classification-mode test-partition eval.
    predictions: list[int] | None = None
    targets: list[int] | None = None
    class_scores: list[list[float]] | None = None
    # Gated InfoNCE fusion (#235) diagnostic. Populated only when the
    # model is a MultiModalForecasterModel; ``gate_summary['mean']``
    # is the scalar mean gate value (>0.5 → market-leaning, <0.5 →
    # text-leaning), ``gate_summary['mean_per_class']`` carries the
    # class-conditional mean so the thesis appendix can say whether
    # the gate shifts modality reliance per regime, and
    # ``gate_summary['n_rows']`` is the eval-partition row count the
    # summary was averaged over.
    gate_summary: dict[str, Any] | None = None
    # #304 dual-head methodology surface. Populated whenever the
    # forecaster carried a ``regression_head`` (``head_mode`` in
    # ``regression`` / ``dual``). The pair is computed in log space --
    # the head predicts ``log(forward_realized_vol_10d)`` and the
    # target is the same -- so the units are dimensionless. ``None`` on
    # classification-only runs so the legacy dataclass shape holds.
    regression_rmse_log_rv: float | None = None
    regression_mae_log_rv: float | None = None
    regression_loss: float | None = None
    # #304 acceptance: per-fold R^2 on log_rv joins the existing
    # RMSE/MAE pair so the three-way comparison table (classification-
    # only, regression-only, dual) can report a scale-free goodness-of-
    # fit alongside the absolute-error metrics. ``1 - SSE / SST`` over
    # the partition's standardised log(forward_realized_vol_10d)
    # values; collapses to ``None`` on a partition where SST is 0
    # (constant target — pathological fixture only) so the consumer
    # can tell ``no head ran`` apart from ``head ran on a degenerate
    # partition``.
    regression_r2_log_rv: float | None = None
    # #292 rates-complex per-head metrics. Keyed on the head short
    # name (``2y`` / ``5y`` / ``terminal``); each value is a dict
    # mirroring the regression-metric panel from
    # :mod:`app.evaluation.regression_metrics`
    # (``mae_bps`` / ``directional_accuracy`` / ``r_squared`` -> CI
    # dicts) plus per-head row arrays
    # (``predictions_bps`` / ``actuals_bps``) the conformal
    # calibrator + the §16 comparison table read off. ``None`` when
    # rates heads are inactive on the run.
    rates_metrics: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    model_config: ModelConfig
    device: str
    epochs_requested: int
    epochs_completed: int
    batch_size: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    sequence_groups: int
    total_windows: int
    train_windows: int
    validation_windows: int
    checkpoint_path: str | None
    checkpoint_saved: bool
    best_epoch: int | None = None
    metrics: EvaluationMetrics | None = None
    train_metrics: EvaluationMetrics | None = None
    # Explicit walk-forward partition metrics. ``val_metrics`` is the
    # best early-stopping checkpoint's validation RMSE; ``test_metrics``
    # is the held-out test-partition RMSE -- the headline number the
    # aggregator emits as ``test-RMSE``. On the legacy 80/20 internal
    # split path both fields collapse to the val partition (no real
    # held-out test set exists), so the back-compat regression contract
    # is preserved while the walk-forward path carries a real test
    # eval. ``protocol`` distinguishes the two paths in the per-trial
    # record.
    val_metrics: EvaluationMetrics | None = None
    test_metrics: EvaluationMetrics | None = None
    fold_id: str | None = None
    protocol: str = "legacy-80-20"
    weight_decay: float = 1e-4
    target_mode: str = "real"
    text_encoder: str | None = None
    text_adapter_dim: int = 0
    text_pool_lambda_inv_days: float = 0.0
    # #304 dual-head: per-fold log_rv standardiser fitted on the train
    # slice only. ``None`` on head_mode='classification' runs (the
    # default) so the regression byte-identity contract holds. When
    # head_mode in {regression, dual}, the dict carries
    # ``{"mean": float, "std": float}`` so downstream consumers can
    # invert the standardised regression head output (multiply by std,
    # add mean, then ``exp(...)`` to recover the original
    # ``forward_realized_vol_10d`` units).
    log_rv_scaler: dict[str, float] | None = None
    # #292 rates-complex per-head standardiser + quantile edges fitted
    # on the train slice only. Persisted onto the run summary so the
    # inference path (``services.forecaster``) can invert the
    # standardised regression output back to raw bps and so the API
    # response carries the per-head bps band in the natural finance
    # unit. ``rates_scalers`` maps the head short name to a
    # ``{"mean": float, "std": float}`` dict; ``rates_quantile_edges``
    # maps it to a ``{"column": str, "lower": float, "upper": float,
    # "n_train_rows": int}`` dict (the per-fold tertile cutoffs used
    # by the auxiliary classification surface).
    rates_scalers: dict[str, dict[str, float]] | None = None
    rates_quantile_edges: dict[str, dict[str, float | int | str]] | None = None
    # #273 multi-task loss. ``None`` on every pre-#273 run and on
    # multi_task_loss=False runs (the default). When the joint loss is
    # active the dict carries the per-axis class-weight vectors fitted on
    # the train slice plus the four lambda coefficients the
    # ``MultiTaskLoss`` module was constructed with, so a resume from the
    # checkpoint reads back the exact loss config the run trained under.
    # Schema: ``{"stance": [...], "certainty": [...], "topic": [...],
    # "lambdas": {"stance": float, "factor": float, "certainty": float,
    # "topic": float}}``. The factor axis is a regression branch with no
    # class weights, so it appears only under ``lambdas``.
    multi_task_class_weights: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    model: "Any"
    summary: TrainingRunSummary


@dataclass
class TrainingDataSourceSummary:
    path: Path
    format: str
    record_groups: int
    records: int
    vectors: int
    usable_sequences: int
    status: str
    message: str
