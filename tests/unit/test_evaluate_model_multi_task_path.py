"""Eval-time multi-task loss wiring regression (#285 follow-up).

The single-task path delegated val loss to ``loss_fn(predictions, y)`` —
the CrossEntropy term over the stance axis. The train side, when
``model_config.multi_task_loss=True``, optimises a lambda-weighted sum
across stance + certainty + time, so the pre-fix eval ranked
checkpoints against a different objective than the train side. These
tests pin the eval-side path:

- with ``multi_task_loss_fn`` provided, ``_evaluate_model`` dispatches
  through ``forward_multi_task`` and reports the lambda-weighted total
  on ``EvaluationMetrics.loss`` (and ``regime_loss`` for the
  classification surface);
- with ``multi_task_loss_fn=None`` the path stays byte-identical to the
  legacy single-task behaviour (same call shape, same per-class
  breakdown, same loss value).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from app.models.config import (
    FeatureVector,
    ModelConfig,
    MULTI_TASK_CERTAINTY_CLASSES,
    MULTI_TASK_TIME_CLASSES,
)
from app.training.loop import (
    _evaluate_model,
    _make_partition_dataset,
    train_model,
)
from app.training.loss import MultiTaskLoss


def _dummy_feature_vector(
    *,
    vol: float,
    day: int,
    stance: int,
    time: int,
    certainty: int,
) -> FeatureVector:
    return FeatureVector(
        date=_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
        target_stance_idx=stance,
        target_stance_present=True,
        target_time_idx=time,
        target_time_present=True,
        target_certainty_idx=certainty,
        target_certainty_present=True,
    )


def _build_classification_groups(n: int = 40) -> list[list[FeatureVector]]:
    """Synthetic walk-forward fold with all 3 axes populated per row."""

    return [
        [
            _dummy_feature_vector(
                day=i + 1,
                vol=0.01 + 0.001 * i,
                stance=i % 3,
                time=i % 2,
                certainty=i % 3,
            )
            for i in range(n)
        ]
    ]


class _ToyMultiTaskModel(nn.Module):
    """Minimal classifier exposing both ``forward`` and ``forward_multi_task``.

    Bypasses the LSTM construction so the unit test stays CPU-tiny: the
    "model" just emits deterministic logits derived from a Linear over a
    pooled view of the input tensor. The shape contract is what the
    eval helper exercises.
    """

    output_mode = "classification"

    def __init__(
        self,
        n_classes: int = 3,
        *,
        certainty_classes: int = MULTI_TASK_CERTAINTY_CLASSES,
        time_classes: int = MULTI_TASK_TIME_CLASSES,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.proj = nn.Linear(6, 16, bias=False)
        self.stance = nn.Linear(16, n_classes)
        self.certainty = nn.Linear(16, certainty_classes)
        self.time = nn.Linear(16, time_classes)
        self._text_path_active = False

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> mean-pool over time, then project.
        return self.proj(x.mean(dim=1))

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.stance(self._pool(x))

    def forward_multi_task(
        self, x: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        pooled = self._pool(x)
        return {
            "stance": self.stance(pooled),
            "certainty": self.certainty(pooled),
            "time": self.time(pooled),
        }


def _make_toy_classifier(n_classes: int = 3) -> _ToyMultiTaskModel:
    torch.manual_seed(0)
    return _ToyMultiTaskModel(n_classes=n_classes)


def _make_eval_loader(
    batch_size: int = 4,
    *,
    with_mt_aux: bool,
    seed: int = 0,
) -> tuple[DataLoader[Any], dict[str, torch.Tensor] | None]:
    """Build a small DataLoader for the eval helper."""

    torch.manual_seed(seed)
    n = 12
    x = torch.randn(n, 5, 6)
    y = torch.tensor([i % 3 for i in range(n)], dtype=torch.long)
    mt_aux: dict[str, torch.Tensor] | None = None
    if with_mt_aux:
        mt_aux = {
            "certainty": torch.tensor([i % MULTI_TASK_CERTAINTY_CLASSES for i in range(n)], dtype=torch.long),
            "certainty_mask": torch.tensor([i % 3 != 0 for i in range(n)], dtype=torch.bool),
            "time": torch.tensor([i % MULTI_TASK_TIME_CLASSES for i in range(n)], dtype=torch.long),
            "time_mask": torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool),
        }
    dataset = _make_partition_dataset(x, y, None, None, mt_aux)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, mt_aux


def test_evaluate_model_multi_task_dispatches_to_forward_multi_task() -> None:
    """When ``multi_task_loss_fn`` is set, eval reads ``forward_multi_task``.

    Sets a sentinel on the model's ``forward_multi_task`` so we can
    confirm the eval helper actually calls it (and does NOT fall back
    to the single-output ``forward`` path used on the single-task
    branch).
    """

    model = _make_toy_classifier()
    fmt_calls: list[int] = []
    original_fmt = model.forward_multi_task

    def _tracing_fmt(x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        fmt_calls.append(int(x.size(0)))
        return original_fmt(x, **kwargs)

    model.forward_multi_task = _tracing_fmt  # type: ignore[method-assign]

    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=True)
    loss_fn = nn.CrossEntropyLoss()
    mt_loss_fn = MultiTaskLoss(
        lambda_stance=1.0,
        lambda_time=0.3,
        lambda_certainty=0.3,
    )

    metrics = _evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        loss_fn,
        multi_task_loss_fn=mt_loss_fn,
    )

    # The tracing forward_multi_task must have fired on every batch.
    assert fmt_calls, "forward_multi_task was never called under multi_task_loss_fn"
    # The eval should have streamed all 12 rows through 3 batches of size 4.
    assert sum(fmt_calls) == 12
    # The classification surface must still populate the headline fields.
    assert metrics.regime_accuracy is not None
    assert metrics.regime_f1_macro is not None
    assert metrics.regime_loss is not None


def test_evaluate_model_multi_task_loss_differs_from_single_task_ce() -> None:
    """Multi-task val loss must NOT equal the single-task CE on the same model.

    Same model + same batch but two eval calls: one with
    ``multi_task_loss_fn=None`` (single-task CE), one with the
    MultiTaskLoss configured. The lambda-weighted multi-task total
    rolls in certainty + time terms with non-trivial lambdas,
    so the headline ``loss`` value must diverge from the single-task CE.
    """

    model = _make_toy_classifier()
    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=True)
    loss_fn = nn.CrossEntropyLoss()

    single_task_metrics = _evaluate_model(
        model, loader, torch.device("cpu"), loss_fn, multi_task_loss_fn=None
    )

    # Rebuild the loader so the same batch ordering is consumed twice.
    loader2, _ = _make_eval_loader(batch_size=4, with_mt_aux=True)
    mt_loss_fn = MultiTaskLoss(
        lambda_stance=1.0,
        lambda_time=0.5,
        lambda_certainty=0.5,
    )
    multi_task_metrics = _evaluate_model(
        model, loader2, torch.device("cpu"), loss_fn, multi_task_loss_fn=mt_loss_fn
    )

    # Single-task CE eval reports just the stance CE; multi-task eval
    # adds certainty + time terms, so the values must differ.
    assert single_task_metrics.loss != pytest.approx(multi_task_metrics.loss, abs=1e-6), (
        "multi-task eval emitted the same loss as single-task CE; the "
        "eval branch did not dispatch to MultiTaskLoss"
    )
    # The accuracy / F1 surface is computed off the stance logits in
    # both paths, so those headline numbers should be identical.
    assert single_task_metrics.regime_accuracy == pytest.approx(
        multi_task_metrics.regime_accuracy, abs=1e-9
    )
    assert single_task_metrics.regime_f1_macro == pytest.approx(
        multi_task_metrics.regime_f1_macro, abs=1e-9
    )


def test_evaluate_model_multi_task_attaches_per_axis_breakdown() -> None:
    """Per-axis losses ride on ``classification_breakdown[multi_task_axis_losses]``."""

    model = _make_toy_classifier()
    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=True)
    loss_fn = nn.CrossEntropyLoss()
    mt_loss_fn = MultiTaskLoss(
        lambda_stance=1.0,
        lambda_time=0.3,
        lambda_certainty=0.3,
    )

    metrics = _evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        loss_fn,
        multi_task_loss_fn=mt_loss_fn,
    )

    assert metrics.classification_breakdown is not None
    axis_losses = metrics.classification_breakdown.get("multi_task_axis_losses")
    assert isinstance(axis_losses, dict), (
        "expected classification_breakdown[multi_task_axis_losses] dict; "
        f"got {type(axis_losses).__name__}"
    )
    for axis in ("stance", "certainty", "time"):
        assert axis in axis_losses
        assert isinstance(axis_losses[axis], float)


def test_evaluate_model_single_task_path_omits_multi_task_breakdown() -> None:
    """Single-task eval leaves ``multi_task_axis_losses`` absent."""

    model = _make_toy_classifier()
    # The single-task path does not need the mt_aux block at all; build
    # a 4-arity dataset (no aux) so the legacy contract is exercised.
    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=False)
    loss_fn = nn.CrossEntropyLoss()

    metrics = _evaluate_model(
        model, loader, torch.device("cpu"), loss_fn, multi_task_loss_fn=None
    )

    assert metrics.classification_breakdown is not None
    assert "multi_task_axis_losses" not in metrics.classification_breakdown


def test_evaluate_model_raises_when_aux_missing_under_multi_task() -> None:
    """An mt-aux-less loader must error when ``multi_task_loss_fn`` is set."""

    model = _make_toy_classifier()
    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=False)
    loss_fn = nn.CrossEntropyLoss()
    mt_loss_fn = MultiTaskLoss()

    with pytest.raises(RuntimeError, match="aux tensors"):
        _evaluate_model(
            model,
            loader,
            torch.device("cpu"),
            loss_fn,
            multi_task_loss_fn=mt_loss_fn,
        )


def test_evaluate_model_multi_task_mask_collapse_zeroes_axis_contribution() -> None:
    """A batch where every axis-mask is False contributes only stance loss.

    Per-row masks must be honoured -- if all certainty / time
    masks are False, the multi-task total collapses to
    ``lambda_stance * stance_loss``. This pins the mask-honouring contract
    on the eval side (the train side already had this guarantee via
    :class:`MultiTaskLoss`).
    """

    model = _make_toy_classifier()
    # Custom loader: all axis masks False so only the stance branch
    # contributes. The stance loss term reduces to the unweighted CE
    # on the stance logits.
    torch.manual_seed(13)
    n = 8
    x = torch.randn(n, 5, 6)
    y = torch.tensor([i % 3 for i in range(n)], dtype=torch.long)
    mt_aux = {
        "certainty": torch.zeros(n, dtype=torch.long),
        "certainty_mask": torch.zeros(n, dtype=torch.bool),
        "time": torch.zeros(n, dtype=torch.long),
        "time_mask": torch.zeros(n, dtype=torch.bool),
    }
    dataset = _make_partition_dataset(x, y, None, None, mt_aux)
    loader = DataLoader(dataset, batch_size=n, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    lambda_stance = 0.7
    mt_loss_fn = MultiTaskLoss(
        lambda_stance=lambda_stance,
        lambda_time=5.0,
        lambda_certainty=5.0,
    )
    metrics = _evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        loss_fn,
        multi_task_loss_fn=mt_loss_fn,
    )

    # Reference: lambda_stance * CE(stance_logits, y) over the whole
    # partition. The expected value uses the same model + same x + same
    # y so the float comparison can pin to abs=1e-6.
    with torch.no_grad():
        stance_logits = model.forward_multi_task(x)["stance"]
        expected_stance_loss = nn.functional.cross_entropy(stance_logits, y).item()
    expected_total = lambda_stance * expected_stance_loss
    assert metrics.loss == pytest.approx(expected_total, abs=1e-5), (
        "multi-task eval did not collapse to lambda_stance * stance_loss "
        "on the all-False mask batch; per-row masks are not being honoured"
    )


def test_evaluate_model_default_none_keeps_legacy_call_shape() -> None:
    """A call without ``multi_task_loss_fn`` must keep the legacy shape.

    Mirrors the byte-identity guarantee: a regression-mode classifier
    invoked exactly the way the legacy regression contract calls
    ``_evaluate_model`` must return an EvaluationMetrics object with
    the same surface as before the patch.
    """

    model = _make_toy_classifier()
    loader, _ = _make_eval_loader(batch_size=4, with_mt_aux=False)
    loss_fn = nn.CrossEntropyLoss()

    # The legacy single-task call site does not pass ``multi_task_loss_fn``.
    metrics = _evaluate_model(model, loader, torch.device("cpu"), loss_fn)
    assert metrics.loss is not None
    assert metrics.regime_accuracy is not None


def test_train_model_multi_task_path_writes_per_axis_breakdown_into_val_metrics() -> None:
    """End-to-end: ``train_model`` with ``multi_task_loss=True`` surfaces the breakdown.

    The eval-side fix means the per-trial JSON now carries
    ``classification_breakdown[multi_task_axis_losses]`` on the val +
    test partitions; before the fix the val/test metrics derived from
    the single-task CE so this key was absent.
    """

    config = ModelConfig(
        output_mode="classification",
        multi_task_loss=True,
        n_classes=3,
    )
    groups = _build_classification_groups(n=40)
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    val_metrics = result.summary.val_metrics
    assert val_metrics is not None
    assert val_metrics.classification_breakdown is not None
    assert "multi_task_axis_losses" in val_metrics.classification_breakdown
    axis_losses = val_metrics.classification_breakdown["multi_task_axis_losses"]
    assert set(axis_losses.keys()) == {"stance", "certainty", "time"}
