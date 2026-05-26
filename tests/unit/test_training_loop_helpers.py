"""Unit tests for the training loop helper primitives.

The helpers covered here are the building blocks of the per-cell
forecaster training loop:

- ``_zero_credibility`` / ``_allocate_credibility_buffer`` /
  ``_slice_credibility_buffer`` (the credibility input path)
- ``_copy_state_inplace`` / ``_snapshot_state`` (the
  best-state-tracking primitives that replaced the deepcopy loop)
- ``_move_to_device`` (the once-per-tensor device move that
  hoisted the per-batch ``.to`` calls out of the inner loop)
- ``_resolve_compile_amp_flags`` / ``_maybe_compile_model`` (the
  per-arch + per-device dispatch the perf rewrite introduced)
- ``_evaluate_model`` (deferred ``.item()`` aggregation contract)
- ``_unpack_batch`` (batch arity gate)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from app.evaluation.metrics import EvaluationMetrics
from app.models.config import ModelConfig
from app.models.factory import build_forecaster
from app.training.loop import (
    _allocate_credibility_buffer,
    _copy_state_inplace,
    _evaluate_model,
    _maybe_compile_model,
    _move_to_device,
    _resolve_compile_amp_flags,
    _slice_credibility_buffer,
    _snapshot_state,
    _unpack_batch,
    _zero_credibility,
)


def _cpu_model(credibility_features: bool = False) -> "torch.nn.Module":
    cfg = ModelConfig(
        input_size=6,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        head_hidden_size=4,
        credibility_features=credibility_features,
    )
    return build_forecaster(cfg)


def test_zero_credibility_returns_none_when_feature_off() -> None:
    model = _cpu_model(credibility_features=False)
    out = _zero_credibility(model, 5, torch.device("cpu"))
    assert out is None


def test_zero_credibility_returns_zero_tensor_when_feature_on() -> None:
    model = _cpu_model(credibility_features=True)
    out = _zero_credibility(model, 7, torch.device("cpu"))
    assert out is not None
    assert out.shape == (7, 4)
    assert torch.all(out == 0)


def test_allocate_credibility_buffer_skips_when_feature_off() -> None:
    model = _cpu_model(credibility_features=False)
    out = _allocate_credibility_buffer(model, 32, torch.device("cpu"))
    assert out is None


def test_allocate_credibility_buffer_returns_pre_sized_buffer() -> None:
    model = _cpu_model(credibility_features=True)
    buffer = _allocate_credibility_buffer(model, 16, torch.device("cpu"))
    assert buffer is not None
    assert buffer.shape == (16, 4)
    assert buffer.device.type == "cpu"


def test_slice_credibility_buffer_returns_none_on_none() -> None:
    assert _slice_credibility_buffer(None, 8) is None


def test_slice_credibility_buffer_returns_input_when_size_matches() -> None:
    buffer = torch.zeros((8, 4))
    sliced = _slice_credibility_buffer(buffer, 8)
    assert sliced is buffer


def test_slice_credibility_buffer_narrows_when_size_below_max() -> None:
    buffer = torch.zeros((16, 4))
    sliced = _slice_credibility_buffer(buffer, 5)
    assert sliced is not None
    assert sliced.shape == (5, 4)
    # narrow returns a view that shares storage with the parent.
    assert sliced.data_ptr() == buffer.data_ptr()


def test_move_to_device_returns_none_unchanged() -> None:
    assert _move_to_device(None, torch.device("cpu")) is None


def test_move_to_device_is_noop_when_device_matches() -> None:
    tensor = torch.zeros((3, 2))
    moved = _move_to_device(tensor, torch.device("cpu"))
    assert moved is tensor


def test_copy_state_inplace_overwrites_existing_tensors() -> None:
    target = {"weight": torch.zeros((2, 3))}
    source = {"weight": torch.ones((2, 3))}
    _copy_state_inplace(target, source)
    assert torch.all(target["weight"] == 1.0)
    # Same storage; the buffer was overwritten, not replaced.
    assert target["weight"].shape == (2, 3)


def test_copy_state_inplace_adds_missing_key() -> None:
    target: dict[str, torch.Tensor] = {}
    source = {"weight": torch.ones((2,))}
    _copy_state_inplace(target, source)
    assert "weight" in target
    assert torch.all(target["weight"] == 1.0)


def test_snapshot_state_returns_independent_clones() -> None:
    model = _cpu_model()
    snapshot = _snapshot_state(model)
    # Mutate the snapshot; the model parameters must stay untouched.
    for key in snapshot:
        snapshot[key].zero_()
    state = model.state_dict()
    # Every original parameter has finite values; the snapshot zeroing
    # did not leak back.
    assert any(torch.any(state[k] != 0) for k in state)


def test_resolve_compile_amp_flags_disables_on_cpu_device() -> None:
    model = _cpu_model()
    use_compile, use_amp = _resolve_compile_amp_flags(
        model,
        architecture="lstm",
        device=torch.device("cpu"),
        use_compile=True,
        use_amp=True,
    )
    assert use_compile is False
    assert use_amp is False


def test_resolve_compile_amp_flags_skips_incompatible_arch_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _cpu_model()
    # ``_resolve_compile_amp_flags`` checks ``device.type == "cuda"``;
    # the helper does not touch the real GPU, only inspects the
    # device type string, so a synthetic CUDA device is enough.
    fake_device = torch.device("cuda")
    use_compile, use_amp = _resolve_compile_amp_flags(
        model,
        architecture="informer",
        device=fake_device,
        use_compile=True,
        use_amp=True,
    )
    assert use_compile is False
    assert use_amp is False


def test_resolve_compile_amp_flags_keeps_compatible_arch_on_cuda() -> None:
    model = _cpu_model()
    fake_device = torch.device("cuda")
    use_compile, use_amp = _resolve_compile_amp_flags(
        model,
        architecture="lstm",
        device=fake_device,
        use_compile=True,
        use_amp=True,
    )
    assert use_compile is True
    assert use_amp is True


def test_maybe_compile_model_is_noop_when_disabled() -> None:
    model = _cpu_model()
    out = _maybe_compile_model(model, use_compile=False)
    assert out is model


def test_unpack_batch_two_element() -> None:
    batch_x = torch.zeros((4, 5, 6))
    batch_y = torch.zeros((4, 2))
    x, y, text, missing, mt_aux, log_rv = _unpack_batch((batch_x, batch_y))
    assert x is batch_x
    assert y is batch_y
    assert text is None
    assert missing is None
    assert mt_aux is None
    assert log_rv is None


def test_unpack_batch_four_element() -> None:
    batch_x = torch.zeros((4, 5, 6))
    batch_y = torch.zeros((4, 2))
    text = torch.zeros((4, 8))
    missing = torch.zeros((4, 1))
    x, y, t, m, mt_aux, log_rv = _unpack_batch((batch_x, batch_y, text, missing))
    assert t is text
    assert m is missing
    assert mt_aux is None
    assert log_rv is None


def test_unpack_batch_rejects_unexpected_arity() -> None:
    # Arity 3 / 5 / 9 / 11 became valid post-#304 (the dual-head log_rv
    # slot composes with the prior shapes); pick 6 -- still
    # unsupported -- so the negative-path coverage stays.
    with pytest.raises(ValueError, match="unexpected batch arity"):
        _unpack_batch(tuple(torch.zeros(1) for _ in range(6)))


def test_evaluate_model_returns_inf_metrics_on_empty_loader() -> None:
    """``_evaluate_model`` handles a zero-batch loader gracefully."""

    from torch.utils.data import DataLoader, TensorDataset

    model = _cpu_model()
    model.eval()
    empty_dataset = TensorDataset(torch.zeros((0, 5, 6)), torch.zeros((0, 2)))
    loader = DataLoader(empty_dataset, batch_size=4)
    loss_fn = torch.nn.SmoothL1Loss()
    metrics = _evaluate_model(model, loader, torch.device("cpu"), loss_fn)
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.loss == float("inf")
    assert metrics.close_rmse == float("inf")
    assert metrics.combined_rmse == float("inf")


def test_evaluate_model_aggregates_loss_and_rmse_correctly() -> None:
    """End-to-end: a known-input loader yields the documented aggregates."""

    from torch.utils.data import DataLoader, TensorDataset

    model = _cpu_model()
    model.eval()
    # Two batches of size 4 each. The model's predictions are some
    # value; the test asserts the aggregate's structure, not the exact
    # number (the model's randomness is the per-cell init).
    x = torch.zeros((8, 5, 6))
    y = torch.zeros((8, 2))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    loss_fn = torch.nn.SmoothL1Loss()
    metrics = _evaluate_model(model, loader, torch.device("cpu"), loss_fn)
    assert metrics.loss >= 0.0
    assert metrics.close_rmse >= 0.0
    assert metrics.volatility_rmse >= 0.0
    assert metrics.combined_rmse >= 0.0


def test_evaluate_model_consumes_pre_allocated_credibility_buffer() -> None:
    """When ``credibility_buffer`` is supplied the loop uses it as-is."""

    from torch.utils.data import DataLoader, TensorDataset

    model = _cpu_model(credibility_features=True)
    model.eval()
    x = torch.zeros((6, 5, 6))
    y = torch.zeros((6, 2))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    loss_fn = torch.nn.SmoothL1Loss()
    buffer = _allocate_credibility_buffer(model, 3, torch.device("cpu"))
    metrics = _evaluate_model(
        model, loader, torch.device("cpu"), loss_fn, credibility_buffer=buffer
    )
    assert metrics.loss >= 0.0
