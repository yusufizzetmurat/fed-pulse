from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.services.forecaster import _capture_rng_state, _restore_rng_state


def test_capture_and_restore_round_trip_for_torch():
    torch.manual_seed(11)
    captured = _capture_rng_state()
    # Mutate the RNG.
    torch.manual_seed(99)
    after_mutation = torch.randn(4).tolist()
    _restore_rng_state(captured)
    after_restore = torch.randn(4).tolist()
    # The post-restore draw must match what we'd have got immediately after capture.
    torch.manual_seed(11)
    expected = torch.randn(4).tolist()
    assert after_restore == expected
    assert after_restore != after_mutation


def test_capture_and_restore_round_trip_for_python_random():
    random.seed(11)
    captured = _capture_rng_state()
    random.seed(99)
    after_mutation = [random.random() for _ in range(3)]
    _restore_rng_state(captured)
    after_restore = [random.random() for _ in range(3)]
    random.seed(11)
    expected = [random.random() for _ in range(3)]
    assert after_restore == expected
    assert after_restore != after_mutation


def test_restore_on_empty_state_is_a_no_op():
    torch.manual_seed(11)
    before = torch.randn(2).tolist()
    _restore_rng_state(None)
    torch.manual_seed(11)
    after = torch.randn(2).tolist()
    assert before == after
