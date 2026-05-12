from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.determinism import enable_deterministic_mode, make_generator, seed_worker


def _shuffle_batches(seed: int) -> list[torch.Tensor]:
    data = torch.arange(64, dtype=torch.float32).unsqueeze(1)
    targets = torch.arange(64, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data, targets)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )
    return [batch[0].clone() for batch in loader]


def test_dataloader_with_seeded_generator_is_reproducible() -> None:
    first = _shuffle_batches(seed=11)
    second = _shuffle_batches(seed=11)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_dataloader_with_different_seeds_diverges() -> None:
    seed_a = _shuffle_batches(seed=11)
    seed_b = _shuffle_batches(seed=29)
    diverged = any(not torch.equal(a, b) for a, b in zip(seed_a, seed_b))
    assert diverged, "different seeds must produce different shuffles"


def test_enable_deterministic_mode_flips_cudnn_flags() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    enable_deterministic_mode(seed=11)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_enable_deterministic_mode_seeds_python_and_numpy() -> None:
    enable_deterministic_mode(seed=11)
    py_first = random.random()
    np_first = np.random.rand()

    enable_deterministic_mode(seed=11)
    py_second = random.random()
    np_second = np.random.rand()

    assert py_first == py_second
    assert np_first == np_second
