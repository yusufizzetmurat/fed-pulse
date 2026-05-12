from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["enable_deterministic_mode", "make_generator", "seed_worker"]


def enable_deterministic_mode(seed: int) -> None:
    # CUBLAS_WORKSPACE_CONFIG is required before use_deterministic_algorithms
    # under recent CUDA; setdefault avoids overriding a caller's choice.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    # Torch reseeds its own worker RNG, but numpy and random stay at their
    # parent values otherwise.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
