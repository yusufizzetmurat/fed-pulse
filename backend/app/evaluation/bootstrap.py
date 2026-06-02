from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float
    coverage: float
    n_resamples: int
    block_size: int


def _moving_blocks(n: int, block_size: int) -> int:
    return max(1, math.ceil(n / block_size))


def _resample_indices(n: int, block_size: int, rng: random.Random) -> list[int]:
    if n <= 0:
        return []
    blocks = _moving_blocks(n, block_size)
    indices: list[int] = []
    for _ in range(blocks):
        start = rng.randint(0, max(0, n - block_size))
        indices.extend(range(start, min(n, start + block_size)))
    return indices[:n]


def block_bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str = "mean",
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> BootstrapCI:
    if not values:
        return BootstrapCI(
            point=float("nan"),
            lo=float("nan"),
            hi=float("nan"),
            coverage=coverage,
            n_resamples=n_resamples,
            block_size=block_size,
        )
    if statistic not in {"mean", "median"}:
        raise ValueError(f"unsupported statistic={statistic!r}")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")

    rng = random.Random(seed)
    arr = list(values)
    n = len(arr)
    point = _summary(arr, statistic)

    samples: list[float] = []
    for _ in range(n_resamples):
        idx = _resample_indices(n, block_size, rng)
        resample = [arr[i] for i in idx]
        samples.append(_summary(resample, statistic))
    samples.sort()
    alpha = (1.0 - coverage) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(0, min(n_resamples - 1, hi_idx))
    return BootstrapCI(
        point=point,
        lo=samples[lo_idx],
        hi=samples[hi_idx],
        coverage=coverage,
        n_resamples=n_resamples,
        block_size=block_size,
    )


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> BootstrapCI:
    if len(a) != len(b):
        raise ValueError(f"paired series must have equal length; got {len(a)} vs {len(b)}")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    return block_bootstrap_ci(
        diffs,
        statistic="mean",
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )


def _summary(values: Sequence[float], statistic: str) -> float:
    if not values:
        return float("nan")
    if statistic == "mean":
        return sum(values) / len(values)
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 1:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
