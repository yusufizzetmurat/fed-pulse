from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

# 2021 is intentionally outside the named windows. It is the post-COVID
# recovery year — neither the zero-rate emergency stance nor the hike cycle
# — and treating it as its own regime adds noise without thesis value.
# Override by passing a custom `regime_windows` tuple to `aggregate_by_regime`.
REGIME_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("pre_2020_calm", "2010-01-01", "2019-12-31"),
    ("covid_shock", "2020-01-01", "2020-12-31"),
    ("hike_cycle", "2022-01-01", "2023-12-31"),
)


@dataclass(frozen=True)
class RegimeRow:
    regime: str
    fold_id: str
    variant: str
    metric: str
    mean: float
    std: float | None
    n: int
    # Raw per-seed values backing this row. Carries forward into the
    # bootstrap CI helper without forcing the aggregator to recompute
    # from upstream artefacts. ``ci_lo`` / ``ci_hi`` are populated when
    # at least two samples are present; rows with a single sample
    # (smoke runs, single-seed cells) carry ``None`` so callers can
    # surface "n/a" in the table.
    samples: tuple[float, ...] = ()
    ci_lo: float | None = None
    ci_hi: float | None = None


def _to_date(value: str) -> date:
    return date.fromisoformat(value)


def aggregate_by_regime(
    holdouts: Iterable[Mapping[str, Any]],
    *,
    regime_windows: Iterable[tuple[str, str, str]] = REGIME_WINDOWS,
    metric_keys: Iterable[str] = ("combined_rmse", "close_rmse", "volatility_rmse", "directional_accuracy"),
    bootstrap_block_size: int = 1,
    bootstrap_resamples: int = 1000,
    bootstrap_coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> list[RegimeRow]:
    """Return one row per (regime, fold, variant, metric).

    Each row carries the per-seed samples it was built from plus a
    moving-block bootstrap CI on the row mean. Bootstrap defaults to
    block size 1 because the per-seed list is a small sample of
    independent runs, not a time series; pass a larger block when the
    samples are autocorrelated (e.g. a per-day series fed into this
    helper).
    """

    from app.evaluation.bootstrap import block_bootstrap_ci

    out: list[RegimeRow] = []
    holdout_list = list(holdouts)
    if not holdout_list:
        return out

    windows = [(name, _to_date(start), _to_date(end)) for name, start, end in regime_windows]

    for hold in holdout_list:
        fold_id = str(hold.get("fold_id", ""))
        test_start = hold.get("test_start") or hold.get("test_window", {}).get("start")
        test_end = hold.get("test_end") or hold.get("test_window", {}).get("end")
        if not test_start or not test_end:
            continue
        ts, te = _to_date(str(test_start)), _to_date(str(test_end))
        variants = hold.get("variants") or {}
        if not isinstance(variants, dict):
            continue
        for regime_name, w_start, w_end in windows:
            if w_end < ts or w_start > te:
                continue
            for variant_name, variant_payload in variants.items():
                if not isinstance(variant_payload, dict):
                    continue
                for metric_key in metric_keys:
                    metric = variant_payload.get(metric_key)
                    if not isinstance(metric, dict):
                        continue
                    samples = tuple(_iter_samples(metric.get("per_seed")))
                    ci_lo: float | None = None
                    ci_hi: float | None = None
                    if len(samples) > 1:
                        ci = block_bootstrap_ci(
                            list(samples),
                            block_size=int(bootstrap_block_size),
                            n_resamples=int(bootstrap_resamples),
                            coverage=float(bootstrap_coverage),
                            seed=int(bootstrap_seed),
                        )
                        ci_lo, ci_hi = float(ci.lo), float(ci.hi)
                    out.append(
                        RegimeRow(
                            regime=regime_name,
                            fold_id=fold_id,
                            variant=str(variant_name),
                            metric=metric_key,
                            mean=float(metric.get("mean", float("nan"))),
                            std=_coerce_optional_float(metric.get("std")),
                            n=int(metric.get("count", 0)),
                            samples=samples,
                            ci_lo=ci_lo,
                            ci_hi=ci_hi,
                        )
                    )
    return out


def _iter_samples(per_seed: Any) -> Iterable[float]:
    """Extract numeric per-seed values from the metric block.

    Tolerates both ``{seed: value}`` and ``{seed: {"value": x}}`` shapes
    so older aggregate.json layouts keep working. Non-finite or
    unparseable entries are dropped so the bootstrap input stays a
    clean numeric list.
    """

    if not isinstance(per_seed, Mapping):
        return
    for _seed, raw in per_seed.items():
        value: float | None
        if isinstance(raw, Mapping):
            value = _coerce_optional_float(raw.get("value"))
        else:
            value = _coerce_optional_float(raw)
        if value is None:
            continue
        yield value


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result
