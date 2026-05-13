from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping

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


def _to_date(value: str) -> date:
    return date.fromisoformat(value)


def aggregate_by_regime(
    holdouts: Iterable[Mapping],
    *,
    regime_windows: Iterable[tuple[str, str, str]] = REGIME_WINDOWS,
    metric_keys: Iterable[str] = ("combined_rmse", "close_rmse", "volatility_rmse", "directional_accuracy"),
) -> list[RegimeRow]:
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
                    out.append(
                        RegimeRow(
                            regime=regime_name,
                            fold_id=fold_id,
                            variant=str(variant_name),
                            metric=metric_key,
                            mean=float(metric.get("mean", float("nan"))),
                            std=_coerce_optional_float(metric.get("std")),
                            n=int(metric.get("count", 0)),
                        )
                    )
    return out


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result
