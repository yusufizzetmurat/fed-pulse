"""Tests for the coverage audit script.

The audit gates the sweep on a narrow REQUIRED set (the supervised
target only) and reports OPTIONAL families as a per-family roll-up plus
a trainer-flag impact list. Sidecar parquets are reported but do not
fail the audit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_training_package_coverage import (
    OPTIONAL_EVENT_FAMILIES,
    REQUIRED_EVENT_FAMILIES,
    TRAINER_FLAG_DEPENDENCIES,
    _check_column,
    audit,
)


def _make_minimal_event_dataframe(n: int = 10) -> pd.DataFrame:
    """The smallest events.parquet shape the audit treats as passing:
    the supervised target column populated, plus event_kind for the
    distribution report.
    """
    return pd.DataFrame(
        {
            "forward_realized_vol_10d": [0.05] * n,
            "event_kind": ["statement"] * n,
        }
    )


def test_required_set_is_exactly_the_supervised_target() -> None:
    """Contract guard: the REQUIRED set must NOT regress into requiring
    optional features. The trainer only refuses to start if the target
    is missing — everything else is a feature flag that degrades.
    """
    required_cols: list[str] = []
    for cols in REQUIRED_EVENT_FAMILIES.values():
        required_cols.extend(cols)
    assert required_cols == ["forward_realized_vol_10d"]


def test_every_optional_family_has_a_trainer_flag_mapping() -> None:
    """Every optional family must declare which trainer flag it gates,
    so the audit report can name the flag rather than just the column.
    A new family without a mapping should fail this test loudly.
    """
    for family in OPTIONAL_EVENT_FAMILIES:
        assert family in TRAINER_FLAG_DEPENDENCIES, (
            f"Optional family {family!r} has no entry in "
            "TRAINER_FLAG_DEPENDENCIES — add the trainer flag(s) it "
            "gates so the audit report can surface the impact."
        )


def test_check_column_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    status, ok, populated, total = _check_column(df, "y", 50.0)
    assert status == "missing"
    assert ok is False
    assert populated == 0
    assert total == 3


def test_check_column_empty() -> None:
    df = pd.DataFrame({"x": [None, None, None]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "empty"
    assert ok is False
    assert populated == 0


def test_check_column_sparse() -> None:
    df = pd.DataFrame({"x": [1.0, None, None, None]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "sparse"
    assert ok is False
    assert populated == 1
    assert total == 4


def test_check_column_ok() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "ok"
    assert ok is True
    assert populated == 4


def test_audit_passes_on_minimal_dataframe(tmp_path: Path) -> None:
    """Bare-minimum events.parquet — only the supervised target — passes.

    Also confirms the audit does not gate on any other column even when
    every optional family is absent.
    """
    df = _make_minimal_event_dataframe(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_audit_fails_when_canonical_target_missing(tmp_path: Path) -> None:
    df = _make_minimal_event_dataframe(n=10).drop(
        columns=["forward_realized_vol_10d"]
    )
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_fails_when_canonical_target_sparse(tmp_path: Path) -> None:
    df = _make_minimal_event_dataframe(n=10)
    df["forward_realized_vol_10d"] = [0.05] + [None] * 9  # 10% populated
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_garch_baseline_and_residual_are_separate_families() -> None:
    """The trainer uses ``forward_realized_vol_10d_garch_residual`` as
    the supervised target under ``--vol-target-mode garch_residual``;
    ``forward_realized_vol_10d_garch_baseline`` is only the GARCH model
    fit. Bundling both into one family would let a TP with only the
    baseline populated falsely report the residual-gated sweep arm as
    healthy. Lock the split here.
    """
    assert "garch_baseline" in OPTIONAL_EVENT_FAMILIES
    assert "garch_residual" in OPTIONAL_EVENT_FAMILIES
    assert OPTIONAL_EVENT_FAMILIES["garch_baseline"] == [
        "forward_realized_vol_10d_garch_baseline"
    ]
    assert OPTIONAL_EVENT_FAMILIES["garch_residual"] == [
        "forward_realized_vol_10d_garch_residual"
    ]


def test_vote_tally_family_includes_is_unanimous() -> None:
    """The loader (``loaders.py`` ~line 1524) derives one of four vote
    scalars from ``is_unanimous``; the column is required=False in the
    schema, so a TP can have ``votes_for`` populated while ``is_unanimous``
    is absent. The audit should surface that gap rather than declare
    the family healthy.
    """
    assert "is_unanimous" in OPTIONAL_EVENT_FAMILIES["vote_tally"]


def test_audit_returns_two_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist.parquet"
    result = audit(target)
    assert result == 2


def test_audit_passes_when_event_kind_includes_unknown(tmp_path: Path) -> None:
    df = _make_minimal_event_dataframe(n=10)
    df.loc[0, "event_kind"] = "unrecognised_kind"
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_audit_passes_when_event_kind_column_missing(tmp_path: Path) -> None:
    df = _make_minimal_event_dataframe(n=10).drop(columns=["event_kind"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_sparse_threshold_override_relaxes_required_check(
    tmp_path: Path,
) -> None:
    """A target column populated below the default 50% threshold can be
    waved through by lowering --sparse-threshold-pct.
    """
    df = _make_minimal_event_dataframe(n=10)
    df["forward_realized_vol_10d"] = [0.05] + [None] * 9  # 10% populated
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    assert audit(parquet) == 1  # fails at default 50%
    assert audit(parquet, sparse_threshold=5.0) == 0  # passes at 5%


def test_audit_reports_sidecar_absence_without_failing(tmp_path: Path) -> None:
    df = _make_minimal_event_dataframe(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0
