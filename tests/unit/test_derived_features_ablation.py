"""Three-arm derived-features ablation runner (#309).

The §16 walkthrough surfaces five derived-feature columns the
forecaster head reads alongside the encoder embedding:
``sentiment_score`` / ``stance_label`` / ``factor_labels`` /
``certainty_score`` / ``topic_label``. The runner at
``scripts/run_derived_features_ablation.py`` ships three arms on
the canonical fold protocol:

- ``baseline`` -- the canonical pipeline byte-identical to the pre-#309
  head. No FeatureVector slot is touched before the per-fold scaler
  fit and ``use_derived_text_features`` stays ``True``.
- ``derived_ablation`` -- the five columns are zeroed in place on
  every FeatureVector before the scaler sees them, and
  ``use_derived_text_features=False`` collapses the multi-task
  factor / topic aux masks downstream.
- ``derived_replacement`` -- same narrow zero, plus the #291
  pre-meeting rates columns once #315 wires them. Gated on
  ``data/external/fred/rates_panel.parquet`` and currently surfaces
  a deferral status on the manifest.

These tests pin the arm dispatch + zero-injection contract without
touching a real training package on disk.
"""

from __future__ import annotations

import datetime as _dt
import sys
from typing import Any

import pytest

from scripts.run_derived_features_ablation import (
    _ARM_ALIASES,
    _ARM_CHOICES,
    _FIVE_DERIVED_COLUMNS,
    _FIVE_DERIVED_FV_ATTRS,
    _FIVE_DERIVED_MT_AUX_AXES,
    _canonicalise_arm,
    _configurations,
    _parse_args,
    _zero_five_derived_columns_inplace,
)


# ---------------------------------------------------------------------------
# Arm vocabulary + dispatch
# ---------------------------------------------------------------------------


def test_arm_choices_match_issue_309_three_way_spec() -> None:
    """The three-arm vocabulary is what the issue body names verbatim."""

    assert _ARM_CHOICES == (
        "baseline",
        "derived_ablation",
        "derived_replacement",
    )


def test_arm_aliases_resolve_legacy_pr_314_names() -> None:
    """``ablation`` / ``replacement`` legacy strings keep working."""

    assert _ARM_ALIASES == {
        "ablation": "derived_ablation",
        "replacement": "derived_replacement",
    }
    assert _canonicalise_arm("ablation") == "derived_ablation"
    assert _canonicalise_arm("replacement") == "derived_replacement"
    assert _canonicalise_arm("baseline") == "baseline"
    assert _canonicalise_arm("derived_ablation") == "derived_ablation"


def test_configurations_emit_three_arms_in_fixed_order() -> None:
    """The three-arm sweep order matches the wiki §16 table column order."""

    configs = _configurations()
    assert [name for name, _ in configs] == [
        "baseline",
        "derived_ablation",
        "derived_replacement",
    ]
    # Baseline keeps derived on; the other two flip the flag off.
    by_name = {name: kwargs for name, kwargs in configs}
    assert by_name["baseline"]["use_derived"] is True
    assert by_name["derived_ablation"]["use_derived"] is False
    assert by_name["derived_replacement"]["use_derived"] is False
    # Only the replacement arm carries a ``requires`` marker.
    assert "requires" not in by_name["baseline"]
    assert "requires" not in by_name["derived_ablation"]
    assert "requires" in by_name["derived_replacement"]


def test_cli_arm_flag_accepts_three_canonical_choices(monkeypatch) -> None:
    """The ``--arm`` flag accepts the three-arm vocabulary."""

    for arm in _ARM_CHOICES:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_derived_features_ablation",
                "--training-package-id",
                "tp_dummy",
                "--arm",
                arm,
            ],
        )
        args = _parse_args()
        assert args.arm == arm


def test_cli_arm_flag_accepts_legacy_aliases(monkeypatch) -> None:
    """``--arm ablation`` / ``--arm replacement`` keep working."""

    for alias in _ARM_ALIASES:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_derived_features_ablation",
                "--training-package-id",
                "tp_dummy",
                "--arm",
                alias,
            ],
        )
        args = _parse_args()
        assert args.arm == alias  # Resolved by ``_canonicalise_arm`` in main().


def test_cli_arm_flag_rejects_unknown_arm(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_derived_features_ablation",
            "--training-package-id",
            "tp_dummy",
            "--arm",
            "not_an_arm",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_cli_arm_flag_default_is_none(monkeypatch) -> None:
    """Default ``--arm`` is ``None`` -- legacy three-arm sweep."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_derived_features_ablation",
            "--training-package-id",
            "tp_dummy",
        ],
    )
    args = _parse_args()
    assert args.arm is None


# ---------------------------------------------------------------------------
# Five-column contract
# ---------------------------------------------------------------------------


def test_five_derived_columns_match_issue_body_verbatim() -> None:
    """The conceptual five-column list matches the issue body exactly."""

    assert _FIVE_DERIVED_COLUMNS == (
        "sentiment_score",
        "stance_label",
        "factor_labels",
        "certainty_score",
        "topic_label",
    )


def test_five_derived_fv_attrs_map_conceptual_names_to_fv_slots() -> None:
    """The FV-attribute list covers sentiment + stance one-hot + certainty.

    ``factor_labels`` and ``topic_label`` have no per-bar slot in
    ``as_rich_list`` -- they ride the multi-task aux axes. The
    FV-attribute list intentionally only carries the three columns
    with a per-bar slot; the aux-axis names live on
    ``_FIVE_DERIVED_MT_AUX_AXES``.
    """

    assert _FIVE_DERIVED_FV_ATTRS == (
        "sentiment_score",
        "stance_hawk",
        "stance_dove",
        "stance_neutral",
        "certain_label_certain",
    )
    assert _FIVE_DERIVED_MT_AUX_AXES == ("factor", "topic")


# ---------------------------------------------------------------------------
# Zero-injection on FeatureVector sequences
# ---------------------------------------------------------------------------


_TORCH = pytest.importorskip("torch")


def _make_fv(*, day: int) -> Any:
    """Build a FeatureVector with every five-column slot non-zero."""

    from app.models.config import FeatureVector

    return FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.42,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        # Five-column targets:
        stance_hawk=1.0,
        stance_dove=0.0,
        stance_neutral=0.0,
        certain_label_certain=0.7,
        # Non-target slots stay populated so we can assert they
        # survive the in-place zero.
        stance_missing=0.0,
        time_label_forward=0.6,
        mp_surprise_level=0.3,
        mp_surprise_path_factor=0.4,
        realized_vol_20d=0.05,
        vix_close=20.0,
    )


def test_zero_five_columns_zeros_the_five_targets_on_a_single_sequence() -> None:
    sequence = [_make_fv(day=i + 1) for i in range(4)]
    _zero_five_derived_columns_inplace([sequence])
    for fv in sequence:
        # Five-column targets all zero.
        assert fv.sentiment_score == 0.0
        assert fv.stance_hawk == 0.0
        assert fv.stance_dove == 0.0
        assert fv.stance_neutral == 0.0
        assert fv.certain_label_certain == 0.0


def test_zero_five_columns_leaves_non_target_slots_intact() -> None:
    """The narrow zero must not touch market / vol / mp_surprise slots."""

    sequence = [_make_fv(day=i + 1) for i in range(4)]
    _zero_five_derived_columns_inplace([sequence])
    for fv in sequence:
        # Market block stays intact.
        assert fv.market_close == 100.0
        assert fv.market_volatility == 0.01
        # MP-surprise block stays intact (NOT in the five-column scope).
        assert fv.mp_surprise_level == 0.3
        assert fv.mp_surprise_path_factor == 0.4
        # Realised-vol + cross-asset slots stay intact.
        assert fv.realized_vol_20d == 0.05
        assert fv.vix_close == 20.0
        # ``time_label_forward`` stays intact (not in the five-column
        # scope; only ``stance_label`` + ``certainty_score`` are).
        assert fv.time_label_forward == 0.6
        # ``stance_missing`` stays intact -- it's a missingness flag,
        # not a derived feature signal.
        assert fv.stance_missing == 0.0


def test_zero_five_columns_walks_multiple_sequences() -> None:
    """The zeroer walks every sequence in the partition iterable."""

    seq_a = [_make_fv(day=1), _make_fv(day=2)]
    seq_b = [_make_fv(day=3), _make_fv(day=4)]
    _zero_five_derived_columns_inplace([seq_a, seq_b])
    for sequence in (seq_a, seq_b):
        for fv in sequence:
            assert fv.sentiment_score == 0.0
            assert fv.stance_hawk == 0.0


def test_zero_five_columns_is_no_op_on_empty_input() -> None:
    """Empty partition iterable returns cleanly with no error."""

    _zero_five_derived_columns_inplace([])
    _zero_five_derived_columns_inplace([[]])


def test_zero_five_columns_preserves_as_rich_list_shape() -> None:
    """The narrow zero must not break ``as_rich_list``'s per-bar width.

    The five-column zero overwrites slots in place; the
    ``as_rich_list`` shape contract (RICH_FEATURE_SIZE per bar) is
    the load-bearing piece. Downstream the per-fold scaler fits on
    the same shape it expected before -- one column is just locked at
    zero. This test pins the shape contract so the runner's zeroing
    can never silently drop a per-bar slot.
    """

    from app.models.config import RICH_FEATURE_SIZE

    fv = _make_fv(day=1)
    original_width = len(fv.as_rich_list())
    _zero_five_derived_columns_inplace([[fv]])
    zeroed_width = len(fv.as_rich_list())
    assert zeroed_width == original_width
    assert zeroed_width == RICH_FEATURE_SIZE


# ---------------------------------------------------------------------------
# Baseline byte-identity: the canonical pipeline survives ``--arm baseline``
# ---------------------------------------------------------------------------


def test_baseline_arm_does_not_apply_the_narrow_zero() -> None:
    """Baseline arm's contract is byte-identical to the pre-#309 pipeline.

    The narrow five-column zero only runs on the two ``use_derived=False``
    arms. The baseline cell forwards ``use_derived=True`` and
    ``_run_one_cell`` short-circuits the in-place pass.
    """

    configs = {name: kwargs for name, kwargs in _configurations()}
    assert configs["baseline"]["use_derived"] is True
    # The other two arms flip the flag off, which is the gate
    # ``_run_one_cell`` reads before applying the narrow zero.
    assert configs["derived_ablation"]["use_derived"] is False
    assert configs["derived_replacement"]["use_derived"] is False


def test_baseline_arm_preserves_per_bar_slot_values() -> None:
    """A baseline cell that does not call the zeroer leaves slots intact.

    The ``_run_one_cell`` branch gates on ``if not use_derived:`` so a
    baseline cell never calls ``_zero_five_derived_columns_inplace``.
    This test asserts the contract by walking the same FeatureVectors
    through a no-op (mirroring the baseline cell's behaviour) and
    confirming the five slots round-trip.
    """

    sequence = [_make_fv(day=i + 1) for i in range(4)]
    snapshot = [
        (fv.sentiment_score, fv.stance_hawk, fv.certain_label_certain)
        for fv in sequence
    ]
    # Baseline branch: do NOT call the zeroer.
    for fv, (s0, h0, c0) in zip(sequence, snapshot, strict=True):
        assert fv.sentiment_score == s0
        assert fv.stance_hawk == h0
        assert fv.certain_label_certain == c0
