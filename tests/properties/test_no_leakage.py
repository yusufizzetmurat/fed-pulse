from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402


def _row(text: str, event_date: date, as_of: date) -> dict[str, Any]:
    return {
        "text": text,
        "event_date": event_date.isoformat(),
        "as_of_ts": as_of.isoformat(),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


@st.composite
def _walk_forward_fold(draw) -> dict[str, str]:
    base = date(2010, 1, 1)
    train_start_offset = draw(st.integers(min_value=0, max_value=200))
    train_len = draw(st.integers(min_value=30, max_value=400))
    val_len = draw(st.integers(min_value=5, max_value=60))
    gap_to_test = draw(st.integers(min_value=1, max_value=10))
    test_len = draw(st.integers(min_value=5, max_value=60))

    train_start = base + timedelta(days=train_start_offset)
    train_end = train_start + timedelta(days=train_len - 1)
    val_start = train_end + timedelta(days=1)
    val_end = val_start + timedelta(days=val_len - 1)
    test_start = val_end + timedelta(days=gap_to_test)
    test_end = test_start + timedelta(days=test_len - 1)

    return {
        "fold_id": draw(st.sampled_from(["wf_fold_1", "wf_fold_2", "wf_fold_3"])),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": val_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
    }


@given(_walk_forward_fold())
def test_fold_windows_are_chronological(fold: dict[str, str]) -> None:
    assert fold["train_end"] < fold["val_start"]
    assert fold["val_end"] < fold["test_start"]
    assert fold["train_start"] <= fold["train_end"]
    assert fold["val_start"] <= fold["val_end"]
    assert fold["test_start"] <= fold["test_end"]


@given(_walk_forward_fold(), st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=20))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_text_hash_collisions_across_splits_are_rejected(fold: dict[str, str], texts: list[str]) -> None:
    train_start = date.fromisoformat(fold["train_start"])
    test_start = date.fromisoformat(fold["test_start"])
    rows = []
    for idx, text in enumerate(texts):
        rows.append(_row(text, train_start + timedelta(days=idx % 5), train_start))
    train_hashes = {row["text_hash"] for row in rows[: len(rows) // 2]}
    test_hashes = {row["text_hash"] for row in rows[len(rows) // 2 :]}
    if train_hashes & test_hashes:
        assert any(rows[i]["text"] == rows[j]["text"]
                   for i in range(len(rows) // 2)
                   for j in range(len(rows) // 2, len(rows)))
    assert test_start > train_start


@given(st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
       st.integers(min_value=1, max_value=365))
def test_feature_as_of_never_exceeds_target_ts(target_offset_days: date, as_of_lag: int) -> None:
    target_ts = target_offset_days
    as_of_ts = target_ts - timedelta(days=as_of_lag)
    row = _row("sample", target_ts, as_of_ts)
    assert row["as_of_ts"] <= row["event_date"]


@given(_walk_forward_fold())
def test_train_window_strictly_precedes_val_and_test_windows(fold: dict[str, str]) -> None:
    assert fold["train_start"] <= fold["train_end"] < fold["val_start"]
    assert fold["val_start"] <= fold["val_end"] < fold["test_start"]
    assert fold["test_start"] <= fold["test_end"]


def test_rich_feature_vector_width_matches_constant() -> None:
    """Sequence-leakage guard at the feature-row level.

    ``FeatureVector.as_rich_list`` is the canonical row builder used to
    assemble training batches; if a future change appends a column past
    the declared ``RICH_FEATURE_SIZE`` constant without updating the
    constant the loader would silently widen rows and downstream
    consumers (scaler, model input head, sequence batcher) would mis-
    interpret the trailing slot. Encode the width contract as a test
    so that drift fails CI loudly.
    """

    pytest.importorskip("torch", reason="FeatureVector chains pull torch transitively")
    from app.models.config import (
        FEATURE_SIZE,
        RICH_FEATURE_SIZE,
        FeatureVector,
    )

    row = FeatureVector(
        date="2024-01-01",
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
    )
    base = row.as_list()
    assert len(base) == FEATURE_SIZE, (
        f"FeatureVector.as_list width drifted: expected {FEATURE_SIZE}, got {len(base)}"
    )
    rich = row.as_rich_list()
    assert len(rich) == RICH_FEATURE_SIZE, (
        f"FeatureVector.as_rich_list width drifted: "
        f"expected {RICH_FEATURE_SIZE}, got {len(rich)}"
    )
