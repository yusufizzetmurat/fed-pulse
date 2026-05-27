"""Unit + smoke tests for the B2 end-to-end fine-tune harness (#213).

Covers three contracts:

1. ``AutoModelForSequenceClassification`` builds with ``num_labels=3``
   when wired through the harness's default head configuration.
2. Per-fold tertile cutoffs are fitted on the train slice only -- a
   row in the test slice never influences the cutoffs the labels are
   assigned against.
3. A 1-epoch synthetic-fixture fine-tune runs to completion on CPU
   under 60 s and produces a 3-class softmax over the test slice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from app.data import finetune_pilot_b2
from app.data.finetune_pilot_b2 import (
    FomcRow,
    N_CLASSES,
    VOL_REGIME_LABELS,
    build_partition_classification_targets,
    load_fomc_rows,
)


# ---------------------------------------------------------------------------
# Pure-Python contract tests (no torch / transformers required).
# ---------------------------------------------------------------------------


def test_vol_regime_labels_cover_three_classes() -> None:
    assert VOL_REGIME_LABELS == ("calm", "normal", "high")
    assert N_CLASSES == 3


def test_build_partition_targets_fits_cutoffs_on_train_slice_only() -> None:
    """The test slice never influences the tertile cutoffs."""

    # Train slice spans a clean 0.0 -> 0.06 range; test slice carries
    # exclusively high-vol rows. If the cutoffs leaked the test slice
    # we'd see a different ``upper`` boundary.
    train = [
        FomcRow(record_id=f"t{i}", text="x", event_date=f"2020-01-{i:02d}", forward_vol=v)
        for i, v in enumerate([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], start=1)
    ]
    test = [
        FomcRow(record_id=f"e{i}", text="y", event_date=f"2021-01-{i:02d}", forward_vol=v)
        for i, v in enumerate([0.50, 0.55, 0.60], start=1)
    ]
    train_labels, train_cutoffs, _ = build_partition_classification_targets(
        train, train_rows=train
    )
    test_labels, test_cutoffs, _ = build_partition_classification_targets(
        test, train_rows=train
    )
    # Same cutoffs across both partitions; they are a function of the
    # train slice and the train slice only.
    assert train_cutoffs == test_cutoffs
    # Cutoffs are bounded by the train-slice range.
    lower, upper = train_cutoffs
    assert 0.0 <= lower <= upper <= 0.06
    # Every test row should land in class 2 (high) because every
    # ``forward_vol`` is above the upper tertile of the train slice.
    assert test_labels == [2, 2, 2]
    # Train labels cover the full 0 / 1 / 2 set.
    assert set(train_labels) == {0, 1, 2}


def test_build_partition_targets_drops_missing_vol() -> None:
    """Non-finite forward-vol rows fall out of the kept index list."""

    train = [
        FomcRow(record_id="t1", text="x", event_date="2020-01-01", forward_vol=0.01),
        FomcRow(record_id="t2", text="x", event_date="2020-01-02", forward_vol=0.02),
        FomcRow(record_id="t3", text="x", event_date="2020-01-03", forward_vol=0.03),
        FomcRow(record_id="t4", text="x", event_date="2020-01-04", forward_vol=float("nan")),
    ]
    labels, _, kept = build_partition_classification_targets(train, train_rows=train)
    # The NaN row is dropped; the three finite rows survive.
    assert len(labels) == 3
    assert kept == [0, 1, 2]


def test_load_fomc_rows_joins_registry_with_events(tmp_path: Path) -> None:
    """``load_fomc_rows`` joins registry rows against events.parquet."""

    pd = pytest.importorskip("pandas")
    package_dir = tmp_path / "tp_test"
    package_dir.mkdir()

    registry_rows = [
        {
            "record_id": "doc_2020",
            "text": "FOMC document text 2020.",
            "event_date": "2020-03-15",
            "source": "fomc_statement",
            "sample_weight": 1.0,
        },
        {
            "record_id": "doc_2021",
            "text": "FOMC document text 2021.",
            "event_date": "2021-06-10",
            "source": "fomc_statement",
            "sample_weight": 1.0,
        },
        {
            "record_id": "cross_bank_2020",
            "text": "ECB document — should be dropped at sample_weight=0.",
            "event_date": "2020-09-01",
            "source": "ecb_press",
            "sample_weight": 0.0,
        },
    ]
    (package_dir / "registry_normalized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in registry_rows), encoding="utf-8"
    )

    events_frame = pd.DataFrame(
        [
            {"event_date": "2020-03-15", "forward_realized_vol_10d": 0.015},
            {"event_date": "2021-06-10", "forward_realized_vol_10d": 0.022},
            # Event row with no matching registry text — joins to nothing.
            {"event_date": "2022-01-01", "forward_realized_vol_10d": 0.030},
        ]
    )
    events_frame.to_parquet(package_dir / "events.parquet")

    rows = load_fomc_rows(package_dir)
    assert {r.record_id for r in rows} == {"doc_2020", "doc_2021"}
    assert all(r.forward_vol > 0.0 for r in rows)


# ---------------------------------------------------------------------------
# Torch / transformers-dependent smoke tests. Skipped when the deps are
# unavailable on the runner; CI installs them via the backend
# requirements lock.
# ---------------------------------------------------------------------------


def _torch_or_skip() -> Any:
    return pytest.importorskip("torch")


def _transformers_or_skip() -> Any:
    return pytest.importorskip("transformers")


def test_model_head_has_three_output_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wiring ``AutoModelForSequenceClassification`` through the harness
    produces a head with three output classes."""

    _torch_or_skip()
    transformers = _transformers_or_skip()
    # Use the smallest BERT-family stub on HF that ships in the CI cache:
    # ``hf-internal-testing/tiny-random-bert``. The model is intentionally
    # tiny (~30 KB) so the test downloads + loads in seconds even on a
    # cold cache.
    stub_alias = "hf-internal-testing/tiny-random-bert"
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            stub_alias,
            num_labels=finetune_pilot_b2.N_CLASSES,
            id2label=finetune_pilot_b2.ID2LABEL,
            label2id=finetune_pilot_b2.LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # noqa: BLE001 -- network/cache flake on CI
        pytest.skip(f"tiny-random-bert unavailable in test env: {exc}")
    assert model.config.num_labels == 3
    # The id2label map carries calm / normal / high in canonical order.
    assert {int(k): v for k, v in model.config.id2label.items()} == finetune_pilot_b2.ID2LABEL


def test_smoke_one_epoch_completes_under_60s_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-epoch fine-tune on a tiny synthetic fixture runs in < 60 s on CPU
    and emits a 3-class softmax over the test slice."""

    torch = _torch_or_skip()
    transformers = _transformers_or_skip()
    stub_alias = "hf-internal-testing/tiny-random-bert"
    try:
        # Touch the stub once so a download failure short-circuits to
        # ``skip`` instead of failing the timing assertion below.
        transformers.AutoTokenizer.from_pretrained(stub_alias)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny-random-bert unavailable in test env: {exc}")

    # Force CPU even if a GPU is visible -- this is a CPU smoke test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_texts = [
        "Inflation remains elevated; further tightening is appropriate.",
        "The committee judges that the policy stance is appropriate.",
        "Conditions are softening; downside risks have increased.",
        "Labour market remains tight; price pressures persist.",
        "Financial conditions have eased meaningfully since the prior meeting.",
        "Growth is moderating; the committee will assess incoming data.",
    ]
    train_labels = [2, 1, 0, 2, 0, 1]
    test_texts = [
        "Risks to the outlook are roughly balanced.",
        "Inflation is well above the committee's longer-run goal.",
    ]
    test_labels = [1, 2]

    t0 = time.perf_counter()
    cell = finetune_pilot_b2._train_and_eval_one_cell(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        encoder_alias=stub_alias,
        seed=11,
        epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        max_length=32,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"smoke fine-tune took {elapsed:.1f}s, expected < 60s"

    # Three-class softmax over the test slice -- per_class block has
    # one row per class.
    breakdown = cell["classification_breakdown"]
    assert breakdown["n_classes"] == 3
    assert len(breakdown["per_class"]) == 3
    # Macro-F1 is a finite scalar between 0 and 1.
    assert 0.0 <= cell["macro_f1"] <= 1.0
