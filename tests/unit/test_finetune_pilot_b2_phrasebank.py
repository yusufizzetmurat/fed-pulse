"""Tests for the PhraseBank auxiliary-task path through B2 (#33).

Covers four contracts:

1. Default-off path through ``_train_and_eval_one_cell`` produces a
   metrics dict whose phrasebank-aux fields are zero / None, so an
   operator who never flips ``--enable-phrasebank-aux`` sees the
   pre-#33 B2 behaviour byte-identically.
2. PhraseBank-on path runs to completion, returns a finite aux-loss
   trace, and reports the operator-supplied lambda.
3. The aux head's parameters receive non-zero gradients during the
   training step (proves the aux gradient actually flows into the
   optimiser).
4. ``run_sweep`` honours the ``--phrasebank-jsonl`` fixture without
   reaching the HF Hub — sweep payloads carry a ``phrasebank_aux``
   meta block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.data import finetune_pilot_b2
from app.data.phrasebank import PhraseBankRow


def _build_synthetic_package(tmp_path: Path) -> Path:
    """Build a minimal training-package fixture under ``tmp_path``.

    Five train rows + one test row spanning a clean walk-forward fold,
    plus a fold manifest. Shared between the JSONL-fixture happy-path
    test and the lambda<=0 disable-the-aux footgun test.
    """

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    package_dir = tmp_path / "tp_phrasebank_smoke"
    package_dir.mkdir()

    registry_rows: list[dict[str, Any]] = []
    events_rows: list[dict[str, Any]] = []
    for i in range(5):
        ed = f"2020-0{i + 1}-15"
        registry_rows.append(
            {
                "record_id": f"doc_{i}",
                "text": f"FOMC statement {i}. Inflation remains a concern.",
                "event_date": ed,
                "source": "fomc_statement",
                "sample_weight": 1.0,
            }
        )
        events_rows.append(
            {"event_date": ed, "forward_realized_vol_10d": 0.010 + 0.005 * i}
        )
    test_ed = "2021-01-15"
    registry_rows.append(
        {
            "record_id": "doc_test",
            "text": "FOMC test statement. Conditions softening.",
            "event_date": test_ed,
            "source": "fomc_statement",
            "sample_weight": 1.0,
        }
    )
    events_rows.append(
        {"event_date": test_ed, "forward_realized_vol_10d": 0.035}
    )

    (package_dir / "registry_normalized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in registry_rows), encoding="utf-8"
    )
    pd.DataFrame(events_rows).to_parquet(package_dir / "events.parquet")

    fold_manifest = {
        "folds": [
            {
                "fold_id": "fold_smoke",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-12-31",
            }
        ]
    }
    (package_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(fold_manifest), encoding="utf-8"
    )
    return package_dir


def _torch_or_skip() -> Any:
    return pytest.importorskip("torch")


def _transformers_or_skip() -> Any:
    return pytest.importorskip("transformers")


def _stub_alias() -> str:
    return "hf-internal-testing/tiny-random-bert"


def _ensure_stub_loadable() -> None:
    transformers = _transformers_or_skip()
    try:
        transformers.AutoTokenizer.from_pretrained(_stub_alias())
    except Exception as exc:  # noqa: BLE001 -- cache flake on CI
        pytest.skip(f"tiny-random-bert unavailable in test env: {exc}")


def _make_phrasebank_rows() -> list[PhraseBankRow]:
    return [
        PhraseBankRow(row_id="pb_0", sentence="Revenue grew strongly.", label_idx=2),
        PhraseBankRow(row_id="pb_1", sentence="Operating costs were flat.", label_idx=1),
        PhraseBankRow(row_id="pb_2", sentence="Profit fell sharply.", label_idx=0),
        PhraseBankRow(row_id="pb_3", sentence="Margins widened.", label_idx=2),
    ]


def _make_fomc_corpus() -> tuple[list[str], list[int], list[str], list[int]]:
    train_texts = [
        "Inflation remains elevated; further tightening is appropriate.",
        "The committee judges that the policy stance is appropriate.",
        "Conditions are softening; downside risks have increased.",
        "Labour market remains tight; price pressures persist.",
    ]
    train_labels = [2, 1, 0, 2]
    test_texts = [
        "Risks to the outlook are roughly balanced.",
        "Inflation is well above the committee's longer-run goal.",
    ]
    test_labels = [1, 2]
    return train_texts, train_labels, test_texts, test_labels


def test_default_off_metrics_have_zero_aux_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With aux off, the metrics dict reports zero rows + zero lambda
    AND two runs with the same seed are byte-identical.

    The default-off path is the byte-identity contract: an operator
    who never flips ``--enable-phrasebank-aux`` sees a B2 cell that
    is equivalent to the pre-#33 harness. The two-run determinism
    check (#425) catches device-placement drift and inadvertent
    optimizer / loader changes that the original 'is the value between
    0 and 1' assertion would miss.
    """

    torch = _torch_or_skip()
    _ensure_stub_loadable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_texts, train_labels, test_texts, test_labels = _make_fomc_corpus()
    cell_kwargs = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "encoder_alias": _stub_alias(),
        "seed": 11,
        "epochs": 1,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "max_length": 32,
        "phrasebank_rows": None,
        "phrasebank_aux_lambda": 0.0,
    }
    cell_a = finetune_pilot_b2._train_and_eval_one_cell(**cell_kwargs)
    cell_b = finetune_pilot_b2._train_and_eval_one_cell(**cell_kwargs)

    assert cell_a["phrasebank_aux_lambda"] == 0.0
    assert cell_a["phrasebank_aux_rows"] == 0
    assert cell_a["phrasebank_aux_train_loss"] is None
    assert 0.0 <= cell_a["macro_f1"] <= 1.0

    # Determinism check: same-seed runs on the same corpus must
    # produce bit-identical metrics. CPU PyTorch with a properly
    # seeded RNG is deterministic; a regression that introduced a
    # device move, an optimizer change, or a loader-order shift would
    # break this exact equality. No tolerance -- if a future change
    # introduces float drift we want the test to flag it loudly.
    assert cell_a["macro_f1"] == cell_b["macro_f1"], (
        f"aux-off determinism broken: macro_f1 {cell_a['macro_f1']} != {cell_b['macro_f1']}"
    )
    assert cell_a["accuracy"] == cell_b["accuracy"]
    assert cell_a["weighted_f1"] == cell_b["weighted_f1"]
    assert cell_a["train_loss"] == cell_b["train_loss"]


def test_aux_on_metrics_report_finite_aux_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With aux on, the cell returns a finite aux-loss trace + lambda."""

    torch = _torch_or_skip()
    _ensure_stub_loadable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_texts, train_labels, test_texts, test_labels = _make_fomc_corpus()
    cell = finetune_pilot_b2._train_and_eval_one_cell(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        encoder_alias=_stub_alias(),
        seed=11,
        epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.0,
        max_length=32,
        phrasebank_rows=_make_phrasebank_rows(),
        phrasebank_aux_lambda=0.3,
    )
    assert cell["phrasebank_aux_lambda"] == pytest.approx(0.3)
    assert cell["phrasebank_aux_rows"] == 4
    aux_loss = cell["phrasebank_aux_train_loss"]
    assert aux_loss is not None
    assert aux_loss == pytest.approx(aux_loss)  # finite (no NaN)
    assert 0.0 <= cell["macro_f1"] <= 1.0


def test_aux_gradient_flows_into_aux_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aux-head parameters receive non-zero gradients.

    Direct check: build the encoder + aux head, run one forward +
    backward through the same code path as the harness, assert the
    aux-head linear layer's weight grad is non-zero.
    """

    torch = _torch_or_skip()
    transformers = _transformers_or_skip()
    _ensure_stub_loadable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    stub = _stub_alias()
    tokenizer = transformers.AutoTokenizer.from_pretrained(stub)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        stub,
        num_labels=finetune_pilot_b2.N_CLASSES,
        ignore_mismatched_sizes=True,
    )
    hidden_size = int(model.config.hidden_size)
    aux_head = torch.nn.Linear(hidden_size, 3)

    rows = _make_phrasebank_rows()
    encodings = tokenizer(
        [r.sentence for r in rows],
        truncation=True,
        max_length=32,
        padding="max_length",
        return_tensors="pt",
    )
    aux_labels = torch.tensor([r.label_idx for r in rows], dtype=torch.long)

    model.train()
    aux_head.train()
    outputs = model.base_model(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
    )
    pooled = finetune_pilot_b2._pooled_from_base_model_output(
        outputs, encodings["attention_mask"]
    )
    logits = aux_head(pooled)
    loss = torch.nn.functional.cross_entropy(logits, aux_labels)
    loss.backward()

    # Aux head's linear weight grad is populated + non-zero somewhere.
    assert aux_head.weight.grad is not None
    assert aux_head.weight.grad.abs().sum().item() > 0.0
    # The encoder body also receives gradient -- proves the aux loss
    # flows back through the shared encoder, not just the new head.
    encoder_grads = [
        p.grad for p in model.base_model.parameters() if p.grad is not None
    ]
    assert any(g.abs().sum().item() > 0.0 for g in encoder_grads)


def test_run_sweep_honours_phrasebank_jsonl_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_sweep`` reads PhraseBank from a local JSONL when supplied.

    Tightly bounds the smoke: 1 seed, 1 fold, 1 epoch, batch 2 over a
    tiny synthetic training-package fixture. The assertion is on the
    sweep payload's ``phrasebank_aux`` meta block, not the macro-F1
    (the random-init stub does not converge on 4 train rows).
    """

    torch = _torch_or_skip()
    _ensure_stub_loadable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    package_dir = _build_synthetic_package(tmp_path)

    # PhraseBank fixture.
    pb_fixture = tmp_path / "phrasebank_fixture.jsonl"
    pb_rows = [
        {"sentence": "Revenue grew sharply.", "label": "positive"},
        {"sentence": "Profit fell sharply.", "label": "negative"},
        {"sentence": "Costs were flat.", "label": "neutral"},
        {"sentence": "Margins narrowed slightly.", "label": "negative"},
    ]
    pb_fixture.write_text(
        "\n".join(json.dumps(r) for r in pb_rows), encoding="utf-8"
    )

    # Redirect the resolver so we don't depend on the global processed/
    # directory inside the worktree.
    monkeypatch.setattr(
        finetune_pilot_b2,
        "_resolve_training_package_dir",
        lambda _id: package_dir,
    )

    args = type(
        "Args",
        (),
        {
            "training_package_id": "tp_phrasebank_smoke",
            "encoder_alias": _stub_alias(),
            "seeds": [11],
            "folds": ["fold_smoke"],
            "epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.0,
            "max_length": 32,
            "enable_phrasebank_aux": True,
            "phrasebank_aux_lambda": 0.3,
            "phrasebank_subset": "sentences_allagree",
            "phrasebank_cache_root": None,
            "phrasebank_jsonl": pb_fixture,
        },
    )()
    payload = finetune_pilot_b2.run_sweep(args)
    meta = payload["phrasebank_aux"]
    assert meta["enabled"] is True
    assert meta["n_rows"] == 4
    assert meta["aux_lambda"] == pytest.approx(0.3)
    assert meta["class_counts"] == [2, 1, 1]
    # The sweep produced exactly one (seed, fold) cell with the aux
    # payload populated.
    cells = payload["trials"][0]["folds"]
    assert len(cells) == 1
    assert "phrasebank_aux" in cells[0]
    assert cells[0]["phrasebank_aux"]["n_rows"] == 4


def test_run_sweep_treats_enable_aux_with_zero_lambda_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--enable-phrasebank-aux`` with ``--phrasebank-aux-lambda=0`` is a
    footgun: PhraseBank would load (and emit ``enabled=true`` meta) while
    the multiplier zeroed every aux gradient. We treat lambda<=0 as
    aux-disabled, emit a WARN line, and never touch the loader.
    """

    torch = _torch_or_skip()
    _ensure_stub_loadable()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    package_dir = _build_synthetic_package(tmp_path)

    # Sentinel: if the loader is reached we explode the test.
    def _exploding_loader(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("PhraseBank loader must not be invoked when lambda<=0")

    monkeypatch.setattr(
        finetune_pilot_b2,
        "_resolve_training_package_dir",
        lambda _id: package_dir,
    )
    monkeypatch.setattr(
        "app.data.phrasebank.load_phrasebank_rows",
        _exploding_loader,
    )

    args = type(
        "Args",
        (),
        {
            "training_package_id": "tp_phrasebank_smoke",
            "encoder_alias": _stub_alias(),
            "seeds": [11],
            "folds": ["fold_smoke"],
            "epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.0,
            "max_length": 32,
            "enable_phrasebank_aux": True,
            "phrasebank_aux_lambda": 0.0,
            "phrasebank_subset": "sentences_allagree",
            "phrasebank_cache_root": None,
            "phrasebank_jsonl": None,
        },
    )()
    payload = finetune_pilot_b2.run_sweep(args)
    captured = capsys.readouterr()
    assert "WARN" in captured.out
    assert payload["phrasebank_aux"] == {"enabled": False}
    # Per-cell schema is also aux-free.
    cells = payload["trials"][0]["folds"]
    assert "phrasebank_aux" not in cells[0]
