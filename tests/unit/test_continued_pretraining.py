"""Smoke tests for the CPU-side of continued_pretraining.

The actual MLM training run is GPU-bound and exercised via `make`-driven
smoke runs, not pytest. These tests cover the pair-collection paths that
shape data before it hits the model.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.data import continued_pretraining as cpt


def test_iter_local_pairs_skips_missing_and_empty(tmp_path: Path) -> None:
    (tmp_path / "speeches.json").write_text(
        json.dumps(
            [
                {"text": "Inflation pressures remain elevated."},
                {"body": "Activity has expanded at a moderate pace."},
                {"text": "   "},  # empty after strip → dropped
                {"unrelated": "no text"},
            ]
        ),
        encoding="utf-8",
    )
    pairs = cpt._iter_local_pairs(tmp_path, ["speeches.json", "missing.json"])
    assert len(pairs) == 2
    assert {p["sequenceA"] for p in pairs} == {
        "Inflation pressures remain elevated.",
        "Activity has expanded at a moderate pace.",
    }
    assert all(p["sequenceB"] == "" for p in pairs)
    assert all(p["next_sentence_label"] == 0 for p in pairs)


def test_iter_local_pairs_skips_non_list_payload(tmp_path: Path) -> None:
    (tmp_path / "wrong.json").write_text(json.dumps({"text": "not a list"}), encoding="utf-8")
    pairs = cpt._iter_local_pairs(tmp_path, ["wrong.json"])
    assert pairs == []


def _install_fake_datasets(monkeypatch, rows: list[dict]) -> None:
    fake = types.SimpleNamespace()
    fake.load_dataset = lambda dataset_id, **kw: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_bis_pair_stream_filters_empty_and_respects_max_rows(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "A1", "sequenceB": "B1", "next_sentence_label": 0},
            {"sequenceA": "", "sequenceB": "B2", "next_sentence_label": 0},  # empty A → dropped
            {"sequenceA": "A3", "sequenceB": "B3", "next_sentence_label": 1},
            {"sequenceA": "A4", "sequenceB": None, "next_sentence_label": 0},
            {"sequenceA": "A5", "sequenceB": "B5", "next_sentence_label": 1},
        ],
    )
    rows = list(
        cpt._bis_pair_stream(
            "samchain/BIS_speeches_97_23_MLM",
            None,
            streaming=False,
            max_rows=3,
        )
    )
    assert len(rows) == 3
    assert [r["sequenceA"] for r in rows] == ["A1", "A3", "A4"]
    assert rows[1]["next_sentence_label"] == 1
    assert rows[2]["sequenceB"] == ""


def test_iter_fomc_pairs_reads_only_minutes_and_statements(tmp_path: Path) -> None:
    """The strict FOMC-only substrate must read only ``fomc_minutes.json``
    and ``fomc_statements.json`` even when other JSON files sit alongside."""

    (tmp_path / "fomc_minutes.json").write_text(
        json.dumps([{"text": "Minutes paragraph."}]),
        encoding="utf-8",
    )
    (tmp_path / "fomc_statements.json").write_text(
        json.dumps([{"text": "Statement paragraph."}]),
        encoding="utf-8",
    )
    # Decoys: these must NOT be picked up by the strict substrate.
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Decoy speech."}]),
        encoding="utf-8",
    )
    (tmp_path / "beige_book.json").write_text(
        json.dumps([{"text": "Decoy beige book."}]),
        encoding="utf-8",
    )

    pairs = cpt._iter_fomc_pairs(tmp_path)
    assert len(pairs) == 2
    bodies = {p["sequenceA"] for p in pairs}
    assert bodies == {"Minutes paragraph.", "Statement paragraph."}
    # Decoys must not have made it in.
    assert "Decoy speech." not in bodies
    assert "Decoy beige book." not in bodies


def test_collect_pairs_substrate_fomc_strips_bis_and_local(tmp_path: Path) -> None:
    """``--substrate fomc`` is strict: no BIS network call, no broader
    local corpus, only the two FOMC JSON files."""

    (tmp_path / "fomc_statements.json").write_text(
        json.dumps([{"text": "FOMC statement body."}]),
        encoding="utf-8",
    )
    (tmp_path / "fomc_minutes.json").write_text(
        json.dumps([{"text": "FOMC minutes body."}]),
        encoding="utf-8",
    )
    # Even though chair_speeches is in --corpus-files (legacy default),
    # the fomc substrate must skip it.
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Decoy chair speech."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "fomc",
            "--data-dir",
            str(tmp_path),
        ]
    )
    pairs = cpt._collect_pairs(args)
    bodies = {p["sequenceA"] for p in pairs}
    assert bodies == {"FOMC statement body.", "FOMC minutes body."}
    assert "Decoy chair speech." not in bodies


def test_collect_pairs_substrate_fomc_respects_max_rows(tmp_path: Path) -> None:
    """``--max-rows`` clamps the strict FOMC pool just like the local
    substrate does."""

    (tmp_path / "fomc_minutes.json").write_text(
        json.dumps([{"text": f"Minutes {i}."} for i in range(5)]),
        encoding="utf-8",
    )
    (tmp_path / "fomc_statements.json").write_text(
        json.dumps([{"text": f"Stmt {i}."} for i in range(5)]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "fomc",
            "--data-dir",
            str(tmp_path),
            "--max-rows",
            "3",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 3


def test_collect_pairs_substrate_local(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Local speech text."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "local",
            "--data-dir",
            str(tmp_path),
            "--corpus-files",
            "chair_speeches.json",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert pairs == [
        {"sequenceA": "Local speech text.", "sequenceB": "", "next_sentence_label": 0}
    ]


def test_collect_pairs_substrate_bis_uses_streaming(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
            {"sequenceA": "BIS A2", "sequenceB": "BIS B2", "next_sentence_label": 1},
        ],
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "bis",
            "--streaming",
            "--max-rows",
            "10",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 2
    assert pairs[0]["sequenceA"] == "BIS A1"


def test_collect_pairs_substrate_both_loads_local_first_then_bis_remainder(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression: --substrate both must give local representation under --max-rows.

    Previous behaviour ran BIS first (which always fills the cap given its size),
    silently emptying the local slice — making --substrate both --max-rows N
    indistinguishable from --substrate bis. Local now loads first; BIS fills the
    remaining capacity.
    """
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
            {"sequenceA": "BIS A2", "sequenceB": "BIS B2", "next_sentence_label": 0},
        ],
    )
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Local 1."}, {"text": "Local 2."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "both",
            "--data-dir",
            str(tmp_path),
            "--corpus-files",
            "chair_speeches.json",
            "--max-rows",
            "3",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 3
    # Local (2 pairs) loads first; BIS contributes the remaining 1.
    assert pairs[0]["sequenceA"] == "Local 1."
    assert pairs[1]["sequenceA"] == "Local 2."
    assert pairs[2]["sequenceA"] == "BIS A1"


def test_collect_pairs_substrate_both_warns_when_local_exhausts_cap(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression: when --max-rows is small enough that local fills it, BIS gets
    dropped — surface this with a warning so the user knows BIS substrate is absent."""
    import warnings as _warnings

    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
        ],
    )
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Local 1."}, {"text": "Local 2."}, {"text": "Local 3."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "both",
            "--data-dir",
            str(tmp_path),
            "--corpus-files",
            "chair_speeches.json",
            "--max-rows",
            "2",
        ]
    )
    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        pairs = cpt._collect_pairs(args)

    # Local got 3 pairs (no cap on local under --substrate both unless local-only),
    # exceeding --max-rows; BIS substrate dropped + warning surfaced.
    assert len(pairs) == 3
    assert all(p["sequenceA"].startswith("Local") for p in pairs)
    bis_drop_warnings = [w for w in captured if "BIS substrate" in str(w.message)]
    assert bis_drop_warnings, "expected a warning when BIS substrate dropped"


def test_resolve_dataset_sha_returns_requested_when_supplied() -> None:
    """When the user pins --bis-dataset-revision, that value passes through verbatim."""
    assert cpt._resolve_dataset_sha("samchain/BIS_speeches_97_23_MLM", "deadbeef") == "deadbeef"


def test_parse_args_objective_validates_choices() -> None:
    args = cpt._parse_args(["--objective", "mlm"])
    assert args.objective == "mlm"
    with pytest.raises(SystemExit):
        cpt._parse_args(["--objective", "nonsense"])


def test_bis_xbank_substrate_is_registered() -> None:
    """The bis_xbank substrate must show up in the valid-choices tuple
    so the CLI accepts ``--substrate bis_xbank`` without an argparse error."""
    assert "bis_xbank" in cpt._VALID_SUBSTRATES


def _install_fake_gtfintechlab(monkeypatch, sentences_per_dataset: dict[str, list[str]]) -> None:
    """Stub the datasets module's load_dataset + config/split helpers used by
    ``_iter_gtfintechlab_xbank_pairs``.

    ``sentences_per_dataset`` keys are HF dataset ids; values are the flat list
    of sentence strings that should be returned (single config / single split).
    """
    fake = types.SimpleNamespace()
    fake.get_dataset_config_names = lambda dataset_id, revision=None, **kw: ["default"]
    fake.get_dataset_split_names = lambda dataset_id, config, revision=None, **kw: ["train"]

    def _load(dataset_id, config=None, split=None, revision=None, **kw):
        sents = sentences_per_dataset.get(dataset_id, [])
        return [{"sentences": s} for s in sents]

    fake.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_iter_gtfintechlab_xbank_pairs_shape(monkeypatch) -> None:
    """The xbank pair iterator must return pair dicts with the same keys as
    ``_iter_fomc_pairs`` / ``_bis_pair_stream`` so the trainer stays uniform."""
    sentences = {
        "gtfintechlab/european_central_bank": ["ECB s1.", "ECB s2.", "ECB s3.", "ECB s4."],
        "gtfintechlab/bank_of_japan": ["BoJ s1.", "BoJ s2."],
        "gtfintechlab/bank_of_england": ["BoE s1.", "BoE s2."],
        "gtfintechlab/bank_of_canada": ["BoC s1.", "BoC s2."],
        "gtfintechlab/reserve_bank_of_australia": ["RBA s1.", "RBA s2."],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)

    pairs = cpt._iter_gtfintechlab_xbank_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for p in pairs:
        assert set(p.keys()) >= {"sequenceA", "sequenceB", "next_sentence_label"}
        assert isinstance(p["sequenceA"], str) and p["sequenceA"]
        assert isinstance(p["sequenceB"], str) and p["sequenceB"]
        # Pairs must be two distinct sentences, not degenerate copies.
        assert p["sequenceA"] != p["sequenceB"]
        assert isinstance(p["next_sentence_label"], int)


def test_iter_gtfintechlab_xbank_pairs_covers_all_five_banks(monkeypatch) -> None:
    """Every one of the 5 cross-bank datasets must contribute pairs — no silent
    drop of a single bank."""
    sentences = {
        "gtfintechlab/european_central_bank": ["ECB a.", "ECB b."],
        "gtfintechlab/bank_of_japan": ["BoJ a.", "BoJ b."],
        "gtfintechlab/bank_of_england": ["BoE a.", "BoE b."],
        "gtfintechlab/bank_of_canada": ["BoC a.", "BoC b."],
        "gtfintechlab/reserve_bank_of_australia": ["RBA a.", "RBA b."],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)
    pairs = cpt._iter_gtfintechlab_xbank_pairs()
    joined = " | ".join(p["sequenceA"] + " " + p["sequenceB"] for p in pairs)
    for marker in ("ECB", "BoJ", "BoE", "BoC", "RBA"):
        assert marker in joined, f"{marker} contribution missing from xbank pair stream"


def test_iter_gtfintechlab_xbank_pairs_dedup_preserves_b(monkeypatch) -> None:
    """When the pairing buffer contains a degenerate (a == b) duplicate,
    only ``a`` should be dropped — ``b`` must roll forward as the
    candidate first-of-pair so it can legitimately pair with the next
    sentence. The pre-fix code reset the buffer to ``[]`` and silently
    discarded ``b`` alongside ``a``.

    Sequence: ["X", "X", "Y"]. The first two collapse into the
    degenerate (X, X) — drop the first X, keep the second X as buffer.
    Pair (X, Y) then emits a single pair.
    """
    sentences = {
        # All five datasets must be present (the iterator walks the full
        # _GTFINTECHLAB_XBANK_DATASETS tuple) but only the first carries
        # the dedup-relevant fixture.
        "gtfintechlab/european_central_bank": ["X", "X", "Y"],
        "gtfintechlab/bank_of_japan": [],
        "gtfintechlab/bank_of_england": [],
        "gtfintechlab/bank_of_canada": [],
        "gtfintechlab/reserve_bank_of_australia": [],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)
    pairs = cpt._iter_gtfintechlab_xbank_pairs()
    # Exactly one pair (X, Y) — not zero (pre-fix would have dropped
    # the second X with the dedup-reset) and not two (would happen if
    # the dedup somehow emitted (X, X) anyway).
    assert pairs == [
        {"sequenceA": "X", "sequenceB": "Y", "next_sentence_label": 0}
    ]


def test_iter_gtfintechlab_xbank_pairs_dedup_does_not_double_count_b(
    monkeypatch,
) -> None:
    """Edge case: when ``b == c`` (the next sentence after a dedup also
    equals ``b``), the loop must emit ONE pair, not two — the rolled-
    forward ``b`` should dedup-cancel against ``c`` rather than
    silently fire (b, c) as a degenerate pair.

    Sequence: ["X", "X", "X", "Y"]. Step 1: (X, X) → drop first X,
    buffer = [X]. Step 2: append X → (X, X) again → drop, buffer =
    [X]. Step 3: append Y → (X, Y) emit. Exactly one pair.
    """
    sentences = {
        "gtfintechlab/european_central_bank": ["X", "X", "X", "Y"],
        "gtfintechlab/bank_of_japan": [],
        "gtfintechlab/bank_of_england": [],
        "gtfintechlab/bank_of_canada": [],
        "gtfintechlab/reserve_bank_of_australia": [],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)
    pairs = cpt._iter_gtfintechlab_xbank_pairs()
    assert pairs == [
        {"sequenceA": "X", "sequenceB": "Y", "next_sentence_label": 0}
    ]


def test_iter_gtfintechlab_xbank_pairs_logs_trailing_odd_sentence(
    monkeypatch, caplog
) -> None:
    """A bucket with an odd number of non-empty sentences leaves the
    last sentence unpaired. The iterator must log how many trailing
    sentences were dropped per (config, split) bucket so
    reproducibility audits can reconcile pair counts against row
    counts."""
    sentences = {
        # 3 sentences in this dataset → 1 pair emitted, 1 trailing.
        "gtfintechlab/european_central_bank": ["A", "B", "C"],
        "gtfintechlab/bank_of_japan": [],
        "gtfintechlab/bank_of_england": [],
        "gtfintechlab/bank_of_canada": [],
        "gtfintechlab/reserve_bank_of_australia": [],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)
    caplog.set_level("INFO", logger="app.data.continued_pretraining")
    pairs = cpt._iter_gtfintechlab_xbank_pairs()
    assert pairs == [
        {"sequenceA": "A", "sequenceB": "B", "next_sentence_label": 0}
    ]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "xbank_trailing_dropped" in messages
    assert "dataset_id=gtfintechlab/european_central_bank" in messages
    assert "n=1" in messages


def test_iter_gtfintechlab_xbank_pairs_raises_on_missing_revision(
    monkeypatch,
) -> None:
    """If a future edit adds a bank to ``_GTFINTECHLAB_XBANK_DATASETS``
    without a corresponding entry in ``_GTFINTECHLAB_XBANK_REVISIONS``,
    the iterator must raise loudly — silently degrading to HF Hub HEAD
    would break the reproducibility invariant. Simulate by removing
    one pin and asserting ``KeyError``."""
    sentences = {
        "gtfintechlab/european_central_bank": ["A", "B"],
        "gtfintechlab/bank_of_japan": ["C", "D"],
        "gtfintechlab/bank_of_england": ["E", "F"],
        "gtfintechlab/bank_of_canada": ["G", "H"],
        "gtfintechlab/reserve_bank_of_australia": ["I", "J"],
    }
    _install_fake_gtfintechlab(monkeypatch, sentences)
    # Drop the BoJ pin so the iterator's first BoJ lookup fails.
    patched = dict(cpt._GTFINTECHLAB_XBANK_REVISIONS)
    patched.pop("gtfintechlab/bank_of_japan")
    monkeypatch.setattr(cpt, "_GTFINTECHLAB_XBANK_REVISIONS", patched)
    with pytest.raises(KeyError, match="bank_of_japan"):
        cpt._iter_gtfintechlab_xbank_pairs()


def test_iter_gtfintechlab_xbank_pairs_scale_floor() -> None:
    """The live HF datasets yield 5 x ~3,000 sentences; even after consecutive
    pairing we should clear 5,000 pairs. Skip when ``datasets`` package or
    network is unavailable (CI runs in offline-only mode)."""
    pytest.importorskip("datasets")
    try:
        pairs = cpt._iter_gtfintechlab_xbank_pairs()
    except Exception as exc:  # pragma: no cover — networkless CI
        pytest.skip(f"HF cross-bank load failed: {exc}")
    assert len(pairs) >= 5000, (
        f"expected >= 5000 xbank pairs across the 5 banks, got {len(pairs)}"
    )


def test_collect_pairs_substrate_bis_xbank_combines_bis_and_xbank(
    monkeypatch, tmp_path: Path
) -> None:
    """``--substrate bis_xbank`` must combine BIS + cross-bank pairs and
    exclude FOMC / local Fed-adjacent JSONs."""
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
            {"sequenceA": "BIS A2", "sequenceB": "BIS B2", "next_sentence_label": 1},
        ],
    )
    # The fake xbank iterator replaces the real HF loader to keep the test
    # offline. Returns a couple of clearly-tagged pairs we can assert on.
    monkeypatch.setattr(
        cpt,
        "_iter_gtfintechlab_xbank_pairs",
        lambda: [
            {"sequenceA": "XBANK A1", "sequenceB": "XBANK B1", "next_sentence_label": 0},
            {"sequenceA": "XBANK A2", "sequenceB": "XBANK B2", "next_sentence_label": 0},
        ],
    )
    # Decoy local + FOMC files that must NOT be loaded under bis_xbank.
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Decoy local speech."}]),
        encoding="utf-8",
    )
    (tmp_path / "fomc_statements.json").write_text(
        json.dumps([{"text": "Decoy FOMC statement."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "bis_xbank",
            "--data-dir",
            str(tmp_path),
        ]
    )
    pairs = cpt._collect_pairs(args)
    bodies_a = [p["sequenceA"] for p in pairs]
    assert "BIS A1" in bodies_a and "BIS A2" in bodies_a
    assert "XBANK A1" in bodies_a and "XBANK A2" in bodies_a
    # Decoys excluded.
    assert "Decoy local speech." not in bodies_a
    assert "Decoy FOMC statement." not in bodies_a


def test_collect_pairs_substrate_bis_xbank_respects_max_rows(
    monkeypatch, tmp_path: Path
) -> None:
    """Under ``--substrate bis_xbank --max-rows N``, xbank loads first (small,
    finite) and BIS fills the remainder — mirroring the local/both convention
    so the xbank contribution is never silently emptied."""
    _install_fake_datasets(
        monkeypatch,
        [{"sequenceA": f"BIS A{i}", "sequenceB": "B", "next_sentence_label": 0} for i in range(10)],
    )
    monkeypatch.setattr(
        cpt,
        "_iter_gtfintechlab_xbank_pairs",
        lambda: [
            {"sequenceA": "XBANK A1", "sequenceB": "XBANK B1", "next_sentence_label": 0},
            {"sequenceA": "XBANK A2", "sequenceB": "XBANK B2", "next_sentence_label": 0},
        ],
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "bis_xbank",
            "--data-dir",
            str(tmp_path),
            "--max-rows",
            "5",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 5
    # First two come from xbank, remaining 3 from BIS.
    assert pairs[0]["sequenceA"].startswith("XBANK")
    assert pairs[1]["sequenceA"].startswith("XBANK")
    assert pairs[2]["sequenceA"].startswith("BIS")


def test_main_bis_xbank_manifest_records_xbank_dataset_revisions(
    monkeypatch, tmp_path: Path
) -> None:
    """Reproducibility: a --substrate bis_xbank run must record every
    cross-bank dataset id + pinned SHA in the manifest, alongside the BIS
    entry. Otherwise a future re-run can't reconstruct the exact text mix.
    """
    _install_fake_datasets(
        monkeypatch,
        [{"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0}],
    )
    monkeypatch.setattr(
        cpt,
        "_iter_gtfintechlab_xbank_pairs",
        lambda: [
            {"sequenceA": "XBANK A1", "sequenceB": "XBANK B1", "next_sentence_label": 0}
        ],
    )
    # Keep the manifest writer offline + skip the real MLM training run.
    monkeypatch.setattr(cpt, "_resolve_dataset_sha", lambda dataset_id, revision: revision)

    def _fake_run_mlm(**kwargs):
        out_dir = kwargs["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        return {
            "base_checkpoint": kwargs["base_checkpoint"],
            "base_revision": None,
            "epochs": kwargs["epochs"],
            "learning_rate": kwargs["learning_rate"],
            "batch_size": kwargs["batch_size"],
            "block_size": kwargs["block_size"],
            "objective": kwargs["objective"],
            "train_runtime_s": 0.0,
            "train_loss": 0.0,
            "num_examples": len(kwargs["pair_records"]),
            "checkpoint_path": str(out_dir / "checkpoint"),
        }

    monkeypatch.setattr(cpt, "run_mlm", _fake_run_mlm)

    rc = cpt.main(
        [
            "--substrate",
            "bis_xbank",
            "--artifact-root",
            str(tmp_path),
            "--data-dir",
            str(tmp_path),
            "--bis-dataset-revision",
            "deadbeef",
        ]
    )
    assert rc == 0

    # Locate the run dir under the temp artifact root.
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected single run dir, got {run_dirs}"
    manifest_path = run_dirs[0] / "run_manifest.json"
    assert manifest_path.exists(), "manifest was not written"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_xbank_ids = {
        "gtfintechlab/european_central_bank",
        "gtfintechlab/bank_of_japan",
        "gtfintechlab/bank_of_england",
        "gtfintechlab/bank_of_canada",
        "gtfintechlab/reserve_bank_of_australia",
    }

    # The 5 xbank ids + the BIS id should all be referenced under
    # hyperparameters.xbank_dataset_revisions (5 ids -> 5 pinned SHAs).
    recorded = manifest["hyperparameters"]["xbank_dataset_revisions"]
    assert set(recorded.keys()) == expected_xbank_ids
    for dataset_id, sha in recorded.items():
        assert sha == cpt._GTFINTECHLAB_XBANK_REVISIONS[dataset_id]
        # Pinned SHAs are 40-char lowercase hex; guard against accidental nulls.
        assert isinstance(sha, str) and len(sha) == 40

    # BIS entry is unaffected: id + requested + resolved still in place.
    assert manifest["hyperparameters"]["bis_dataset_id"] == cpt.DEFAULT_BIS_DATASET_ID
    assert manifest["hyperparameters"]["bis_dataset_revision_resolved"] == "deadbeef"
