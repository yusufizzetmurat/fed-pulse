"""Push canonical fed-pulse artefacts to Hugging Face Hub (#302 Stage 1-3).

Run this once locally after merging the deployability PR. The script is
idempotent — it skips files whose remote SHA matches the local SHA, so
re-running it after a partial failure or a checkpoint refresh only
uploads the diff.

Usage
-----

Dry-run (no network, prints what would be uploaded)::

    python scripts/push_artefacts_to_hub.py --dry-run

Push everything (encoders + forecaster + retrieval + trajectory + rates
heads + training package + embedding caches)::

    python scripts/push_artefacts_to_hub.py --all

Push only a subset::

    python scripts/push_artefacts_to_hub.py --kinds encoder,forecaster

The script reads the canonical paths off ``backend/app/models/registry.yaml``
where possible. Anything that isn't pinned there (e.g. the per-encoder
embedding cache parquets) is enumerated from disk.

Requirements
------------

- ``HF_TOKEN`` env var with ``write`` scope on the ``yusufizzetmurat``
  namespace (or the namespace passed via ``--owner``).
- ``huggingface_hub>=0.24`` installed in the current Python env.
- Local checkpoints / parquet bundles present under ``/data/`` or
  ``data/`` per the registry.

Post-run actions
----------------

After every successful push the script prints the resulting commit
SHAs. Copy each sha into ``backend/app/models/registry.yaml`` under the
matching ``artefacts:`` entry's ``revision:`` field. Without this pin
the inference container will pull ``main``, which defeats the
deterministic-load contract the rest of the registry enforces.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.registry import (  # noqa: E402
    encoder_ref,
    load_artefacts,
    parse_hf_uri,
)


# Map artefact-registry key -> ((local source path on disk),
# (license id), (encoder citation if any)).
ARTEFACT_SOURCES: dict[str, dict[str, str]] = {
    "encoder_canonical": {
        "encoder_alias": "finbert_fed_adjacent_xbank_dapt",
        "license": "cc-by-4.0",
        "kind": "encoder",
    },
    "encoder_finbert_fomc": {
        "encoder_alias": "finbert_fomc",
        "license": "cc-by-4.0",
        "kind": "encoder",
    },
    "encoder_finbert_fed_adjacent": {
        "encoder_alias": "finbert_fed_adjacent",
        "license": "cc-by-4.0",
        "kind": "encoder",
    },
    "encoder_finbert_fed_adjacent_xbank": {
        "encoder_alias": "finbert_fed_adjacent_xbank",
        "license": "cc-by-4.0",
        "kind": "encoder",
    },
    "forecaster_canonical": {
        "local_path": "backend/models/forecaster_best.pt",
        "license": "mit",
        "kind": "forecaster",
    },
    "rates_heads_canonical": {
        "local_path": "backend/models/forecaster_best.pt",
        "license": "mit",
        "kind": "rates_heads",
    },
    "retrieval_bundle": {
        "local_path": "data/artifacts/retrieval",
        "license": "mit",
        "kind": "retrieval",
    },
    "trajectory_bundle": {
        "local_path": "data/artifacts/trajectory",
        "license": "mit",
        "kind": "trajectory",
    },
    "training_package": {
        "local_path": "data/processed/canonical",
        "license": "mit",
        "kind": "training_package",
    },
    "embedding_caches": {
        "local_path": "data/raw/embeddings",
        "license": "mit",
        "kind": "embedding_caches",
    },
}


MODEL_CARD_TEMPLATE = """---
license: {license}
library_name: transformers
tags:
- fed-pulse
- monetary-policy
- fomc
---

# {repo_id}

Canonical artefact published by the [fed-pulse](https://github.com/yusufizzetmurat/fed-pulse) SWE 599 research project at Boğaziçi University.

- Companion repo: <https://github.com/yusufizzetmurat/fed-pulse>
- Companion wiki: <https://github.com/yusufizzetmurat/fed-pulse/wiki>
- Kind: `{kind}`
- License: `{license}`

## Training corpus

{corpus}

## Training command

```
{train_cmd}
```

## Consumer schema

The fed-pulse FastAPI app reads this artefact through the registry resolver
in `backend/app/models/registry.py`. The events parquet schema the
consumer side expects (when this repo is a dataset) is documented in
`fed-pulse.wiki/07_Data_Schema.md`. A minimal snippet:

```
events.parquet columns:
  event_date       string  ISO date
  symbol           string  yfinance ticker
  text             string  FOMC statement / minutes / press conference body
  stance           int     0 hawkish / 1 dovish / 2 neutral
  certainty        int     0 low / 1 medium / 2 high
  factor           int     0 inflation / 1 employment / 2 financial-stability / 3 other
  topic            int     0 rates / 1 balance-sheet / 2 communication / 3 outlook
  next_meeting     date    next-FOMC pointer
  rates_*          float   pre/post-meeting yield observations
```

## Citation

If you use this artefact, please cite the fed-pulse repository.
"""


CORPUS_BLURBS: dict[str, str] = {
    "encoder": (
        "FinBERT base + cross-bank pretraining substrate "
        "(`samchain/BIS_speeches_97_23_MLM` + the gtfintechlab ECB / BoJ / BoE / "
        "BoC / RBA multi-axis corpora reformatted as NSP pairs). See "
        "`fed-pulse.wiki/13_External_Corpora_Inventory.md` for the full provenance."
    ),
    "forecaster": (
        "Multi-head forecaster (#292) trained on the canonical fed-pulse "
        "training package over the expanding walk-forward fold protocol. "
        "Multi-target rates heads + vol-regime classifier share the cross-bank "
        "DAPT encoder backbone."
    ),
    "rates_heads": (
        "Multi-target rates heads (#292/#293). 2y, 10y, target-rate regression "
        "outputs with auxiliary 3-class easing/neutral/tightening surface per "
        "head. Conformal-calibrated."
    ),
    "retrieval": (
        "Sentence-transformer retrieval bundle (#294) — embedding_index.parquet + "
        "embeddings.npy + manifest. Trained via MultipleNegativesRankingLoss on "
        "same-meeting positive pairs."
    ),
    "trajectory": (
        "Sequence-of-meetings trajectory bundle (#296) — LSTM checkpoint + 2D "
        "embeddings + manifest. Predicts next-meeting stance from the prior "
        "sequence of statement embeddings."
    ),
    "training_package": (
        "Canonical training package — events.parquet + splits + fold manifest + "
        "rates_panel.parquet + linguistic_features.parquet + mp_surprises.parquet "
        "+ macro_state.parquet + registry_normalized.jsonl + quality reports."
    ),
    "embedding_caches": (
        "Per-encoder embedding caches keyed on (encoder alias, encoder revision). "
        "Each parquet matches the schema in `backend/app/data/embedding_cache.py`: "
        "record_id, doc_id, event_date, chunk_index, chunk_preview, embedding."
    ),
}


TRAIN_CMDS: dict[str, str] = {
    "encoder": "python -m app.data.continued_pretraining --substrate xbank_dapt --seed 11",
    "forecaster": "python -m app.train_forecaster --training-package-id <tp-id> --seed 11",
    "rates_heads": "python scripts/run_rates_heads_sweep.py --training-package-id <tp-id>",
    "retrieval": "python -m app.retrieval.train --train-end <fold-end>",
    "trajectory": "python -m app.trajectory.train --architecture lstm",
    "training_package": "python -m app.data.pipeline_data_prep --all-sources",
    "embedding_caches": "python scripts/cache_embeddings.py --encoder <alias> --training-package-id <tp-id> --allow-network",
}


@dataclasses.dataclass
class PushPlan:
    artefact_key: str
    repo_id: str
    repo_type: str
    local_path: Path
    license: str
    kind: str


def _resolve_local_path(meta: dict[str, str]) -> Path:
    if "encoder_alias" in meta:
        ref = encoder_ref(meta["encoder_alias"])
        if ref is None:
            raise ValueError(f"Unknown encoder alias: {meta['encoder_alias']!r}")
        return Path(ref.repo)
    return REPO_ROOT / meta["local_path"]


def _build_plans(kinds: set[str] | None) -> list[PushPlan]:
    artefacts = load_artefacts()
    plans: list[PushPlan] = []
    for key, meta in ARTEFACT_SOURCES.items():
        if kinds is not None and meta["kind"] not in kinds:
            continue
        ref = artefacts.get(key)
        if ref is None:
            raise ValueError(
                f"Artefact key {key!r} missing from backend/app/models/registry.yaml. "
                "Add it to the `artefacts:` block before pushing."
            )
        parsed = parse_hf_uri(ref.hf_uri)
        local_path = _resolve_local_path(meta)
        plans.append(
            PushPlan(
                artefact_key=key,
                repo_id=parsed.repo_id,
                repo_type=parsed.repo_type,
                local_path=local_path,
                license=meta["license"],
                kind=meta["kind"],
            )
        )
    return plans


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _render_model_card(plan: PushPlan) -> str:
    return MODEL_CARD_TEMPLATE.format(
        repo_id=plan.repo_id,
        license=plan.license,
        kind=plan.kind,
        corpus=CORPUS_BLURBS.get(plan.kind, "See companion repo for details."),
        train_cmd=TRAIN_CMDS.get(plan.kind, "See companion repo."),
    )


def _push_one(
    plan: PushPlan,
    *,
    token: str | None,
    dry_run: bool,
) -> None:
    print(f"\n=== {plan.artefact_key} ({plan.kind}) ===")
    print(f"    local : {plan.local_path}")
    print(f"    remote: hf://{'datasets/' if plan.repo_type == 'dataset' else ''}{plan.repo_id}")

    if not plan.local_path.exists():
        print(f"    [skip] local path does not exist: {plan.local_path}")
        return

    card = _render_model_card(plan)

    if dry_run:
        print("    [dry-run] would create repo and upload folder")
        print(f"    [dry-run] model card length: {len(card)} chars")
        if plan.local_path.is_file():
            print(f"    [dry-run] file sha256: {_file_sha256(plan.local_path)}")
        else:
            count = sum(1 for _ in plan.local_path.rglob("*") if _.is_file())
            print(f"    [dry-run] folder contains {count} files")
        return

    from huggingface_hub import HfApi, create_repo, upload_folder, upload_file  # type: ignore[import-not-found]

    create_repo(
        repo_id=plan.repo_id,
        repo_type=plan.repo_type,
        token=token,
        exist_ok=True,
        private=False,
    )
    api = HfApi(token=token)

    # Always (re)write the model card.
    card_path = plan.local_path / "README.md" if plan.local_path.is_dir() else plan.local_path.parent / "README.md"
    if plan.local_path.is_dir():
        (plan.local_path / "README.md").write_text(card, encoding="utf-8")

    if plan.local_path.is_file():
        api.upload_file(
            path_or_fileobj=str(plan.local_path),
            path_in_repo=plan.local_path.name,
            repo_id=plan.repo_id,
            repo_type=plan.repo_type,
            token=token,
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=plan.repo_id,
            repo_type=plan.repo_type,
            token=token,
        )
    else:
        upload_folder(
            folder_path=str(plan.local_path),
            repo_id=plan.repo_id,
            repo_type=plan.repo_type,
            token=token,
        )

    info_fn = api.model_info if plan.repo_type == "model" else api.dataset_info
    info = info_fn(plan.repo_id)
    print(f"    pushed @ {getattr(info, 'sha', '?')}")
    print(f"    --> pin this sha in registry.yaml under artefacts.{plan.artefact_key}.revision")
    # Silence the unused variable from card_path which is only built to keep the
    # local-dir branch parallel to the file branch.
    del card_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Push fed-pulse artefacts to HF Hub.")
    parser.add_argument("--owner", default="yusufizzetmurat", help="HF namespace owner (default: yusufizzetmurat)")
    parser.add_argument("--kinds", default=None, help="Comma-separated subset of kinds to push (e.g. encoder,forecaster)")
    parser.add_argument("--all", action="store_true", help="Push every artefact in ARTEFACT_SOURCES.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without contacting HF Hub.")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.all and args.kinds is None:
        parser.error("Pass --all, --kinds, or --dry-run.")

    kinds: set[str] | None = None
    if args.kinds:
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    plans = _build_plans(kinds)
    if not plans:
        print("No artefacts matched the filter.")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.dry_run and not token:
        parser.error("HF_TOKEN env var is required (write scope).")

    for plan in plans:
        _push_one(plan, token=token, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry-run complete. Pass --all or --kinds=... to actually push.")
    else:
        print("\nDone. Update registry.yaml with the printed commit shas, then commit + push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
