"""Jackknife-by-token robustness probe on the hawk / dove lexicons (#506).

The 15-dim ``RICH_LINGUISTIC_SLICE`` block carries one in-house feature,
``hawk_dove_asymmetry``, that is computed against two hand-curated
lexicons in :mod:`app.features.linguistic`:

- ``HAWK_TOKENS`` -- 7 single-token hawkish markers
- ``HAWK_PHRASES`` -- 1 multi-token phrase
- ``DOVE_TOKENS`` -- 9 single-token dovish markers
- ``DOVE_PHRASES`` -- 1 multi-token phrase

This runner leaves each single-token entry out in turn, recomputes the
``hawk_dove_asymmetry`` column on every event in a training package
under the patched lexicon, swaps the patched ``linguistic_features``
parquet into the package directory for the duration of one cell, and
calls :mod:`scripts.run_per_family_ablation` with a single-seed, single-
fold ``baseline`` smoke. The result is a per-token (with-F1, without-F1,
delta) triple plus a ``fragile_tokens`` list of every token whose
``|delta|`` exceeds :data:`FRAGILE_DELTA_THRESHOLD` (0.005 macro-F1).

Why this matters: the asymmetry feature is the only place in the rich-
feature block where a hand-curated 16-token lexicon picks up signal. A
single fragile token (one whose removal flips the family lift) would
warrant rephrasing the wiki claim from "the in-house hawk-dove block
carries X macro-F1" to "the block carries X macro-F1 contingent on a
small handful of tokens." The jackknife exposes the dependency cleanly.

Execution cost
--------------

The runner is GPU-bound. At the default canonical-comparison settings
(1 seed x 1 fold x baseline cell x 40 epochs) one cell takes roughly
8-12 minutes on a single Runpod A100 against the post-#350 training
package. The union of the four lexicon collections holds 18 entries
(7 hawk tokens + 1 hawk phrase + 9 dove tokens + 1 dove phrase) so a
full sweep is ~3-4 GPU-hours. A faster smoke (``--epochs 5``,
``--smoke``) reduces the cost to ~30 minutes total but trades CI
fidelity for sweep budget.

This script ships in the repo as a runner; the JSON artefact populates
after a separate GPU pass. The Makefile target ``hawk-dove-jackknife``
invokes it without arguments other than ``TRAINING_PACKAGE_ID``.

Output
------

JSON artefact at ``backend/artifacts/experiments/hawk_dove_jackknife.json``::

    {
      "training_package_id": "...",
      "baseline_f1": 0.4538,
      "fragile_delta_threshold": 0.005,
      "tokens": [
        {
          "token": "tightening",
          "kind": "hawk",
          "without_f1": 0.4502,
          "delta": -0.0036
        },
        ...
      ],
      "fragile_tokens": ["restrictive"]
    }

The ``with_f1`` is omitted from each per-token row because it is the
same baseline number every cell shares; the top-level ``baseline_f1``
carries it once.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from app.config import BACKEND_ROOT


#: |delta| floor above which a token is considered fragile. 0.005 is the
#: tightest band the per-family ablation table treats as a real lift; a
#: token whose removal moves macro-F1 by more than that warrants a
#: separate caveat in the wiki write-up.
FRAGILE_DELTA_THRESHOLD: float = 0.005


@dataclass(frozen=True)
class TokenJackknifeResult:
    """Per-token smoke result.

    ``without_f1`` is the macro-F1 the per-family ablation runner
    reported with the token leave-one-out applied; ``delta`` is the
    signed difference vs the baseline (negative = removing the token
    hurt the model).
    """

    token: str
    kind: str
    without_f1: float
    delta: float


def fragile_tokens_from_results(
    results: Iterable[TokenJackknifeResult],
    *,
    threshold: float = FRAGILE_DELTA_THRESHOLD,
) -> list[str]:
    """Return the tokens whose ``|delta|`` exceeds ``threshold``.

    Sorted lexicographically so re-runs against the same delta vector
    hit the same JSON bytes.
    """

    fragile = [r.token for r in results if abs(r.delta) > threshold]
    return sorted(fragile)


def patched_lexicons(
    leave_out_token: str,
    *,
    base_hawk: frozenset[str],
    base_dove: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(hawk, dove)`` lexicons with ``leave_out_token`` removed.

    The token is dropped from whichever collection contains it; if it
    appears in neither, both inputs come back unchanged so the caller's
    fixture round-trips losslessly. Symmetric removal (a token that
    accidentally lives in both lists) drops from both.
    """

    return (base_hawk - {leave_out_token}, base_dove - {leave_out_token})


def _build_token_inventory(
    hawk_tokens: frozenset[str],
    dove_tokens: frozenset[str],
) -> list[tuple[str, str]]:
    """Return ``(token, kind)`` pairs covering both single-token lists.

    Output is sorted by ``(kind, token)`` so the runner walks the same
    order every call and the JSON line-up matches the wiki citation
    schedule.
    """

    items: list[tuple[str, str]] = []
    for token in sorted(hawk_tokens):
        items.append((token, "hawk"))
    for token in sorted(dove_tokens):
        items.append((token, "dove"))
    return items


def _rebuild_patched_parquet(
    package_dir: Path,
    *,
    patched_hawk: frozenset[str],
    patched_dove: frozenset[str],
    output_parquet: Path,
) -> None:
    """Recompute ``hawk_dove_asymmetry`` on the patched lexicon and write
    the resulting frame as parquet at ``output_parquet``.

    Reads the canonical ``linguistic_features.parquet`` for every other
    column (the LDA shares, the densities, the pivot distance) so we
    only re-do the cheap per-document hawk/dove pass. Falls back to
    rebuilding the whole frame via :func:`build_linguistic_feature_frame`
    when the canonical parquet is absent.
    """

    import pandas as pd

    from app.features import linguistic as linguistic_mod

    canonical_parquet = package_dir / "linguistic_features.parquet"
    if not canonical_parquet.exists():
        # Cold rebuild: the runner has no cached frame to splice into,
        # so we recompute the whole frame against the patched lexicon.
        original_hawk = linguistic_mod.HAWK_TOKENS
        original_dove = linguistic_mod.DOVE_TOKENS
        try:
            linguistic_mod.HAWK_TOKENS = patched_hawk
            linguistic_mod.DOVE_TOKENS = patched_dove
            frame, _ = linguistic_mod.build_linguistic_feature_frame(
                package_dir=package_dir
            )
        finally:
            linguistic_mod.HAWK_TOKENS = original_hawk
            linguistic_mod.DOVE_TOKENS = original_dove
        frame.to_parquet(output_parquet, engine="pyarrow", index=False)
        return

    registry = package_dir / "registry_normalized.jsonl"
    if not registry.exists():
        raise FileNotFoundError(
            f"Cannot recompute hawk/dove asymmetry: {registry} missing"
        )

    # Walk the registry once and assemble ``text_hash -> patched
    # asymmetry`` under the patched lexicon. The hash-keyed dict folds
    # any sentence-level shards together by hash, mirroring the
    # ``_aggregate_corpus`` join the canonical feature builder uses.
    text_by_hash: dict[str, list[str]] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        text_hash = str(payload.get("text_hash", "") or "").strip()
        text = str(payload.get("text", "") or "")
        if not text_hash or not text:
            continue
        text_by_hash.setdefault(text_hash, []).append(text)

    original_hawk = linguistic_mod.HAWK_TOKENS
    original_dove = linguistic_mod.DOVE_TOKENS
    try:
        linguistic_mod.HAWK_TOKENS = patched_hawk
        linguistic_mod.DOVE_TOKENS = patched_dove
        patched_asymmetry: dict[str, float] = {}
        for text_hash, shards in text_by_hash.items():
            joined = "\n".join(shards)
            patched_asymmetry[text_hash] = linguistic_mod.hawk_dove_asymmetry(joined)
    finally:
        linguistic_mod.HAWK_TOKENS = original_hawk
        linguistic_mod.DOVE_TOKENS = original_dove

    frame = pd.read_parquet(canonical_parquet, engine="pyarrow")
    frame["hawk_dove_asymmetry"] = frame["text_hash"].map(patched_asymmetry).astype(float)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, engine="pyarrow", index=False)


def _invoke_per_family_baseline(
    *,
    training_package_id: str,
    seeds: list[int],
    folds: list[str] | None,
    epochs: int,
    output_json: Path,
) -> float:
    """Run the per-family ablation runner under the ``baseline`` cell.

    Shells out to :mod:`scripts.run_per_family_ablation` with
    ``--cells baseline`` so the smoke covers exactly one cell. Parses
    the resulting JSON and returns the cell's bootstrap mean macro-F1.
    """

    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.run_per_family_ablation",
        "--training-package-id",
        training_package_id,
        "--cells",
        "baseline",
        "--epochs",
        str(epochs),
        "--seeds",
        *[str(s) for s in seeds],
        "--output",
        str(output_json),
    ]
    if folds:
        cmd.extend(["--folds", *folds])
    subprocess.run(cmd, check=True)
    payload = json.loads(output_json.read_text())
    baseline_summary = payload["summary"]["baseline"]
    return float(baseline_summary["regime_f1_macro"]["mean"])


def _run_jackknife(
    *,
    training_package_id: str,
    seeds: list[int],
    folds: list[str] | None,
    epochs: int,
    package_dir: Path,
    tmp_artefact_dir: Path,
    macro_f1_for_lexicon: Callable[[frozenset[str], frozenset[str]], float],
) -> tuple[float, list[TokenJackknifeResult]]:
    """Walk every single-token entry in the union and collect deltas.

    ``macro_f1_for_lexicon`` is the per-cell driver. The production path
    swaps in :func:`_macro_f1_for_lexicon_real`; tests inject a stub
    that returns a deterministic synthetic delta vector.
    """

    from app.features.linguistic import DOVE_TOKENS, HAWK_TOKENS

    inventory = _build_token_inventory(HAWK_TOKENS, DOVE_TOKENS)

    baseline_f1 = macro_f1_for_lexicon(HAWK_TOKENS, DOVE_TOKENS)
    results: list[TokenJackknifeResult] = []
    for token, kind in inventory:
        patched_hawk, patched_dove = patched_lexicons(
            token, base_hawk=HAWK_TOKENS, base_dove=DOVE_TOKENS
        )
        without_f1 = macro_f1_for_lexicon(patched_hawk, patched_dove)
        results.append(
            TokenJackknifeResult(
                token=token,
                kind=kind,
                without_f1=without_f1,
                delta=without_f1 - baseline_f1,
            )
        )
    return baseline_f1, results


def _macro_f1_for_lexicon_real(
    *,
    training_package_id: str,
    seeds: list[int],
    folds: list[str] | None,
    epochs: int,
    package_dir: Path,
    tmp_artefact_dir: Path,
) -> Callable[[frozenset[str], frozenset[str]], float]:
    """Bind a per-cell driver that swaps the lexicon and shells out."""

    canonical_parquet = package_dir / "linguistic_features.parquet"
    backup_parquet = tmp_artefact_dir / "linguistic_features.canonical.parquet"
    tmp_artefact_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the start-state: either the canonical parquet exists (and
    # the backup is its byte-identical copy), or it does not (and the
    # restore step deletes the patched file rather than restoring from
    # a non-existent backup). Without the start-not-exists branch a
    # crash after the first swap would leave the canonical path holding
    # a patched parquet permanently.
    canonical_existed_at_startup = canonical_parquet.exists()
    if canonical_existed_at_startup and not backup_parquet.exists():
        backup_parquet.write_bytes(canonical_parquet.read_bytes())

    cell_counter = {"n": 0}

    def driver(patched_hawk: frozenset[str], patched_dove: frozenset[str]) -> float:
        cell_counter["n"] += 1
        patched_parquet = (
            tmp_artefact_dir / f"linguistic_features.patched_cell_{cell_counter['n']:03d}.parquet"
        )
        _rebuild_patched_parquet(
            package_dir,
            patched_hawk=patched_hawk,
            patched_dove=patched_dove,
            output_parquet=patched_parquet,
        )
        canonical_parquet.write_bytes(patched_parquet.read_bytes())
        try:
            cell_output = (
                tmp_artefact_dir / f"per_family_baseline_cell_{cell_counter['n']:03d}.json"
            )
            return _invoke_per_family_baseline(
                training_package_id=training_package_id,
                seeds=seeds,
                folds=folds,
                epochs=epochs,
                output_json=cell_output,
            )
        finally:
            if canonical_existed_at_startup and backup_parquet.exists():
                canonical_parquet.write_bytes(backup_parquet.read_bytes())
            elif canonical_parquet.exists():
                canonical_parquet.unlink()

    return driver


def _serialise_payload(
    *,
    training_package_id: str,
    baseline_f1: float,
    results: list[TokenJackknifeResult],
    threshold: float,
) -> dict[str, object]:
    return {
        "training_package_id": training_package_id,
        "baseline_f1": baseline_f1,
        "fragile_delta_threshold": threshold,
        "tokens": [
            {
                "token": r.token,
                "kind": r.kind,
                "without_f1": r.without_f1,
                "delta": r.delta,
            }
            for r in results
        ],
        "fragile_tokens": fragile_tokens_from_results(
            results, threshold=threshold
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package ID (local id or ``hf://datasets/...`` URI).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11],
        help="Seeds passed through to the per-family runner (default: 11).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Optional fold-id subset; defaults to every fold in the package.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epochs per cell (default 40; --smoke drops to 5).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for a fast pass (epochs=5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "``backend/artifacts/experiments/hawk_dove_jackknife.json``."
        ),
    )
    parser.add_argument(
        "--tmp-artefact-dir",
        type=Path,
        default=None,
        help=(
            "Scratch dir for per-cell parquet / JSON artefacts. Defaults "
            "to ``backend/artifacts/experiments/hawk_dove_jackknife_cells/``."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    return BACKEND_ROOT / "artifacts" / "experiments" / "hawk_dove_jackknife.json"


def _resolve_tmp_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    return BACKEND_ROOT / "artifacts" / "experiments" / "hawk_dove_jackknife_cells"


def main() -> int:
    from app.training.loaders import _resolve_training_package_dir

    args = _parse_args()
    epochs = 5 if args.smoke else int(args.epochs)
    output_path = _resolve_output_path(args.output)
    tmp_dir = _resolve_tmp_dir(args.tmp_artefact_dir)
    package_dir = _resolve_training_package_dir(args.training_package_id)

    driver = _macro_f1_for_lexicon_real(
        training_package_id=args.training_package_id,
        seeds=list(args.seeds),
        folds=list(args.folds) if args.folds else None,
        epochs=epochs,
        package_dir=package_dir,
        tmp_artefact_dir=tmp_dir,
    )
    baseline_f1, results = _run_jackknife(
        training_package_id=args.training_package_id,
        seeds=list(args.seeds),
        folds=list(args.folds) if args.folds else None,
        epochs=epochs,
        package_dir=package_dir,
        tmp_artefact_dir=tmp_dir,
        macro_f1_for_lexicon=driver,
    )
    payload = _serialise_payload(
        training_package_id=args.training_package_id,
        baseline_f1=baseline_f1,
        results=results,
        threshold=FRAGILE_DELTA_THRESHOLD,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[hawk_dove_jackknife] wrote {output_path}")
    print(
        f"[hawk_dove_jackknife] baseline_f1={baseline_f1:.4f} "
        f"fragile_tokens={payload['fragile_tokens']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
