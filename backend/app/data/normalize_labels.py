"""Map source labels into the canonical three-class stance taxonomy.

The current Trillion Dollar Words (gtfintechlab/fomc_communication)
integer mapping consumed throughout the pipeline is:

    class 0 -> dovish
    class 1 -> hawkish
    class 2 -> neutral

The mapping is pinned to ``MAPPING_VERSION = "label_map_v1.0"``; any
change to the integer-to-stance correspondence must bump the version
string and any artefact produced against the old mapping must be
re-emitted. Downstream metrics that depend on per-class names
(precision/recall tables, confusion matrices, the multi-axis stance axis)
key off this mapping; macro-F1 is invariant to the names, so a swap
inside the mapping is silent in macro-F1 alone.

The token lists below are case-folded synonyms each source labelling
convention has used at some point; the integer literals (``"0"``,
``"1"``, ``"2"``) are the canonical TDW codes.

(Why ``MAPPING_VERSION`` exists: an earlier revision of this file
flipped the class-1 / class-2 codes and trained the v0 fine-tunes
against the inverse mapping, producing artefacts whose hawkish-vs-
neutral labels were swapped at the API surface even though macro-F1 was
unchanged. ``MAPPING_VERSION`` is the contract that catches a future
regression of the same shape: any caller that persists labelled rows
should stamp them with this version so a future re-emit can detect a
mapping drift instead of silently relaying bad labels. See
``../../../fed-pulse.wiki/09_Risk_Register.md`` R-16.)
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

from app.config import DATA_DIR as DEFAULT_DATA_DIR
from app.data.label_schemas import sample_weight_for

DEFAULT_INPUT = DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_labeled.jsonl"
DEFAULT_EXCEPTIONS = DEFAULT_DATA_DIR / "interim" / "phase2" / "label_mapping_exceptions.json"
DEFAULT_META = DEFAULT_DATA_DIR / "interim" / "phase2" / "label_mapping_metadata.json"
MAPPING_VERSION = "label_map_v1.0"


# Canonical token sets for the three stance axes. The integer literals
# match the canonical TDW codes (see module docstring); the alphabetic
# tokens are case-folded synonyms accumulated across source conventions.
HAWKISH_TOKENS = {
    "hawkish",
    "hawk",
    "tightening",
    "restrictive",
    "positive",
    "label_1",
    "1",
}
DOVISH_TOKENS = {
    "dovish",
    "dove",
    "easing",
    "accommodative",
    "negative",
    "label_0",
    "0",
}
NEUTRAL_TOKENS = {
    "neutral",
    "mixed",
    "balanced",
    "label_2",
    "2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map source labels into deterministic 3-class taxonomy (hawkish/dovish/neutral)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Source registry JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output labeled registry JSONL.")
    parser.add_argument(
        "--exceptions-output",
        default=str(DEFAULT_EXCEPTIONS),
        help="Output JSON file for unmappable labels.",
    )
    parser.add_argument(
        "--metadata-output",
        default=str(DEFAULT_META),
        help="Output JSON file for mapping metadata.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _clean_label(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip().lower()
    value = value.replace("-", "_")
    return value


def _map_label(raw_label: str) -> str | None:
    label = _clean_label(raw_label)
    if not label:
        return None
    if label in HAWKISH_TOKENS:
        return "hawkish"
    if label in DOVISH_TOKENS:
        return "dovish"
    if label in NEUTRAL_TOKENS:
        return "neutral"

    # soft matching for compound labels
    if "hawk" in label or "tight" in label or "restrict" in label:
        return "hawkish"
    if "dov" in label or "ease" in label or "accommod" in label:
        return "dovish"
    if "neutral" in label or "mixed" in label or "balance" in label:
        return "neutral"
    return None


_PEER_REVIEWED_SOURCES = {
    "hf_fomc_communication",
    "fomc_communication",
    "trillion_dollar_words",
    "op_fed",
    "gss_factor",
    "aruoba_drechsel_shocks",
    "cieslak_schrimpf_news",
    "hansen_mcmahon_topics",
    "gtfintechlab_federal_reserve_system",
}
_KAGGLE_SOURCES = {"kaggle_fed_statements_minutes"}


def _provenance_for_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("provenance") or "").strip().lower()
    if explicit in {"peer_reviewed", "kaggle", "peer_reviewed_cross_bank", "scraped"}:
        return explicit
    source = str(row.get("source") or "").strip().lower()
    if source in _PEER_REVIEWED_SOURCES:
        return "peer_reviewed"
    if source in _KAGGLE_SOURCES:
        return "kaggle"
    return "scraped"


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_normalized_rows(rows: list[dict[str, Any]]) -> None:
    """Run ``NormalizedDocSchema`` on the labeled-registry frame.

    Lazy mode collects every column / row violation in one
    ``SchemaErrors``. ``FED_PULSE_SKIP_SCHEMA_VALIDATION=1`` bypasses
    validation for diagnostic re-runs.
    """

    if not rows:
        return
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return
    from app.data.schemas import NormalizedDocSchema, validate_frame

    frame = pd.DataFrame(rows)
    validate_frame(NormalizedDocSchema, frame)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _validate_normalized_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    warnings.warn(
        "app.data.normalize_labels is deprecated. Use app.data.label_normalization instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    exceptions_path = Path(args.exceptions_output)
    metadata_path = Path(args.metadata_output)

    rows = _read_jsonl(input_path)
    if not rows:
        print(f"No input rows found at {input_path}")
        return 1

    exceptions: list[dict[str, str]] = []
    counts = {"hawkish": 0, "dovish": 0, "neutral": 0, "unlabeled": 0, "unmapped": 0}

    for row in rows:
        raw_label = str(row.get("label", "")).strip()
        mapped = _map_label(raw_label) if raw_label else None
        row["mapped_label"] = mapped
        row["label_map_version"] = MAPPING_VERSION
        row["label_taxonomy"] = "hawkish_dovish_neutral"

        provenance = _provenance_for_row(row)
        row["provenance"] = provenance
        row["sample_weight"] = sample_weight_for(provenance) if mapped else 0.0
        # Audit Tier 1.8: ``axes`` was constructed by reading flat
        # ``axis_factor`` / ``axis_certainty`` / ``axis_topic`` columns
        # that no upstream source actually emits, so the dict was
        # ``{stance: <mapped>, factor: None, certainty: None, topic: None}``
        # on every row even when the upstream had the data. The
        # gtfintechlab and Op-Fed ingesters park their per-axis labels in
        # ``multi_axis_extras`` (gtfintechlab_time_label,
        # gtfintechlab_certain_label, op_fed_stance_nli, gss_target_factor,
        # gss_path_factor, ...). Lift those into ``axes`` so the
        # downstream event-row builder, multi-axis slot, and any future
        # per-topic stance head actually see them. The flat ``axis_*``
        # path stays as a fallback for sources that one day emit them.
        extras = row.get("multi_axis_extras") or {}
        if not isinstance(extras, dict):
            extras = {}
        factor_value = _coerce_optional_float(
            row.get("axis_factor")
            if row.get("axis_factor") is not None
            else extras.get("gss_target_factor")
        )
        certainty_value = _coerce_optional_str(
            row.get("axis_certainty")
            if row.get("axis_certainty") is not None
            else extras.get("gtfintechlab_certain_label")
        )
        time_value = _coerce_optional_str(extras.get("gtfintechlab_time_label"))
        topic_value = _coerce_optional_str(row.get("axis_topic"))
        row["axes"] = {
            "stance": mapped,
            "factor": factor_value,
            "certainty": certainty_value,
            "time": time_value,
            "topic": topic_value,
        }

        if not raw_label:
            counts["unlabeled"] += 1
            continue
        if mapped is None:
            counts["unmapped"] += 1
            exceptions.append(
                {
                    "record_id": str(row.get("record_id", "")),
                    "source": str(row.get("source", "")),
                    "raw_label": raw_label,
                }
            )
        else:
            counts[mapped] += 1

    _write_jsonl(output_path, rows)
    exceptions_path.parent.mkdir(parents=True, exist_ok=True)
    exceptions_path.write_text(json.dumps({"exceptions": exceptions}, indent=2), encoding="utf-8")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "label_map_version": MAPPING_VERSION,
                "label_taxonomy": "hawkish_dovish_neutral",
                "counts": counts,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "exceptions_path": str(exceptions_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Labeled registry written to {output_path}")
    print(f"Mapping exceptions written to {exceptions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

