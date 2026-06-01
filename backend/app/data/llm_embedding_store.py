"""LLM embedding precompute for Variant C of the forecaster text feature.

Reads the source registry, embeds each document's text with a Gemini
embedding model (one vector per document, not per chunk), and writes
the result as parquet keyed on document_id. Mirrors
chunk_embedding_store.py's surface but with a single-vector-per-doc
representation rather than per-chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from app.services.gemini_client import embed_text


from app.config import DATA_DIR as DEFAULT_DATA_DIR

DEFAULT_REGISTRY = DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "llm_embeddings.parquet"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Gemini embeddings (one per document) for Variant C of the forecaster."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def precompute_embeddings(
    *,
    registry_path: Path,
    output_path: Path,
    embedding_client: Any,
    max_rows: int = 0,
    request_interval_seconds: float = 0.0,
) -> int:
    """Embed each document in registry and persist as parquet.

    Returns the number of rows written. Skips documents with empty text.
    """

    rows = _read_jsonl(registry_path)
    if max_rows > 0:
        rows = rows[:max_rows]

    records: list[dict[str, Any]] = []
    eligible = [(i, r) for i, r in enumerate(rows) if str(r.get("text", "") or "")]
    for index, (_orig_index, row) in enumerate(eligible):
        text = str(row.get("text", "") or "")
        embedding = embed_text(text, model=embedding_client)
        records.append(
            {
                "document_id": str(row.get("record_id", "")),
                "event_date": str(row.get("event_date", "")),
                "embedding": embedding,
            }
        )
        if request_interval_seconds > 0 and index < len(eligible) - 1:
            time.sleep(request_interval_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame.from_records(records).to_parquet(output_path, index=False)
    except Exception:
        # Fallback: jsonl alongside the parquet path so the smoke does not
        # silently produce nothing if pandas/pyarrow are missing.
        jsonl_path = output_path.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")
    return len(records)


def main() -> int:
    args = _parse_args()
    from app.services.gemini_client import load_embedding_model

    client = load_embedding_model(args.embedding_model)
    written = precompute_embeddings(
        registry_path=Path(args.registry),
        output_path=Path(args.output),
        embedding_client=client,
        max_rows=args.max_rows,
        request_interval_seconds=args.request_interval_seconds,
    )
    print(f"LLM embeddings written: {written}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
