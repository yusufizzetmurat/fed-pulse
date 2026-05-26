"""Proxy-task warm-start for the text-embedding adapter (issue #327).

The forecaster's ``TextEmbeddingAdapter`` is zero-initialised by
default so an activated text path forwards to the same point in
feature space as the rich-features-only baseline. That makes the
broadcast-static A/B comparison clean, but it also means the text
adapter has no signal-aware starting point: every gradient through
the adapter is the first non-zero gradient it sees.

This module fits the adapter on a proxy stance task ("given the
pooled FOMC text embedding, predict {hawkish, dovish, neutral}") and
persists the warmed weights to disk. The forecaster CLI then loads
the persisted state_dict into the adapter at construction time, so
the text path has a small but signal-aware init instead of zeros from
epoch 0.

The proxy task is intentionally narrow:

- One linear classifier head ``adapter_dim -> 3``.
- One epoch budget defaulting to 20 (the corpus is small).
- Adam optimiser, fixed LR, no warmup; the goal is to lift the
  adapter weights off zero, not to chase macro-F1.

The output checkpoint is a plain torch ``state_dict`` round-trippable
via :func:`torch.load`. The forecaster reads only the keys whose
prefixes match the ``text_adapter.*`` submodule; head weights are
discarded by the loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from app.models.text_embedding_adapter import TextEmbeddingAdapter

STANCE_LABEL_MAP: dict[str, int] = {"hawkish": 0, "dovish": 1, "neutral": 2}


class TextAdapterProxyClassifier(nn.Module):
    """Adapter + linear classifier head for the proxy stance task."""

    def __init__(self, in_dim: int, adapter_dim: int, n_classes: int = 3) -> None:
        super().__init__()
        # zero_init=False is the deliberate departure from the
        # forecaster's default. The whole point of warm-starting is to
        # leave the adapter at a non-zero point in weight space, so a
        # zero-initialised proxy classifier would defeat the exercise.
        self.adapter = TextEmbeddingAdapter(
            in_dim=in_dim, out_dim=adapter_dim, zero_init=False
        )
        self.classifier = nn.Linear(adapter_dim, n_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        projected = self.adapter(pooled)
        return cast(torch.Tensor, self.classifier(projected))


def _load_corpus_rows(corpus_path: Path) -> list[dict[str, Any]]:
    """Read the proxy-task corpus from a JSON / JSONL / parquet file.

    Accepts a list-of-dicts JSON, a JSONL stream, or a parquet file
    with columns ``text_embedding_pooled`` and ``stance_label``. The
    parquet path lazy-imports pandas / pyarrow so the rest of the
    backend stays import-clean when those libraries are not installed.
    """

    if not corpus_path.exists():
        raise FileNotFoundError(f"warm-start corpus not found: {corpus_path}")
    suffix = corpus_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [row for row in payload["rows"] if isinstance(row, dict)]
        raise ValueError(
            f"warm-start corpus JSON shape not recognised: {corpus_path}"
        )
    if suffix == ".parquet":
        import pandas as pd  # noqa: PLC0415 -- optional dependency

        frame = pd.read_parquet(corpus_path)
        return cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    raise ValueError(
        f"warm-start corpus suffix not supported: {suffix!r} ({corpus_path})"
    )


def _row_payload(row: dict[str, Any]) -> tuple[list[float], int] | None:
    """Extract the pooled embedding + stance index from a corpus row."""

    pooled = row.get("text_embedding_pooled")
    if pooled is None:
        pooled = row.get("pooled")
    if not isinstance(pooled, list) or not pooled:
        return None
    label = row.get("stance_label")
    if label is None:
        label = row.get("axis_stance")
    if isinstance(label, str):
        label_idx = STANCE_LABEL_MAP.get(label.lower())
        if label_idx is None:
            return None
    elif isinstance(label, int):
        if label not in {0, 1, 2}:
            return None
        label_idx = int(label)
    else:
        return None
    return [float(v) for v in pooled], int(label_idx)


def pretrain_text_adapter(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    adapter_dim: int = 64,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 11,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Fit the text adapter on a (pooled_embedding -> stance) proxy task.

    The function reads the corpus, builds a small classifier on top of
    a fresh (non-zero-init) :class:`TextEmbeddingAdapter`, trains for
    ``epochs`` epochs, and persists the adapter's state_dict +
    metadata to ``output_path`` so the forecaster CLI can load it into
    its own ``text_adapter`` submodule at construction time.

    Returns a metadata dict with the resolved hyperparameters and the
    final training loss for logging.
    """

    corpus_path_p = Path(corpus_path)
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_corpus_rows(corpus_path_p)
    pooled_rows: list[list[float]] = []
    label_rows: list[int] = []
    in_dim = 0
    for row in rows:
        payload = _row_payload(row)
        if payload is None:
            continue
        pooled, label_idx = payload
        if in_dim == 0:
            in_dim = len(pooled)
        if len(pooled) != in_dim:
            continue
        pooled_rows.append(pooled)
        label_rows.append(label_idx)
    if not pooled_rows:
        raise ValueError(
            f"warm-start corpus produced no (pooled, stance) rows: {corpus_path_p}"
        )

    torch.manual_seed(int(seed))
    device_obj = torch.device(device)
    pooled_tensor = torch.tensor(pooled_rows, dtype=torch.float32, device=device_obj)
    label_tensor = torch.tensor(label_rows, dtype=torch.long, device=device_obj)
    n_classes = int(label_tensor.max().item()) + 1
    n_classes = max(n_classes, 3)
    model = TextAdapterProxyClassifier(
        in_dim=in_dim, adapter_dim=int(adapter_dim), n_classes=n_classes
    ).to(device_obj)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    n_samples = pooled_tensor.shape[0]
    indices = torch.arange(n_samples, device=device_obj)
    last_epoch_loss = 0.0
    for epoch in range(int(epochs)):
        # Deterministic per-epoch shuffle using torch's seeded RNG.
        permutation = torch.randperm(n_samples, device=device_obj)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, n_samples, int(batch_size)):
            end = min(start + int(batch_size), n_samples)
            batch_idx = permutation[start:end]
            optimiser.zero_grad(set_to_none=True)
            logits = model(pooled_tensor[batch_idx])
            loss = F.cross_entropy(logits, label_tensor[batch_idx])
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        last_epoch_loss = epoch_loss / max(batch_count, 1)

    # Persist the adapter's state_dict only. The classifier head was a
    # proxy-task scaffold and should not bleed into the forecaster.
    adapter_state = {
        f"text_adapter.{key}": value.detach().cpu()
        for key, value in model.adapter.state_dict().items()
    }
    metadata: dict[str, Any] = {
        "in_dim": int(in_dim),
        "adapter_dim": int(adapter_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "final_epoch_loss": float(last_epoch_loss),
        "task": "stance_proxy_classification",
        "indices_sampled": int(indices.numel()),
    }
    checkpoint_payload: dict[str, Any] = {
        "state_dict": adapter_state,
        "metadata": metadata,
    }
    torch.save(checkpoint_payload, output_path_p)
    return metadata


def load_warm_start_into_adapter(
    adapter: TextEmbeddingAdapter,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load a warm-start checkpoint into an existing adapter.

    Reads the persisted ``state_dict`` and copies the ``text_adapter.*``
    entries onto ``adapter``. Returns the metadata block so the
    forecaster training loop can log the warm-start origin onto the
    run summary. Raises ``ValueError`` when the persisted ``in_dim``
    or ``adapter_dim`` does not match the live adapter.
    """

    checkpoint_path_p = Path(checkpoint_path)
    if not checkpoint_path_p.exists():
        raise FileNotFoundError(
            f"warm-start checkpoint not found: {checkpoint_path_p}"
        )
    payload = torch.load(checkpoint_path_p, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    persisted_in = int(metadata.get("in_dim", 0) or 0)
    persisted_out = int(metadata.get("adapter_dim", 0) or 0)
    if persisted_in and persisted_in != adapter.in_dim:
        raise ValueError(
            "warm-start in_dim mismatch: checkpoint="
            f"{persisted_in}, adapter={adapter.in_dim}"
        )
    if persisted_out and persisted_out != adapter.out_dim:
        raise ValueError(
            "warm-start adapter_dim mismatch: checkpoint="
            f"{persisted_out}, adapter={adapter.out_dim}"
        )
    raw_state = payload.get("state_dict") or {}
    rekeyed: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if key.startswith("text_adapter."):
            rekeyed[key[len("text_adapter.") :]] = value
        else:
            rekeyed[key] = value
    adapter.load_state_dict(rekeyed, strict=strict)
    return metadata


__all__ = [
    "STANCE_LABEL_MAP",
    "TextAdapterProxyClassifier",
    "load_warm_start_into_adapter",
    "pretrain_text_adapter",
]
