from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

MODEL_REGISTRY_PATH = Path(__file__).resolve().parent / "registry.yaml"

# URI prefix indicating an artefact hosted on Hugging Face Hub. The
# resolver below pulls these into the local HF cache on first use and
# returns the resolved path that ``from_pretrained`` consumes. Local
# filesystem paths in ``registry.yaml`` stay primary; ``hf://`` is the
# additional mode introduced by #302 so a fresh machine can boot the
# inference container without the ``/data/`` mount.
HF_URI_PREFIX = "hf://"
HF_DATASET_PREFIX = "hf://datasets/"


@dataclass(frozen=True)
class EncoderRef:
    alias: str
    repo: str
    revision: str
    gated: bool
    task: str
    description: str


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> dict[str, EncoderRef]:
    target = path or MODEL_REGISTRY_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    encoders = (raw or {}).get("encoders") or {}
    by_repo: dict[str, EncoderRef] = {}
    for alias, fields in encoders.items():
        if not isinstance(fields, dict):
            continue
        ref = EncoderRef(
            alias=alias,
            repo=str(fields["repo"]),
            revision=str(fields["revision"]),
            gated=bool(fields.get("gated", False)),
            task=str(fields.get("task", "classification")),
            description=str(fields.get("description", "")),
        )
        by_repo[ref.repo] = ref
        by_repo[ref.alias] = ref
        for repo_alias in fields.get("repo_aliases") or ():
            by_repo[str(repo_alias)] = ref
    return by_repo


def revision_for(repo_or_alias: str) -> str | None:
    registry = load_registry()
    ref = registry.get(repo_or_alias)
    if ref is None or not ref.revision:
        return None
    return ref.revision


def is_pinned(repo_or_alias: str) -> bool:
    ref = encoder_ref(repo_or_alias)
    return ref is not None and bool(ref.revision)


def encoder_ref(repo_or_alias: str) -> EncoderRef | None:
    return load_registry().get(repo_or_alias)


def from_pretrained_kwargs(repo_or_alias: str, **extra: Any) -> dict[str, Any]:
    revision = revision_for(repo_or_alias)
    if revision is None:
        raise ValueError(
            f"Model '{repo_or_alias}' is not pinned in models/registry.yaml. "
            "Add a revision before loading."
        )
    return {"revision": revision, **extra}


@dataclass(frozen=True)
class ArtefactRef:
    """One entry from the ``artefacts:`` block in ``registry.yaml``.

    These are the HF Hub mirrors of every model / dataset the inference
    container pulls at boot. ``eager=True`` artefacts are downloaded by
    the droplet entrypoint before uvicorn starts; ``eager=False`` rows
    are lazy-fetched on first request.
    """

    name: str
    hf_uri: str
    revision: str
    eager: bool
    description: str


@lru_cache(maxsize=1)
def load_artefacts(path: Path | None = None) -> dict[str, ArtefactRef]:
    target = path or MODEL_REGISTRY_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    block = raw.get("artefacts") or {}
    out: dict[str, ArtefactRef] = {}
    for name, fields in block.items():
        if not isinstance(fields, dict):
            continue
        out[name] = ArtefactRef(
            name=name,
            hf_uri=str(fields["hf_uri"]),
            revision=str(fields.get("revision") or ""),
            eager=bool(fields.get("eager", False)),
            description=str(fields.get("description", "")),
        )
    return out


def artefact_ref(name: str) -> ArtefactRef | None:
    return load_artefacts().get(name)


def eager_artefacts() -> list[ArtefactRef]:
    """Return the artefacts the boot entrypoint should pre-warm."""

    return [ref for ref in load_artefacts().values() if ref.eager]


@dataclass(frozen=True)
class HFRef:
    """Parsed ``hf://`` or ``hf://datasets/`` URI.

    ``repo_id`` is the ``owner/name`` slug, ``revision`` is the optional
    pinned commit / tag / branch, and ``repo_type`` is either ``model``
    or ``dataset``.
    """

    repo_id: str
    revision: str | None
    repo_type: str

    @property
    def is_dataset(self) -> bool:
        return self.repo_type == "dataset"


def is_hf_uri(value: str) -> bool:
    return value.startswith(HF_URI_PREFIX)


def parse_hf_uri(uri: str) -> HFRef:
    """Parse an ``hf://[datasets/]owner/name[:revision]`` URI.

    The grammar is intentionally narrow: a single ``:`` separates the
    repo id from an optional revision pin. Branch names containing
    colons are not supported (mirroring HF's own URI-style references).
    """

    if not is_hf_uri(uri):
        raise ValueError(f"Not an hf:// URI: {uri!r}")
    body = uri[len(HF_URI_PREFIX) :]
    repo_type = "model"
    if body.startswith("datasets/"):
        repo_type = "dataset"
        body = body[len("datasets/") :]
    if not body:
        raise ValueError(f"Empty hf:// URI body: {uri!r}")
    revision: str | None
    if ":" in body:
        repo_id, raw_revision = body.split(":", 1)
        revision = raw_revision or None
    else:
        repo_id, revision = body, None
    if "/" not in repo_id:
        raise ValueError(
            f"hf:// URI must be 'owner/name', got {repo_id!r} from {uri!r}"
        )
    return HFRef(repo_id=repo_id, revision=revision, repo_type=repo_type)


def resolve_hf_uri(
    uri: str,
    *,
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> Path:
    """Pull an ``hf://`` URI into the local HF cache and return its path.

    Calls :func:`huggingface_hub.snapshot_download` and returns the
    cache directory ``from_pretrained`` (or a parquet reader) can use
    directly. The HF token is taken from the explicit argument first,
    then from ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` env vars so the
    droplet entrypoint only needs to set one secret.
    """

    from huggingface_hub import snapshot_download

    ref = parse_hf_uri(uri)
    resolved_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    kwargs: dict[str, Any] = {
        "repo_id": ref.repo_id,
        "repo_type": ref.repo_type,
    }
    if ref.revision:
        kwargs["revision"] = ref.revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if resolved_token:
        kwargs["token"] = resolved_token
    local_path = snapshot_download(**kwargs)
    return Path(local_path)


def resolve_repo(
    repo_or_uri: str,
    *,
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> str:
    """Resolve a registry ``repo`` field to a path / repo id consumable by HF.

    - ``hf://owner/name[:rev]`` -> :func:`resolve_hf_uri` cache path
    - everything else (local path, plain HF repo id) -> passed through
    """

    if is_hf_uri(repo_or_uri):
        return str(resolve_hf_uri(repo_or_uri, cache_dir=cache_dir, token=token))
    return repo_or_uri
