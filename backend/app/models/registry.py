from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

# Canonical encoder roles (ADR 0019). Two encoders win the canonical
# slots: the FOMC-only DAPT substrate as the headline classifier, and
# the cross-bank DAPT encoder as the retrieval base. The pre-#330
# arrangement pinned the cross-bank DAPT substrate as the single
# canonical encoder despite Bundle A.2 / A.4 returning null on the
# vol-regime classifier — one substrate doing two jobs poorly.
EncoderRole = Literal["classifier", "retrieval"]
KNOWN_ROLES: tuple[str, ...] = ("classifier", "retrieval")

# HF Hub repo-id format: ``owner/name`` where each side starts with an
# alphanumeric and may contain ``[a-zA-Z0-9_.-]``. The grammar matches
# what huggingface_hub itself accepts; the stricter point is that paths
# like ``../../etc/passwd`` or ``owner/`` are rejected outright at the
# resolver boundary instead of being passed to ``snapshot_download``.
_HF_REPO_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*$"
)
# Revision pin: commit sha, tag, or branch. No path separators, no ``..``.
_HF_REVISION_RE = re.compile(r"^[a-zA-Z0-9._-]+$")

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
    # Canonical role tag (ADR 0019). ``None`` for entries that do not
    # claim either canonical slot — bake-off siblings, control ablations,
    # placeholder rows. Two tagged entries ship: ``classifier`` for the
    # headline substrate, ``retrieval`` for the retrieval base.
    role: str | None = None
    # Inference-feature aliases the encoder contributes to a serving
    # forecaster (#341). Empty tuple for encoders that have never been
    # threaded into the serving forward path (bake-off siblings,
    # placeholder rows). The serving loader cross-checks the
    # checkpoint's inference contract against this set so a registry
    # that drops a feature mid-flight refuses to bind a checkpoint
    # trained against the old declaration.
    inference_features: tuple[str, ...] = ()
    # #548 opt-in for HF encoders that ship custom modeling code
    # (e.g. nomic-ai/nomic-bert-2048). Default False so every existing
    # AutoConfig / AutoModel load stays in the standard transformers
    # path. Flip to True at the registry row only after a security
    # review of the repo — the flag tells transformers it is OK to
    # download and execute Python from the HF repo at load time.
    trust_remote_code: bool = False


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> dict[str, EncoderRef]:
    target = path or MODEL_REGISTRY_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    encoders = (raw or {}).get("encoders") or {}
    by_repo: dict[str, EncoderRef] = {}
    for alias, fields in encoders.items():
        if not isinstance(fields, dict):
            continue
        raw_role = fields.get("role")
        raw_features = fields.get("inference_features") or ()
        ref = EncoderRef(
            alias=alias,
            repo=str(fields["repo"]),
            revision=str(fields["revision"]),
            gated=bool(fields.get("gated", False)),
            task=str(fields.get("task", "classification")),
            description=str(fields.get("description", "")),
            role=str(raw_role) if raw_role is not None else None,
            inference_features=tuple(str(v) for v in raw_features),
            trust_remote_code=bool(fields.get("trust_remote_code", False)),
        )
        by_repo[ref.repo] = ref
        by_repo[ref.alias] = ref
        for repo_alias in fields.get("repo_aliases") or ():
            by_repo[str(repo_alias)] = ref
    return by_repo


def resolve_by_role(role: EncoderRole) -> str:
    """Return the canonical encoder alias for the given role (ADR 0019).

    Two roles are recognised: ``classifier`` (headline classification
    substrate) and ``retrieval`` (retrieval base). The resolver scans
    the encoder block once, dedup-keyed by ``alias`` so callers see
    each registered entry once, and returns the first ``alias`` whose
    ``role:`` matches.

    Raises :class:`KeyError` when the role is unknown (not in
    :data:`KNOWN_ROLES`) or when no encoder in the registry carries
    that role tag. Callers that want to gracefully fall back to a
    legacy default should resolve through :func:`encoder_ref` or a
    callsite-specific default constant, not through this function.
    """

    if role not in KNOWN_ROLES:
        raise KeyError(
            f"unknown encoder role {role!r}; known roles: {KNOWN_ROLES}"
        )
    registry = load_registry()
    # Dedup by alias — the loader fans an entry out across ``repo`` /
    # ``alias`` / ``repo_aliases`` keys, so scanning ``.values()``
    # directly would visit the same ref multiple times.
    seen: set[str] = set()
    for ref in registry.values():
        if ref.alias in seen:
            continue
        seen.add(ref.alias)
        if ref.role == role:
            return ref.alias
    raise KeyError(
        f"no encoder with role={role!r} registered in models/registry.yaml; "
        f"known roles: {KNOWN_ROLES}"
    )


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
    # #341: inference-feature aliases the artefact contributes when the
    # serving container binds it. Empty for non-forecaster artefacts
    # (training package, embedding caches, retrieval bundle).
    inference_features: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def load_artefacts(path: Path | None = None) -> dict[str, ArtefactRef]:
    target = path or MODEL_REGISTRY_PATH
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    block = raw.get("artefacts") or {}
    out: dict[str, ArtefactRef] = {}
    for name, fields in block.items():
        if not isinstance(fields, dict):
            continue
        raw_features = fields.get("inference_features") or ()
        out[name] = ArtefactRef(
            name=name,
            hf_uri=str(fields["hf_uri"]),
            revision=str(fields.get("revision") or ""),
            eager=bool(fields.get("eager", False)),
            description=str(fields.get("description", "")),
            inference_features=tuple(str(v) for v in raw_features),
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

    ``repo_id`` is validated against the HF Hub format
    (``[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*``) so
    pathological inputs (path traversal like ``../../etc/passwd``,
    multi-colon URIs, trailing slashes, empty revisions) raise
    :class:`ValueError` at the boundary instead of being forwarded to
    :func:`huggingface_hub.snapshot_download` where a malformed value
    could escape into the local filesystem cache.
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
    if body.count(":") > 1:
        raise ValueError(
            f"hf:// URI carries multiple ':' separators; only owner/name:revision is allowed: {uri!r}"
        )
    if ":" in body:
        repo_id, raw_revision = body.split(":", 1)
        if raw_revision == "":
            raise ValueError(
                f"hf:// URI has trailing ':' with empty revision: {uri!r}"
            )
        revision = raw_revision
    else:
        repo_id, revision = body, None
    if repo_id.endswith("/"):
        raise ValueError(f"hf:// URI repo_id has trailing slash: {uri!r}")
    if not _HF_REPO_ID_RE.match(repo_id):
        raise ValueError(
            f"hf:// URI repo_id {repo_id!r} does not match owner/name "
            f"format (alphanumeric start, [a-zA-Z0-9_.-]*); URI={uri!r}"
        )
    if revision is not None and not _HF_REVISION_RE.match(revision):
        raise ValueError(
            f"hf:// URI revision {revision!r} contains illegal characters "
            f"(allowed: [a-zA-Z0-9._-]+); URI={uri!r}"
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
