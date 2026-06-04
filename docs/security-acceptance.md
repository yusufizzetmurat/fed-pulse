# Security acceptance — outstanding advisories

CI runs `npm audit` and `pip-audit` on every PR. CI fails on **critical**
findings; **high** and **moderate** findings stay in the report but do not
block merges as long as the accepted-risk list below matches.

This file records each advisory that has been explicitly accepted and the
reasoning attached to it. When a fix lands or the risk model changes, the
entry is removed and CI gates on it again.

## Accepted high-severity findings (npm)

| Package | Advisories | Why accepted | Mitigation owner / next step |
|---|---|---|---|
| `next` 14.2.x | DoS via image optimizer (`GHSA-9g9p-9gw9-jx7f`, `GHSA-3x4c-7xq6-9pq8`, `GHSA-h64f-5h5j-jqjh`), request smuggling in rewrites (`GHSA-ggv3-7p47-pfv8`), SSR-component DoS (`GHSA-h25m-26qc-wcjf`, `GHSA-q4gf-8mx6-v5v3`, `GHSA-8h8q-6873-q5fj`), CSP-nonce / beforeInteractive XSS (`GHSA-ffhc-5mcf-pf4q`, `GHSA-gx5p-jg67-6x7h`), cache poisoning in RSC + redirects (`GHSA-vfv6-92ff-j949`, `GHSA-wfc6-r584-vfw7`, `GHSA-3g8h-86w9-wvmq`), Middleware bypass in Pages Router i18n (`GHSA-36qx-fr4f-26g5`), SSRF in WebSocket upgrade (`GHSA-c4j6-fc7j-m34r`) | Self-hosted research dashboard. The live deployment at https://fedpulse.yusufizzetmurat.com/ does not serve untrusted-tenant traffic. Fixes require Next 16 (App Router migration). | Tracked under the Next 16 + App Router migration window. |
| `postcss` <8.5.10 | `</style>` XSS in stringify (`GHSA-qx2v-qp2m-jg93`) | Transitive via Next's bundled postcss. User-supplied CSS is never run through it in this codebase; every CSS source file is authored in-tree. | Resolves when Next 16 lands. |

## Accepted findings (pip)

| Package | Advisories | Why accepted | Mitigation owner / next step |
|---|---|---|---|
| `transformers` 4.57.6 | `PYSEC-2025-217`, `GHSA-69w3-r845-3855` | Both advisories' fix lands in `5.0.0rc3`. The 5.x line dropped `GenerationMixin` from `transformers.models.auto.auto_factory`, which broke the encoder loader on the post_306 Runpod sweep. The pin `>=4.57,<5` keeps the loader working; the CVEs ride on the pin. The advisories cover code paths (a custom safetensors-deserialization edge case and the remote-code-loading path) that this codebase does not exercise: `trust_remote_code` is `False` at every loader call site, and checkpoints are pulled from a known-clean HF repo owned by the project. | Resolves when either (a) a 4.x backport ships, or (b) the encoder loader is migrated to the 5.x AutoFactory layout. Tracked under #409. |

## Accepted moderate-severity findings

None right now. `follow-redirects` and `axios` highs were resolved by bumping
the lockfile.

## How to update this list

1. Run `make audit-npm` / `make audit-python` to surface the current report.
2. For a new finding, decide: fix now (preferred), accept with a documented
   reason (this file), or downgrade CI threshold (last resort).
3. Re-run CI to confirm the new state matches the table above.
