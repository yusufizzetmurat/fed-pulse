# Security acceptance — outstanding advisories

CI runs `npm audit` and `pip-audit` on every PR. CI fails on **critical**
findings; **high** and **moderate** findings stay in the report but do not
block merges as long as the accepted-risk list below matches.

This file is the record of what we've explicitly decided to live with and the
reasoning behind each decision. When a fix lands or the risk model changes,
remove the entry and let CI gate on it again.

## Accepted high-severity findings (npm)

| Package | Advisories | Why accepted | Mitigation owner / next step |
|---|---|---|---|
| `next` 14.2.x | DoS via image optimizer (`GHSA-9g9p-9gw9-jx7f`, `GHSA-3x4c-7xq6-9pq8`, `GHSA-h64f-5h5j-jqjh`), request smuggling in rewrites (`GHSA-ggv3-7p47-pfv8`), SSR-component DoS (`GHSA-h25m-26qc-wcjf`, `GHSA-q4gf-8mx6-v5v3`, `GHSA-8h8q-6873-q5fj`), CSP-nonce / beforeInteractive XSS (`GHSA-ffhc-5mcf-pf4q`, `GHSA-gx5p-jg67-6x7h`), cache poisoning in RSC + redirects (`GHSA-vfv6-92ff-j949`, `GHSA-wfc6-r584-vfw7`, `GHSA-3g8h-86w9-wvmq`), Middleware bypass in Pages Router i18n (`GHSA-36qx-fr4f-26g5`), SSRF in WebSocket upgrade (`GHSA-c4j6-fc7j-m34r`) | Self-hosted research dashboard, not internet-exposed, no untrusted-tenant traffic. Fixes require Next 16 (App Router migration). | Tracked under the planned "Next 16 + App Router" migration window after the thesis is submitted. |
| `postcss` <8.5.10 | `</style>` XSS in stringify (`GHSA-qx2v-qp2m-jg93`) | Transitive via Next's bundled postcss; user-supplied CSS is never run through it in this codebase (we author every CSS source file). | Resolves when Next 16 lands. |

## Accepted moderate-severity findings

None right now. `follow-redirects` and `axios` highs were resolved by bumping
the lockfile.

## How to update this list

1. Run `make audit-npm` / `make audit-python` to surface the current report.
2. For a new finding, decide: fix now (preferred), accept with a documented
   reason (this file), or downgrade CI threshold (last resort).
3. Re-run CI to confirm the new state matches the table above.
