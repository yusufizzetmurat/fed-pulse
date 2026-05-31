# Stage-2 findings (F HP-refine + F2 × bge) — 2026-05-31

Locked: --no-mp-surprise --use-doc-length --regime-loss focal. 40-sample 5-dim sweep (hidden/ralpha/gamma/dropout/lr).
Matrix best-balanced baseline: dual 0.4359 / cls 0.4241. (dropout/lr via local CLI tweak, uncommitted.)

## F — top 10 by geomean(dual,cls)
| rank | dual | cls | geomean | Δdual | hidden | ralpha | gamma | dropout | lr |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 0.4513 | 0.4307 | 0.4409 | +0.0154 | 64 | 0.7 | 1.0 | 0.3 | 0.001 |
| 2 | 0.4535 | 0.4239 | 0.4384 | +0.0176 | 32 | 0.7 | 1.0 | 0.3 | 0.001 |
| 3 | 0.4516 | 0.4239 | 0.4375 | +0.0157 | 32 | 0.3 | 1.0 | 0.3 | 0.001 |
| 4 | 0.4559 | 0.4173 | 0.4362 | +0.0200 | 32 | 0.3 | 2.0 | 0.3 | 0.001 |
| 5 | 0.4467 | 0.4239 | 0.4351 | +0.0108 | 32 | 0.5 | 1.0 | 0.3 | 0.001 |
| 6 | 0.4312 | 0.4377 | 0.4344 | -0.0047 | 64 | 0.7 | 1.5 | 0.3 | 0.0003 |
| 7 | 0.4440 | 0.4239 | 0.4338 | +0.0081 | 32 | 0.5 | 2.0 | 0.2 | 0.001 |
| 8 | 0.4360 | 0.4311 | 0.4335 | +0.0001 | 64 | 0.3 | 2.0 | 0.3 | 0.001 |
| 9 | 0.4293 | 0.4364 | 0.4328 | -0.0066 | 32 | 0.5 | 1.5 | 0.2 | 0.0003 |
| 10 | 0.4166 | 0.4468 | 0.4314 | -0.0193 | 32 | 0.3 | 2.0 | 0.2 | 0.0003 |

## F2 — top-5 F cells × bge
| config | dual | cls | geomean |
|--|--|--|--|
| h64_a0.7_g1.0_d0.3_l0.001 | 0.4698 | 0.4154 | 0.4418 |
| h32_a0.7_g1.0_d0.3_l0.001 | 0.4368 | 0.4066 | 0.4215 |
| h32_a0.3_g1.0_d0.3_l0.001 | 0.4446 | 0.4066 | 0.4252 |
| h32_a0.3_g2.0_d0.3_l0.001 | 0.4403 | 0.4253 | 0.4327 |
| h32_a0.5_g1.0_d0.3_l0.001 | 0.4575 | 0.4066 | 0.4313 |

## Reads
- F winner: h64_a0.7_g1.0_d0.3_l0.001 (dual 0.4513 / cls 0.4307 / gm 0.4409), beats matrix baseline geomean.
- Regime: small hidden (32-64) + dropout 0.3 + lr 1e-3 dominates.
- F2: bge on the best F cell (h64/a0.7/g1.0/d0.3/lr1e-3) -> dual 0.4698 (highest dual overall, +0.094 vs 0.3773 no-text baseline); best balanced gm 0.4418. No cell crossed 0.50 dual / 0.45 cls gates.
- JOB3 (VIX-on bake-off) NOT run: pod's canonical/canonical_vix events.parquet lack the 6 _VIX_FEATURE_COLUMNS (dataset rev 347d3e30 still serves old events sha c9a5e823, not f9d2bff2). Needs correct dataset-repo revision.
