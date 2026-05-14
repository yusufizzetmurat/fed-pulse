# Pseudo-label production-run + audit runbook

End-to-end procedure for running the chunk-aggregated teacher (PR #120) on the unlabelled scraped pool, sampling a stratified audit, hand-labelling it, and deciding whether the result enters the source registry as production pseudo-labels. Updated 2026-05-14 after the chunk-vote default landed.

The whole flow runs through `make` targets; this runbook is the prose explanation of what each target does and which decision points the user owns.

## 0. Prerequisites

- **Rebuild the backend image once** against the current `pyproject.toml`. If you haven't rebuilt since PR #71 (which added `pydantic-settings`), you'll hit `ModuleNotFoundError: No module named 'pydantic_settings'`. Fix: `docker compose --profile gpu build backend-gpu` (or `docker compose build backend` for the CPU service). Both images install from the same `pyproject.toml`, so the content is identical; pick the one you'll actually use.
- **The `make pseudo-labels*` targets default to the GPU service** (`backend-gpu` with `--profile gpu`) — a 9,696-row pass on CPU takes days. To run on CPU instead, override the variables: `make pseudo-labels PSEUDO_SERVICE=backend PSEUDO_PROFILE_FLAG=`.
- A FinBERT-FOMC seed-71 checkpoint exists under `data/artifacts/phase3/pilot_finetune_*/hf_checkpoints/`. The Makefile defaults `TEACHER_CHECKPOINT` to `/data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints`.
- `data/raw/phase2/source_registry.jsonl` has unlabelled rows ready to score. As of 2026-05-14 there are 92 scraped Fed rows in the registry; bringing the 9,696 Kaggle rows in requires re-running `make data-prep --all-sources` (which is gated on a Kaggle API key — see `app.data.ingest_sources --include-kaggle`).
- Wall-clock estimates on an RTX 4080: 92-row pool ~2-4 minutes, full 9,696-row pool ~4-6 hours.

## 1. Run the chunk-aggregated teacher

```bash
make pseudo-labels                                # uses chunk_vote default, tau_chunk=0.50, tau_doc=0.85
make pseudo-labels PSEUDO_STRATEGY=chunk_mean_pool  # conservative fallback if vote fails
```

This calls `python -m app.data.pseudo_labeling` inside the backend container. The output JSONL lands at `/data/interim/phase2/registry_pseudo_<strategy>.jsonl` (e.g. `registry_pseudo_chunk_vote.jsonl`).

Defaults the Makefile exposes (override via the env vars):

| Knob | Default | What it controls |
| --- | --- | --- |
| `TEACHER_CHECKPOINT` | `/data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints` | The fine-tuned FinBERT-FOMC seed 71 weights |
| `PSEUDO_STRATEGY` | `chunk_vote` | Aggregation rule: `chunk_vote` (modal label across kept chunks; default), `chunk_mean_pool` (conservative, per-class average), `chunk_max_pool` (collapses to hawkish on this corpus; see §2.5.8 of the DL roadmap), `doc_truncated` (legacy 512-token path; for audit reproduction only) |
| `PSEUDO_TAU_CHUNK` | `0.50` | Chunk-level confidence floor — chunks below this drop out of the aggregation |
| `PSEUDO_TAU_DOC` | `0.85` | Doc-level confidence floor — docs whose aggregated max_score < this are dropped from the pseudo set |
| `PSEUDO_AUDIT_SIZE` | `100` | Stratified audit-sample size (Wilson-95% CI ±0.07 at n=100, ±0.30 at n=10) |
| `SEED` | `11` | Audit sampler seed |

**Expected output:**

```
Pseudo-labelled rows written: <N>
Strategy: chunk_vote, tau_doc=0.85, tau_chunk=0.5
Output: /data/interim/phase2/registry_pseudo_chunk_vote.jsonl
```

`N` should be substantially less than the input row count — `tau_doc=0.85` is intentionally tight. Yield in the 10-30% range of input is normal at this threshold; if yield is >80% the gate is probably too loose.

## 2. Sample the stratified audit set

```bash
make pseudo-labels-audit-sample                  # uses default PSEUDO_AUDIT_SIZE=100
```

Writes `data/artifacts/pseudo_label_audits/audit_set_<strategy>_n<size>.csv` with one row per audit sample. Columns:

- `record_id` (unique key)
- `event_date`
- `source_type` (fomc_minutes / fomc_statement / …)
- `teacher_label` (what the chunk-aggregated teacher assigned)
- `teacher_max_score` (doc-level confidence)
- `text_excerpt` (first 500 chars of the document so you can read it without opening the full file)
- `human_label` (**empty — this is the column you fill in**)

The sampler is `app.data.llm_judge.sample_audit_set(rows, n=100, seed=11)`, stratified by `teacher_label` so the per-class precision is estimable (e.g. with a 60-30-10 hawkish/dovish/neutral split, the sample gets ≥10 of each class as long as that class is present in the pseudo set).

## 3. Hand-label the audit

Open `audit_set_<strategy>_n<size>.csv` in Excel / Sheets / Numbers / a text editor. For each row, read the excerpt (and the full document under `data/*.json` if needed) and write **one** of `hawkish` / `dovish` / `neutral` / `ambiguous` in the `human_label` column. Use `ambiguous` for rows where you genuinely cannot pick (the audit metrics handler excludes these from precision computation, so don't force a label).

Save the filled rows as **JSONL** (not CSV) at `data/artifacts/pseudo_label_audits/audit_set_<strategy>_filled.jsonl`. One JSON object per line; each object must carry at least `record_id`, `teacher_label`, and `human_label`. The other columns can stay.

A quick Excel → JSONL conversion in Python:

```python
import csv, json
with open("audit_set_chunk_vote_n100.csv") as fh, \
     open("audit_set_chunk_vote_filled.jsonl", "w") as out:
    for row in csv.DictReader(fh):
        out.write(json.dumps(row) + "\n")
```

The hand-labelling itself takes ~3-4 minutes per FOMC document (most of the time is reading the relevant policy-action section, not deciding). 100 rows = ~4-6 hours of focused work; spread across one or two afternoons.

## 4. Compute audit metrics

```bash
make pseudo-labels-audit-metrics                 # reads audit_set_<strategy>_filled.jsonl
```

Output is the dict returned by `app.data.llm_judge.audit_metrics(rows)`:

```json
{
  "audit_size": 100,
  "labelled_size": 95,                            // some rows may be 'ambiguous' or blank
  "teacher_accuracy": 0.78,                       // top-line metric — NOT the gate
  "judge_accuracy": 0.0,                          // empty when judge wasn't run
  "kappa_teacher_human": 0.62,                    // Cohen's κ
  "kappa_judge_human": 0.0,
  "kappa_teacher_judge": 0.0,
  "teacher_precision": {                          // PER-CLASS — this is the gate
    "hawkish": 0.92,
    "dovish": 0.81,
    "neutral": 0.55
  },
  "judge_precision": { ... }
}
```

## 5. Apply the gate

Per `docs/benchmark-policy.md §NLP Baseline Selection` + the audit-precision rule from the 2026-05-05 audit: **every class with non-trivial support must clear ≥0.90 precision**. Aggregate accuracy is not enough — a teacher that's 0.95 on hawkish and 0.40 on dovish flunks because the dovish slice is unreliable, even if the headline is 0.85.

| Outcome | What to do |
| --- | --- |
| All classes ≥ 0.90 precision | **Pass.** Add the pseudo set to the source registry: `python -m app.data.source_ingestion --include-pseudo /data/interim/phase2/registry_pseudo_<strategy>.jsonl` (the loader sets `label_origin=pseudo`). Rebuild the training package. Re-run the Phase-4 ablation against the expanded corpus and report the result alongside the original 2.4k-tuple numbers. |
| 2 of 3 classes ≥ 0.90; one class fails | **Conditional pass.** You can still use the pseudo set, but the failing class must be **excluded** from the pseudo rows (only ingest the rows the teacher assigned to the passing classes). Add a note in `dataset_metadata.json` explaining the partial use. The expanded corpus is then biased toward the passing classes — disclose this in the thesis. |
| 2 or more classes fail | **Fail.** Try the next strategy in the fallback chain: chunk_vote → chunk_mean_pool → judge_only (Gemini 2.5 Pro confirmation). If all three fail, the labelled-corpus expansion via pseudo-labelling is infeasible at this teacher quality. Document as a negative result in the thesis (this is publishable — see §2.5.8 of `06_Deep_Learning_Roadmap.md`). |

## 6. Re-run Phase-4 ablation against the expanded corpus

If the gate passes (full or conditional):

```bash
make data-prep DATASET_VERSION=<v_post_pseudo> FEATURE_VERSION=<v_post_pseudo> OWNER=<who>
# Note the new training_package_id from the output, then:
make train-batch TRAINING_PACKAGE_ID=<new_id> OWNER=<who>
docker compose run --rm backend python -m app.data.phase4_attention_ablation \
    --training-package-id <new_id> --variants A,B,C --seeds 11,29,47,71,97
```

Compare against the 2026-05-05 wf_fold_3 holdout (Variant A 0.00475, Variant B 0.21264 collapse, Variant C 0.00564) — the question is whether the larger labelled pool unblocks Variant B / C convergence. If yes, that's a thesis-changing finding. If not, the data-starvation diagnosis is reinforced with one more data point.

## 7. Update the wiki

Whatever the outcome:

- `01_Progress_Snapshot.md` — add a "Status update YYYY-MM-DD" entry with the audit precision numbers and the decision (pass / conditional pass / fail).
- `06_Deep_Learning_Roadmap.md §2.5.8` — extend the "Audit outcome" with the new precision tables and the v2 ablation result if the gate passed.
- `09_Risk_Register.md` R-14 (time-coverage gap) — if you ran the audit on post-2022 FOMC text and it passed, mark Path A as the chosen route and update the status.

## Judge-only audit (no human pass)

The user-vs-teacher audit above is the strictest gate but expensive (one focused afternoon of hand-labelling). The judge-only path drops the human pass entirely: the Gemini judge becomes the gold reference per `docs/benchmark-policy.md §Contamination handling`. The judge is architecturally distinct from the FinBERT-FOMC teacher (different family, different pretraining, different vocabulary), so its labels are a legitimate independent annotator.

**When to use this path:**

- Time pressure makes human labelling infeasible.
- You want a fast first signal on whether the chunk-aggregated teacher's pseudo-labels are even close to plausible.
- You want a methodology contribution ("we audited at N=K with an LLM judge as gold; teacher precision was X") without committing to a 4-6 hour human pass.

**When NOT to use this path:**

- The judge has its own bias (in the 2026-05-05 audit, Gemini was anti-hawkish at temperature 0). Don't treat judge-only precision as ground truth.
- For the published thesis result you still want a small human-pass on disagreements (judge-vs-teacher mismatched rows) so the conclusion isn't "two models agreed with each other".

### Workflow

```bash
# 1. Score the pseudo set with the Gemini judge. Writes
#    /data/interim/phase2/registry_pseudo_<strategy>_judged.jsonl
#    with judge_label / judge_confidence / judge_model_id per row.
make pseudo-labels-judge-pass
# Override knobs:
#   JUDGE_MODEL=gemini-2.5-flash JUDGE_REQUEST_INTERVAL=35 make pseudo-labels-judge-pass
#   (free-tier flash is 2 req/min so 35s spacing keeps under the cap)

# 2. Compute teacher-vs-judge precision + Cohen's kappa.
make pseudo-labels-audit-metrics-judge
```

Output is a JSON dict with the new fields:

```json
{
  "audit_size": 91,
  "gold_source": "judge_only",
  "teacher_judge_accuracy": 0.78,
  "cohen_kappa_teacher_judge": 0.55,
  "teacher_per_class": {
    "hawkish": {"precision": 0.94, "recall": 0.82, "tp": 31, "fp": 2, "fn": 7, "support_in_gold": 38},
    "dovish":  {"precision": 1.00, "recall": 0.75, "tp": 9,  "fp": 0, "fn": 3, "support_in_gold": 12},
    "neutral": {"precision": 0.83, "recall": 0.93, "tp": 38, "fp": 8, "fn": 3, "support_in_gold": 41}
  },
  "teacher_label_distribution": {"hawkish": 33, "dovish": 9, "neutral": 46},
  "judge_label_distribution":   {"hawkish": 38, "dovish": 12, "neutral": 41},
  "audit_gate_per_class":       {"hawkish": true, "dovish": true, "neutral": false},
  "audit_gate_passed": false,
  "judge_model_id_distribution": {"gemini-2.5-pro": 91}
}
```

(Numbers above are illustrative.)

The gate passes only if **every supported class** clears 0.90 teacher precision. Partial-pass interpretations live in `§5` above.

## Recommended path through the runbook

For your specific situation as of 2026-05-14 (chunk_vote is the default; the 9,696-row Kaggle pool is the main target):

1. `make data-prep --include-kaggle` (you need the Kaggle API key configured in `.env`).
2. `make pseudo-labels PSEUDO_STRATEGY=chunk_vote` — ~4-6 GPU hours.
3. `make pseudo-labels-audit-sample` — ~30 seconds.
4. Hand-label `audit_set_chunk_vote_n100.csv` over an afternoon or two.
5. `make pseudo-labels-audit-metrics` — instant.
6. Branch based on §5.
