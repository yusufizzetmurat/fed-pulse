# Run Templates

## Purpose
This file provides lightweight templates so each official run can be reproduced and audited later.

## 1) Pre-Run Checklist Template

Use this before launching a run:

- `run_id`:
- `owner`:
- `timestamp_utc`:
- `evaluation_protocol`:
- `dataset_version`:
- `feature_version`:
- `training_package_id`:
- `model_version`:
- `runtime_mode` (`fast` or `quick_train`):
- `seed`:
- `fold_scope`:
- `horizons`:

Sanity checks:
- data package exists and is immutable
- fold manifest matches protocol
- no unresolved leakage warnings
- model checkpoint version is pinned

## 2) Post-Run Report Template

Complete this immediately after a run:

- `run_id`:
- `completed_at_utc`:
- `status` (`success` or `failed`):
- `failure_reason` (if failed):

Metrics summary:
- RMSE (mean/std across folds/seeds):
- MAE (mean/std across folds/seeds):
- MAPE (mean/std across folds/seeds):
- Macro-F1 (for NLP baselines):
- Weighted-F1:
- latency p50:
- latency p95:

Observations:
- strongest regime:
- weakest regime:
- anomalies worth investigating:

## 3) Artifact Manifest Template

Track exactly what was produced:

- `run_id`:
- `artifact_root`:
- `model_artifact_path`:
- `metrics_file_path`:
- `fold_metrics_path`:
- `logs_path`:
- `plots_path`:
- `config_snapshot_path`:

Checksums (recommended):
- `metrics_sha256`:
- `model_sha256`:
- `config_sha256`:

## 4) Minimal Acceptance Gate

A run is considered report-ready only when:

1. required identifiers are complete (`dataset_version`, `feature_version`, `model_version`, `run_id`, `training_package_id`)
2. metrics are present and parseable
3. artifacts are stored and path-valid
4. no unresolved leakage or split-policy violations
# Run Templates

These templates are intentionally short so they can be copied into run logs without heavy editing.

## 1) Pre-Run Configuration

### Run Header
- `run_id`:
- `evaluation_protocol`: `evaluation_protocol_v1`
- `dataset_version`:
- `feature_version`:
- `model_version`:
- `runtime_mode`:
- `seed`:
- `owner`:
- `started_at_utc`:

### Data + Split
- source datasets:
- final record count:
- split strategy: expanding walk-forward
- fold count:
- horizon set: `1d`, `3d`, `5d`, `10d`
- excluded ranges (if any):

### Model Setup
- model family:
- architecture parameters:
- optimizer and learning rate:
- loss function:
- regularization:
- adaptation policy (if enabled):

### Runtime Constraints
- max training/adaptation budget:
- expected latency target (`p50`, `p95`):
- hardware profile:

### Leakage Validation
- no future-aware feature generation: [ ] pass [ ] fail
- scaler fit on train only: [ ] pass [ ] fail
- duplicate/source leakage scan complete: [ ] pass [ ] fail
- pseudo-label holdout isolation verified (if applicable): [ ] pass [ ] fail

### Approval
- reviewer:
- approved: [ ] yes [ ] no
- comments:

---

## 2) Post-Run Report

### Run Identity
- `run_id`:
- `evaluation_protocol`: `evaluation_protocol_v1`
- `dataset_version`:
- `feature_version`:
- `model_version`:
- `runtime_mode`:
- `seed`:
- `ended_at_utc`:

### Execution Summary
- status: [ ] completed [ ] failed [ ] aborted
- failure reason (if any):
- total runtime:

### Aggregate Metrics
- RMSE:
- MAE:
- MAPE:
- coverage (if available):
- calibration error (if available):
- latency `p50`:
- latency `p95`:
- adaptation overhead (if applicable):

### Per-Horizon Table
| Horizon | RMSE | MAE | MAPE | Coverage | Notes |
|---|---:|---:|---:|---:|---|
| `1d` |  |  |  |  |  |
| `3d` |  |  |  |  |  |
| `5d` |  |  |  |  |  |
| `10d` |  |  |  |  |  |

### Fold/Seed Stats
- fold count:
- mean across seeds:
- std across seeds:
- outlier seeds or folds:

### Compliance
- protocol-compliant: [ ] yes [ ] no
- versioning-compliant: [ ] yes [ ] no
- leakage checks passed: [ ] yes [ ] no

### Interpretation
- key observations:
- known limitations:
- follow-up actions:

---

## 3) Artifact Manifest

### Manifest Header
- `run_id`:
- `evaluation_protocol`: `evaluation_protocol_v1`
- generated_at_utc:
- owner:

### Required IDs
- `dataset_version`:
- `feature_version`:
- `model_version`:

### Produced Artifacts
| Artifact Type | Path | Checksum/Hash | Notes |
|---|---|---|---|
| checkpoint |  |  |  |
| metrics_json |  |  |  |
| metrics_csv |  |  |  |
| run_log |  |  |  |
| plots |  |  |  |
| config_snapshot |  |  |  |

### Registry Mapping
- `model_registry` reference:
- `experiment_runs` reference:
- `online_predictions` reference (if applicable):

### Retention
- retained for benchmark table: [ ] yes [ ] no
- promoted to active model candidate: [ ] yes [ ] no
- archival policy tag:

### Integrity
- all required files present: [ ] yes [ ] no
- hashes verified: [ ] yes [ ] no
- metadata complete: [ ] yes [ ] no
- paths reproducible in repo/container context: [ ] yes [ ] no
