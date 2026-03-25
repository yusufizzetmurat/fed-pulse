# Run Templates

Use these templates for every official run.

## 1) Pre-Run

- `run_id`:
- `owner`:
- `started_at_utc`:
- `evaluation_protocol`:
- `dataset_version`:
- `feature_version`:
- `training_package_id`:
- `model_version`:
- `runtime_mode`:
- `seed`:
- `horizons`:

Checks:
- [ ] fold manifest matches protocol
- [ ] no unresolved leakage warnings
- [ ] checkpoint/version pinned

## 2) Post-Run

- `run_id`:
- `ended_at_utc`:
- `status`:
- `failure_reason` (if failed):

Metrics:
- RMSE:
- MAE:
- MAPE:
- Macro-F1 / Weighted-F1 (if NLP baseline):
- latency `p50`:
- latency `p95`:

Notes:
- strongest regime:
- weakest regime:
- follow-up action:

## 3) Artifact Manifest

- `run_id`:
- `artifact_root`:
- `model_artifact_path`:
- `metrics_path`:
- `logs_path`:
- `plots_path`:
- `config_snapshot_path`:
- `model_sha256`:
- `metrics_sha256`:

## Acceptance Gate
A run is report-ready only if:
1. required IDs are complete
2. metrics are parseable
3. artifacts are present and reproducible
4. leakage/split checks pass
