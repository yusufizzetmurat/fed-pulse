SHELL := /bin/bash

DATASET_VERSION ?=
FEATURE_VERSION ?=
TRAINING_PACKAGE_ID ?=
OWNER ?= unknown
SEED ?= 11

TEACHER_CHECKPOINT ?= /data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints
PSEUDO_STRATEGY ?= chunk_vote
PSEUDO_TAU_CHUNK ?= 0.50
PSEUDO_TAU_DOC ?= 0.85
PSEUDO_AUDIT_SIZE ?= 100

.PHONY: help dev dev-cpu dev-gpu down logs lock verify openapi-snapshot data-prep train-smoke train-batch changelog audit-python audit-npm pseudo-labels pseudo-labels-audit-sample pseudo-labels-audit-metrics

help:
	@echo "Targets:"
	@echo "  make dev              - Start CPU backend + frontend"
	@echo "  make dev-cpu          - Start CPU backend + frontend"
	@echo "  make dev-gpu          - Start GPU backend + frontend (requires NVIDIA runtime)"
	@echo "  make down             - Stop all containers"
	@echo "  make logs             - Tail compose logs"
	@echo "  make lock             - Regenerate backend/requirements.lock from pyproject.toml"
	@echo "  make verify           - Build toy snapshot, run full pytest suite, check imports"
	@echo "  make openapi-snapshot - Regenerate tests/snapshots/openapi.json"
	@echo "  make data-prep        - Run capability-first data preparation pipeline"
	@echo "  make train-smoke      - Run Phase 3 single-seed smoke execution"
	@echo "  make train-batch      - Run Phase 3 full official batch execution"
	@echo "  make changelog        - Regenerate CHANGELOG.md from Conventional Commits via git-cliff"
	@echo "  make audit-python     - Run pip-audit on the backend deps (mirrors CI)"
	@echo "  make audit-npm        - Run npm audit on the frontend deps (mirrors CI)"
	@echo "  make pseudo-labels    - Run chunk-aggregated teacher on the unlabelled pool (chunk_vote default)"
	@echo "  make pseudo-labels-audit-sample - Sample a stratified PSEUDO_AUDIT_SIZE-row CSV for human labelling"
	@echo "  make pseudo-labels-audit-metrics - Compute teacher precision against the human-labelled audit set"

dev: dev-cpu

dev-cpu:
	docker compose up -d --build backend frontend

dev-gpu:
	docker compose --profile gpu up -d --build backend-gpu frontend

down:
	docker compose --profile gpu down

logs:
	docker compose logs -f --tail=200

lock:
	docker compose run --rm backend bash -c "pip install --quiet uv && uv pip compile --generate-hashes --output-file requirements.lock pyproject.toml"

verify:
	docker compose run --rm backend bash /app/scripts/verify_smoke.sh

openapi-snapshot:
	@mkdir -p tests/snapshots
	docker compose run --rm -T backend python /app/scripts/regen_openapi_snapshot.py --stdout > tests/snapshots/openapi.json
	@echo "wrote tests/snapshots/openapi.json"

data-prep:
	@test -n "$(DATASET_VERSION)" || (echo "DATASET_VERSION is required"; exit 1)
	@test -n "$(FEATURE_VERSION)" || (echo "FEATURE_VERSION is required"; exit 1)
	docker compose run --rm backend \
		python -m app.data.pipeline_data_prep \
		--all-sources \
		--dataset-version "$(DATASET_VERSION)" \
		--feature-version "$(FEATURE_VERSION)" \
		--owner "$(OWNER)" \
		$(if $(TRAINING_PACKAGE_ID),--training-package-id "$(TRAINING_PACKAGE_ID)",)

train-smoke:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.data.phase3_training_execution \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--mode smoke \
		--seed "$(SEED)" \
		--owner "$(OWNER)"

train-batch:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.data.phase3_training_execution \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--mode full \
		--owner "$(OWNER)"

changelog:
	@command -v git-cliff >/dev/null 2>&1 || { \
		echo "git-cliff not installed. Install via: cargo install git-cliff  OR  pipx install git-cliff"; exit 1; \
	}
	git-cliff --config cliff.toml --output CHANGELOG.md

audit-python:
	docker compose run --rm backend bash -c "pip install --quiet pip-audit==2.7.3 && pip-audit --strict"

audit-npm:
	docker compose run --rm frontend npm audit --audit-level=high --production

# Pseudo-label production run (chunk-aggregated teacher; default strategy chunk_vote).
# Reads /data/raw/phase2/source_registry.jsonl, writes a pseudo set + threshold
# sweep under /data/interim/phase2/. Override defaults via PSEUDO_STRATEGY,
# PSEUDO_TAU_CHUNK, PSEUDO_TAU_DOC, TEACHER_CHECKPOINT. See docs/pseudo-label-runbook.md.
pseudo-labels:
	docker compose run --rm backend \
		python -m app.data.pseudo_labeling \
		--teacher-checkpoint "$(TEACHER_CHECKPOINT)" \
		--teacher-model-id finbert_fomc_s71 \
		--teacher-model-version chunk_aggregated_v1 \
		--input /data/raw/phase2/source_registry.jsonl \
		--output /data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY).jsonl \
		--strategy "$(PSEUDO_STRATEGY)" \
		--tau-chunk "$(PSEUDO_TAU_CHUNK)" \
		--threshold "$(PSEUDO_TAU_DOC)"

# Stratified PSEUDO_AUDIT_SIZE-row sample for human labelling. Reads the pseudo set
# at /data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY).jsonl, writes the
# audit CSV + accompanying JSONL under /data/artifacts/pseudo_label_audits/.
pseudo-labels-audit-sample:
	docker compose run --rm backend \
		python -c "import json; from pathlib import Path; from app.data.llm_judge import sample_audit_set, write_audit_csv; \
		rows = [json.loads(l) for l in open('/data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY).jsonl')]; \
		sample = sample_audit_set(rows, n=$(PSEUDO_AUDIT_SIZE), seed=$(SEED)); \
		Path('/data/artifacts/pseudo_label_audits').mkdir(parents=True, exist_ok=True); \
		write_audit_csv(sample, Path('/data/artifacts/pseudo_label_audits/audit_set_$(PSEUDO_STRATEGY)_n$(PSEUDO_AUDIT_SIZE).csv')); \
		print(f'wrote audit_set_$(PSEUDO_STRATEGY)_n$(PSEUDO_AUDIT_SIZE).csv with {len(sample)} rows')"

# Compute teacher precision against the human-labelled audit. Expects a
# human_label column added to the CSV (open in Excel/Sheets, fill, save as
# audit_set_<strategy>_filled.jsonl). Prints Cohen's kappa + per-class precision.
pseudo-labels-audit-metrics:
	docker compose run --rm backend \
		python -c "import json; from app.data.llm_judge import audit_metrics; \
		rows = [json.loads(l) for l in open('/data/artifacts/pseudo_label_audits/audit_set_$(PSEUDO_STRATEGY)_filled.jsonl')]; \
		print(json.dumps(audit_metrics(rows), indent=2))"
