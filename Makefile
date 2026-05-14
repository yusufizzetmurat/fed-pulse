SHELL := /bin/bash

DATASET_VERSION ?=
FEATURE_VERSION ?=
TRAINING_PACKAGE_ID ?=
OWNER ?= unknown
SEED ?= 11

.PHONY: help dev dev-cpu dev-gpu down logs lock verify openapi-snapshot data-prep train-smoke train-batch changelog audit-python audit-npm

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
