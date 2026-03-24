SHELL := /bin/bash

DATASET_VERSION ?=
FEATURE_VERSION ?=
TRAINING_PACKAGE_ID ?=
OWNER ?= unknown
SEED ?= 11

.PHONY: help dev dev-cpu dev-gpu down logs data-prep train-smoke train-batch

help:
	@echo "Targets:"
	@echo "  make dev             - Start CPU backend + frontend"
	@echo "  make dev-cpu         - Start CPU backend + frontend"
	@echo "  make dev-gpu         - Start GPU backend + frontend (requires NVIDIA runtime)"
	@echo "  make down            - Stop all containers"
	@echo "  make logs            - Tail compose logs"
	@echo "  make data-prep       - Run capability-first data preparation pipeline"
	@echo "  make train-smoke     - Run Phase 3 single-seed smoke execution"
	@echo "  make train-batch     - Run Phase 3 full official batch execution"

dev: dev-cpu

dev-cpu:
	docker compose up -d --build backend frontend

dev-gpu:
	docker compose --profile gpu up -d --build backend-gpu frontend

down:
	docker compose --profile gpu down

logs:
	docker compose logs -f --tail=200

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
