SHELL := /bin/bash

DATASET_VERSION ?=
FEATURE_VERSION ?=
TRAINING_PACKAGE_ID ?=
OWNER ?= unknown
SEED ?= 11
ARCHITECTURE ?= lstm

TEACHER_CHECKPOINT ?= /data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints
PSEUDO_STRATEGY ?= chunk_vote
PSEUDO_TAU_CHUNK ?= 0.50
PSEUDO_TAU_DOC ?= 0.85
PSEUDO_AUDIT_SIZE ?= 100

# Judge-only audit knobs. JUDGE_REQUEST_INTERVAL spaces calls to respect
# Gemini free-tier rate limits (2 req/min on gemini-2.5-flash, so 30s
# interval is the safe floor; paid tiers can leave it at 0.0).
JUDGE_MODEL ?= gemini-2.5-pro
JUDGE_MODEL_VERSION ?= 20260514_v1
JUDGE_REQUEST_INTERVAL ?= 0.0

# Pseudo-label workflow runs through the GPU backend by default — a 9,696-row pass
# on CPU takes days. Override on a CPU-only box with:
#   make pseudo-labels PSEUDO_SERVICE=backend PSEUDO_PROFILE_FLAG=
# Both services share Dockerfile + pyproject.toml, so the image content is
# identical; the GPU service simply reserves an NVIDIA device.
PSEUDO_SERVICE ?= backend-gpu
PSEUDO_PROFILE_FLAG ?= --profile gpu

.PHONY: help dev dev-cpu dev-gpu down logs lock verify openapi-snapshot data-prep train-smoke train-batch changelog audit-python audit-npm pseudo-labels pseudo-labels-audit-sample pseudo-labels-audit-metrics pseudo-labels-judge-pass pseudo-labels-audit-metrics-judge macro-state next-fomc cross-asset forecaster-sweep forecaster-sweep-aggregate forecaster-credibility-train

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
	@echo "  make pseudo-labels-judge-pass - Score the pseudo set with Gemini (judge-only audit gold)"
	@echo "  make pseudo-labels-audit-metrics-judge - Compute teacher precision against the LLM judge gold"
	@echo "  make macro-state      - Build the FRED macro-state parquet (Phase 8 #147)"
	@echo "  make next-fomc        - Predict next-FOMC decision (Phase 8 headline, #147)"
	@echo "  make cross-asset      - Cross-asset abnormal-return response head (Phase 8, #148)"
	@echo "  make forecaster-sweep         - 6-arch x 5-seed forecaster sweep (Phase 8, #70)"
	@echo "  make forecaster-sweep-aggregate - Aggregate sweep trials into per-arch CIs"
	@echo "  make forecaster-credibility-train - Single training run with credibility features on"

dev: dev-cpu

dev-cpu:
	docker compose up -d --build redis backend worker frontend

dev-gpu:
	docker compose --profile gpu up -d --build redis backend-gpu worker-gpu frontend

down:
	docker compose --profile gpu down

logs:
	docker compose logs -f --tail=200

worker-logs:
	docker compose logs -f --tail=200 worker

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
		python -m app.data.nlp_baseline_batch \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--mode smoke \
		--seed "$(SEED)" \
		--owner "$(OWNER)"

train-batch:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.data.nlp_baseline_batch \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--mode full \
		--owner "$(OWNER)"

# Forecaster architecture sweep: 6 architectures (lstm, lstm_attn, gru, tcn,
# transformer, dlinear) x official 5-seed set {11, 29, 47, 71, 97}.
# Writes forecaster_sweep_results.json + .csv next to the checkpoint.
#
# Runs against backend-gpu under the gpu compose profile so the sweep hits
# the RTX 4080. Override FORECASTER_COMPOSE_SERVICE=backend and
# FORECASTER_COMPOSE_PROFILE=default for a CPU-only smoke run. The
# aggregate target uses the same overrides so artefact paths line up.
FORECASTER_COMPOSE_SERVICE ?= backend-gpu
FORECASTER_COMPOSE_PROFILE ?= gpu

forecaster-sweep:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--sweep \
		--architectures lstm lstm_attn gru tcn transformer dlinear \
		--seeds 11 29 47 71 97 \
		--report-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/forecaster_sweep_results.json"

# Aggregate per-trial JSONs into a per-architecture headline (block-bootstrap CIs).
# The aggregator is CPU-bound; the default backend service is fine.
forecaster-sweep-aggregate:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.evaluation.forecaster_sweep_aggregator \
		--artifact-dir "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)"

# Single-architecture training with --credibility-features ON. Used for
# isolated credibility-vs-baseline comparisons; defaults to ARCHITECTURE=lstm
# and SEED=11. Override either as needed. GPU service by default; the same
# FORECASTER_COMPOSE_* overrides switch to CPU for a smoke run.
forecaster-credibility-train:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--architecture "$(ARCHITECTURE)" \
		--seed "$(SEED)" \
		--credibility-features \
		--checkpoint-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/credibility_$(ARCHITECTURE)_seed$(SEED).pt"

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
# PSEUDO_TAU_CHUNK, PSEUDO_TAU_DOC, TEACHER_CHECKPOINT.
pseudo-labels:
	docker compose $(PSEUDO_PROFILE_FLAG) run --rm $(PSEUDO_SERVICE) \
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
	docker compose $(PSEUDO_PROFILE_FLAG) run --rm $(PSEUDO_SERVICE) \
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
	docker compose $(PSEUDO_PROFILE_FLAG) run --rm $(PSEUDO_SERVICE) \
		python -c "import json; from app.data.llm_judge import audit_metrics; \
		rows = [json.loads(l) for l in open('/data/artifacts/pseudo_label_audits/audit_set_$(PSEUDO_STRATEGY)_filled.jsonl')]; \
		print(json.dumps(audit_metrics(rows), indent=2))"

# Score the pseudo set with the Gemini judge — the judge-only audit path
# uses these labels as gold without requiring a human pass. Requires
# GEMINI_API_KEY in .env. Free-tier rate limit is 2 req/min for flash;
# default --request-interval-seconds 0.0 assumes a paid quota.
pseudo-labels-judge-pass:
	docker compose $(PSEUDO_PROFILE_FLAG) run --rm $(PSEUDO_SERVICE) \
		python -m app.data.llm_judge \
		--input /data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY).jsonl \
		--output /data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY)_judged.jsonl \
		--judge-model "$(JUDGE_MODEL)" \
		--judge-model-version "$(JUDGE_MODEL_VERSION)" \
		--request-interval-seconds "$(JUDGE_REQUEST_INTERVAL)" \
		--resume

# Judge-only audit: use the Gemini judge labels as gold and compute
# teacher per-class precision + Cohen's kappa(teacher, judge). Pass
# criterion: every supported class >= 0.90 precision.
pseudo-labels-audit-metrics-judge:
	docker compose $(PSEUDO_PROFILE_FLAG) run --rm $(PSEUDO_SERVICE) \
		python -c "import json; from app.data.llm_judge import audit_metrics_judge_only; \
		rows = [json.loads(l) for l in open('/data/interim/phase2/registry_pseudo_$(PSEUDO_STRATEGY)_judged.jsonl')]; \
		print(json.dumps(audit_metrics_judge_only(rows), indent=2))"

# Continued pretraining (FinBERT-FedAdjacent).
# Defaults to samchain/BIS_speeches_97_23_MLM as the substrate (909,877 NSP pairs).
# Override with SUBSTRATE=local for the legacy 44-doc JSON corpus.
# Override with SUBSTRATE=both to mix BIS + local in a single run.
SUBSTRATE ?= bis
PRETRAIN_SEED ?= 11
PRETRAIN_EPOCHS ?= 2
PRETRAIN_BATCH_SIZE ?= 8
PRETRAIN_BLOCK_SIZE ?= 256
PRETRAIN_LR ?= 2e-5
PRETRAIN_MAX_ROWS ?= 0
PRETRAIN_OBJECTIVE ?= mlm_nsp
PRETRAIN_BASE_CHECKPOINT ?= ProsusAI/finbert
PRETRAIN_OUT_NAME ?= finbert_fed_adjacent
finbert-fed-adjacent-pretrain:
	docker compose run --rm backend \
		python -m app.data.continued_pretraining \
		--substrate "$(SUBSTRATE)" \
		--seed "$(PRETRAIN_SEED)" \
		--epochs "$(PRETRAIN_EPOCHS)" \
		--batch-size "$(PRETRAIN_BATCH_SIZE)" \
		--block-size "$(PRETRAIN_BLOCK_SIZE)" \
		--learning-rate "$(PRETRAIN_LR)" \
		--max-rows "$(PRETRAIN_MAX_ROWS)" \
		--objective "$(PRETRAIN_OBJECTIVE)" \
		--base-checkpoint "$(PRETRAIN_BASE_CHECKPOINT)" \
		--checkpoint-name "$(PRETRAIN_OUT_NAME)"

# Control ablation: same substrate + recipe, but starting from bert-base-uncased
# instead of ProsusAI/finbert. Pair the resulting checkpoint with the FinBERT-
# based one in the bake-off to isolate the contribution of the finance prior.
finbert-fed-adjacent-pretrain-bert-control:
	$(MAKE) finbert-fed-adjacent-pretrain \
		PRETRAIN_BASE_CHECKPOINT=bert-base-uncased \
		PRETRAIN_OUT_NAME=bert_base_fed_adjacent \
		$(if $(SUBSTRATE),SUBSTRATE=$(SUBSTRATE),)

finbert-fed-adjacent-pretrain-smoke:
	docker compose run --rm backend \
		python -m app.data.continued_pretraining \
		--substrate bis \
		--streaming \
		--max-rows 200 \
		--epochs 1 \
		--batch-size 4 \
		--block-size 128 \
		--objective mlm

# Build (or reuse) the per-encoder embedding cache for a training package.
# Required: ENCODER (alias from models/registry.yaml), TRAINING_PACKAGE_ID.
# Set ALLOW_NETWORK=1 the first time you cache a new encoder.
cache-embeddings:
	@if [ -z "$(ENCODER)" ]; then echo "Set ENCODER=<alias>"; exit 2; fi
	@if [ -z "$(TRAINING_PACKAGE_ID)" ]; then echo "Set TRAINING_PACKAGE_ID=<id>"; exit 2; fi
	docker compose run --rm backend \
		python -m app.data.embedding_cache \
		--encoder "$(ENCODER)" \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		$(if $(ALLOW_NETWORK),--allow-network,) \
		$(if $(FORCE),--force,)

# Aggregate one or more finetune_batch aggregate.json files into a headline
# table with block-bootstrap CIs. ARTIFACT_DIR defaults to data/artifacts/phase3.
bakeoff-aggregate:
	docker compose run --rm backend \
		python -m app.evaluation.bakeoff_aggregator \
		--artifact-dir "$(or $(ARTIFACT_DIR),data/artifacts/phase3)"

# Zero-shot cross-CB transfer matrix for an NLP checkpoint. Repeat the same
# alias in MODEL_CHECKPOINTS to feed per-seed checkpoints for that alias —
# CI bands appear when >=2 distinct checkpoints land in the same cell.
# Required: TRAINING_PACKAGE_ID, MODEL_CHECKPOINTS (alias=path[,alias=path]).
# Optional: EVAL_BANKS (ecb,boj,boe,boc,rba; default = all 5).
eval-cross-bank:
	@if [ -z "$(TRAINING_PACKAGE_ID)" ]; then echo "Set TRAINING_PACKAGE_ID=<id>"; exit 2; fi
	@if [ -z "$(MODEL_CHECKPOINTS)" ]; then echo "Set MODEL_CHECKPOINTS=alias=path[,alias=path]"; exit 2; fi
	docker compose run --rm backend \
		python -m app.evaluation.transfer_matrix \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--model-checkpoints "$(MODEL_CHECKPOINTS)" \
		$(if $(EVAL_BANKS),--eval-banks "$(EVAL_BANKS)",)

# Build the FRED macro-state snapshot at data/external/fred/macro_state.parquet.
# Reads UNRATE, CPIAUCSL, PCEPILFE, MANEMP, PAYEMS, RSAFS. Requires FRED_API_KEY
# in .env on first run (cached JSON afterwards).
MACRO_STATE_START ?= 2010-01-01
MACRO_STATE_END ?= today
macro-state:
	docker compose run --rm backend \
		python -m app.data.macro_state \
		--start "$(MACRO_STATE_START)" \
		--end "$(MACRO_STATE_END)"

# Predict the FOMC's next decision using text + macro + OIS + credibility +
# linguistic features (Phase 8 headline, #147). Required: TRAINING_PACKAGE_ID
# pointing at a package under data/processed/<id>/ containing events.parquet
# and linguistic_features.parquet. Reads mp_surprises.parquet and
# macro_state.parquet from data/external/fred/.
next-fomc:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.forecasting.next_fomc_decision \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--seed "$(SEED)"

# Predict the cross-section of asset abnormal returns to FOMC events using
# text + macro + OIS + credibility + linguistic features (Phase 8, #148).
# Required: TRAINING_PACKAGE_ID pointing at a package under
# data/processed/<id>/ that contains events.parquet (with per-asset rows)
# and linguistic_features.parquet. Reads mp_surprises.parquet and
# macro_state.parquet from data/external/fred/.
cross-asset:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.forecasting.cross_asset_response \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--seed "$(SEED)"
