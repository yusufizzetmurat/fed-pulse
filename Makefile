SHELL := /bin/bash

DATASET_VERSION ?=
FEATURE_VERSION ?=
TRAINING_PACKAGE_ID ?=
OWNER ?= unknown
SEED ?= 11
ARCHITECTURE ?= lstm
ENCODER_ALIAS ?= finbert_fed_adjacent
EPOCHS ?= 4
BATCH_SIZE ?= 16
LEARNING_RATE ?= 2e-5

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

.PHONY: help dev dev-cpu dev-gpu down logs lock verify openapi-snapshot data-prep audit-training-package train-smoke train-batch changelog audit-python audit-npm pseudo-labels pseudo-labels-audit-sample pseudo-labels-audit-metrics pseudo-labels-judge-pass pseudo-labels-audit-metrics-judge macro-state build-macro-state build-mp-surprises build-rates-panel rebuild-linguistic-features cache-voyage-embeddings next-fomc cross-asset forecaster-sweep forecaster-sweep-exhaustive forecaster-sweep-baseline forecaster-sweep-aggregate forecaster-sweep-shuffled-control forecaster-credibility-train regime-baseline-tiers regime-arch-sweep regime-pooled-aggregate regime-ensemble-aggregate regime-capacity-push train-text-multi-axis-classifier dual-head-comparison derived-features-ablation rates-heads-sweep canonical-comparison canonical-comparison-fomc-attributable canonical-comparison-retrieval-analogs canonical-comparison-regime-conditioning text-path-ab per-family-ablation finetune-pilot-b2 finetune-pilot-b2-phrasebank cross-source-transfer reproduce-all reproduce-smoke push-artefacts deploy-prod-build

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
	@echo "  make train-text-multi-axis-classifier TRAINING_PACKAGE_ID=<id>"
	@echo "                         - Fine-tune the FinBERT-FedAdjacent + MultiTaskHead text classifier the /analyze cards read"
	@echo "  make changelog        - Regenerate CHANGELOG.md from Conventional Commits via git-cliff"
	@echo "  make audit-python     - Run pip-audit on the backend deps (mirrors CI)"
	@echo "  make audit-npm        - Run npm audit on the frontend deps (mirrors CI)"
	@echo "  make pseudo-labels    - Run chunk-aggregated teacher on the unlabelled pool (chunk_vote default)"
	@echo "  make pseudo-labels-audit-sample - Sample a stratified PSEUDO_AUDIT_SIZE-row CSV for human labelling"
	@echo "  make pseudo-labels-audit-metrics - Compute teacher precision against the human-labelled audit set"
	@echo "  make pseudo-labels-judge-pass - Score the pseudo set with Gemini (judge-only audit gold)"
	@echo "  make pseudo-labels-audit-metrics-judge - Compute teacher precision against the LLM judge gold"
	@echo "  make macro-state      - Build the FRED macro-state parquet (Phase 8 #147)"
	@echo "  make build-macro-state - Same as macro-state, named for the data-asset suite"
	@echo "  make build-mp-surprises - Build the MP-surprise time-series parquet"
	@echo "  make build-rates-panel - Build the daily rates panel (DGS1/2/5/10, DFEDTARU, #291)"
	@echo "  make rebuild-linguistic-features TRAINING_PACKAGE_ID=<id>"
	@echo "                         - Re-emit linguistic_features.parquet for a training package"
	@echo "  make cache-voyage-embeddings"
	@echo "                         - Cache voyage-finance-2 embeddings for the FOMC corpus"
	@echo "  make next-fomc        - Predict next-FOMC decision (Phase 8 headline, #147)"
	@echo "  make cross-asset      - Cross-asset abnormal-return response head (Phase 8, #148)"
	@echo "  make forecaster-sweep         - 8-arch x 5-seed forecaster sweep, random-search subset of the HP grid, bucketed runner (BATCHING_MODE=auto)"
	@echo "  make forecaster-sweep-exhaustive - 8-arch x 5-seed forecaster sweep, full HP cross-product, single worker, BATCHING_MODE=off (back-compat)"
	@echo "  make forecaster-sweep-baseline - 6-arch x 5-seed forecaster sweep, legacy 6-feature input"
	@echo "  make forecaster-sweep-shuffled-control - Memorisation-control row: same architectures + seeds, median HP, --shuffle-targets-control on"
	@echo "  make forecaster-sweep-aggregate - Aggregate sweep trials into per-arch CIs"
	@echo "  make regime-baseline-tiers     - Phase 9 V2 3-tier regime classifier (Market / +Rich / +Rich+NLP)"
	@echo "  make forecaster-credibility-train - Single training run with credibility features on"
	@echo "  make dual-head-comparison TRAINING_PACKAGE_ID=<id>"
	@echo "                         - Three-way head-mode comparison (classification / regression / dual) across the official seed set"
	@echo "  make derived-features-ablation TRAINING_PACKAGE_ID=<id>"
	@echo "  make rates-heads-sweep TRAINING_PACKAGE_ID=<id> [SEED=<seed>]"
	@echo "                         - Three-way derived-text-features ablation (baseline / ablation / replacement)"
	@echo "  make canonical-comparison TRAINING_PACKAGE_ID=<id>"
	@echo "  make text-path-ab TRAINING_PACKAGE_ID=<id>"
	@echo "                         - Canonical dual-head comparison (5 seeds x 40 epochs, regression-alpha=0.5, canonical output JSON)"
	@echo "  make per-family-ablation TRAINING_PACKAGE_ID=<id>"
	@echo "                         - Per-family rich-feature ablation (#334; backs the §6 substitution-finding table)"
	@echo "  make finetune-pilot-b2 TRAINING_PACKAGE_ID=<id> [ENCODER_ALIAS=<alias>]"
	@echo "                         - B2 end-to-end fine-tune on vol-regime (#213; AutoModelForSequenceClassification, 5 seeds x 4 folds x 5 epochs)"
	@echo "  make finetune-pilot-b2-phrasebank TRAINING_PACKAGE_ID=<id> [PHRASEBANK_AUX_LAMBDA=0.3] [PHRASEBANK_SUBSET=sentences_allagree]"
	@echo "                         - B2 fine-tune with the PhraseBank auxiliary 3-way sentiment CE on top (#33 Path B; ADR 0033)"
	@echo "  make cross-source-transfer TRAINING_PACKAGE_ID=<id> ENCODER_CHECKPOINTS=alias=path[,alias=path]"
	@echo "                         - Cross-source transfer matrix (#72 + #83; inference-only per source_type stratum)"
	@echo "  make reproduce-smoke TRAINING_PACKAGE_ID=<id> [SEED=11]"
	@echo "                         - 1-seed x 1-fold dual-head smoke + numerical-contract assertion (#335 CI guard)"

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

audit-training-package:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	python -m scripts.audit_training_package_coverage \
		--training-package-id "$(TRAINING_PACKAGE_ID)"

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

# Fresh-machine reproduction smoke (#302 Stage 5). Pulls the canonical
# training package + embedding caches from HF Hub and runs a single-
# seed, single-epoch forecaster training so a reviewer with only
# Docker + an HF token can confirm the pipeline runs end-to-end. The
# expected wall time on the 8 GB / 4 vCPU droplet is ~15 minutes
# (~10 min for the artefact pull on a cold cache, ~5 min for the
# one-epoch training pass). Sets ``FED_PULSE_REPRODUCE_SMOKE=1`` so
# downstream code paths know they are in the smoke run and can shrink
# expensive sub-steps without changing the canonical training script.
reproduce-all:
	docker compose run --rm \
		-e FED_PULSE_REPRODUCE_SMOKE=1 \
		-e HF_TOKEN=$$HF_TOKEN \
		backend bash -c '\
			set -e && \
			python scripts/reproduce_all.py'

# Numerical-contract CI guard (#335). Runs a 1-seed x 1-fold smoke
# variant of the canonical dual-head training and asserts the resulting
# macro-F1 stays within the pinned tolerance in
# ``tests/regression/reproducibility_reference.json``. Designed to run
# on the ``ubuntu-latest`` GitHub runner without docker compose so the
# ``reproduce-smoke`` workflow can call it directly. Gates on
# ``TRAINING_PACKAGE_ID`` so a typo on the workflow input fails fast
# instead of pulling the wrong artefact.
reproduce-smoke:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	FED_PULSE_REPRODUCE_SMOKE=1 \
	PYTHONPATH=backend \
	python -m scripts.run_reproducibility_smoke \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--seed "$(SEED)" \
		--reference-path tests/regression/reproducibility_reference.json

# One-time push of every canonical artefact to HF Hub. Runs the
# idempotent uploader in dry-run by default so the operator can sanity
# check the plan before flipping --all on. See
# ``scripts/push_artefacts_to_hub.py`` for full documentation.
push-artefacts:
	python scripts/push_artefacts_to_hub.py --dry-run

# Local build of the production container — useful for verifying the
# Dockerfile builds clean before pushing to main. Does NOT publish the
# image anywhere.
deploy-prod-build:
	docker compose -f compose.prod.yml build

# Train the text-only multi-axis classifier (#78 follow-up). Reads
# events.parquet from the supplied training package, fine-tunes
# finbert_fed_adjacent + MultiTaskHead end-to-end on the supervised
# axis rows, writes the best-epoch checkpoint to
# ``backend/models/text_multi_axis_best.pt`` which the /analyze service
# singleton picks up on next backend restart.
train-text-multi-axis-classifier:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile gpu run --rm backend-gpu \
		python -m app.data.train_text_multi_axis_classifier \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--encoder-alias "$(ENCODER_ALIAS)" \
		--epochs $(EPOCHS) \
		--seed $(SEED) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LEARNING_RATE)

# Forecaster architecture sweep. The default target runs the
# rich-feature path (35-dim per-bar input) across the seven canonical
# architectures (lstm, lstm_attn, gru, tcn, transformer, dlinear,
# informer) x the official 5-seed set {11, 29, 47, 71, 97}, so the
# forecaster sees the four feature families the data pipeline already
# ships (credibility, linguistic, MP-surprise, multi-axis) on every bar.
#
# TFT is excluded from the canonical sweep targets per ADR 0020 (the
# generic classifier head strips the native quantile-output + Variable
# Selection Network inductive bias). The ``tft`` identifier and module
# are kept for back-compat with existing checkpoints; opt back in by
# passing ``--architectures tft`` explicitly on the trainer command line.
#
# The default target draws a random subset of HP combos from the full
# cross-product (--random-search-samples=50, seed=42) and runs eight
# cells concurrently on the same GPU via the spawn-mode process pool.
# The 8-worker default matches the RTX 4080's 16 GB; the larger
# architectures (transformer hidden=128, layers=3) hold roughly 1 GB
# per cell, leaving headroom for the CUDA allocator's fragmentation
# pool.
#
# ``forecaster-sweep-exhaustive`` enumerates every cell in the HP
# cross-product sequentially. It is the back-compat path for the
# byte-identity regression test and for diagnostic re-runs that need
# the deterministic candidate ordering of the pre-speedup sweep.
#
# The earlier 6-feature path is still available as
# ``forecaster-sweep-baseline`` for back-compat smoke checks against
# pre-PR-#173 sweep numbers.
#
# All three targets write forecaster_sweep_results.json + .csv under
# data/artifacts/forecaster_sweep/<TRAINING_PACKAGE_ID>/; the baseline
# variant lands under a ``baseline_`` filename prefix so the two
# artefact sets coexist on disk.
#
# Runs against backend-gpu under the gpu compose profile so the sweep
# hits the RTX 4080. Override FORECASTER_COMPOSE_SERVICE=backend and
# FORECASTER_COMPOSE_PROFILE=default for a CPU-only smoke run. The
# aggregate target uses the same overrides so artefact paths line up.
FORECASTER_COMPOSE_SERVICE ?= backend-gpu
FORECASTER_COMPOSE_PROFILE ?= gpu

# Random-search + parallel-worker + batching-mode knobs the user can
# override on the command line. ``RANDOM_SEARCH_SAMPLES=216`` against
# the full grid collapses to the exhaustive enumeration (the sampler
# clamps to the grid size). ``BATCHING_MODE=auto`` (the default)
# groups cells by model topology and runs each bucket as one
# concurrent unit inside one CUDA context; ``BATCHING_MODE=off``
# reverts to the legacy ProcessPoolExecutor path with
# ``PARALLEL_WORKERS`` spawn-mode workers.
RANDOM_SEARCH_SAMPLES ?= 50
RANDOM_SEARCH_SEED ?= 42
PARALLEL_WORKERS ?= 8
BATCHING_MODE ?= auto
# Gradient-norm clip applied to every training step. The training-loop
# CLI default is 0.0 (off); the sweep target pins 1.0 here so the
# post-#180 numbers stay comparable with earlier runs that used the
# pre-rewrite implicit 1.0 clip. Override with ``GRAD_CLIP_NORM=0.0``
# to ablate clipping explicitly.
GRAD_CLIP_NORM ?= 1.0

forecaster-sweep:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	@test -n "$(TEXT_ENCODER)" || (echo "TEXT_ENCODER is required (e.g. finbert, voyage_finance_2, or 'none' for the text-off row)"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--sweep \
		--rich-features \
		--architectures lstm lstm_attn gru tcn transformer dlinear informer \
		--seeds 11 29 47 71 97 \
		--folds wf_fold_1 wf_fold_2 wf_fold_3 wf_fold_4 \
		--hidden-sizes 32 64 128 \
		--num-layers-grid 1 2 3 \
		--dropouts 0.1 0.2 0.3 0.4 \
		--learning-rates 1e-3 3e-4 \
		--weight-decays 0 1e-4 1e-3 \
		--text-encoder "$(TEXT_ENCODER)" \
		--text-adapter-dims 32 64 128 \
		--random-search \
		--random-search-samples $(RANDOM_SEARCH_SAMPLES) \
		--random-search-seed $(RANDOM_SEARCH_SEED) \
		--parallel-workers $(PARALLEL_WORKERS) \
		--batching-mode $(BATCHING_MODE) \
		--grad-clip-norm $(GRAD_CLIP_NORM) \
		--report-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/forecaster_sweep_results.json"

# Exhaustive sweep: every cell in the HP cross-product, single worker,
# bucketed runner OFF. Reproduces the pre-PR forecaster_sweep_results.json
# byte-identically on the same package and seed set, which is the
# contract the byte-identity regression test pins.
forecaster-sweep-exhaustive:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	@test -n "$(TEXT_ENCODER)" || (echo "TEXT_ENCODER is required (e.g. finbert, voyage_finance_2, or 'none' for the text-off row)"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--sweep \
		--rich-features \
		--architectures lstm lstm_attn gru tcn transformer dlinear informer \
		--seeds 11 29 47 71 97 \
		--hidden-sizes 32 64 128 \
		--num-layers-grid 1 2 3 \
		--dropouts 0.1 0.2 0.3 0.4 \
		--learning-rates 1e-3 3e-4 \
		--weight-decays 0 1e-4 1e-3 \
		--text-encoder "$(TEXT_ENCODER)" \
		--text-adapter-dims 32 64 128 \
		--batching-mode off \
		--parallel-workers 1 \
		--report-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/forecaster_sweep_results.json"

# Memorisation control: one median-HP combo across all architectures
# and the five seeds, with --shuffle-targets-control on. A model whose
# real-targets RMSE is close to its shuffled-targets RMSE is memorising
# rather than learning the input-target mapping. Output lands beside
# the main sweep under a distinct filename so the aggregator picks it
# up as the shuffled-control row.
forecaster-sweep-shuffled-control:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	@test -n "$(TEXT_ENCODER)" || (echo "TEXT_ENCODER is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--sweep \
		--rich-features \
		--architectures lstm lstm_attn gru tcn transformer dlinear informer \
		--seeds 11 29 47 71 97 \
		--hidden-sizes 64 \
		--num-layers-grid 2 \
		--dropouts 0.2 \
		--learning-rates 1e-3 \
		--weight-decays 0 \
		--text-encoder "$(TEXT_ENCODER)" \
		--text-adapter-dims 64 \
		--shuffle-targets-control \
		--report-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/shuffled_control_forecaster_sweep_results.json"

forecaster-sweep-baseline:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m app.train_forecaster \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--sweep \
		--no-rich-features \
		--architectures lstm lstm_attn gru tcn transformer dlinear \
		--seeds 11 29 47 71 97 \
		--report-path "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)/baseline_forecaster_sweep_results.json"

# Aggregate per-trial JSONs into a per-architecture headline (block-bootstrap CIs).
# The aggregator is CPU-bound; the default backend service is fine.
forecaster-sweep-aggregate:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.evaluation.forecaster_sweep_aggregator \
		--artifact-dir "/data/artifacts/forecaster_sweep/$(TRAINING_PACKAGE_ID)"

# Phase 9 V2 (#195) 3-tier vol-regime classification baseline harness.
# Runs Market-Only, Market+Rich, and Market+Rich+NLP-Embeddings as
# separate classification sweeps so the marginal lift of each input
# family on the regime-classification axis is measurable. Per-tier
# JSON lands under
# data/artifacts/regime_baseline_tiers/$(TRAINING_PACKAGE_ID)/<tier>/.
# Override NLP_TEXT_ENCODER to swap the tier-3 encoder.
NLP_TEXT_ENCODER ?= finbert_fed_adjacent
regime-baseline-tiers:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python scripts/run_regime_baseline_tiers.py \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--nlp-text-encoder "$(NLP_TEXT_ENCODER)" \
		--report-root /data/artifacts/regime_baseline_tiers

# Phase A (#226) architecture sweep -- per-architecture random-search
# HP at the A5 best-cell neighbourhood, all on the Tier 5 surface
# (rich + LLM, no NLP) by default. The downstream
# ``regime-ensemble-aggregate`` target consumes the per-architecture
# JSONs and reports mean-logit / mean-softmax / plurality-vote macro-F1.
USE_LLM_FEATURES ?= on
regime-arch-sweep:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python scripts/run_regime_architecture_sweep.py \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		$(if $(filter on,$(USE_LLM_FEATURES)),--use-llm-features,--no-llm-features) \
		$(if $(NLP_TEXT_ENCODER),--text-encoder "$(NLP_TEXT_ENCODER)",) \
		--report-root /data/artifacts/regime_arch_sweep

# Round 5 (#244) LoRA + in-loop FinBERT ceiling probe. Reads the
# best architecture + HP cell from the post-correction
# regime_arch_sweep, then trains one cell (seed 97 x 4 folds) with
# the encoder pulled into the loop and wrapped in PEFT LoRA.
# Output lands at data/artifacts/encoder_lora_ceiling_probe/<pkg>/.
lora-ceiling-probe:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python scripts/run_lora_ceiling_probe.py \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		$(if $(LORA_ARCHITECTURE),--architecture "$(LORA_ARCHITECTURE)",)

# GARCH(1,1) classical-finance reference baseline. Fits per fold on
# SPX log-returns up to train_end, forecasts 10-day forward conditional
# vol, bins via train-slice quantile cutoffs, reports pooled macro-F1
# with a moving-block bootstrap CI under
# data/artifacts/garch_baseline/$(TRAINING_PACKAGE_ID)/.
garch-baseline:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python scripts/garch_baseline.py \
		--training-package-id "$(TRAINING_PACKAGE_ID)"

# Pooled-fold macro-F1 with bootstrap CIs across every per-fold trial in
# INPUT_DIR. INPUT_DIR is recursively walked for
# forecaster_sweep_results.json files.
regime-pooled-aggregate:
	@test -n "$(INPUT_DIR)" || (echo "INPUT_DIR is required"; exit 1)
	docker compose run --rm backend \
		python -m app.evaluation.regime_pooled_aggregator \
		--input-dir "$(INPUT_DIR)"

# Multi-architecture ensemble macro-F1. ARCH_SWEEP_DIR is the parent of
# the per-architecture subdirectories produced by ``regime-arch-sweep``.
regime-ensemble-aggregate:
	@test -n "$(ARCH_SWEEP_DIR)" || (echo "ARCH_SWEEP_DIR is required"; exit 1)
	docker compose run --rm backend \
		python -m app.evaluation.ensemble_aggregator \
		--arch-sweep-dir "$(ARCH_SWEEP_DIR)"

# Phase B (#227) capacity push -- random-search across hidden / schedule
# / weight-decay at the Tier 5 surface (rich + LLM, no NLP). LR_SCHEDULES
# defaults to both options so the cosine_warmup branch lands alongside
# the legacy plateau path.
regime-capacity-push:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python scripts/run_regime_capacity_push.py \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--report-root /data/artifacts/regime_capacity_push

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

# Round 3 (#242) corpus ablation: strict FOMC-only continued-pretrain.
# Loads only fomc_statements.json + fomc_minutes.json (~48 docs, ~240k
# tokens), bypassing BIS entirely. Pair with finbert-bis-only-pretrain
# for the 3-way ablation against the legacy mixed substrate.
finbert-fomc-only-pretrain:
	$(MAKE) finbert-fed-adjacent-pretrain \
		SUBSTRATE=fomc \
		PRETRAIN_OUT_NAME=finbert_fomc_only

# Round 3 (#242) corpus ablation: strict BIS-only continued-pretrain.
# Drops the legacy local Fed-adjacent JSON corpus so the encoder learns
# only from samchain/BIS_speeches_97_23_MLM. Pair with the FOMC-only
# variant above to compare narrow-but-on-target against broad-but-mixed.
finbert-bis-only-pretrain:
	$(MAKE) finbert-fed-adjacent-pretrain \
		SUBSTRATE=bis \
		PRETRAIN_OUT_NAME=finbert_bis_only

# DAPT substrate extension: BIS speeches + 5 cross-bank gtfintechlab corpora
# (ECB / BoJ / BoE / BoC / RBA) reformatted as consecutive-sentence NSP pairs.
# Same hyperparameters as the headline finbert-fed-adjacent-pretrain run; only
# the substrate changes. Output checkpoint registers under
# ``finbert_fed_adjacent_xbank_dapt`` in models/registry.yaml after the first
# GPU run lands and the embedding cache builder self-pins the revision.
finbert-fed-adjacent-xbank-dapt-pretrain:
	$(MAKE) finbert-fed-adjacent-pretrain \
		SUBSTRATE=bis_xbank \
		PRETRAIN_OUT_NAME=finbert_fed_adjacent_xbank_dapt

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
# Reads the activity + inflation panel (UNRATE, CPIAUCSL, PCEPILFE, MANEMP,
# PAYEMS, RSAFS) plus the rates + financial-conditions panel (DGS10, T10Y2Y,
# T10Y3M, BAMLH0A0HYM2, NFCI, DFII10). Requires FRED_API_KEY in .env on first
# run (cached JSON afterwards).
MACRO_STATE_START ?= 2010-01-01
MACRO_STATE_END ?= today
macro-state: build-macro-state

build-macro-state:
	@test -f .env || (echo ".env required for FRED_API_KEY"; exit 1)
	docker compose run --rm backend \
		python -m app.data.macro_state \
		--output data/external/fred/macro_state.parquet \
		--start "$(MACRO_STATE_START)" \
		--end "$(MACRO_STATE_END)"

# Build the monetary-policy surprise time-series at
# data/external/fred/mp_surprises.parquet. Requires FRED_API_KEY in .env on
# first run; subsequent runs reuse the per-series JSON cache.
build-mp-surprises:
	@test -f .env || (echo ".env required for FRED_API_KEY"; exit 1)
	docker compose run --rm backend \
		python -m app.data.mp_surprise \
		--output data/external/fred/mp_surprises.parquet

# Build the daily rates panel at data/external/fred/rates_panel.parquet
# (#291). Carries DGS1/2/5/10, T10Y2Y, T10Y3M, DFEDTARU, DFEDTARL with
# strict-backward publication-date indexing. Required by the
# event_dataset_builder when emitting rates-complex forward targets and
# pre-meeting expectation features. Requires FRED_API_KEY in .env on
# first run; subsequent runs reuse the per-series JSON cache.
RATES_PANEL_START ?= 2008-01-01
RATES_PANEL_END ?= today
build-rates-panel:
	@test -f .env || (echo ".env required for FRED_API_KEY"; exit 1)
	docker compose run --rm backend \
		python -m app.data.rates_panel \
		--start "$(RATES_PANEL_START)" \
		--end "$(RATES_PANEL_END)" \
		--output data/external/fred/rates_panel.parquet

# Re-emit linguistic_features.parquet for a given training package. The
# default output filename and LDA-artifact filenames live under the package
# directory. Required: TRAINING_PACKAGE_ID.
rebuild-linguistic-features:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose run --rm backend \
		python -m app.features.linguistic \
		--training-package-id "$(TRAINING_PACKAGE_ID)"

# Cache voyage-finance-2 sentence embeddings for a training package under
# data/raw/embeddings/. Requires VOYAGE_API_KEY in .env. The Voyage REST
# API is contacted only when --allow-network is passed.
cache-voyage-embeddings:
	@test -f .env || (echo ".env required for VOYAGE_API_KEY"; exit 1)
	docker compose run --rm backend \
		python scripts/cache_voyage_embeddings.py --allow-network \
		$(if $(TRAINING_PACKAGE_ID),--training-package-id "$(TRAINING_PACKAGE_ID)",) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(FORCE),--force,)

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

# #304 dual-head methodology runner. Runs the three head-mode
# configurations (classification / regression / dual) across the
# official seed set and a 40-epoch budget; aggregates per-fold
# regime_f1_macro + regression_rmse_log_rv into a single comparison
# JSON the §16 finalization-roadmap table reads.
dual-head-comparison:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m scripts.run_dual_head_comparison \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--seeds 11 29 47 71 97 \
		--epochs 40

# #309 derived-text-features ablation runner. Trains the forecaster
# under baseline / ablation / replacement configurations across the
# official seed set so the §16 table can quantify whether the
# per-sentence derived text features carry forecaster-relevant signal
# over the document-level encoder path.
derived-features-ablation:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m scripts.run_derived_features_ablation \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		--seeds 11 29 47 71 97 \
		--epochs 40


# #292 / #317 finding #16 rates-heads sweep runner. Trains the three
# rates heads (2y / 5y / terminal) on the official 5-seed x 4-fold
# protocol and emits a per-head MAE-bps + directional-accuracy + R^2
# panel keyed by fold and seed. SEED is the single-seed override
# (defaults to the full official seed set when unset).
rates-heads-sweep:
	@test -n "$(TRAINING_PACKAGE_ID)" || (echo "TRAINING_PACKAGE_ID is required"; exit 1)
	docker compose --profile "$(FORECASTER_COMPOSE_PROFILE)" run --rm "$(FORECASTER_COMPOSE_SERVICE)" \
		python -m scripts.run_rates_heads_sweep \
		--training-package-id "$(TRAINING_PACKAGE_ID)" \
		$(if $(SEED),--seeds $(SEED),--seeds 11 29 47 71 97)

# #322 canonical dual-head comparison. Pins the regression-alpha,
# output path, seed set, and epoch budget the §16 finalization-roadmap
# table reads, so the canonical run is reproducible without remembering
# the flag combination. Output lands at
# ``artifacts/experiments/dual_head_comparison_canonical.json``.
canonical-comparison:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_dual_head_comparison \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/dual_head_comparison_canonical.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--regression-alpha 0.5

# #305 opt-in variant. Same canonical sweep with the FOMC-attributable
# rates target. Output path is distinct so a #305 sweep can run
# concurrently with the canonical sweep without overwriting it.
canonical-comparison-fomc-attributable:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_dual_head_comparison \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/dual_head_comparison_post_305.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--regression-alpha 0.5 \
		--rates-target-mode fomc_attributable

# #306 opt-in variant. Attaches the 5-dim retrieval-analog summary
# block to every supervised event. Distinct output path.
canonical-comparison-retrieval-analogs:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_dual_head_comparison \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/dual_head_comparison_post_306.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--regression-alpha 0.5 \
		--use-retrieval-analogs

# #307 opt-in variant. Attaches the 3-scalar macro-regime block and
# mounts the multiplicative gate over the rich-feature slice. Distinct
# output path.
canonical-comparison-regime-conditioning:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_dual_head_comparison \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/dual_head_comparison_post_307.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--regression-alpha 0.5 \
		--use-regime-conditioning

# #327 text-path A/B comparison. Runs the three configurations
# (broadcast-static / per-bar / flat MLP) across the official seed set
# and canonical fold protocol; emits a per-arm JSON the §6.15 wiki
# table reads. Output lands at
# ``artifacts/experiments/text_path_ab.json``.
text-path-ab:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_text_path_ab \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/text_path_ab.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--head-mode dual \
		--regression-alpha 0.5

# #334 per-family rich-feature ablation runner. Zeros each rich-feature
# family one at a time (linguistic / credibility / mp_surprise /
# multi_axis / realised_vol / cross_asset / llm_features) plus a
# cumulative chain that drops text -> text+market-aux -> everything-
# except-legacy-market. Backs the §6 substitution-finding table.
per-family-ablation:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m scripts.run_per_family_ablation \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/per_family_ablation.json \
		--seeds 11 29 47 71 97 \
		--epochs 40 \
		--head-mode dual \
		--regression-alpha 0.5

# #213 B2 end-to-end fine-tune harness. Fine-tunes
# AutoModelForSequenceClassification directly on FOMC document text
# against the per-fold vol_regime_10d 3-class label. The encoder
# defaults to the classifier-role alias per ADR 0019; override via
# ENCODER_ALIAS=<alias>. Output lands at
# ``artifacts/experiments/finetune_pilot_b2.json``.
finetune-pilot-b2:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m app.data.finetune_pilot_b2 \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/finetune_pilot_b2.json \
		--seeds 11 29 47 71 97 \
		--epochs 5 \
		--train-batch-size 16 \
		--learning-rate 2e-5 \
		--weight-decay 0.01 \
		$(if $(ENCODER_ALIAS),--encoder-alias $(ENCODER_ALIAS),)

# #33 Path B — PhraseBank as a supervised auxiliary task on top of the
# B2 fine-tune. Same harness as ``finetune-pilot-b2`` with
# ``--enable-phrasebank-aux`` flipped on; output lands at a sibling
# artefact so the §6.x tier table can compare the two cells head-to-head.
# PHRASEBANK_AUX_LAMBDA defaults to 0.3 (per ADR 0033); sweep ``{0.1,
# 0.3, 0.5, 1.0}`` to isolate the lambda knob.
finetune-pilot-b2-phrasebank:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	docker compose run --rm backend python -m app.data.finetune_pilot_b2 \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--output artifacts/experiments/finetune_pilot_b2_phrasebank.json \
		--seeds 11 29 47 71 97 \
		--epochs 5 \
		--train-batch-size 16 \
		--learning-rate 2e-5 \
		--weight-decay 0.01 \
		--enable-phrasebank-aux \
		--phrasebank-aux-lambda $(if $(PHRASEBANK_AUX_LAMBDA),$(PHRASEBANK_AUX_LAMBDA),0.3) \
		--phrasebank-subset $(if $(PHRASEBANK_SUBSET),$(PHRASEBANK_SUBSET),sentences_allagree) \
		$(if $(ENCODER_ALIAS),--encoder-alias $(ENCODER_ALIAS),)

cross-source-transfer:
	@if [ -z "$$TRAINING_PACKAGE_ID" ]; then echo "TRAINING_PACKAGE_ID required" >&2; exit 1; fi
	@if [ -z "$$ENCODER_CHECKPOINTS" ]; then echo "ENCODER_CHECKPOINTS required (alias=path[,alias=path])" >&2; exit 1; fi
	docker compose run --rm backend python -m app.evaluation.cross_source_transfer \
		--training-package-id $$TRAINING_PACKAGE_ID \
		--encoder-checkpoints "$$ENCODER_CHECKPOINTS" \
		$(if $(SOURCE_TYPES),--source-types "$(SOURCE_TYPES)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)
