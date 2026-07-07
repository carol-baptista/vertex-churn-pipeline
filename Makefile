# Usage:
#   make test
#   make train
#   make package          # Phase 3: assemble RF serving bundle
#   make deploy           # Phase 3: upload + register + deploy (GCP cost)

TARGET ?= tests/
ARGS ?=

# Training defaults (override on the command line, e.g. make train METRIC=f2)
METRIC ?= f1
POS_WEIGHT ?= sqrt
THRESHOLD_STRATEGY ?= f1
RECALL_FLOOR ?= 0.75
FEATURE_SET ?= baseline
MODEL ?= random_forest
ROW ?= 0
CUSTOMER_ID ?=
LIMIT ?= 500

# Phase 3 deploy flags
DRY_RUN ?= 0
REGISTER_ONLY ?= 0

COMMON_TRAIN_ARGS = \
	--metric $(METRIC) \
	--pos-weight $(POS_WEIGHT) \
	--threshold-strategy $(THRESHOLD_STRATEGY) \
	--recall-floor $(RECALL_FLOOR)

TRAIN_ARGS = \
	--feature-set $(FEATURE_SET) \
	$(COMMON_TRAIN_ARGS)

DEPLOY_EXTRA :=
ifeq ($(DRY_RUN),1)
DEPLOY_EXTRA += --dry-run
endif
ifeq ($(REGISTER_ONLY),1)
DEPLOY_EXTRA += --register-only
endif

.PHONY: help sync test train train-baseline train-smoke train-fast train-probe train-probe-compare fairness predict package package-test deploy undeploy seed-scoring score-local score-vertex warm-cache cache-lookup

help:
	@echo "Targets:"
	@echo "  make sync              Install dependencies (incl. dev / pytest)"
	@echo "  make test              Run all tests under tests/"
	@echo "  make test TARGET=...   Run one file or one test (pytest node id)"
	@echo "  make test ARGS='...'   Extra pytest flags, e.g. -k tenure_zero -v"
	@echo ""
	@echo "Phase 2 pipeline (BigQuery -> train -> verify):"
	@echo "  make train-baseline    Optional: freeze baseline snapshot for comparisons"
	@echo "  make train             Full train (default: baseline features)"
	@echo "  make fairness          Print test fairness slices (MODEL=$(MODEL))"
	@echo "  make predict           Score one BigQuery row with saved artifact"
	@echo "  make predict CUSTOMER_ID=7590-VHVEG"
	@echo ""
	@echo "Phase 3 deploy (Random Forest champion only):"
	@echo "  make package           Copy RF artifact -> serving/churn-rf/v1/"
	@echo "  make package-test      Local bundle smoke test (free)"
	@echo "  make deploy DRY_RUN=1  Print deploy plan without GCP changes"
	@echo "  make deploy            Upload GCS + CPR image + Registry + Endpoint"
	@echo "  make deploy REGISTER_ONLY=1  Register model, skip endpoint"
	@echo "  make undeploy          Stop endpoint billing"
	@echo ""
	@echo "Phase 4 batch scoring (BQ predictions table):"
	@echo "  make seed-scoring      Sample customers_scoring (LIMIT=$(LIMIT), no Churn label)"
	@echo "  make score-local       Score with local model -> churn_ml.predictions"
	@echo "  make score-vertex      Score via Vertex BatchPredictionJob -> predictions"
	@echo "  make warm-cache        Export latest scores -> data/cache/ (hybrid read path)"
	@echo "  make cache-lookup CUSTOMER_ID=7590-VHVEG  Read one customer from cache"
	@echo ""
	@echo "Training options (feature_set=$(FEATURE_SET), metric=$(METRIC), pos=$(POS_WEIGHT)):"
	@echo "  make train FEATURE_SET=engineered  Demo engineered features vs baseline"
	@echo "  make train-smoke       Quick run on 2000 rows, no tuning"
	@echo "  make train-fast        Full data, skip tree-model grid searches"
	@echo "  make train-probe       Probe audit only (--probe-feature)"
	@echo "  make train-probe-compare  Full vs probe-selected training comparison"
	@echo ""
	@echo "Artifacts (after make train):"
	@echo "  make fairness MODEL=xgboost"
	@echo "  make predict MODEL=xgboost ROW=3"

sync:
	uv sync --extra dev

test: sync
	uv run pytest $(TARGET) $(ARGS)

train: sync
	uv run python -m src.train $(TRAIN_ARGS)

train-baseline: sync
	uv run python -m src.train --feature-set baseline --save-baseline $(COMMON_TRAIN_ARGS)

train-smoke: sync
	uv run python -m src.train --sample 2000 --no-tune $(TRAIN_ARGS)

train-fast: sync
	uv run python -m src.train --no-tune $(TRAIN_ARGS)

train-probe: sync
	uv run python -m src.train --probe-feature --no-tune $(TRAIN_ARGS)

train-probe-compare: sync
	uv run python -m src.train --probe-train --no-tune $(TRAIN_ARGS)

fairness:
	uv run python -m src.inspect --model $(MODEL)

predict:
ifeq ($(strip $(CUSTOMER_ID)),)
	uv run python -m src.predict --model $(MODEL) --row $(ROW)
else
	uv run python -m src.predict --model $(MODEL) --customer-id $(CUSTOMER_ID)
endif

package:
	uv run python -m src.package package

package-test: package
	uv run python -m src.package smoke-test

deploy: package
	uv run python -m src.deploy $(DEPLOY_EXTRA)

undeploy:
	uv run python -m src.deploy --undeploy

seed-scoring:
	uv run python -m src.batch seed --limit $(LIMIT)

score-local:
	uv run python -m src.batch score-local

score-vertex:
	uv run python -m src.batch score-vertex

warm-cache:
	uv run python -m src.cache_warm warm

cache-lookup:
	uv run python -m src.cache_warm lookup --customer-id $(CUSTOMER_ID)
