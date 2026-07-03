# Usage:
#   make test
#   make test TARGET=tests/test_preprocess.py
#   make test TARGET=tests/test_preprocess.py::test_tenure_zero_fills_blank_total_charges
#   make test ARGS="-k tenure_zero -v"
#   make train
#   make train METRIC=f1 POS_WEIGHT=sqrt

TARGET ?= tests/
ARGS ?=

# Training defaults (override on the command line, e.g. make train METRIC=f1)
METRIC ?= f2
POS_WEIGHT ?= sqrt
THRESHOLD_STRATEGY ?= recall_floor
RECALL_FLOOR ?= 0.75

TRAIN_ARGS = \
	--metric $(METRIC) \
	--pos-weight $(POS_WEIGHT) \
	--threshold-strategy $(THRESHOLD_STRATEGY) \
	--recall-floor $(RECALL_FLOOR)

.PHONY: help sync test train train-smoke train-fast train-probe train-probe-compare

help:
	@echo "Targets:"
	@echo "  make sync              Install dependencies (incl. dev / pytest)"
	@echo "  make test              Run all tests under tests/"
	@echo "  make test TARGET=...   Run one file or one test (pytest node id)"
	@echo "  make test ARGS='...'   Extra pytest flags, e.g. -k tenure_zero -v"
	@echo ""
	@echo "Training (defaults: metric=$(METRIC), pos=$(POS_WEIGHT), threshold=$(THRESHOLD_STRATEGY), recall>=$(RECALL_FLOOR)):"
	@echo "  make train             Full train + compare (4 models, tree tuning)"
	@echo "  make train-smoke       Quick run on 2000 rows, no tuning"
	@echo "  make train-fast        Full data, skip tree-model grid searches"
	@echo "  make train-probe       Probe audit only (--probe-feature)"
	@echo "  make train-probe-compare  Full vs probe-selected training comparison"
	@echo ""
	@echo "Override training knobs, e.g.:"
	@echo "  make train METRIC=f1 POS_WEIGHT=full THRESHOLD_STRATEGY=f1"

sync:
	uv sync --extra dev

test: sync
	uv run pytest $(TARGET) $(ARGS)

train: sync
	uv run python -m src.train $(TRAIN_ARGS)

train-smoke: sync
	uv run python -m src.train --sample 2000 --no-tune $(TRAIN_ARGS)

train-fast: sync
	uv run python -m src.train --no-tune $(TRAIN_ARGS)

train-probe: sync
	uv run python -m src.train --probe-feature --no-tune $(TRAIN_ARGS)

train-probe-compare: sync
	uv run python -m src.train --probe-train --no-tune $(TRAIN_ARGS)
