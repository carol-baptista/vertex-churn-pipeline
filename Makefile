# Usage:
#   make test
#   make test TARGET=tests/test_preprocess.py
#   make test TARGET=tests/test_preprocess.py::test_tenure_zero_fills_blank_total_charges
#   make test ARGS="-k tenure_zero -v"

TARGET ?= tests/
ARGS ?=

.PHONY: help sync test train train-smoke train-fast

help:
	@echo "Targets:"
	@echo "  make sync              Install dependencies (incl. dev / pytest)"
	@echo "  make test              Run all tests under tests/"
	@echo "  make test TARGET=...   Run one file or one test (pytest node id)"
	@echo "  make test ARGS='...'   Extra pytest flags, e.g. -k tenure_zero -v"
	@echo "  make train             Full train + compare (BigQuery, XGBoost tuning)"
	@echo "  make train-smoke       Quick run on 2000 rows, no tuning"
	@echo "  make train-fast        Full data, skip XGBoost grid search"

sync:
	uv sync --extra dev

test: sync
	uv run pytest $(TARGET) $(ARGS)

train: sync
	uv run python -m src.train

train-smoke: sync
	uv run python -m src.train --sample 2000 --no-tune

train-fast: sync
	uv run python -m src.train --no-tune
