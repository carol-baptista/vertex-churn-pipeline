# Usage:
#   make test
#   make test TARGET=tests/test_preprocess.py
#   make test TARGET=tests/test_preprocess.py::test_tenure_zero_fills_blank_total_charges
#   make test ARGS="-k tenure_zero -v"

TARGET ?= tests/
ARGS ?=

.PHONY: help sync test

help:
	@echo "Targets:"
	@echo "  make sync              Install dependencies (incl. dev / pytest)"
	@echo "  make test              Run all tests under tests/"
	@echo "  make test TARGET=...   Run one file or one test (pytest node id)"
	@echo "  make test ARGS='...'   Extra pytest flags, e.g. -k tenure_zero -v"

sync:
	uv sync --extra dev

test: sync
	uv run pytest $(TARGET) $(ARGS)
