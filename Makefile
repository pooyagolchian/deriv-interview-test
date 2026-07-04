# RAG evaluation harness — common commands.
# Uses `python3` by default; override with `make PY=python run`.
PY ?= python3

.PHONY: help setup run validate test clean all

help:
	@echo "Targets:"
	@echo "  setup     Install optional deps (anthropic + pytest) for the real LLM call and tests"
	@echo "  run       Regenerate all evaluation artifacts (python run.py)"
	@echo "  validate  Validate artifacts and recommendation reproducibility (python validate.py)"
	@echo "  test      Run the pytest suite"
	@echo "  all       run + validate + test"
	@echo "  clean     Remove generated artifacts and caches"

setup:
	$(PY) -m pip install -e ".[dev]"

run:
	$(PY) run.py

validate:
	$(PY) validate.py

test:
	$(PY) -m pytest

all: run validate test

clean:
	rm -f retrieval.json automated_scores.json llm_review.json \
	      recommendation.md recommendation.json review_report.md \
	      failure_taxonomy.json llm_calls.jsonl
	rm -rf .pytest_cache **/__pycache__ __pycache__
