# RAG evaluation harness — common commands.
# Uses `python3` by default; override with `make PY=python run`.
PY ?= python3

# Project virtual environment. `.venv/` is gitignored; recreate it with `make venv`
# from the pinned requirements-dev.txt (which IS versioned) for a reproducible env.
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
# Auto-prefer the project venv when it exists; fall back to system $(PY) otherwise.
RUN_PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(PY))

.PHONY: help venv setup run validate test clean all

help:
	@echo "Targets:"
	@echo "  venv      Create $(VENV) and install pinned dev deps from requirements-dev.txt"
	@echo "  setup     Editable install with dev extras into the venv (needs package code)"
	@echo "  run       Regenerate all evaluation artifacts (python run.py)"
	@echo "  validate  Validate artifacts and recommendation reproducibility (python validate.py)"
	@echo "  test      Run the pytest suite"
	@echo "  all       run + validate + test"
	@echo "  clean     Remove generated artifacts and caches"

# Create/refresh the virtual environment from the pinned lock file.
venv: $(VENV_PY)

$(VENV_PY): requirements-dev.txt
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-dev.txt

setup: venv
	$(VENV_PY) -m pip install -e ".[dev]"

run:
	$(RUN_PY) run.py

validate:
	$(RUN_PY) validate.py

test:
	$(RUN_PY) -m pytest

all: run validate test

clean:
	rm -f retrieval.json automated_scores.json llm_review.json \
	      recommendation.md recommendation.json review_report.md \
	      failure_taxonomy.json llm_calls.jsonl
	rm -rf .pytest_cache **/__pycache__ __pycache__
