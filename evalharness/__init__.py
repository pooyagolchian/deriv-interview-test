"""Deterministic, replayable offline evaluator for retrieval-augmented QA prompt variants.

Pipeline stages (see ``pipeline.py``):

    retrieval -> rule checks -> (one) LLM review -> aggregation -> reports/validation

Design principles: everything except the single LLM review call is deterministic and runs
from local files; the promotion recommendation is computed in code and is reproducible from
stored artifacts.
"""

__version__ = "0.1.0"
