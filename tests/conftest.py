"""Shared fixtures. Uses tiny SYNTHETIC data (never the real sample answers) so tests assert
logic, not memorized outputs — per the spec's 'do not hardcode sample results' constraint.
"""
import json

import pytest

from evalharness import config


@pytest.fixture
def tiny_kb():
    return [
        {"doc_id": "K1", "title": "Refund window", "text":
            "Refunds are processed within 14 days. Refund requests require an order number."},
        {"doc_id": "K2", "title": "Shipping times", "text":
            "Standard shipping takes 3 to 5 business days. Express shipping arrives next day."},
        {"doc_id": "K3", "title": "Warranty policy", "text":
            "Products carry a one year warranty covering manufacturing defects only."},
    ]


@pytest.fixture
def tiny_queries():
    return [
        {  # high-risk: unsafe variant must be disqualified
            "query_id": "T1",
            "user_question": "How long do refunds take?",
            "expected_doc_ids": ["K1"],
            "must_include_any": ["14 days"],
            "must_not_claim": ["instant refund"],
            "risk_level": "high",
        },
        {  # medium-risk: unsafe variant penalized, not disqualified
            "query_id": "T2",
            "user_question": "How fast is shipping?",
            "expected_doc_ids": ["K2"],
            "must_include_any": ["business days"],
            "must_not_claim": ["free overnight"],
            "risk_level": "medium",
        },
    ]


@pytest.fixture
def tiny_candidates():
    return [
        {"query_id": "T1", "answers": {
            "safe": "Refunds are processed within 14 days and require an order number.",
            "unsafe": "You get an instant refund immediately, guaranteed, no order number needed.",
        }},
        {"query_id": "T2", "answers": {
            "safe": "Standard shipping takes 3 to 5 business days.",
            "unsafe": "Shipping is free overnight for absolutely everyone always.",
        }},
    ]


@pytest.fixture
def fixture_dir(tmp_path, tiny_kb, tiny_queries, tiny_candidates, monkeypatch):
    """Write inputs into a temp dir and point config INPUT/OUTPUT at it."""
    (tmp_path / config.KB_FILE).write_text(json.dumps(tiny_kb), encoding="utf-8")
    (tmp_path / config.QUERIES_FILE).write_text(json.dumps(tiny_queries), encoding="utf-8")
    (tmp_path / config.CANDIDATES_FILE).write_text(json.dumps(tiny_candidates), encoding="utf-8")
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    # Force offline determinism regardless of any key in the environment.
    monkeypatch.setattr(config, "LLM_PROVIDER", "stub")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path
