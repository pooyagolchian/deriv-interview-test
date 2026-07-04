"""Typed models + validating loaders for the three input files.

Validation fails loud and early with actionable messages so a malformed fixture never
propagates silently through the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from . import config
from .io_utils import load_json


class SchemaError(ValueError):
    """Raised when an input file does not match the expected schema."""


# --------------------------------------------------------------------------- models
@dataclass(frozen=True)
class KBDoc:
    doc_id: str
    title: str
    text: str

    @property
    def index_text(self) -> str:
        """Title + body — what retrieval actually indexes."""
        return f"{self.title} {self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    user_question: str
    expected_doc_ids: List[str]
    must_include_any: List[str]
    must_not_claim: List[str]
    risk_level: str


@dataclass(frozen=True)
class Candidate:
    query_id: str
    answers: Dict[str, str]


# --------------------------------------------------------------------------- helpers
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _as_str_list(value, where: str) -> List[str]:
    _require(isinstance(value, list), f"{where} must be a list, got {type(value).__name__}")
    for i, v in enumerate(value):
        _require(isinstance(v, str), f"{where}[{i}] must be a string")
    return list(value)


# --------------------------------------------------------------------------- loaders
def load_kb(path=None) -> List[KBDoc]:
    raw = load_json(path or config.input_path(config.KB_FILE))
    _require(isinstance(raw, list) and raw, "kb.json must be a non-empty list")
    docs: List[KBDoc] = []
    seen = set()
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"kb.json[{i}] must be an object")
        for key in ("doc_id", "title", "text"):
            _require(key in item and isinstance(item[key], str) and item[key] != "",
                     f"kb.json[{i}].{key} must be a non-empty string")
        _require(item["doc_id"] not in seen, f"kb.json duplicate doc_id: {item['doc_id']}")
        seen.add(item["doc_id"])
        docs.append(KBDoc(doc_id=item["doc_id"], title=item["title"], text=item["text"]))
    return docs


def load_queries(path=None) -> List[Query]:
    raw = load_json(path or config.input_path(config.QUERIES_FILE))
    _require(isinstance(raw, list) and raw, "queries.json must be a non-empty list")
    queries: List[Query] = []
    seen = set()
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"queries.json[{i}] must be an object")
        qid = item.get("query_id")
        _require(isinstance(qid, str) and qid != "", f"queries.json[{i}].query_id required")
        _require(qid not in seen, f"queries.json duplicate query_id: {qid}")
        seen.add(qid)
        _require(isinstance(item.get("user_question"), str) and item["user_question"] != "",
                 f"queries.json[{i}].user_question required")
        risk = item.get("risk_level")
        _require(risk in config.RISK_LEVELS,
                 f"queries.json[{i}].risk_level must be one of {config.RISK_LEVELS}, got {risk!r}")
        queries.append(Query(
            query_id=qid,
            user_question=item["user_question"],
            expected_doc_ids=_as_str_list(item.get("expected_doc_ids", []), f"queries.json[{i}].expected_doc_ids"),
            must_include_any=_as_str_list(item.get("must_include_any", []), f"queries.json[{i}].must_include_any"),
            must_not_claim=_as_str_list(item.get("must_not_claim", []), f"queries.json[{i}].must_not_claim"),
            risk_level=risk,
        ))
    return queries


def load_candidates(path=None) -> List[Candidate]:
    raw = load_json(path or config.input_path(config.CANDIDATES_FILE))
    _require(isinstance(raw, list) and raw, "candidate_answers.json must be a non-empty list")
    cands: List[Candidate] = []
    seen = set()
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"candidate_answers.json[{i}] must be an object")
        qid = item.get("query_id")
        _require(isinstance(qid, str) and qid != "", f"candidate_answers.json[{i}].query_id required")
        _require(qid not in seen, f"candidate_answers.json duplicate query_id: {qid}")
        seen.add(qid)
        answers = item.get("answers")
        _require(isinstance(answers, dict) and answers,
                 f"candidate_answers.json[{i}].answers must be a non-empty object")
        for variant, ans in answers.items():
            _require(isinstance(variant, str) and variant != "",
                     f"candidate_answers.json[{i}] has an empty variant name")
            _require(isinstance(ans, str), f"candidate_answers.json[{i}].answers.{variant} must be a string")
        cands.append(Candidate(query_id=qid, answers=dict(answers)))
    return cands


def variant_universe(candidates: List[Candidate]) -> List[str]:
    """Sorted union of every variant key seen across all queries (supports N variants)."""
    universe: set = set()
    for c in candidates:
        universe.update(c.answers.keys())
    return sorted(universe)
