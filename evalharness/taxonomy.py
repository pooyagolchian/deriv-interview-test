"""Failure taxonomy: map deterministic check results (+ the LLM overclaim flag) to a small
controlled vocabulary of failure tags. Zero or more tags per answer.
"""
from __future__ import annotations

from typing import Dict, List

from . import config

# Controlled vocabulary.
TAG_UNSUPPORTED_CLAIM = "unsupported_claim"
TAG_MISSED_KEY_FACT = "missed_key_fact"
TAG_POLICY_VIOLATION = "policy_violation"
TAG_RETRIEVAL_MISS = "retrieval_miss"
TAG_OVERCONFIDENT_TONE = "overconfident_tone"
TAG_IRRELEVANT_ANSWER = "irrelevant_answer"

VOCABULARY = (
    TAG_UNSUPPORTED_CLAIM,
    TAG_MISSED_KEY_FACT,
    TAG_POLICY_VIOLATION,
    TAG_RETRIEVAL_MISS,
    TAG_OVERCONFIDENT_TONE,
    TAG_IRRELEVANT_ANSWER,
)


def tag_answer(score: dict, overclaim: bool, cfg: config.EvalConfig = config.DEFAULT_CONFIG) -> List[str]:
    tags: List[str] = []
    grounding = score.get("grounding_score", 0.0)
    inc = score.get("must_include_pass", True)

    if "retrieval_miss" in score.get("risk_flags", []):
        tags.append(TAG_RETRIEVAL_MISS)
    if not inc:
        tags.append(TAG_MISSED_KEY_FACT)
    if not score.get("must_not_claim_pass", True):
        tags.append(TAG_POLICY_VIOLATION)
    if grounding < cfg.tau_grounding:
        tags.append(TAG_UNSUPPORTED_CLAIM)
    if overclaim:
        tags.append(TAG_OVERCONFIDENT_TONE)
    # Off-topic: barely grounded AND missing the key fact (covers empty answers too).
    if (grounding < cfg.tau_irrelevant and not inc) or score.get("empty_answer", False):
        tags.append(TAG_IRRELEVANT_ANSWER)
    return tags


def build_taxonomy(scores: List[dict], llm_review: List[dict],
                   cfg: config.EvalConfig = config.DEFAULT_CONFIG) -> List[dict]:
    overclaim_lookup: Dict[tuple, bool] = {}
    for rev in llm_review:
        qid = rev.get("query_id")
        for variant, flag in (rev.get("overclaim_flags") or {}).items():
            overclaim_lookup[(qid, variant)] = bool(flag)

    out: List[dict] = []
    for s in scores:
        key = (s["query_id"], s["variant"])
        out.append({
            "query_id": s["query_id"],
            "variant": s["variant"],
            "tags": tag_answer(s, overclaim_lookup.get(key, False), cfg),
        })
    return out
