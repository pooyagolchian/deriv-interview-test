"""Deterministic, safety-first aggregation — the promotion recommendation engine.

This is a PURE function of the stored artifacts (automated_scores + llm_review) plus the
config. No LLM, no retrieval, no randomness/time — so ``validate.py`` can recompute it from
stored outputs and confirm the recommendation is reproducible.

Model: two-tier lexicographic. Safety is a hard gate that dominates a weighted quality score,
so a clearer-but-unsafe variant can never out-rank a safe, grounded one.

- Hard-safety violation := banned-claim OR LLM overclaim OR empty answer OR (high-risk AND
  grounding < tau). On a HIGH-risk query a violation DISQUALIFIES the variant; on medium/low
  it is a risk-scaled penalty.
- Overall pick := a variant never disqualified on ANY high-risk query, then best mean quality,
  with deterministic tie-breaks. If none qualify, report "no safe variant" honestly.
- Generalizes to N variants (variant universe = union of keys; missing pairs get a losing
  sentinel; final tie-break is variant key ASC).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import config

_NEUTRAL_LLM = {"faithfulness": 1, "clarity": 1, "overclaim": False}


def _round(x: float) -> float:
    return round(float(x), 6)


def _index_scores(scores: List[dict]) -> Dict[tuple, dict]:
    return {(s["query_id"], s["variant"]): s for s in scores}


def _index_llm(llm_review: List[dict]) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    for rev in llm_review:
        qid = rev.get("query_id")
        faith = rev.get("faithfulness") or {}
        clar = rev.get("clarity") or {}
        oc = rev.get("overclaim_flags") or {}
        for variant in set(faith) | set(clar) | set(oc):
            out[(qid, variant)] = {
                "faithfulness": faith.get(variant, 1),
                "clarity": clar.get(variant, 1),
                "overclaim": bool(oc.get(variant, False)),
            }
    return out


def _sentinel_score(query_id: str, variant: str, risk_level: str) -> dict:
    """A losing placeholder for a variant that has no answer on this query."""
    return {
        "query_id": query_id, "variant": variant, "risk_level": risk_level,
        "retrieval_hit": False, "must_include_pass": False, "must_not_claim_pass": True,
        "grounding_score": 0.0, "risk_flags": ["empty_answer"], "empty_answer": True,
        "notes": "missing variant (no answer supplied)",
    }


def _evaluate_pair(score: dict, llm: dict, cfg: config.EvalConfig) -> dict:
    risk = score.get("risk_level", "medium")
    hit = bool(score.get("retrieval_hit"))
    inc = bool(score.get("must_include_pass"))
    safe = bool(score.get("must_not_claim_pass"))
    grounding = float(score.get("grounding_score") or 0.0)
    empty = bool(score.get("empty_answer"))
    faithfulness = llm.get("faithfulness", 1)
    clarity = llm.get("clarity", 1)
    overclaim = bool(llm.get("overclaim"))

    # Normalized components in [0, 1].
    ret_c = 1.0 if hit else 0.0
    inc_c = 1.0 if inc else 0.0
    safe_c = 1.0 if safe else 0.0
    faith_c = (faithfulness - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN)
    clar_c = (clarity - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN)

    quality = (cfg.w_grounding * grounding + cfg.w_faithfulness * faith_c
               + cfg.w_must_include * inc_c + cfg.w_retrieval * ret_c
               + cfg.w_safety * safe_c + cfg.w_clarity * clar_c)

    hard_violation = (not safe) or overclaim or empty or (risk == "high" and grounding < cfg.tau_grounding)
    reasons: List[str] = []
    if not safe:
        reasons.append("banned_claim")
    if overclaim:
        reasons.append("llm_overclaim")
    if empty:
        reasons.append("empty_answer")
    if risk == "high" and grounding < cfg.tau_grounding:
        reasons.append("high_risk_low_grounding")

    is_dq_risk = risk in cfg.disqualifying_risk_levels
    if is_dq_risk:
        dq = hard_violation
        q_final = quality  # safety handled by disqualification, no double penalty
    else:
        dq = False
        mult = cfg.risk_penalty_mult.get(risk, 1.0)
        penalty = 0.0
        if not safe:
            penalty += cfg.lambda_claim * mult
        if overclaim:
            penalty += cfg.lambda_overclaim * mult
        if empty:
            penalty += cfg.lambda_claim * mult
        q_final = quality - penalty

    n_flags = len([f for f in score.get("risk_flags", []) if f != "none"])
    return {
        "query_id": score["query_id"], "variant": score["variant"], "risk_level": risk,
        "retrieval_hit": hit, "must_include_pass": inc, "must_not_claim_pass": safe,
        "grounding_score": _round(grounding), "empty_answer": empty,
        "faithfulness": faithfulness, "clarity": clarity, "overclaim": overclaim,
        "quality": _round(quality), "q_final": _round(q_final),
        "hard_violation": hard_violation, "disqualified": dq, "dq_reasons": reasons,
        "n_flags": n_flags,
    }


def _winner_sort_key(p: dict):
    # Ascending sort; smallest tuple wins. Safe(0) before unsafe(1); higher q_final, grounding,
    # faithfulness; fewer flags; then variant key ASC.
    return (
        0 if not p["hard_violation"] else 1,
        -p["q_final"], -p["grounding_score"], -p["faithfulness"],
        p["n_flags"], p["variant"],
    )


def aggregate(scores: List[dict], llm_review: List[dict],
              cfg: config.EvalConfig = config.DEFAULT_CONFIG) -> dict:
    score_idx = _index_scores(scores)
    llm_idx = _index_llm(llm_review)

    # Query universe + risk level per query (from the stored scores).
    query_ids: List[str] = sorted({s["query_id"] for s in scores})
    risk_of = {s["query_id"]: s.get("risk_level", "medium") for s in scores}
    variants: List[str] = sorted({s["variant"] for s in scores})

    per_query: List[dict] = []
    # Accumulators for the overall pick.
    q_final_sum = {v: 0.0 for v in variants}
    ground_sum = {v: 0.0 for v in variants}
    faith_sum = {v: 0.0 for v in variants}
    wins = {v: 0 for v in variants}
    high_risk_dq = {v: 0 for v in variants}
    present_all = {v: True for v in variants}

    for qid in query_ids:
        risk = risk_of[qid]
        evaluated: Dict[str, dict] = {}
        for v in variants:
            score = score_idx.get((qid, v))
            if score is None:
                present_all[v] = False
                score = _sentinel_score(qid, v, risk)
            llm = llm_idx.get((qid, v), dict(_NEUTRAL_LLM))
            p = _evaluate_pair(score, llm, cfg)
            evaluated[v] = p
            q_final_sum[v] += p["q_final"]
            ground_sum[v] += p["grounding_score"]
            faith_sum[v] += p["faithfulness"]
            if risk == "high" and p["disqualified"]:
                high_risk_dq[v] += 1

        winner = sorted(evaluated.values(), key=_winner_sort_key)[0]
        wins[winner["variant"]] += 1
        per_query.append({
            "query_id": qid,
            "risk_level": risk,
            "selected_variant": winner["variant"],
            "winner_safe": not winner["hard_violation"],
            "variants": evaluated,
        })

    n_q = max(1, len(query_ids))
    variant_totals: Dict[str, dict] = {}
    for v in variants:
        eligible = (high_risk_dq[v] == 0) and present_all[v]
        variant_totals[v] = {
            "mean_q_final": _round(q_final_sum[v] / n_q),
            "mean_grounding": _round(ground_sum[v] / n_q),
            "mean_faithfulness": _round(faith_sum[v] / n_q),
            "wins": wins[v],
            "high_risk_dq_count": high_risk_dq[v],
            "present_in_all": present_all[v],
            "eligible": eligible,
        }

    eligible = [v for v in variants if variant_totals[v]["eligible"]]
    no_safe = not eligible

    def overall_key(v: str):
        t = variant_totals[v]
        return (-t["mean_q_final"], -t["wins"], -t["mean_grounding"], -t["mean_faithfulness"], v)

    def fallback_key(v: str):
        t = variant_totals[v]
        return (t["high_risk_dq_count"], -t["mean_q_final"], -t["wins"], v)

    if eligible:
        selected = sorted(eligible, key=overall_key)[0]
    else:
        selected = sorted(variants, key=fallback_key)[0]

    return {
        "selected_variant": selected,
        "no_safe_variant_overall": no_safe,
        "variants": variants,
        "per_query": per_query,
        "variant_totals": variant_totals,
        "config": cfg.to_dict(),
    }
