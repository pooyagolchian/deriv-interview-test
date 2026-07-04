"""Render human + machine artifacts from the aggregation result.

- ``recommendation.json`` — machine-readable sidecar (the aggregation result + input hashes);
  the source of truth that makes the recommendation reproducible and checkable.
- ``recommendation.md`` — the human promotion writeup, rendered from the sidecar.
- ``review_report.md`` — per-query explainability view for debugging.

No timestamps are written into these files so they stay byte-reproducible across runs.
"""
from __future__ import annotations

from typing import Dict, List

from . import config
from .io_utils import content_hash
from .schema import Candidate, Query


# --------------------------------------------------------------------------- sidecar
def build_recommendation_json(agg: dict, kb: list, queries_raw: list, candidates_raw: list) -> dict:
    return {
        "selected_variant": agg["selected_variant"],
        "no_safe_variant_overall": agg["no_safe_variant_overall"],
        "variants": agg["variants"],
        "variant_totals": agg["variant_totals"],
        "per_query": agg["per_query"],
        "config": agg["config"],
        "input_hashes": {
            "kb": content_hash(kb),
            "queries": content_hash(queries_raw),
            "candidate_answers": content_hash(candidates_raw),
        },
    }


# --------------------------------------------------------------------------- helpers
def _failed_checks(pq_variant: dict) -> List[str]:
    fails = []
    if not pq_variant["retrieval_hit"]:
        fails.append("retrieval_miss")
    if not pq_variant["must_include_pass"]:
        fails.append("missed_key_fact")
    if not pq_variant["must_not_claim_pass"]:
        fails.append("banned_claim")
    if pq_variant["overclaim"]:
        fails.append("llm_overclaim")
    if pq_variant["empty_answer"]:
        fails.append("empty_answer")
    return fails


# --------------------------------------------------------------------------- recommendation.md
def render_recommendation_md(agg: dict, queries: List[Query]) -> str:
    q_by_id = {q.query_id: q for q in queries}
    selected = agg["selected_variant"]
    totals = agg["variant_totals"]
    variants = agg["variants"]

    lines: List[str] = []
    lines.append("# Promotion Recommendation\n")
    if agg["no_safe_variant_overall"]:
        lines.append(f"> ⚠️ **No variant is safe to promote.** Least-unsafe fallback: "
                     f"**`{selected}`**. Do not ship without addressing the high-risk failures below.\n")
    else:
        lines.append(f"## ✅ Promote: `{selected}`\n")
        lines.append("Selected as the highest-quality variant that is **never disqualified on any "
                     "high-risk query**. Safety and grounding dominate the decision; clarity is a "
                     "tie-breaker only.\n")

    # Summary table.
    lines.append("## Summary\n")
    lines.append("| Variant | Wins | High-risk DQ | Eligible | Mean quality | Mean grounding |")
    lines.append("|---|---|---|---|---|---|")
    for v in variants:
        t = totals[v]
        mark = " ⬅️ promoted" if v == selected else ""
        lines.append(f"| `{v}`{mark} | {t['wins']} | {t['high_risk_dq_count']} | "
                     f"{'yes' if t['eligible'] else 'no'} | {t['mean_q_final']:.3f} | {t['mean_grounding']:.3f} |")
    lines.append("")

    # Per-query reasoning.
    lines.append("## Per-query reasoning\n")
    for pq in agg["per_query"]:
        q = q_by_id.get(pq["query_id"])
        question = q.user_question if q else ""
        lines.append(f"### {pq['query_id']} · risk: **{pq['risk_level']}** · winner: "
                     f"`{pq['selected_variant']}`")
        if question:
            lines.append(f"_{question}_\n")
        lines.append("| Variant | Grounding | Faith | Clarity | q_final | Disqualified | Failed checks |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in variants:
            pv = pq["variants"].get(v)
            if not pv:
                continue
            fails = ", ".join(_failed_checks(pv)) or "—"
            dq = "yes" if pv["disqualified"] else ("unsafe" if pv["hard_violation"] else "no")
            lines.append(f"| `{v}` | {pv['grounding_score']:.2f} | {pv['faithfulness']} | "
                         f"{pv['clarity']} | {pv['q_final']:.3f} | {dq} | {fails} |")
        lines.append("")

    # Top reasons.
    lines.append("## Top reasons for the decision\n")
    for reason in _top_reasons(agg, selected):
        lines.append(f"- {reason}")
    lines.append("")

    # Clarity vs safety tradeoff.
    lines.append("## Clarity vs. safety tradeoff\n")
    lines.append(_tradeoff_text(agg, selected))
    lines.append("")

    # Limitations.
    lines.append("## Known limitations of this evaluation harness\n")
    for lim in _LIMITATIONS:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def _top_reasons(agg: dict, selected: str) -> List[str]:
    totals = agg["variant_totals"]
    reasons: List[str] = []
    st = totals[selected]
    reasons.append(f"`{selected}` is never disqualified on a high-risk query "
                   f"(high-risk DQ count = {st['high_risk_dq_count']}) and is eligible for promotion.")
    reasons.append(f"`{selected}` has the best safety-weighted mean quality "
                   f"({st['mean_q_final']:.3f}) and mean grounding ({st['mean_grounding']:.3f}).")
    # Cite the strongest rejected variant.
    rejected = [v for v in agg["variants"] if v != selected]
    dq_rejects = sorted(rejected, key=lambda v: -totals[v]["high_risk_dq_count"])
    if dq_rejects and totals[dq_rejects[0]]["high_risk_dq_count"] > 0:
        v = dq_rejects[0]
        reasons.append(f"`{v}` was rejected: disqualified on {totals[v]['high_risk_dq_count']} "
                       f"high-risk query(ies) for unsafe or ungrounded claims, despite any surface clarity.")
    return reasons


def _tradeoff_text(agg: dict, selected: str) -> str:
    # Find a query where a rejected variant looked clear but was unsafe.
    for pq in agg["per_query"]:
        for v, pv in pq["variants"].items():
            if v != selected and pv["hard_violation"] and pv["clarity"] >= 3:
                return (f"On **{pq['query_id']}** ({pq['risk_level']} risk), `{v}` reads clearly "
                        f"(clarity {pv['clarity']}/5) but its claim is unsupported "
                        f"(grounding {pv['grounding_score']:.2f}) and it "
                        f"{'overcommits beyond the evidence' if pv['overclaim'] else 'trips a banned-claim rule'}. "
                        f"The scoring caps clarity's weight at {agg['config']['w_clarity']:.2f} while "
                        f"grounding + faithfulness + safety together carry "
                        f"{agg['config']['w_grounding'] + agg['config']['w_faithfulness'] + agg['config']['w_safety']:.2f}, "
                        f"and a high-risk safety violation is disqualifying — so the clearer answer "
                        f"correctly loses to the safer `{selected}`.")
    return (f"`{selected}` is promoted on safety and grounding. Where a rejected variant was more "
            f"fluent, clarity is weighted only {agg['config']['w_clarity']:.2f} and can never override "
            f"a safety or grounding failure.")


_LIMITATIONS = [
    "Retrieval is lexical (TF-IDF); it can miss purely semantic paraphrase matches on a larger corpus.",
    "Grounding is token-overlap precision — a coarse proxy that can be fooled by answers that reuse "
    "evidence vocabulary while distorting meaning (e.g. dropped negations).",
    "Banned-claim detection is safety-biased and may over-flag; it is one signal, backed by grounding "
    "and the LLM overclaim flag rather than trusted alone.",
    "The LLM review is a single bounded call; its scores are advisory and validated/fallback-stubbed, "
    "not authoritative. With no API key the stub derives scores from rule signals only.",
    "Weights and thresholds are hand-tuned defaults (in `config.py` / the sidecar), not learned from "
    "labeled outcomes.",
]


# --------------------------------------------------------------------------- review_report.md
def render_review_report_md(agg: dict, queries: List[Query], retrieval: List[dict],
                            candidates: List[Candidate], llm_review: List[dict],
                            taxonomy: List[dict]) -> str:
    q_by_id = {q.query_id: q for q in queries}
    retr_by_q = {r["query_id"]: r["retrieved"] for r in retrieval}
    ans_by_q = {c.query_id: c.answers for c in candidates}
    llm_by_q = {r["query_id"]: r for r in llm_review}
    tax_by_qv = {(t["query_id"], t["variant"]): t["tags"] for t in taxonomy}

    lines: List[str] = ["# Review Report (per-query explainability)\n"]
    lines.append(f"**Overall selected variant:** `{agg['selected_variant']}`"
                 + (" (⚠️ no safe variant — least-unsafe fallback)" if agg["no_safe_variant_overall"] else "")
                 + "\n")

    for pq in agg["per_query"]:
        qid = pq["query_id"]
        q = q_by_id.get(qid)
        lines.append(f"## {qid} — {q.user_question if q else ''}")
        lines.append(f"- **Risk level:** {pq['risk_level']}")
        lines.append(f"- **LLM winner:** `{llm_by_q.get(qid, {}).get('winner', 'n/a')}`  ·  "
                     f"**Final selected:** `{pq['selected_variant']}`")

        # Retrieved evidence.
        lines.append("- **Retrieved evidence:**")
        for r in retr_by_q.get(qid, []):
            lines.append(f"  - `{r['doc_id']}` (score {r['score']:.3f}) — {r['title']}")

        # Answers + checks + tags.
        lines.append("- **Candidate answers:**")
        for v in sorted(ans_by_q.get(qid, {})):
            pv = pq["variants"].get(v, {})
            fails = ", ".join(_failed_checks(pv)) if pv else "n/a"
            tags = ", ".join(tax_by_qv.get((qid, v), [])) or "none"
            lines.append(f"  - **`{v}`**: {ans_by_q[qid][v]}")
            lines.append(f"    - checks: grounding={pv.get('grounding_score', 0):.2f}, "
                         f"faith={pv.get('faithfulness', '?')}, clarity={pv.get('clarity', '?')}, "
                         f"q_final={pv.get('q_final', 0):.3f}")
            lines.append(f"    - failed: {fails or '—'}  ·  taxonomy: {tags}")
        lines.append("")
    return "\n".join(lines)
