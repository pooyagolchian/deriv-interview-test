"""The single controlled LLM review stage.

Contract (spec):
- Exactly ONE model call across all query/answer pairs.
- The model judges only clarity, faithfulness, overclaim, and a per-query winner. It never
  performs retrieval and never makes the final promotion decision.
- Retrieved evidence + deterministic check results are passed in as context.
- Every returned value is validated in code; invalid fields FAIL CLOSED to a deterministic
  stub value so the pipeline stays safe and always produces schema-valid output.
- Every call is logged to llm_calls.jsonl.

Providers:
- ``anthropic``: real Claude call (used when a key + SDK are available).
- ``stub``: deterministic, derives scores ONLY from allowed rule signals (grounding,
  must_include, must_not_claim) — never from gold answers or variant names. Keeps the whole
  pipeline runnable offline and reproducible.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import config
from .io_utils import append_jsonl, content_hash
from .schema import Candidate, Query
from .textnorm import content_tokens

SYSTEM_PROMPT = (
    "You are a careful evaluation judge for a retrieval-augmented QA system. "
    "You compare candidate answers ONLY on the evidence provided. You must NOT invent facts, "
    "perform retrieval, or decide deployment. Judge only: clarity, faithfulness to the given "
    "evidence, whether an answer overcommits beyond the evidence, and a per-query winner. "
    "Return STRICT JSON only, matching the requested schema exactly."
)


# --------------------------------------------------------------------------- context
def build_contexts(queries: List[Query], retrieval: List[dict], scores: List[dict],
                   candidates: List[Candidate]) -> List[dict]:
    """One context object per query, bundling evidence + answers + deterministic checks."""
    retr_by_q = {r["query_id"]: r["retrieved"] for r in retrieval}
    ans_by_q = {c.query_id: c.answers for c in candidates}
    score_by_qv = {(s["query_id"], s["variant"]): s for s in scores}

    contexts: List[dict] = []
    for q in queries:
        variants = {}
        for variant, answer in sorted(ans_by_q.get(q.query_id, {}).items()):
            s = score_by_qv.get((q.query_id, variant), {})
            variants[variant] = {
                "answer": answer,
                "checks": {
                    "retrieval_hit": s.get("retrieval_hit"),
                    "must_include_pass": s.get("must_include_pass"),
                    "must_not_claim_pass": s.get("must_not_claim_pass"),
                    "grounding_score": s.get("grounding_score"),
                    "risk_flags": s.get("risk_flags"),
                },
            }
        contexts.append({
            "query_id": q.query_id,
            "user_question": q.user_question,
            "risk_level": q.risk_level,
            "evidence": [{"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]}
                        for r in retr_by_q.get(q.query_id, [])],
            "variants": variants,
        })
    return contexts


def build_prompt(contexts: List[dict]) -> str:
    variant_universe = sorted({v for c in contexts for v in c["variants"]})
    schema_hint = {
        "reviews": [{
            "query_id": "<id>",
            "winner": "<one of the variant keys for that query, or 'tie'>",
            "faithfulness": {v: "<int 1-5>" for v in variant_universe},
            "clarity": {v: "<int 1-5>" for v in variant_universe},
            "overclaim_flags": {v: "<true|false>" for v in variant_universe},
            "justification": "<= 2 sentences>",
        }]
    }
    return (
        "Evaluate the following queries. For EACH query, score every variant and pick a winner.\n"
        "Rules: use ONLY the provided evidence; faithfulness = supported by evidence; "
        "overclaim = asserts more than the evidence supports; clarity = how clearly written. "
        "Scores are integers 1-5. Do not perform retrieval. Do not choose deployment.\n\n"
        f"Return JSON with exactly this shape (one entry per query):\n{json.dumps(schema_hint, indent=2)}\n\n"
        f"INPUT:\n{json.dumps(contexts, ensure_ascii=False, indent=2)}\n"
    )


# --------------------------------------------------------------------------- stub
def _clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _stub_review_for(ctx: dict, cfg: config.EvalConfig) -> dict:
    faith, clar, oc = {}, {}, {}
    for variant, data in ctx["variants"].items():
        checks = data["checks"]
        grounding = checks.get("grounding_score") or 0.0
        safe = checks.get("must_not_claim_pass", True)
        inc = checks.get("must_include_pass", True)
        empty = not content_tokens(data["answer"])

        faith[variant] = _clamp_int(round(1 + 4 * grounding), config.SCORE_MIN, config.SCORE_MAX)
        base = 3 + (1 if (inc and not empty) else 0) - (2 if empty else 0)
        clar[variant] = _clamp_int(base, config.SCORE_MIN, config.SCORE_MAX)
        oc[variant] = bool((not safe) or (grounding < cfg.tau_grounding) or empty)

    # Winner: highest faithfulness, then grounding, then clarity, then variant key ASC.
    def sort_key(v: str):
        grounding = ctx["variants"][v]["checks"].get("grounding_score") or 0.0
        return (-faith[v], -grounding, -clar[v], v)

    winner = sorted(ctx["variants"].keys(), key=sort_key)[0] if ctx["variants"] else "tie"
    return {
        "query_id": ctx["query_id"],
        "winner": winner,
        "faithfulness": faith,
        "clarity": clar,
        "overclaim_flags": oc,
        "justification": (
            f"[stub] Derived from rule signals: winner {winner} has the strongest grounding/"
            f"faithfulness; overclaim flags set where grounding<{cfg.tau_grounding} or a banned "
            f"claim was detected."
        ),
    }


def derive_stub_reviews(contexts: List[dict], cfg: config.EvalConfig) -> Dict[str, dict]:
    return {c["query_id"]: _stub_review_for(c, cfg) for c in contexts}


# --------------------------------------------------------------------------- real call
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if start != -1 and end > start else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    import anthropic  # imported lazily so the stub path needs no dependency

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


# --------------------------------------------------------------------------- validation
def _validate_and_merge(raw: Optional[dict], contexts: List[dict],
                        stub: Dict[str, dict]) -> tuple[List[dict], List[str]]:
    """Validate the model's JSON per field; substitute stub values for anything invalid."""
    warnings: List[str] = []
    raw_by_q: Dict[str, dict] = {}
    if isinstance(raw, dict) and isinstance(raw.get("reviews"), list):
        for r in raw["reviews"]:
            if isinstance(r, dict) and isinstance(r.get("query_id"), str):
                raw_by_q[r["query_id"]] = r

    def valid_score(x) -> bool:
        return isinstance(x, int) and not isinstance(x, bool) and config.SCORE_MIN <= x <= config.SCORE_MAX

    reviews: List[dict] = []
    for ctx in contexts:
        qid = ctx["query_id"]
        variants = list(ctx["variants"].keys())
        base = stub[qid]
        got = raw_by_q.get(qid, {})
        rec = {"query_id": qid}

        # faithfulness / clarity: per-variant int in range, else stub value.
        for field in ("faithfulness", "clarity"):
            merged = {}
            got_map = got.get(field) if isinstance(got.get(field), dict) else {}
            for v in variants:
                val = got_map.get(v)
                if valid_score(val):
                    merged[v] = val
                else:
                    merged[v] = base[field][v]
                    if got:  # only warn when the model actually returned something for this query
                        warnings.append(f"{qid}.{field}.{v} invalid ({val!r}) -> stub {base[field][v]}")
            rec[field] = merged

        # overclaim flags: per-variant bool, else stub.
        merged_oc = {}
        got_oc = got.get("overclaim_flags") if isinstance(got.get("overclaim_flags"), dict) else {}
        for v in variants:
            val = got_oc.get(v)
            merged_oc[v] = val if isinstance(val, bool) else base["overclaim_flags"][v]
        rec["overclaim_flags"] = merged_oc

        # winner: must be a variant for this query or "tie", else stub winner.
        winner = got.get("winner")
        rec["winner"] = winner if winner in variants or winner == "tie" else base["winner"]
        if got and rec["winner"] != winner:
            warnings.append(f"{qid}.winner invalid ({winner!r}) -> stub {base['winner']}")

        just = got.get("justification")
        rec["justification"] = just if isinstance(just, str) and just.strip() else base["justification"]
        reviews.append(rec)

    return reviews, warnings


# --------------------------------------------------------------------------- orchestration
def resolve_provider(cfg_provider: str) -> str:
    """Decide which provider to use given config + environment."""
    if cfg_provider == "stub":
        return "stub"
    if cfg_provider == "anthropic":
        return "anthropic"
    # auto
    if config.get_api_key():
        try:
            import anthropic  # noqa: F401
            return "anthropic"
        except ImportError:
            return "stub"
    return "stub"


def run_llm_review(queries: List[Query], retrieval: List[dict], scores: List[dict],
                   candidates: List[Candidate], cfg: config.EvalConfig = config.DEFAULT_CONFIG,
                   provider: Optional[str] = None, model: Optional[str] = None,
                   log_path=None) -> List[dict]:
    contexts = build_contexts(queries, retrieval, scores, candidates)
    prompt = build_prompt(contexts)
    stub = derive_stub_reviews(contexts, cfg)
    provider = provider or resolve_provider(config.LLM_PROVIDER)
    model = model or config.DEFAULT_MODEL
    log_path = log_path or config.output_path(config.LLM_CALLS_FILE)
    ts = datetime.now(timezone.utc).isoformat()

    if provider == "anthropic":
        api_key = config.get_api_key()
        try:
            response_text = _call_anthropic(prompt, model, api_key)
            raw = _extract_json(response_text)
            reviews, warnings = _validate_and_merge(raw, contexts, stub)
            status = "ok" if raw is not None else "unparseable_response_used_stub"
        except Exception as exc:  # network/auth/SDK error -> fall back to stub, never crash
            response_text = f"<error: {type(exc).__name__}: {exc}>"
            reviews, warnings = list(stub.values()), [f"anthropic call failed: {exc}"]
            status = "error_used_stub"
        append_jsonl(log_path, {
            "timestamp": ts, "provider": "anthropic", "model": model, "status": status,
            "input_hash": content_hash(contexts), "num_queries": len(contexts),
            "system": SYSTEM_PROMPT, "prompt": prompt, "response": response_text,
            "validation_warnings": warnings,
        })
        return reviews

    # stub provider
    reviews = list(stub.values())
    append_jsonl(log_path, {
        "timestamp": ts, "provider": "stub", "model": None, "status": "ok",
        "input_hash": content_hash(contexts), "num_queries": len(contexts),
        "system": SYSTEM_PROMPT, "prompt": prompt,
        "response": json.dumps({"reviews": reviews}, ensure_ascii=False),
        "validation_warnings": [],
    })
    return reviews
