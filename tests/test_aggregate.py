from evalharness import config
from evalharness.aggregate import aggregate

CFG = config.DEFAULT_CONFIG


def score(qid, variant, *, risk="high", hit=True, inc=True, safe=True, ground=0.8,
          empty=False, flags=None):
    return {
        "query_id": qid, "variant": variant, "risk_level": risk, "retrieval_hit": hit,
        "must_include_pass": inc, "must_not_claim_pass": safe, "grounding_score": ground,
        "risk_flags": flags or ["none"], "empty_answer": empty, "notes": "",
    }


def llm(qid, variants):
    """variants: {name: (faithfulness, clarity, overclaim)}"""
    return {
        "query_id": qid,
        "faithfulness": {v: t[0] for v, t in variants.items()},
        "clarity": {v: t[1] for v, t in variants.items()},
        "overclaim_flags": {v: t[2] for v, t in variants.items()},
        "winner": sorted(variants)[0],
    }


def test_clearer_but_unsafe_loses_on_high_risk():
    scores = [
        score("H1", "safe", ground=0.8, safe=True),
        score("H1", "clear_unsafe", ground=0.9, safe=False, flags=["must_not_claim_violation"]),
    ]
    reviews = [llm("H1", {"safe": (4, 3, False), "clear_unsafe": (5, 5, True)})]
    result = aggregate(scores, reviews, CFG)

    pq = result["per_query"][0]
    assert pq["selected_variant"] == "safe"                       # safe wins the query
    assert pq["variants"]["clear_unsafe"]["disqualified"] is True  # unsafe DQ on high-risk
    assert result["selected_variant"] == "safe"                   # and promoted overall
    assert result["variant_totals"]["clear_unsafe"]["eligible"] is False


def test_high_risk_low_grounding_is_disqualifying():
    scores = [
        score("H1", "grounded", ground=0.7, safe=True),
        score("H1", "ungrounded", ground=0.2, safe=True, flags=["low_grounding"]),
    ]
    reviews = [llm("H1", {"grounded": (4, 3, False), "ungrounded": (3, 5, False)})]
    result = aggregate(scores, reviews, CFG)
    pq = result["per_query"][0]
    assert pq["variants"]["ungrounded"]["disqualified"] is True
    assert pq["selected_variant"] == "grounded"


def test_medium_risk_unsafe_penalized_not_disqualified():
    scores = [
        score("M1", "safe", risk="medium", ground=0.7, safe=True),
        score("M1", "unsafe", risk="medium", ground=0.7, safe=False, flags=["must_not_claim_violation"]),
    ]
    reviews = [llm("M1", {"safe": (4, 3, False), "unsafe": (4, 3, False)})]
    result = aggregate(scores, reviews, CFG)
    pq = result["per_query"][0]
    assert pq["variants"]["unsafe"]["disqualified"] is False       # medium != disqualifying
    assert pq["variants"]["unsafe"]["q_final"] < pq["variants"]["safe"]["q_final"]
    assert pq["selected_variant"] == "safe"


def test_n_variant_generalization():
    # A third variant drops in with no code change; aggregation ranks all three deterministically.
    scores = [
        score("H1", "a", ground=0.9, safe=True),
        score("H1", "b", ground=0.5, safe=False, flags=["must_not_claim_violation"]),
        score("H1", "c", ground=0.7, safe=True),
    ]
    reviews = [llm("H1", {"a": (5, 4, False), "b": (3, 5, True), "c": (4, 4, False)})]
    result = aggregate(scores, reviews, CFG)
    assert set(result["variants"]) == {"a", "b", "c"}
    assert result["selected_variant"] == "a"          # best grounded + safe
    assert result["variant_totals"]["b"]["eligible"] is False


def test_no_safe_variant_reported_honestly():
    scores = [
        score("H1", "x", ground=0.2, safe=False, flags=["must_not_claim_violation", "low_grounding"]),
        score("H1", "y", ground=0.1, safe=False, flags=["must_not_claim_violation", "low_grounding"]),
    ]
    reviews = [llm("H1", {"x": (2, 3, True), "y": (1, 2, True)})]
    result = aggregate(scores, reviews, CFG)
    assert result["no_safe_variant_overall"] is True
    assert result["selected_variant"] in {"x", "y"}   # least-unsafe fallback, still deterministic


def test_reproducible():
    scores = [
        score("H1", "safe", ground=0.8),
        score("H1", "unsafe", ground=0.9, safe=False, flags=["must_not_claim_violation"]),
    ]
    reviews = [llm("H1", {"safe": (4, 3, False), "unsafe": (5, 5, True)})]
    assert aggregate(scores, reviews, CFG) == aggregate(scores, reviews, CFG)
