from evalharness import config, llm_review
from evalharness.schema import Candidate, Query

CFG = config.DEFAULT_CONFIG


def _ctx(qid="Q1", variants=None):
    variants = variants or {
        "safe": {"answer": "refunds take fourteen days", "checks": {
            "retrieval_hit": True, "must_include_pass": True, "must_not_claim_pass": True,
            "grounding_score": 0.9, "risk_flags": ["none"]}},
        "unsafe": {"answer": "instant refund guaranteed", "checks": {
            "retrieval_hit": True, "must_include_pass": False, "must_not_claim_pass": False,
            "grounding_score": 0.2, "risk_flags": ["must_not_claim_violation", "low_grounding"]}},
    }
    return {"query_id": qid, "user_question": "?", "risk_level": "high",
            "evidence": [], "variants": variants}


# ---- stub determinism + derivation --------------------------------------------------------
def test_stub_is_deterministic():
    ctxs = [_ctx()]
    assert llm_review.derive_stub_reviews(ctxs, CFG) == llm_review.derive_stub_reviews(ctxs, CFG)


def test_stub_faithfulness_tracks_grounding():
    rev = llm_review.derive_stub_reviews([_ctx()], CFG)["Q1"]
    assert rev["faithfulness"]["safe"] > rev["faithfulness"]["unsafe"]
    assert config.SCORE_MIN <= rev["faithfulness"]["safe"] <= config.SCORE_MAX


def test_stub_overclaim_when_unsafe_or_ungrounded():
    rev = llm_review.derive_stub_reviews([_ctx()], CFG)["Q1"]
    assert rev["overclaim_flags"]["unsafe"] is True
    assert rev["overclaim_flags"]["safe"] is False
    assert rev["winner"] == "safe"


# ---- validation / fail-closed merge -------------------------------------------------------
def test_invalid_scores_fall_back_to_stub():
    ctxs = [_ctx()]
    stub = llm_review.derive_stub_reviews(ctxs, CFG)
    raw = {"reviews": [{
        "query_id": "Q1",
        "faithfulness": {"safe": 99, "unsafe": 2},        # 99 is out of range -> stub
        "clarity": {"safe": 4, "unsafe": 3},
        "overclaim_flags": {"safe": "nope", "unsafe": True},  # non-bool -> stub
        "winner": "does_not_exist",                        # invalid -> stub winner
    }]}
    merged, warnings = llm_review._validate_and_merge(raw, ctxs, stub)
    rec = merged[0]
    assert rec["faithfulness"]["safe"] == stub["Q1"]["faithfulness"]["safe"]  # substituted
    assert rec["faithfulness"]["unsafe"] == 2                                 # valid kept
    assert isinstance(rec["overclaim_flags"]["safe"], bool)
    assert rec["winner"] in {"safe", "unsafe", "tie"}
    assert warnings  # violations were recorded


def test_valid_scores_are_used():
    ctxs = [_ctx()]
    stub = llm_review.derive_stub_reviews(ctxs, CFG)
    raw = {"reviews": [{
        "query_id": "Q1",
        "faithfulness": {"safe": 5, "unsafe": 1},
        "clarity": {"safe": 4, "unsafe": 2},
        "overclaim_flags": {"safe": False, "unsafe": True},
        "winner": "safe",
    }]}
    merged, _ = llm_review._validate_and_merge(raw, ctxs, stub)
    assert merged[0]["faithfulness"] == {"safe": 5, "unsafe": 1}
    assert merged[0]["winner"] == "safe"


def test_missing_query_uses_full_stub():
    ctxs = [_ctx()]
    stub = llm_review.derive_stub_reviews(ctxs, CFG)
    merged, _ = llm_review._validate_and_merge({"reviews": []}, ctxs, stub)
    assert merged[0]["faithfulness"] == stub["Q1"]["faithfulness"]


# ---- json extraction ----------------------------------------------------------------------
def test_extract_json_bare_and_fenced_and_garbage():
    assert llm_review._extract_json('{"reviews": []}') == {"reviews": []}
    assert llm_review._extract_json('```json\n{"reviews": [1]}\n```') == {"reviews": [1]}
    assert llm_review._extract_json("no json here") is None


def test_build_contexts_shape():
    queries = [Query("Q1", "?", ["K1"], ["x"], ["y"], "high")]
    retrieval = [{"query_id": "Q1", "retrieved": [{"doc_id": "K1", "title": "t", "text": "b"}]}]
    scores = [{"query_id": "Q1", "variant": "safe", "retrieval_hit": True,
               "must_include_pass": True, "must_not_claim_pass": True,
               "grounding_score": 0.9, "risk_flags": ["none"]}]
    candidates = [Candidate("Q1", {"safe": "an answer"})]
    ctxs = llm_review.build_contexts(queries, retrieval, scores, candidates)
    assert ctxs[0]["query_id"] == "Q1"
    assert "safe" in ctxs[0]["variants"]
    assert ctxs[0]["variants"]["safe"]["checks"]["grounding_score"] == 0.9
