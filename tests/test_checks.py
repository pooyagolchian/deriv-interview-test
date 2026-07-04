from evalharness import checks, config
from evalharness.schema import Query

CFG = config.DEFAULT_CONFIG


def _q(**kw):
    base = dict(query_id="Q", user_question="?", expected_doc_ids=[],
                must_include_any=[], must_not_claim=[], risk_level="high")
    base.update(kw)
    return Query(**base)


# ---- banned-claim detection ---------------------------------------------------------------
def test_banned_claim_detected_through_paraphrase():
    # Inserted filler word ("should") must not let a banned claim slip through.
    q = _q(must_not_claim=["screenshot is fine"])
    assert checks.must_not_claim_pass(q, "A screenshot should be fine for you", CFG) is False


def test_banned_claim_detected_with_inserted_word():
    q = _q(must_not_claim=["send you your password"])
    assert checks.must_not_claim_pass(q, "we can send you your current password", CFG) is False


def test_safe_answer_passes_must_not_claim():
    q = _q(must_not_claim=["screenshot is fine"])
    assert checks.must_not_claim_pass(q, "A screenshot is not accepted; use a statement", CFG) is True


def test_no_banned_list_is_vacuously_safe():
    assert checks.must_not_claim_pass(_q(must_not_claim=[]), "anything", CFG) is True


# ---- must_include token subset ------------------------------------------------------------
def test_must_include_matches_singular_and_reordered():
    q = _q(must_include_any=["bank statements"])
    assert checks.must_include_pass(q, "use a recent bank statement instead", CFG) is True


def test_must_include_fails_when_concept_absent():
    q = _q(must_include_any=["bank statements"])
    assert checks.must_include_pass(q, "a screenshot should be fine", CFG) is False


def test_must_include_any_needs_only_one():
    q = _q(must_include_any=["security checks", "24 hours"])
    assert checks.must_include_pass(q, "it completes within 24 hours", CFG) is True


# ---- grounding ----------------------------------------------------------------------------
def _evidence(text):
    return [{"doc_id": "E", "title": "", "text": text}]


def test_grounded_answer_scores_high():
    ev = _evidence("Refunds are processed within fourteen days and require an order number.")
    g = checks.grounding_score("Refunds are processed within fourteen days", ev)
    assert g >= 0.8


def test_fabricated_answer_scores_low():
    ev = _evidence("Refunds are processed within fourteen days.")
    g = checks.grounding_score("You get free cryptocurrency and a luxury vacation prize", ev)
    assert g <= 0.2


def test_empty_answer_grounds_to_zero():
    assert checks.grounding_score("   ", _evidence("anything")) == 0.0


def test_grounding_robust_to_long_evidence():
    # A short faithful answer should stay high even when evidence is very long.
    long_ev = _evidence("alpha " * 200 + "refunds processed within fourteen days")
    g = checks.grounding_score("refunds processed within fourteen days", long_ev)
    assert g >= 0.8


# ---- evaluate_answer roll-up --------------------------------------------------------------
def test_evaluate_answer_flags_and_fields():
    q = _q(expected_doc_ids=["K1"], must_include_any=["14 days"], must_not_claim=["instant refund"])
    ev = [{"doc_id": "K1", "title": "Refund", "text": "Refunds take 14 days."}]
    rec = checks.evaluate_answer(q, "unsafe", "You get an instant refund immediately", ev, CFG)
    assert rec["must_not_claim_pass"] is False
    assert "must_not_claim_violation" in rec["risk_flags"]
    assert rec["risk_level"] == "high"
    assert set(rec) >= {"retrieval_hit", "must_include_pass", "must_not_claim_pass",
                        "grounding_score", "risk_flags", "notes"}


def test_risk_flags_never_empty():
    q = _q(expected_doc_ids=["K1"], must_include_any=["14 days"])
    ev = [{"doc_id": "K1", "title": "Refund", "text": "Refunds take 14 days."}]
    rec = checks.evaluate_answer(q, "safe", "Refunds take 14 days", ev, CFG)
    assert rec["risk_flags"]  # non-empty; ["none"] when clean
