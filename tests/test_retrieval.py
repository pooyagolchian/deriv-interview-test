from evalharness import config, retrieval
from evalharness.schema import KBDoc, Query


def _docs():
    return [
        KBDoc("A", "Refund window", "Refunds are processed within fourteen days."),
        KBDoc("B", "Shipping times", "Standard shipping takes three to five business days."),
        KBDoc("C", "Warranty", "Products carry a one year warranty for defects."),
    ]


def _q(text):
    return Query("Q", text, [], [], [], "medium")


def test_returns_top_k_with_scores_in_range():
    docs = _docs()
    idx = retrieval.build_index(docs)
    got = retrieval.retrieve(_q("how long do refunds take"), docs, idx, config.DEFAULT_CONFIG.top_k)
    assert len(got) == config.DEFAULT_CONFIG.top_k
    for p in got:
        assert set(p) >= {"doc_id", "score", "title", "text"}
        assert 0.0 <= p["score"] <= 1.0


def test_scores_descending_and_relevant_doc_first():
    docs = _docs()
    idx = retrieval.build_index(docs)
    got = retrieval.retrieve(_q("refund processed days"), docs, idx, 3)
    scores = [p["score"] for p in got]
    assert scores == sorted(scores, reverse=True)
    assert got[0]["doc_id"] == "A"  # the refund doc is the best match


def test_deterministic_across_runs():
    docs = _docs()
    idx = retrieval.build_index(docs)
    a = retrieval.retrieve(_q("shipping business days"), docs, idx, 2)
    b = retrieval.retrieve(_q("shipping business days"), docs, idx, 2)
    assert a == b


def test_tie_break_by_doc_id_ascending():
    # Two identical docs (same tokens) tie on score; the ASC doc_id must come first.
    docs = [
        KBDoc("Z", "t", "alpha beta gamma"),
        KBDoc("A", "t", "alpha beta gamma"),
        KBDoc("M", "t", "unrelated words entirely"),
    ]
    idx = retrieval.build_index(docs)
    got = retrieval.retrieve(_q("alpha beta gamma"), docs, idx, 2)
    assert got[0]["score"] == got[1]["score"]
    assert [p["doc_id"] for p in got] == ["A", "Z"]


def test_all_stopword_query_is_deterministic():
    docs = _docs()
    idx = retrieval.build_index(docs)
    got = retrieval.retrieve(_q("the a of to"), docs, idx, 2)
    assert [p["doc_id"] for p in got] == ["A", "B"]  # all-zero scores -> doc_id ASC
    assert all(p["score"] == 0.0 for p in got)
