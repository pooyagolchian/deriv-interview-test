"""Deterministic TF-IDF + cosine retrieval over the local knowledge base.

Hand-rolled (no numpy / sklearn) so results are byte-stable across library versions and a
clean checkout needs zero dependencies. Cosine over non-negative TF-IDF weights is naturally
bounded to [0, 1], so scores are stored directly with no post-hoc normalization.

    idf(t) = ln((N + 1) / (df(t) + 1)) + 1        # smoothed: never divides by zero / negative
    weight(t, d) = tf(t, d) * idf(t)
    score(q, d) = cosine(vec(q), vec(d)) in [0, 1]
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

from . import config
from .schema import KBDoc, Query
from .textnorm import tokens


def build_index(docs: List[KBDoc]) -> dict:
    """Precompute idf, per-doc TF-IDF vectors, and vector norms."""
    doc_tokens = {d.doc_id: tokens(d.index_text) for d in docs}
    n_docs = len(docs)

    df: Counter = Counter()
    for toks in doc_tokens.values():
        df.update(set(toks))
    idf = {t: math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in df}

    doc_vectors: Dict[str, Dict[str, float]] = {}
    doc_norms: Dict[str, float] = {}
    for doc_id, toks in doc_tokens.items():
        tf = Counter(toks)
        vec = {t: tf[t] * idf[t] for t in tf}
        doc_vectors[doc_id] = vec
        doc_norms[doc_id] = math.sqrt(sum(w * w for w in vec.values()))
    return {"idf": idf, "doc_vectors": doc_vectors, "doc_norms": doc_norms}


def _vectorize(text: str, idf: Dict[str, float]) -> tuple[Dict[str, float], float]:
    # Vocabulary is the corpus; query terms outside it carry no signal and are dropped
    # (they cannot match any document), which keeps cosine well-defined.
    tf = Counter(tokens(text))
    vec = {t: tf[t] * idf[t] for t in tf if t in idf}
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return vec, norm


def _cosine(qvec: Dict[str, float], qnorm: float, dvec: Dict[str, float], dnorm: float) -> float:
    if qnorm == 0.0 or dnorm == 0.0:
        return 0.0
    dot = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
    return dot / (qnorm * dnorm)


def retrieve(query: Query, docs: List[KBDoc], index: dict, top_k: int) -> List[dict]:
    qvec, qnorm = _vectorize(query.user_question, index["idf"])
    scored = []
    for d in docs:
        s = _cosine(qvec, qnorm, index["doc_vectors"][d.doc_id], index["doc_norms"][d.doc_id])
        scored.append((d, s))
    # Deterministic ordering: score DESC, then doc_id ASC as a stable tie-break.
    scored.sort(key=lambda ds: (-ds[1], ds[0].doc_id))
    return [
        {"doc_id": d.doc_id, "score": round(s, 6), "title": d.title, "text": d.text}
        for d, s in scored[:top_k]
    ]


def run_retrieval(queries: List[Query], docs: List[KBDoc], cfg: config.EvalConfig = config.DEFAULT_CONFIG) -> List[dict]:
    index = build_index(docs)
    return [{"query_id": q.query_id, "retrieved": retrieve(q, docs, index, cfg.top_k)} for q in queries]
