# ADR-0002: Deterministic TF-IDF cosine retrieval

- **Status:** Accepted
- **Date:** 2026-07-04
- **Implements:** [`evalharness/retrieval.py`](../../evalharness/retrieval.py), shared tokenizer in [`evalharness/textnorm.py`](../../evalharness/textnorm.py)

## Context

The spec requires retrieval that is (a) implemented in code, (b) run from the local `kb.json`,
(c) deterministic and reproducible from a clean checkout, and (d) explicitly **without external
web calls for knowledge lookup**. The knowledge base is tiny (5 short passages) and the example
`retrieval.json` shows scores in `[0, 1]` (e.g. `0.83`, `0.21`).

## Decision

Hand-rolled **TF-IDF with cosine similarity**, no third-party libraries:

- `idf(t) = ln((N + 1) / (df(t) + 1)) + 1` — smoothed, so it never divides by zero and never
  goes negative.
- `weight(t, d) = tf(t, d) · idf(t)`; `score(q, d) = cosine(vec(q), vec(d))`, which over
  non-negative TF-IDF weights is naturally bounded to `[0, 1]` — **no post-hoc normalization**,
  matching the example format directly.
- Documents are indexed on **title + body**; query terms outside the corpus vocabulary carry no
  signal and are dropped (they can match no document), keeping cosine well-defined.
- Shared tokenizer (one source of truth for all stages): casefold → `[a-z0-9]+` → drop a small
  stopword list (negations kept) → conservative plural stemmer.
- Return **top-2**, tie-break `(score DESC, doc_id ASC)`; store `round(score, 6)`.

## Alternatives considered

- **BM25** — defensible, but raw scores are unbounded; squeezing them into `[0, 1]` needs
  per-query min-max normalization that pins the top document to exactly `1.0` (won't reproduce
  a `0.83`) and breaks cross-query comparability. On a 5-doc corpus BM25's length-normalization
  advantages are marginal.
- **Embedding similarity** (remote) — rejected: an external call for retrieval violates the
  spec, is non-reproducible, and the provided key is Anthropic, which has no embeddings
  endpoint. (See [ADR-0008](0008-methodology-and-tooling.md).)
- **`sklearn.TfidfVectorizer`** — rejected: adds a heavy dependency and its float outputs can
  drift across library versions, undermining byte-level reproducibility.

## Consequences

- Byte-stable results across machines/versions; zero dependencies for retrieval.
- Purely **lexical** — can miss semantic paraphrase matches on a larger corpus (documented
  limitation). The shared stemmer mitigates simple morphology (e.g. `withdrawal`~`withdrawals`).
- A query with no in-vocabulary tokens yields all-zero scores and falls back deterministically
  to `doc_id`-ascending order.
