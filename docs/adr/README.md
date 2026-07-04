# Architecture Decision Records

Design rationale for the RAG-QA evaluation harness. Format: MADR-style
(Context → Decision → Alternatives → Consequences). See [ADR-0001](0001-record-architecture-decisions.md).

| ADR | Decision |
|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions (this process) |
| [0002](0002-deterministic-tfidf-retrieval.md) | Deterministic TF-IDF cosine retrieval |
| [0003](0003-rule-checks-and-grounding.md) | Normalized rule checks + precision grounding (spec deviation) |
| [0004](0004-single-llm-review-stage.md) | One controlled LLM review stage + provider abstraction + stub |
| [0005](0005-safety-first-aggregation.md) | Safety-first, two-tier aggregation |
| [0006](0006-reproducibility-sidecar.md) | Reproducibility via `recommendation.json` sidecar |
| [0007](0007-n-variant-extensibility.md) | N-variant extensibility (regression guard) |
| [0008](0008-methodology-and-tooling.md) | Methodology & tooling (AIDLC + superpowers; no Strands; local retrieval) |
