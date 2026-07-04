# ADR-0004: One controlled LLM review stage with provider abstraction + stub

- **Status:** Accepted
- **Date:** 2026-07-04
- **Implements:** [`evalharness/llm_review.py`](../../evalharness/llm_review.py)

## Context

The spec allows exactly **one** controlled LLM call across all query/answer pairs. The LLM may
judge only clarity, faithfulness, overclaim, and a per-query winner; it must **not** perform
retrieval and must **not** produce the final deployment recommendation. Its output must be
validated in code, every call logged to `llm_calls.jsonl`, and — critically — the pipeline must
still run when **no API key is available**.

## Decision

A single call over all pairs, behind a small provider abstraction:

- **Context passed in:** for each query, the retrieved evidence plus the deterministic check
  results per variant. The prompt forbids retrieval and deployment decisions and requests strict
  JSON (`reviews[]` with `winner`, `faithfulness`, `clarity`, `overclaim_flags`, `justification`).
- **Providers:**
  - `anthropic` — real Claude call (default model `claude-sonnet-5`) when a key
    (`CLAUDE_API_KEY` / `ANTHROPIC_API_KEY`) and the SDK are available. We do **not** send a
    `temperature` param (deprecated on newer models); LLM-level determinism is unnecessary
    because reproducibility is guaranteed downstream ([ADR-0006](0006-reproducibility-sidecar.md)).
  - `stub` — deterministic; derives scores **only** from allowed rule signals (grounding,
    must_include, must_not_claim) — never from gold answers, expected doc ids, or variant names.
- **Validation (fail-closed):** every returned value is checked — winner ∈ the query's variants
  or `"tie"`, faithfulness/clarity are ints in `[1, 5]`, overclaim is boolean. Any invalid field
  is **substituted with the deterministic stub value** and a warning is logged, so the pipeline
  never crashes and `llm_review.json` is always schema-valid.
- **Logging:** every call (real or stub) appends a record to `llm_calls.jsonl` with provider,
  status, input hash, prompt, and response.

## Alternatives considered

- **Multiple LLM calls / per-query calls** — rejected: violates the "one controlled call"
  constraint and costs more.
- **Trusting raw LLM JSON** — rejected: unvalidated model output could inject out-of-range
  scores or unknown variants into the decision.
- **Strands Agents / an agentic loop** — rejected: an autonomous tool-using loop conflicts with
  the single-call, no-LLM-retrieval, deterministic, reproducible requirements
  ([ADR-0008](0008-methodology-and-tooling.md)).

## Consequences

- The harness runs fully offline via the stub; a real key upgrades only the review judgments.
- The LLM's winner is **advisory**; code makes the final call ([ADR-0005](0005-safety-first-aggregation.md)).
- Reproducibility holds at the aggregation level, not the LLM level; stub clarity is a weak
  surface proxy (documented). The real and stub paths share identical schema and validation.
