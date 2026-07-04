# ADR-0006: Reproducibility via a `recommendation.json` sidecar

- **Status:** Accepted
- **Date:** 2026-07-04
- **Implements:** [`evalharness/report.py`](../../evalharness/report.py), [`evalharness/aggregate.py`](../../evalharness/aggregate.py), [`validate.py`](../../validate.py)

## Context

The spec requires that "the recommendation is reproducible from stored outputs." But one stage —
the LLM review — is not perfectly reproducible (model outputs can vary, and newer models don't
accept a `temperature` seed). We need reproducibility that does **not** depend on re-running the
LLM or retrieval.

## Decision

Make the **aggregation a pure function** of the stored artifacts and split reproducibility from
the LLM:

- `aggregate(automated_scores, llm_review, config)` uses no LLM, no retrieval, no time, no
  randomness, and iterates `sorted()` keys — so it is a deterministic function of its inputs.
- Emit a machine-readable **`recommendation.json`** sidecar containing the selected variant, the
  full per-query decision trace, variant totals, a snapshot of the `EvalConfig` used, and content
  **hashes of the three input files**. `recommendation.md` is rendered from this sidecar and
  contains **no timestamps**, so both are byte-reproducible.
- `validate.py` proves reproducibility by re-loading `automated_scores.json` + `llm_review.json`
  + the config snapshot, recomputing `aggregate()`, and asserting the recomputed selected variant
  **and** per-query winners equal what is recorded — without touching the LLM or retrieval.
- Numbers are rounded at serialization (≥ 6 dp) and the recompute runs on the stored rounded
  values, so float drift cannot change the outcome.

## Alternatives considered

- **Cache LLM responses only** — rejected: proves the LLM inputs were stable but not that the
  *decision* is reproducible from stored artifacts.
- **Re-run the entire pipeline to reproduce** — rejected: the LLM stage is non-deterministic, so
  a full re-run is not a reliable reproducibility check.
- **Store only `recommendation.md`** — rejected: prose is not reliably machine-checkable.

## Consequences

- `llm_calls.jsonl` timestamps are audit-only and do **not** affect reproducibility (that log is
  never consumed by aggregation).
- The decision is fully auditable and re-derivable; changing a weight is a config change captured
  in the sidecar, so an old recommendation stays reproducible against its own config.
