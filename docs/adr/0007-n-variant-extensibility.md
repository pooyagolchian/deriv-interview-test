# ADR-0007: N-variant extensibility (pairwise regression guard)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Implements:** [`evalharness/schema.py`](../../evalharness/schema.py) (`variant_universe`), [`evalharness/aggregate.py`](../../evalharness/aggregate.py), [`evalharness/checks.py`](../../evalharness/checks.py) (`run_checks`)

## Context

The sample has two variants (`prompt_a`, `prompt_b`), but the spec's stretch goal asks that a
**third variant drop into the same schema later without changing core pipeline logic**, and the
technical constraints forbid hardcoding query IDs, phrases, or variant/answer text.

## Decision

Treat the set of variants as data, discovered at runtime:

- The **variant universe** is the sorted union of the answer keys across all queries
  (`schema.variant_universe`). No variant name is ever hardcoded.
- Every stage iterates variants generically: rule checks evaluate each `(query, variant)` pair;
  the LLM prompt and validation are built per the variants present; aggregation partitions,
  scores, and ranks over an arbitrary variant set, with `variant_key ASC` as the final
  deterministic tie-break.
- A variant missing on some query is given a **losing sentinel** record (all checks fail,
  grounding 0), so a partially-present variant can never win, and `validate.py` flags the
  coverage gap.

## Alternatives considered

- **Hardcode two variants** — rejected: brittle, and directly violates the "don't hardcode"
  constraint; adding a variant would require code changes.
- **Pairwise-only comparison (A vs B)** — rejected: does not scale to N variants and would need
  a redesign to add a third.

## Consequences

- Adding a third strategy = drop its answers into `candidate_answers.json` and rerun; no core
  code changes. Tests (`test_aggregate.py::test_n_variant_generalization`) cover the 3-variant
  path.
- The sentinel makes partial coverage safe and detectable rather than silently favoring a
  variant that only answered the easy queries.
