# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-04
- **Deciders:** AI engineering (evaluation harness)

## Context

This evaluation harness makes several non-obvious engineering choices (deterministic lexical
retrieval, a deliberate deviation from the spec's literal string matching, a safety-first
aggregation model, a stubbed LLM path). Reviewers and future maintainers need to understand
*why* each choice was made, what alternatives were weighed, and what the consequences are —
without reverse-engineering it from code.

## Decision

We record architecturally significant decisions as **Architecture Decision Records** (ADRs),
one file per decision, in [`docs/adr/`](.), using a lightweight MADR-style template:
Context → Decision → Alternatives considered → Consequences.

- Files are numbered sequentially: `NNNN-kebab-title.md`.
- Each ADR has a status (`Proposed` / `Accepted` / `Superseded by ADR-XXXX`).
- ADRs are immutable once accepted; a change is a new ADR that supersedes the old one.

## Alternatives considered

- **No formal record** (rely on code comments / commit messages) — rejected: rationale and
  rejected alternatives get lost; comments explain *what*, not *why-not*.
- **A single DESIGN.md** — rejected: one growing document obscures the decision boundaries and
  loses the "what did we reject and why" history that ADRs preserve.

## Consequences

- Design intent is discoverable and reviewable; the README links to the relevant ADR per topic.
- Small ongoing cost: a new ADR per significant decision.
- The remaining ADRs (0002–0008) capture the harness's substantive decisions.
