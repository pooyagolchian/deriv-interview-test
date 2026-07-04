# ADR-0008: Development methodology & tooling (AIDLC + superpowers; no Strands; local retrieval)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Affects:** `.clinerules/`, `.aidlc-rule-details/`, `.claude/`, retrieval & LLM choices

## Context

Beyond the harness itself, we were asked to (1) build using **AIDLC** (AWS AI-Driven
Development Lifecycle) and **superpowers** skills from skills.sh, (2) configure `.claude/` for
spec-driven planning, and (3) evaluate whether to build the app on **Strands Agents**. A real
Anthropic key was later supplied "for embedding + usage."

## Decision

**Methodology / `.claude` setup**

- Install AIDLC v1.0.1 steering rules at [`.clinerules/core-workflow.md`](../../.clinerules/core-workflow.md)
  + [`.aidlc-rule-details/`](../../.aidlc-rule-details/) (AIDLC's recommended non-destructive
  Claude Code layout) for the spec → plan → build workflow with approval gates.
- Install the **superpowers** skill set (`obra/superpowers`) into `.claude/skills/` (planning,
  TDD, systematic debugging, verification-before-completion, code review).
- Add [`.claude/CLAUDE.md`](../../.claude/CLAUDE.md) as the operating guide wiring these
  together. The root `CLAUDE.MD` (the assessment spec) is left untouched — note macOS is
  case-insensitive, so a root `CLAUDE.md` would collide with it.

**Strands Agents — rejected as the architecture.** Strands is built for autonomous, tool-using
agent loops. That conflicts with this assessment's hard requirements: a deterministic pipeline,
exactly one controlled LLM call, no LLM-driven retrieval, and a reproducible recommendation. The
single review call uses the Anthropic SDK directly.

**Retrieval — local TF-IDF, not embeddings.** The spec forbids external web calls for retrieval,
and reproducibility argues for local computation. The supplied key is Anthropic, which has **no
embeddings endpoint**, so remote embeddings were not even feasible. The key powers only the one
LLM review stage. (See [ADR-0002](0002-deterministic-tfidf-retrieval.md).)

## Alternatives considered

- **Thin Strands wrapper for the single call** — rejected: adds a dependency (Bedrock-leaning)
  with no benefit over a direct SDK call.
- **Full Strands agent** — rejected: violates determinism / single-call / no-LLM-retrieval /
  reproducibility.
- **Local embedding model** (offline sentence-transformers) — considered and rejected for the
  default: heavier dependency + model download for negligible gain on a 5-doc corpus; the design
  keeps retrieval pluggable if this changes.

## Consequences

- AIDLC + superpowers shape *how* we build (process); they are not runtime dependencies of the
  harness.
- The harness stays deterministic, reproducible, and runnable offline.
- `.env` (the key) is gitignored and never printed.
