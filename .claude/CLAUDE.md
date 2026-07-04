# Project operating guide — spec-driven planning + superpowers

This project is configured for **spec-driven development**: turn a spec into a plan, execute
with review checkpoints, and verify before claiming done. Two toolkits are installed.

> The authoritative product spec is [`CLAUDE.MD`](../CLAUDE.MD) at the repo root (a take-home
> assessment for a deterministic RAG-QA evaluation harness). Do not edit it. This guide governs
> *how* we build, not *what* we build.
>
> ⚠️ macOS filesystem is case-insensitive, so `CLAUDE.MD` and `CLAUDE.md` are the same file at the
> repo root. This methodology guide deliberately lives at `.claude/CLAUDE.md` to leave the
> assessment untouched.

## 1. AIDLC — the spec/planning methodology

AWS AI-DLC (v1.0.1) steering rules are installed at:

- [`.clinerules/core-workflow.md`](../.clinerules/core-workflow.md) — the core workflow ruleset
- [`.aidlc-rule-details/`](../.aidlc-rule-details/) — detailed inception / construction / operations rules

Invoke by starting a request with **"Using AI-DLC, …"**. Phases: **Inception** (requirements,
design, risk) → **Construction** (component design, code, tests) → **Operations**. Each phase asks
structured questions and pauses at **approval gates**. Generated planning docs land in `aidlc-docs/`
(gitignored).

## 2. Superpowers — the execution skills (`.claude/skills/`)

Reach for these skills at the matching moment (they auto-surface, but invoke deliberately):

| When | Skill |
|---|---|
| Before any creative/feature work — explore intent & requirements | `brainstorming` |
| Turning a spec into a step-by-step plan | `writing-plans` |
| Executing a written plan with checkpoints | `executing-plans` |
| Implementing any feature/bugfix — tests first | `test-driven-development` |
| Any bug, test failure, or surprise — before proposing a fix | `systematic-debugging` |
| Independent tasks that can run concurrently | `dispatching-parallel-agents`, `subagent-driven-development` |
| Before claiming work complete/passing | `verification-before-completion` |
| Completing a feature or before merge | `requesting-code-review` → `receiving-code-review` |
| Isolating risky work | `using-git-worktrees` |
| Wrapping up a branch | `finishing-a-development-branch` |

## 3. Operating loop for this repo

1. **Spec → Plan.** Read `CLAUDE.MD`; use `brainstorming` + `writing-plans` (or "Using AI-DLC,").
2. **Plan → Code, test-first.** `test-driven-development`; keep stages deterministic; the single
   LLM call is bounded and validated.
3. **Debug methodically.** `systematic-debugging` — root cause before patch.
4. **Verify before done.** `verification-before-completion`: run `python run.py`, `python validate.py`,
   `pytest` and confirm output before asserting success.
5. **Review.** `requesting-code-review`; record decisions as ADRs in [`docs/adr/`](../docs/adr/).

## 4. Guardrails specific to this project

- Retrieval is **local and deterministic** (TF-IDF from `kb.json`) — no external calls for retrieval.
- Exactly **one** controlled LLM call (the review stage); it never does retrieval and never produces
  the final recommendation. The recommendation is computed in deterministic code.
- Validate all LLM output before use; log every call to `llm_calls.jsonl`.
- Secrets live in `.env` (gitignored); never print the key.
