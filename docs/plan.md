# Plan: Spec-driven RAG Evaluation Harness (+ AIDLC / Superpowers `.claude` setup)

## Context

`CLAUDE.MD` in this repo is a take-home assessment: build a small, **replayable, deterministic** offline evaluator for retrieval-augmented QA. It reads `kb.json` / `queries.json` / `candidate_answers.json`, retrieves evidence, runs rule-based checks, makes **one controlled LLM call**, and produces a **reproducible** recommendation of which prompt variant to promote — treating safety/grounding failures seriously.

You additionally asked to (1) install **AIDLC** (AWS spec-driven planning methodology) and **superpowers** skills from skills.sh, and (2) configure `.claude/` for "spec of planning + superpower to reach the result." Both are AI-development _methodology_ tooling that shapes how we build; they are not runtime dependencies of the app.

**Confirmed decisions:** Skip Strands Agents (its autonomous agent loop conflicts with the required determinism/reproducibility — the single review call uses the Anthropic SDK directly). Retrieval is **local TF-IDF cosine** from `kb.json` (spec forbids external retrieval; the `.env` `CLAUDE_API_KEY` is Anthropic, which has no embeddings endpoint anyway). The Claude key powers only the one LLM review stage.

Environment: Python 3.14, Node 22, git available (repo is **not** yet git-initialized). `.env` holds a real Claude key → default runs make one real API call; a deterministic stub keeps the pipeline fully runnable offline.

---

## Part A — Tooling & `.claude` configuration (spec planning + superpowers)

1. **Superpowers skills** → `.claude/skills/` via `npx skills add obra/superpowers -a claude-code -s '*' -y`. Gives spec-driven planning (`brainstorming`, `writing-plans`, `executing-plans`) + execution superpowers (`test-driven-development`, `systematic-debugging`, `verification-before-completion`, `requesting-code-review`, `dispatching-parallel-agents`).
2. **AIDLC methodology** (already cloned to scratchpad) → copy `aidlc-rules/aws-aidlc-rules/core-workflow.md` to `.clinerules/core-workflow.md` and `aidlc-rules/aws-aidlc-rule-details/` to `.aidlc-rule-details/` (AIDLC's recommended non-destructive Claude Code layout).
3. **`.claude/CLAUDE.md`** — a project operating-guide that wires it together: names AIDLC as the spec→plan→build methodology, lists the superpowers skills and when to use each, and points to the assessment (`CLAUDE.MD`) + the ADRs.
   - ⚠️ macOS FS is case-insensitive: root `CLAUDE.MD` (the assessment) == `CLAUDE.md`. We must **not** create a root `CLAUDE.md` (it would overwrite the assessment). The methodology guide lives at `.claude/CLAUDE.md` and `.clinerules/`, leaving `CLAUDE.MD` pristine.
4. **`.claude/settings.json`** — minimal project settings (permissions allowlist for `python`/`pytest`/`npx`) so future runs prompt less.

## Part B — The application (deterministic RAG eval harness)

**Layout** (inputs + generated artifacts at repo root per the spec's flat artifact list; logic in a package):

```
run.py            validate.py        pyproject.toml   Makefile   README.md   .gitignore
kb.json  queries.json  candidate_answers.json                 # inputs (seeded from spec)
evalharness/
  config.py     # paths, weights, thresholds (tau_g=0.5), model id, code_version
  textnorm.py   # ONE shared normalizer: casefold→[a-z0-9]+→stopwords→stem; content_tokens()
  schema.py     # dataclasses + validators for inputs & every output artifact
  io_utils.py   # load/save json, append jsonl
  retrieval.py  # hand-rolled TF-IDF cosine, top-2, deterministic tie-break
  checks.py     # rule checks + grounding + risk_flags
  taxonomy.py   # failure-taxonomy tagging
  llm_review.py # provider abstraction (real Anthropic + deterministic stub), schema validation, logging
  aggregate.py  # PURE two-tier safety-first aggregation (the recommendation engine)
  report.py     # recommendation.md + recommendation.json (sidecar) + review_report.md
  pipeline.py   # orchestrates stages
```

**Design (from a pressure-tested review; drives the ADRs):**

- **Retrieval (`retrieval.py`)** — TF-IDF cosine over hand-rolled weights (`idf = ln((N+1)/(df+1))+1`), cosine naturally in `[0,1]` (matches the example's 0.83/0.21 without post-hoc scaling). Shared tokenizer with stemming so `withdrawal`~`withdrawals`. Top-2, tie-break `(score DESC, doc_id ASC)`. Writes `retrieval.json`.
- **Rule checks (`checks.py`)** — `retrieval_hit`, `must_include_pass` (≥1 required item via **normalized token-subset** match — order/plural-insensitive), `must_not_claim_pass` (**pass = SAFE = no banned claim**, via normalized + gapped-subsequence match), `grounding_score` = `|content_tokens(answer) ∩ content_tokens(top2)| / |content_tokens(answer)|` (precision → robust to long evidence), `risk_flags` from a fixed vocab. Writes `automated_scores.json`.
  - ⚠️ **Deliberate, documented spec deviation:** the sample's `must_include`/`must_not_claim` values do **not** exact-substring-match the real answers (naive substring would pass 3/4 unsafe answers and fail the correct one). We use normalized token/subsequence matching and back safety with grounding — recorded in an ADR.
- **LLM review (`llm_review.py`)** — exactly ONE call over all pairs. Judges only clarity / faithfulness / overclaim / winner. `provider="anthropic"` when key present (temp 0, model `claude-sonnet-5`, reads `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY`), else `provider="stub"` deriving schema-valid output from rule signals only (no gold answers). Validate every returned value (allowed variants, int 1–5, bool); **fail closed** to stub value + warning on violation. Every call logged to `llm_calls.jsonl`. Writes `llm_review.json`.
- **Aggregation (`aggregate.py`, pure)** — two-tier lexicographic, safety dominates quality. Hard-safety violation = banned-claim OR LLM overclaim OR empty answer OR (high-risk AND low grounding). On **high-risk** a violation **disqualifies** the variant; on medium/low it is a risk-scaled penalty. Quality score weights grounding/faithfulness highest, **clarity lowest (0.10)** so a clearer-but-unsafe variant cannot win. Overall pick = variant never DQ'd on any high-risk query, then best mean quality; deterministic tie-breaks; honest "no safe variant" outcome if none qualify. Generalizes to **N variants** (variant universe = union of answer keys; `variant_key ASC` final tie-break) → a 3rd variant drops in with zero core changes.
- **Reports (`report.py`)** — `recommendation.md` (selected variant, wins/failures table, top reasons, clarity-vs-safety tradeoff, limitations) rendered from a machine-readable **`recommendation.json`** sidecar (config + per-query decisions + input hashes). `review_report.md` = per-query explainability (question, retrieved docs+scores, both answers, failed checks, LLM winner, final pick). `failure_taxonomy.json` = controlled-vocab tags per answer, deterministically mapped from the checks.

**`run.py`** regenerates every artifact end-to-end. **`validate.py`** checks: artifacts exist + valid JSON; all queries × all variants processed; top-2 per query; LLM values use only allowed variants and score ranges; and **reproducibility** — recompute `aggregate()` from stored `retrieval.json`+`automated_scores.json`+`llm_review.json` and assert it yields the selected variant recorded in `recommendation.json`/`.md` (no LLM/retrieval re-run).

## Part C — Tests (`tests/`, pytest)

`test_textnorm`, `test_retrieval` (determinism, top-2, tie-break), `test_checks` (banned-claim paraphrase detection, grounding high-vs-fabricated, token-subset include), `test_aggregate` (clearer-but-unsafe **loses** on high-risk; N-variant generalization; safe-variant promoted), `test_llm_review` (stub determinism, schema validation, fail-closed on bad values), `test_validate` (end-to-end on tiny fixtures → all validation checks pass). Fixtures use their own tiny kb/queries/answers (no hardcoding of the sample).

## Part D — ADR documentation (`docs/adr/`, MADR style)

`0001` record-architecture-decisions · `0002` deterministic TF-IDF retrieval · `0003` rule checks + token-subset matching (spec deviation) + precision grounding · `0004` single controlled LLM stage + provider abstraction + stub · `0005` safety-first two-tier aggregation · `0006` reproducibility via `recommendation.json` sidecar · `0007` N-variant extensibility / regression guard · `0008` methodology & tooling (AIDLC + superpowers; Strands rejected; local retrieval over embeddings).

## Execution approach (ultracode)

Implement the pipeline coherently as a single author (tightly coupled stages). Use background **workflows** for the parallelizable, verify-heavy phases: (a) fan-out authoring of the 8 ADRs, (b) a final **multi-dimensional adversarial review** (spec-compliance, correctness, safety-logic, determinism/reproducibility, test coverage) with per-finding verification before I fix. Track progress with a todo list.

## Verification

1. `python -m pip install -e ".[dev]"` (installs `anthropic` + `pytest`). 2. `python run.py` → regenerates all 8 artifacts, making one real Claude call (logged to `llm_calls.jsonl`). 3. `python validate.py` → all checks pass, incl. reproducibility. 4. `pytest -q` → green. 5. Sanity-read `recommendation.md` + `review_report.md`: expect **prompt_a promoted**, prompt_b flagged unsafe/overclaiming on high-risk Q's. 6. Confirm offline path: unset key → `run.py` still completes via stub, `validate.py` still passes.

## Notes / safety

`.env` is added to `.gitignore` and its key is never printed. If we `git init`, `.env` and generated artifacts stay untracked as configured.
