# RAG-QA Evaluation Harness

A small, **replayable, deterministic** offline evaluator for retrieval-augmented question
answering. Given a local knowledge base, a set of user questions, and answers from two (or more)
prompt strategies, it retrieves evidence, runs code-based quality/safety checks, makes **one
controlled LLM review call**, and computes a **reproducible recommendation** of which prompt
variant to promote — with safety and grounding failures weighted heavily.

Built to be extended, not thrown away. See [`docs/adr/`](docs/adr/) for the design rationale.

---

## Quickstart

```bash
# 1. (optional) create a venv with the dev/LLM deps
make venv                      # or: python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# 2. regenerate every artifact from kb.json / queries.json / candidate_answers.json
python run.py                  # uses the real LLM if a key is set, else a deterministic stub

# 3. validate artifacts + recommendation reproducibility
python validate.py

# 4. run the tests
python -m pytest
```

`make all` runs `run` + `validate` + `test`. The **core pipeline needs no third-party
packages** — retrieval, checks, aggregation, reporting, and validation are pure standard
library, so `python run.py` works from a clean checkout. `anthropic` and `pytest` are only
needed for the real LLM call and the tests.

### LLM configuration

The single review call uses Anthropic. Put a key in `.env` (gitignored):

```bash
cp .env.example .env
# CLAUDE_API_KEY=sk-ant-...      (ANTHROPIC_API_KEY also works)
```

- **Key present + `anthropic` installed** → one real Claude call (`claude-sonnet-5` by default).
- **No key / SDK / offline** → a deterministic **stub** derives review scores from the rule
  signals, so the pipeline still produces every artifact.

Override with `python run.py --provider stub` (force offline) or `--model <id>`, or the env
vars `EVAL_LLM_PROVIDER` / `EVAL_LLM_MODEL`.

---

## Pipeline

```
kb.json ─┐
queries.json ─┼─► 1. retrieval ─► 2. rule checks ─► 3. one LLM review ─► 4. aggregation ─► reports
candidate_answers.json ─┘   (TF-IDF)    (deterministic)   (validated+logged)  (safety-first, pure)
```

| #   | Stage                                                                                | Module                      | Output                                                         |
| --- | ------------------------------------------------------------------------------------ | --------------------------- | -------------------------------------------------------------- |
| 1   | **Retrieval** — deterministic TF-IDF cosine, top-2, scores preserved                 | `evalharness/retrieval.py`  | `retrieval.json`                                               |
| 2   | **Rule checks** — retrieval hit, must-include, banned-claim, grounding, risk flags   | `evalharness/checks.py`     | `automated_scores.json`                                        |
| 3   | **LLM review** — one call; clarity/faithfulness/overclaim/winner; validated + logged | `evalharness/llm_review.py` | `llm_review.json`, `llm_calls.jsonl`                           |
| 4   | **Aggregation** — pure, safety-first, two-tier; the recommendation engine            | `evalharness/aggregate.py`  | —                                                              |
| 5   | **Reports** — human writeup + machine sidecar + explainability                       | `evalharness/report.py`     | `recommendation.md`, `recommendation.json`, `review_report.md` |
| —   | **Failure taxonomy** — controlled-vocab tags per answer                              | `evalharness/taxonomy.py`   | `failure_taxonomy.json`                                        |

### Key design choices (full rationale in ADRs)

- **Deterministic, local retrieval** (TF-IDF cosine, hand-rolled) — no external calls, scores
  naturally in `[0,1]`. [ADR-0002](docs/adr/0002-deterministic-tfidf-retrieval.md)
- **Normalized, token-aware matching** instead of literal substring — the sample's
  `must_include` / `must_not_claim` values do not exact-match the answers; literal matching
  would pass unsafe answers and fail correct ones. Safety never rests on `must_not_claim`
  alone. [ADR-0003](docs/adr/0003-rule-checks-and-grounding.md)
- **One controlled LLM call**, strictly validated, with a deterministic stub fallback; the LLM
  never retrieves and never makes the final call. [ADR-0004](docs/adr/0004-single-llm-review-stage.md)
- **Safety-first, two-tier aggregation** — a banned claim / overclaim / ungrounded answer on a
  **high-risk** query is disqualifying; clarity is weighted lowest so a clearer-but-unsafe
  variant cannot win. [ADR-0005](docs/adr/0005-safety-first-aggregation.md)
- **Reproducible recommendation** — the decision is recomputable from stored artifacts via the
  `recommendation.json` sidecar; `validate.py` enforces it. [ADR-0006](docs/adr/0006-reproducibility-sidecar.md)
- **N-variant ready** — a third prompt variant drops in with no core changes. [ADR-0007](docs/adr/0007-n-variant-extensibility.md)

---

## Artifacts

| File                    | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `retrieval.json`        | Top-2 evidence passages + scores per query                                   |
| `automated_scores.json` | Per (query, variant) rule-check results                                      |
| `llm_review.json`       | Validated LLM judgments per query                                            |
| `llm_calls.jsonl`       | Append-only audit log of every LLM call (provider, status, prompt, response) |
| `recommendation.md`     | Human promotion writeup: pick, summary table, reasons, tradeoffs, limits     |
| `recommendation.json`   | Machine-readable decision sidecar (config + per-query trace + input hashes)  |
| `review_report.md`      | Per-query explainability view for debugging                                  |
| `failure_taxonomy.json` | Controlled-vocab failure tags per answer                                     |

---

## Configuration & extension

All tunables (retrieval `top_k`, grounding threshold, scoring weights, penalties) live in
`EvalConfig` in [`evalharness/config.py`](evalharness/config.py) and are snapshotted into
`recommendation.json` for reproducibility. To add a stage or a variant, the input schema and
aggregation already generalize over an arbitrary set of variant keys.

## Development methodology

This repo is set up for spec-driven development: **AIDLC** steering rules
([`.clinerules/`](.clinerules/), [`.aidlc-rule-details/`](.aidlc-rule-details/)) for the
spec→plan→build workflow, and the **superpowers** skill set in `.claude/skills/` for planning,
TDD, debugging, and verification. See [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

## Limitations

Lexical retrieval misses pure semantic paraphrase on larger corpora; grounding is token-overlap
precision (a coarse proxy); banned-claim detection is safety-biased and may over-flag; weights
are hand-tuned defaults, not learned. The `recommendation.md` "Known limitations" section is
regenerated each run.
