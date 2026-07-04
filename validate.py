#!/usr/bin/env python3
"""Validate the evaluation artifacts and the reproducibility of the recommendation.

Checks (spec section 5):
- required artifacts exist and JSON is valid
- all queries were processed for every variant
- retrieval output contains the top-2 passages per query
- LLM review values use only allowed variants and the expected score range (1-5)
- the recommendation is reproducible: recomputing the aggregation from the stored
  retrieval/automated_scores/llm_review yields the recorded selected variant

Exit code 0 = all checks pass, 1 = at least one failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evalharness import config  # noqa: E402
from evalharness.aggregate import aggregate  # noqa: E402
from evalharness.io_utils import load_json  # noqa: E402
from evalharness.schema import load_candidates, load_queries, variant_universe  # noqa: E402
from evalharness.taxonomy import VOCABULARY  # noqa: E402


class Validator:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passes.append(label)
            print(f"  PASS  {label}")
        else:
            msg = f"{label}" + (f" — {detail}" if detail else "")
            self.failures.append(msg)
            print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        return ok

    def report(self) -> int:
        print("-" * 60)
        print(f"{len(self.passes)} passed, {len(self.failures)} failed")
        if self.failures:
            print("\nFailures:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print("All validation checks passed.")
        return 0


def _valid_score(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and config.SCORE_MIN <= x <= config.SCORE_MAX


def main() -> int:
    v = Validator()
    print("Validating evaluation artifacts...\n")

    # ---- inputs load & validate -----------------------------------------------------------
    try:
        queries = load_queries()
        candidates = load_candidates()
        variants = variant_universe(candidates)
        answers_by_q = {c.query_id: c.answers for c in candidates}
        v.check(True, "inputs load and validate", f"{len(queries)} queries, variants={variants}")
    except Exception as exc:  # noqa: BLE001
        v.check(False, "inputs load and validate", str(exc))
        return v.report()

    query_ids = [q.query_id for q in queries]

    # ---- artifacts exist & JSON valid -----------------------------------------------------
    json_artifacts = {
        "retrieval": config.RETRIEVAL_FILE,
        "automated_scores": config.AUTOMATED_SCORES_FILE,
        "llm_review": config.LLM_REVIEW_FILE,
        "recommendation_json": config.RECOMMENDATION_JSON_FILE,
        "failure_taxonomy": config.FAILURE_TAXONOMY_FILE,
    }
    text_artifacts = {
        "recommendation_md": config.RECOMMENDATION_MD_FILE,
        "review_report": config.REVIEW_REPORT_FILE,
    }
    loaded: dict = {}
    for name, fname in json_artifacts.items():
        path = config.output_path(fname)
        if not v.check(path.exists(), f"artifact exists: {fname}"):
            continue
        try:
            loaded[name] = load_json(path)
            v.check(True, f"valid JSON: {fname}")
        except json.JSONDecodeError as exc:
            v.check(False, f"valid JSON: {fname}", str(exc))

    for name, fname in text_artifacts.items():
        path = config.output_path(fname)
        if v.check(path.exists(), f"artifact exists: {fname}"):
            loaded[name] = path.read_text(encoding="utf-8")

    llm_calls_path = config.output_path(config.LLM_CALLS_FILE)
    v.check(llm_calls_path.exists(), f"artifact exists: {config.LLM_CALLS_FILE}")

    # Bail on structural checks if core artifacts failed to load.
    for req in ("retrieval", "automated_scores", "llm_review", "recommendation_json"):
        if req not in loaded:
            return v.report()

    retrieval = loaded["retrieval"]
    scores = loaded["automated_scores"]
    llm_review = loaded["llm_review"]
    rec = loaded["recommendation_json"]

    # ---- retrieval: top-2 per query -------------------------------------------------------
    retr_by_q = {r.get("query_id"): r for r in retrieval if isinstance(r, dict)}
    v.check(set(retr_by_q) == set(query_ids), "retrieval covers exactly the input queries",
            f"got {sorted(retr_by_q)}")
    expected_k = config.DEFAULT_CONFIG.top_k
    for qid in query_ids:
        rec_r = retr_by_q.get(qid, {})
        got = rec_r.get("retrieved", [])
        v.check(len(got) == expected_k, f"retrieval {qid}: exactly top-{expected_k} passages",
                f"got {len(got)}")
        for i, p in enumerate(got):
            ok = all(k in p for k in ("doc_id", "score", "title", "text")) and \
                isinstance(p.get("score"), (int, float)) and 0.0 <= p["score"] <= 1.0
            v.check(ok, f"retrieval {qid}[{i}]: well-formed passage with score in [0,1]")
        scores_desc = [p.get("score", 0) for p in got]
        v.check(scores_desc == sorted(scores_desc, reverse=True),
                f"retrieval {qid}: scores in descending order")

    # ---- automated_scores: all (query, variant) processed ---------------------------------
    score_keys = {(s.get("query_id"), s.get("variant")) for s in scores}
    for qid in query_ids:
        for variant in sorted(answers_by_q.get(qid, {})):
            v.check((qid, variant) in score_keys,
                    f"automated_scores processed {qid}/{variant}")
    for s in scores:
        label = f"automated_scores {s.get('query_id')}/{s.get('variant')}"
        ok_fields = all(k in s for k in ("retrieval_hit", "must_include_pass", "must_not_claim_pass",
                                         "grounding_score", "risk_flags", "notes"))
        v.check(ok_fields, f"{label}: has required fields")
        g = s.get("grounding_score")
        v.check(isinstance(g, (int, float)) and 0.0 <= g <= 1.0, f"{label}: grounding_score in [0,1]")
        v.check(isinstance(s.get("risk_flags"), list) and len(s["risk_flags"]) >= 1,
                f"{label}: non-empty risk_flags")

    # ---- llm_review: allowed variants + score ranges --------------------------------------
    llm_by_q = {r.get("query_id"): r for r in llm_review}
    v.check(set(llm_by_q) == set(query_ids), "llm_review covers exactly the input queries")
    for qid in query_ids:
        r = llm_by_q.get(qid, {})
        qvariants = set(answers_by_q.get(qid, {}))
        winner = r.get("winner")
        v.check(winner in qvariants or winner == "tie",
                f"llm_review {qid}: winner is an allowed variant or 'tie'", f"got {winner!r}")
        for field in ("faithfulness", "clarity"):
            fmap = r.get(field, {})
            v.check(isinstance(fmap, dict) and set(fmap) >= qvariants,
                    f"llm_review {qid}: {field} covers all variants")
            v.check(all(_valid_score(fmap.get(vt)) for vt in qvariants),
                    f"llm_review {qid}: {field} values are ints in [{config.SCORE_MIN},{config.SCORE_MAX}]")
        oc = r.get("overclaim_flags", {})
        v.check(isinstance(oc, dict) and all(isinstance(oc.get(vt), bool) for vt in qvariants),
                f"llm_review {qid}: overclaim_flags are booleans for all variants")

    # ---- failure taxonomy: known tags -----------------------------------------------------
    if "failure_taxonomy" in loaded:
        for t in loaded["failure_taxonomy"]:
            bad = [tag for tag in t.get("tags", []) if tag not in VOCABULARY]
            v.check(not bad, f"failure_taxonomy {t.get('query_id')}/{t.get('variant')}: tags in vocabulary",
                    f"unknown {bad}")

    # ---- reproducibility: recompute aggregation from stored outputs ------------------------
    stored_variant = rec.get("selected_variant")
    cfg = config.EvalConfig.from_dict(rec.get("config", {}))
    recomputed = aggregate(scores, llm_review, cfg)
    v.check(recomputed["selected_variant"] == stored_variant,
            "recommendation reproducible: recomputed selected_variant matches stored",
            f"recomputed={recomputed['selected_variant']} stored={stored_variant}")

    stored_pq = {p["query_id"]: p["selected_variant"] for p in rec.get("per_query", [])}
    recomputed_pq = {p["query_id"]: p["selected_variant"] for p in recomputed["per_query"]}
    v.check(stored_pq == recomputed_pq, "recommendation reproducible: per-query winners match",
            f"stored={stored_pq} recomputed={recomputed_pq}")

    if "recommendation_md" in loaded:
        v.check(stored_variant and stored_variant in loaded["recommendation_md"],
                "recommendation.md names the selected variant")

    return v.report()


if __name__ == "__main__":
    raise SystemExit(main())
