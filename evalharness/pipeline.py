"""Orchestrate the full evaluation pipeline and write every artifact.

    retrieval -> rule checks -> one LLM review -> aggregation -> reports + taxonomy
"""
from __future__ import annotations

from typing import Optional

from . import aggregate as agg_mod
from . import checks
from . import config
from . import llm_review as llm_mod
from . import report
from . import retrieval as retrieval_mod
from . import taxonomy as taxonomy_mod
from .io_utils import load_json, save_json, write_text
from .schema import load_candidates, load_kb, load_queries


def run_pipeline(cfg: config.EvalConfig = config.DEFAULT_CONFIG,
                 provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    # Load + validate inputs.
    kb = load_kb()
    queries = load_queries()
    candidates = load_candidates()
    kb_raw = load_json(config.input_path(config.KB_FILE))
    queries_raw = load_json(config.input_path(config.QUERIES_FILE))
    candidates_raw = load_json(config.input_path(config.CANDIDATES_FILE))

    # 1. Deterministic retrieval.
    retrieval_records = retrieval_mod.run_retrieval(queries, kb, cfg)
    save_json(config.output_path(config.RETRIEVAL_FILE), retrieval_records)

    # 2. Deterministic rule checks.
    scores = checks.run_checks(queries, candidates, retrieval_records, cfg)
    save_json(config.output_path(config.AUTOMATED_SCORES_FILE), scores)

    # 3. One controlled LLM review call (validated + logged).
    resolved_provider = provider or llm_mod.resolve_provider(config.LLM_PROVIDER)
    llm_records = llm_mod.run_llm_review(queries, retrieval_records, scores, candidates,
                                         cfg, provider=resolved_provider, model=model)
    save_json(config.output_path(config.LLM_REVIEW_FILE), llm_records)

    # 4. Deterministic aggregation (the recommendation engine).
    agg = agg_mod.aggregate(scores, llm_records, cfg)

    # 5. Failure taxonomy.
    tax = taxonomy_mod.build_taxonomy(scores, llm_records, cfg)
    save_json(config.output_path(config.FAILURE_TAXONOMY_FILE), tax)

    # 6. Reports (machine sidecar + human writeups).
    rec_json = report.build_recommendation_json(agg, kb_raw, queries_raw, candidates_raw)
    save_json(config.output_path(config.RECOMMENDATION_JSON_FILE), rec_json)
    write_text(config.output_path(config.RECOMMENDATION_MD_FILE),
               report.render_recommendation_md(agg, queries))
    write_text(config.output_path(config.REVIEW_REPORT_FILE),
               report.render_review_report_md(agg, queries, retrieval_records, candidates,
                                              llm_records, tax))

    return {
        "selected_variant": agg["selected_variant"],
        "no_safe_variant_overall": agg["no_safe_variant_overall"],
        "provider": resolved_provider,
        "num_queries": len(queries),
        "variants": agg["variants"],
    }
