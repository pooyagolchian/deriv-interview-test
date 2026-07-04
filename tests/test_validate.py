"""End-to-end: run the full pipeline on synthetic fixtures, then run the real validator."""
import validate as validator

from evalharness import config
from evalharness.pipeline import run_pipeline

ARTIFACTS = [
    config.RETRIEVAL_FILE, config.AUTOMATED_SCORES_FILE, config.LLM_REVIEW_FILE,
    config.RECOMMENDATION_JSON_FILE, config.RECOMMENDATION_MD_FILE, config.REVIEW_REPORT_FILE,
    config.FAILURE_TAXONOMY_FILE, config.LLM_CALLS_FILE,
]


def test_pipeline_produces_all_artifacts(fixture_dir):
    run_pipeline(cfg=config.DEFAULT_CONFIG, provider="stub")
    for name in ARTIFACTS:
        assert (fixture_dir / name).exists(), f"missing artifact: {name}"


def test_pipeline_promotes_the_safe_variant(fixture_dir):
    summary = run_pipeline(cfg=config.DEFAULT_CONFIG, provider="stub")
    # The unsafe variant is disqualified on the high-risk query, so the safe one is promoted.
    assert summary["selected_variant"] == "safe"
    assert summary["no_safe_variant_overall"] is False


def test_validator_passes_on_generated_artifacts(fixture_dir):
    run_pipeline(cfg=config.DEFAULT_CONFIG, provider="stub")
    assert validator.main() == 0


def test_recommendation_is_reproducible_from_stored_outputs(fixture_dir):
    from evalharness.aggregate import aggregate
    from evalharness.io_utils import load_json

    run_pipeline(cfg=config.DEFAULT_CONFIG, provider="stub")
    scores = load_json(fixture_dir / config.AUTOMATED_SCORES_FILE)
    reviews = load_json(fixture_dir / config.LLM_REVIEW_FILE)
    rec = load_json(fixture_dir / config.RECOMMENDATION_JSON_FILE)
    cfg = config.EvalConfig.from_dict(rec["config"])
    recomputed = aggregate(scores, reviews, cfg)
    assert recomputed["selected_variant"] == rec["selected_variant"]
