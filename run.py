#!/usr/bin/env python3
"""Regenerate every derived artifact from the input files.

    python run.py                 # auto: real LLM call if a key is available, else stub
    python run.py --provider stub # force the offline deterministic stub
    python run.py --model X       # override the review model

Outputs (repo root): retrieval.json, automated_scores.json, llm_review.json,
recommendation.md, recommendation.json, review_report.md, failure_taxonomy.json, llm_calls.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    """Minimal, dependency-free .env loader (does not overwrite existing env vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run the RAG evaluation pipeline.")
    parser.add_argument("--provider", choices=["auto", "anthropic", "stub"], default=None,
                        help="LLM provider for the review stage (default: config/auto).")
    parser.add_argument("--model", default=None, help="Override the review model id.")
    args = parser.parse_args()

    from evalharness import config
    from evalharness.pipeline import run_pipeline

    provider = None if args.provider in (None, "auto") else args.provider
    summary = run_pipeline(cfg=config.DEFAULT_CONFIG, provider=provider, model=args.model)

    print("=" * 60)
    print("Evaluation pipeline complete.")
    print(f"  LLM review provider : {summary['provider']}")
    print(f"  Variants evaluated  : {', '.join(summary['variants'])}")
    if summary["no_safe_variant_overall"]:
        print(f"  ⚠️  NO SAFE VARIANT — least-unsafe fallback: {summary['selected_variant']}")
    else:
        print(f"  ✅ Promote          : {summary['selected_variant']}")
    print("=" * 60)
    print("Artifacts written to repo root. Run `python validate.py` to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
