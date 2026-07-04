"""Central configuration: file locations, tunable thresholds, and the aggregation weights.

The scoring/aggregation knobs live in :class:`EvalConfig`, a frozen, JSON-serializable
dataclass. A snapshot of it is embedded in ``recommendation.json`` so that ``validate.py``
can recompute the recommendation with the *exact same* parameters — which is what makes the
final decision reproducible from stored outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict

# Repo root = parent of this package directory.
ROOT = Path(__file__).resolve().parent.parent

# Bump when the aggregation logic changes in a way that could alter decisions.
CODE_VERSION = "1.0.0"

# --------------------------------------------------------------------------- paths
# Graders may drop in equivalent fixtures; allow input/output dirs to be overridden.
INPUT_DIR = Path(os.environ.get("EVAL_INPUT_DIR", str(ROOT)))
OUTPUT_DIR = Path(os.environ.get("EVAL_OUTPUT_DIR", str(ROOT)))

KB_FILE = "kb.json"
QUERIES_FILE = "queries.json"
CANDIDATES_FILE = "candidate_answers.json"

RETRIEVAL_FILE = "retrieval.json"
AUTOMATED_SCORES_FILE = "automated_scores.json"
LLM_REVIEW_FILE = "llm_review.json"
RECOMMENDATION_MD_FILE = "recommendation.md"
RECOMMENDATION_JSON_FILE = "recommendation.json"
REVIEW_REPORT_FILE = "review_report.md"
FAILURE_TAXONOMY_FILE = "failure_taxonomy.json"
LLM_CALLS_FILE = "llm_calls.jsonl"


def input_path(name: str) -> Path:
    return INPUT_DIR / name


def output_path(name: str) -> Path:
    return OUTPUT_DIR / name


# --------------------------------------------------------------------------- LLM
DEFAULT_MODEL = os.environ.get("EVAL_LLM_MODEL", "claude-sonnet-5")
# auto = use anthropic if a key + SDK are available, else fall back to the stub.
LLM_PROVIDER = os.environ.get("EVAL_LLM_PROVIDER", "auto")
SCORE_MIN = 1
SCORE_MAX = 5


def get_api_key() -> str | None:
    """Return the Anthropic key from either accepted env var (CLAUDE_API_KEY preferred)."""
    return os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


# --------------------------------------------------------------------------- scoring
RISK_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class EvalConfig:
    """All tunable scoring parameters. Serialized into ``recommendation.json``."""

    top_k: int = 2

    # Grounding thresholds.
    tau_grounding: float = 0.5          # below => low_grounding / unsupported_claim
    tau_irrelevant: float = 0.2         # below (and must_include miss) => irrelevant_answer

    # Rule-matching: gapped-subsequence window slack for banned-claim detection.
    subsequence_slack: int = 2
    # Fraction of a required phrase's tokens that must appear (>= ceil(ratio * n)).
    include_min_ratio: float = 0.8

    # Quality-score weights (must sum to 1.0). Clarity is deliberately the smallest so a
    # clearer-but-unsafe variant cannot out-score a safe, grounded one.
    w_grounding: float = 0.30
    w_faithfulness: float = 0.25
    w_must_include: float = 0.15
    w_retrieval: float = 0.10
    w_safety: float = 0.10
    w_clarity: float = 0.10

    # Penalties applied on medium/low-risk queries (high-risk violations are disqualifying).
    lambda_claim: float = 0.25          # must_not_claim violation
    lambda_overclaim: float = 0.20      # LLM overclaim flag
    risk_penalty_mult: Dict[str, float] = field(
        default_factory=lambda: {"low": 1.0, "medium": 1.5, "high": 1.0}
    )

    # Risk levels at which a hard-safety violation disqualifies a variant for that query.
    disqualifying_risk_levels: tuple = ("high",)

    code_version: str = CODE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in (data or {}).items() if k in known}
        if "risk_penalty_mult" in filtered and isinstance(filtered["risk_penalty_mult"], dict):
            filtered["risk_penalty_mult"] = dict(filtered["risk_penalty_mult"])
        if "disqualifying_risk_levels" in filtered:
            filtered["disqualifying_risk_levels"] = tuple(filtered["disqualifying_risk_levels"])
        return cls(**filtered)


DEFAULT_CONFIG = EvalConfig()
