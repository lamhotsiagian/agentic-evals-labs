"""Evaluation metric implementations."""

from shared.metrics.quality import (
    compute_exact_match,
    compute_substring_match,
    compute_similarity,
    detect_hallucination,
    compute_relevance,
    compute_faithfulness,
)
from shared.metrics.performance import (
    estimate_tokens,
    estimate_cost_usd,
    compute_latency_stats,
)
from shared.metrics.security import (
    detect_prompt_injection,
    detect_data_leakage,
    detect_policy_violation,
)

__all__ = [
    "compute_exact_match",
    "compute_substring_match",
    "compute_similarity",
    "detect_hallucination",
    "compute_relevance",
    "compute_faithfulness",
    "estimate_tokens",
    "estimate_cost_usd",
    "compute_latency_stats",
    "detect_prompt_injection",
    "detect_data_leakage",
    "detect_policy_violation",
]
