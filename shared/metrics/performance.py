"""Performance and cost evaluation metrics."""

from __future__ import annotations
import math
from typing import Dict, List


# Synthetic model pricing table per 1M tokens ($)
MODEL_PRICING_PER_1M = {
    "qwen2.5:3b": {"prompt": 0.0, "completion": 0.0},  # Local Ollama = $0.00
    "qwen3:1.7b": {"prompt": 0.0, "completion": 0.0},
    "llama3.2:1b": {"prompt": 0.0, "completion": 0.0},
    "cloud_reference_small": {"prompt": 0.15, "completion": 0.60},
    "cloud_reference_large": {"prompt": 2.50, "completion": 10.00},
}


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count based on character and word heuristics."""
    if not text:
        return 0
    words = text.split()
    chars = len(text)
    # Average 4 characters per token or ~1.3 tokens per word
    return max(len(words), math.ceil(chars / 4.0))


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "qwen2.5:3b",
    reference_cloud: bool = False,
) -> float:
    """Calculates estimated cost in USD. Returns $0.00 for local models unless reference_cloud is True."""
    key = "cloud_reference_small" if reference_cloud else model
    pricing = MODEL_PRICING_PER_1M.get(key, {"prompt": 0.0, "completion": 0.0})
    cost = (prompt_tokens / 1_000_000) * pricing["prompt"] + (
        completion_tokens / 1_000_000
    ) * pricing["completion"]
    return round(cost, 6)


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculates p50, p90, p99, mean, min, max latencies."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)

    def percentile(p: float) -> float:
        idx = min(int(math.ceil(p * n)) - 1, n - 1)
        return sorted_lats[max(0, idx)]

    return {
        "mean": round(sum(sorted_lats) / n, 3),
        "p50": round(percentile(0.50), 3),
        "p90": round(percentile(0.90), 3),
        "p99": round(percentile(0.99), 3),
        "min": round(sorted_lats[0], 3),
        "max": round(sorted_lats[-1], 3),
    }
