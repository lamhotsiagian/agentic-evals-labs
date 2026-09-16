"""Unit tests for Chapter 5 LLM-as-a-Judge and Calibration."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
    sys.modules.pop(m, None)

from judge import MultiJudgeSystem
from calibration import compute_calibration_metrics
from shared.models.provider import get_model_provider
from shared.datasets.loader import load_judge_benchmark_cases


def test_multi_judge_system():
    provider = get_model_provider()
    system = MultiJudgeSystem(primary_model="qwen3:1.7b", alt_model="llama3.2:1b", provider=provider)
    res = system.evaluate_pair(
        prompt="Explain symmetric encryption.",
        response="Symmetric encryption uses one key.",
    )
    assert "primary" in res
    assert "alternative" in res
    assert res["primary"].overall_score >= 3.0
    assert "correctness" in res["primary"].dimension_scores


def test_calibration_metrics():
    benchmarks = load_judge_benchmark_cases()
    provider = get_model_provider()
    system = MultiJudgeSystem(provider=provider)
    evals = [system.primary_judge.evaluate(b["prompt"], b["response"]) for b in benchmarks]

    metrics = compute_calibration_metrics(evals, benchmarks)
    assert "agreement_pct" in metrics
    assert "pearson_correlation" in metrics
    assert metrics["samples_evaluated"] == len(benchmarks)
