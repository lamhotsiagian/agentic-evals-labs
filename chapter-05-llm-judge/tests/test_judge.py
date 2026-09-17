"""Unit tests for Chapter 5 LLM-as-a-Judge and Calibration."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "graders", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
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
    assert "needs_human_review" in res


def test_calibration_metrics():
    benchmarks = load_judge_benchmark_cases()
    provider = get_model_provider()
    system = MultiJudgeSystem(provider=provider)
    evals = [system.primary_judge.evaluate(b["prompt"], b["response"]) for b in benchmarks]

    metrics = compute_calibration_metrics(evals, benchmarks)
    assert "agreement_pct" in metrics
    assert "kappa" in metrics
    assert "kappa_ci95" in metrics
    assert metrics["samples_evaluated"] == len(benchmarks)
    # Only 5 real human labels exist -- the report must say so honestly
    # rather than presenting the number as a trustworthy calibration.
    assert metrics["underpowered"] is True


def test_unparseable_judge_output_abstains_instead_of_passing():
    # This is Chapter 5's central lesson made executable: a judge that
    # cannot be parsed must abstain (passed=None), never silently pass.
    from shared.models.provider import LLMProvider
    from shared.evaluators.llm_judge import LLMJudgeEvaluator

    class Garbage(LLMProvider):
        def generate(self, **kwargs):
            return "I think this response is terrible and wrong."
        def embed(self, model, text):
            return []
        def is_available(self):
            return True
        def chat(self, **kwargs):
            raise NotImplementedError

    verdict = LLMJudgeEvaluator(provider=Garbage()).evaluate("What is the capital of Australia?", "Sydney")
    assert verdict.status == "abstain"
    assert verdict.passed is None


def test_safety_floor_fails_verdict_even_with_high_overall():
    # A safety score below the hard floor must fail the verdict even
    # when the weighted overall clears the pass threshold.
    from shared.models.provider import LLMProvider
    from shared.evaluators.llm_judge import LLMJudgeEvaluator
    import json as _json

    class LowSafety(LLMProvider):
        def generate(self, **kwargs):
            return _json.dumps({"correctness": 5, "relevance": 5, "groundedness": 5,
                                  "safety": 1, "task_completion": 5,
                                  "reasoning": "Otherwise excellent but unsafe.", "evidence": []})
        def embed(self, model, text):
            return []
        def is_available(self):
            return True
        def chat(self, **kwargs):
            raise NotImplementedError

    verdict = LLMJudgeEvaluator(provider=LowSafety()).evaluate("prompt", "response")
    assert verdict.status == "scored"
    assert verdict.passed is False  # floor violation overrides a high weighted overall
