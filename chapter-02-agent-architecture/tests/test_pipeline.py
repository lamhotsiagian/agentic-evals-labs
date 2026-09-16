"""Unit tests for Chapter 2 Agent Architecture Pipeline."""

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

from pipeline import ArchitecturePipeline
from evaluator import ArchitectureEvaluator
from shared.models.provider import get_model_provider


def test_architecture_pipeline_clean_run():
    provider = get_model_provider()
    pipeline = ArchitecturePipeline(provider=provider)
    evaluator = ArchitectureEvaluator()

    task = "Plan a 3-day trip to Tokyo under $1200"
    constraints = {"destination": "Tokyo", "days": 3, "budget": 1200}
    result = pipeline.run(task, constraints, inject_failure_on_first_try=False)

    assert result["success"] is True
    assert result["retries"] == 0
    assert len(result["pipeline_log"]) >= 3

    scores = evaluator.evaluate_run(result)
    assert scores["planning_accuracy"].score == 1.0
    assert scores["execution_accuracy"].score == 1.0
    assert scores["verification_accuracy"].score == 1.0
    assert scores["end_to_end_success"].score == 1.0


def test_architecture_pipeline_with_retry():
    provider = get_model_provider()
    pipeline = ArchitecturePipeline(provider=provider, max_retries=2)
    evaluator = ArchitectureEvaluator()

    task = "Plan a weekend in Rome under 500 EUR"
    constraints = {"destination": "Rome", "days": 2, "budget": 500}
    result = pipeline.run(task, constraints, inject_failure_on_first_try=True)

    # Should have recovered on attempt 2
    assert result["success"] is True
    assert result["retries"] == 1
    scores = evaluator.evaluate_run(result)
    assert scores["end_to_end_success"].score == 1.0
