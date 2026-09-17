"""Unit tests for Chapter 1 evaluation logic."""

import os
import sys

# Ensure chapter and shared are in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CHAPTER_DIR not in sys.path:
    sys.path.insert(0, CHAPTER_DIR)

from agent import CustomerSupportAgent
from evaluator import CustomerSupportEvaluator
from shared.models.schemas import EvaluationCase
from shared.models.provider import get_model_provider


def test_customer_support_agent_response():
    provider = get_model_provider()
    agent = CustomerSupportAgent(model_name="qwen2.5:3b", provider=provider)
    res = agent.respond("I was charged twice for my subscription. What should I do?")
    assert "response" in res
    assert len(res["response"]) > 0
    assert res["model"] == "qwen2.5:3b"


def test_customer_support_evaluator():
    # A real reference answer, not a placeholder label -- this is the
    # exact bug an earlier version of this lab had: expected_output used to be
    # captured on every case and never once read by the grader.
    evaluator = CustomerSupportEvaluator()
    case = EvaluationCase(
        id="CS-TEST-01",
        input_prompt="I was charged twice for my subscription. What should I do?",
        expected_output="We verified the duplicate charge and will refund it to your card within 3-5 business days.",
    )
    actual_output = "We verified the duplicate charge and issued a full refund to your card within 3-5 business days."
    result = evaluator.evaluate_case(case, actual_output, latency_seconds=1.2)

    assert result.passed is True
    assert result.case_id == "CS-TEST-01"
    assert "relevance" in result.metrics
    assert result.metrics["relevance"].score >= 0.25
    assert result.metrics["hallucination"].passed is True


def test_grader_fails_the_wrong_answer_and_passes_the_faithful_one():
    # This is Chapter 1's central lesson made executable: an unvalidated
    # grader can be anti-correlated with quality. The faithful answer
    # must pass and the answer that contradicts policy must fail.
    evaluator = CustomerSupportEvaluator()
    case = EvaluationCase(
        id="CS-001",
        input_prompt="I was charged twice for my subscription this morning ($49.99 x 2). "
                      "What should I do? (Case Reference #001)",
        expected_output="We verified the duplicate charge and will refund the extra $49.99 "
                         "within 3-5 business days.",
    )
    faithful = "We are sorry. We verified the duplicate charge and will refund the extra $49.99 within 3-5 business days."
    wrong = "Your subscription charge refund is impossible; charges are final forever."

    faithful_result = evaluator.evaluate_case(case, faithful, latency_seconds=1.0)
    wrong_result = evaluator.evaluate_case(case, wrong, latency_seconds=1.0)

    assert faithful_result.passed is True
    assert wrong_result.passed is False
    assert faithful_result.metrics["correctness"].score > wrong_result.metrics["correctness"].score
