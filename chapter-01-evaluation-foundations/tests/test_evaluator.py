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
    evaluator = CustomerSupportEvaluator()
    case = EvaluationCase(
        id="CS-TEST-01",
        input_prompt="I was charged twice for my subscription. What should I do?",
        expected_output="Refund instructions",
    )
    actual_output = "We verified the duplicate charge and issued a full refund to your card."
    result = evaluator.evaluate_case(case, actual_output, latency_seconds=1.2)

    assert result.passed is True
    assert result.case_id == "CS-TEST-01"
    assert "relevance" in result.metrics
    assert result.metrics["relevance"].score >= 0.25
    assert result.metrics["hallucination"].passed is True
