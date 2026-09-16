"""Unit tests for Chapter 3 Tool-Calling Evals."""

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

from tools import get_order, calculate_refund, search_customer, get_weather, send_email
from agent import EcommerceCustomerAgent
from evaluator import ToolCallingEvaluator


def test_individual_tools():
    o = get_order("1234")
    assert o["status"] == "success"
    assert o["order"]["customer_id"] == "C-101"

    r = calculate_refund("1234")
    assert r["status"] == "success"
    assert r["eligible"] is True

    bad = get_order("non_existent")
    assert bad["status"] == "error"

    w = get_weather("94105")
    assert w["status"] == "success"


def test_agent_tool_execution_flow():
    agent = EcommerceCustomerAgent()
    evaluator = ToolCallingEvaluator()

    # Normal order inquiry
    res = agent.execute_task("Find order #1234 and calculate refund eligibility.")
    assert len(res["tool_calls"]) == 2
    assert res["recovered"] is False

    scores = evaluator.evaluate_execution(["get_order", "calculate_refund"], res)
    assert scores["tool_selection"].score == 1.0
    assert scores["tool_order"].score == 1.0
    assert scores["result_handling"].score == 1.0


def test_agent_recovery_on_bad_arg():
    agent = EcommerceCustomerAgent()
    evaluator = ToolCallingEvaluator()

    res = agent.execute_task("Find order #99999", inject_bad_arg=True)
    assert res["recovered"] is True
    scores = evaluator.evaluate_execution(["get_order"], res, expect_error_recovery=True)
    assert scores["agent_recovery"].score == 1.0
