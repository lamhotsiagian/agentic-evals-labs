"""Unit tests for Chapter 3 Tool-Calling Evals."""

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

from tools import get_order, calculate_refund, search_customer, get_weather, send_email
from agent import EcommerceCustomerAgent
from evaluator import ToolCallingEvaluator
from agent import ToolRuntime


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

    # Normal order inquiry -- order #1234 is real in the lab database
    res = agent.execute_task("Find order #1234 and calculate refund eligibility.")
    assert len(res["tool_calls"]) == 2
    assert res["recovered"] is False

    scores = evaluator.evaluate_execution(["get_order", "calculate_refund"], res)
    assert scores["tool_selection"].score == 1.0
    assert scores["tool_order"].score == 1.0
    assert scores["result_handling"].score == 1.0
    assert scores["groundedness"].score == 1.0  # order_id 1234 comes straight from the prompt


def test_agent_recovery_on_missing_order():
    # #99999 does not exist in the lab database -- a real not-found error,
    # not an artificially injected bad argument.
    agent = EcommerceCustomerAgent()
    evaluator = ToolCallingEvaluator()

    res = agent.execute_task("Find order #99999")
    assert res["recovered"] is True
    scores = evaluator.evaluate_execution(["get_order"], res, expect_error_recovery=True)
    assert scores["agent_recovery"].score == 1.0


def test_hallucinated_argument_is_caught_by_groundedness():
    # This is Chapter 3's central lesson made executable: a value the
    # agent invents (not in the prompt, not in any prior tool result)
    # must fail groundedness even though it is well-formed.
    runtime = ToolRuntime()
    runtime.execute("get_weather", {"zip_code": "94105"})  # prompt asked for a DIFFERENT zip
    fake_result = {
        "prompt": "Check weather for zip 10001",
        "tool_calls": runtime.calls,
        "final_answer": "Weather checked.",
        "recovered": False,
    }
    scores = ToolCallingEvaluator().evaluate_execution(None, fake_result)
    assert scores["groundedness"].score < 1.0
