"""Unit tests for Chapter 9 Chaos Testing and Reliability."""

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

from chaos import ChaosInjector
from resilient_agent import BaselineAgent, ResilientAgent
from evaluator import ChaosExperimentEvaluator


def test_baseline_vs_resilient_under_chaos():
    faults = {"tool_timeout": True, "http_500": False, "invalid_json": False, "context_corruption": False, "tool_unavailable": False}
    injector = ChaosInjector(active_faults=faults)

    baseline = BaselineAgent(injector)
    res_b = baseline.process_order_request("ORD-001")
    assert res_b["success"] is False
    assert res_b["error_type"] == "TimeoutError"

    resilient = ResilientAgent(injector)
    res_r = resilient.process_order_request("ORD-001")
    assert res_r["success"] is True
    assert res_r["recovered"] is True


def test_chaos_experiment_evaluator():
    evaluator = ChaosExperimentEvaluator()
    faults = {"tool_timeout": True, "http_500": True, "invalid_json": False, "context_corruption": False, "tool_unavailable": False}
    results = evaluator.run_experiment(faults, test_cases=["ORD-1", "ORD-2", "ORD-3"])

    assert results["normal_success_rate"] == 100.0
    assert results["baseline_chaos_success_rate"] == 0.0
    assert results["resilient_chaos_success_rate"] == 100.0
    assert results["recovery_rate"] == 100.0
