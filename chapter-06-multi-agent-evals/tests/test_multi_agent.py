"""Unit tests for Chapter 6 Multi-Agent Systems."""

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

from system import MultiAgentResearchSystem
from evaluator import MultiAgentEvaluator


def test_clean_multi_agent_flow():
    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()

    res = system.run_research("Agent Evaluation Frameworks", inject_handoff_failure=False)
    assert res["completed"] is True
    assert len(res["messages"]) >= 4

    scores = evaluator.evaluate_system_run(res)
    assert scores["handoff_success_rate"].score == 1.0
    assert scores["role_adherence"].score == 1.0
    assert scores["duplicate_work"].score == 0.0
    assert scores["final_synthesis"].score == 1.0


def test_multi_agent_with_handoff_failure():
    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()

    res = system.run_research("Agent Evaluation Frameworks", inject_handoff_failure=True)
    assert any(m.handoff_status == "failed" for m in res["messages"])

    scores = evaluator.evaluate_system_run(res)
    assert scores["handoff_success_rate"].metadata["failed"] >= 1
    assert scores["final_synthesis"].score == 1.0
