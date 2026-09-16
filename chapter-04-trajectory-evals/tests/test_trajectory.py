"""Unit tests for Chapter 4 Agent Trajectory Evals."""

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

from engine import ITHelpdeskAgent
from evaluator import ITTrajectoryEvaluator


def test_clean_trajectory():
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    trace = agent.diagnose_issue("Diagnose access issue", inject_failure_at_step=0)
    assert trace.success is True
    assert len(trace.steps) == 4

    scores = evaluator.evaluate_trace(trace, optimal_steps=4)
    assert scores["step_success_rate"].score == 1.0
    assert scores["recovery_rate"].score == 1.0
    assert scores["loop_penalty"].score == 0.0
    assert scores["trajectory_score"].score >= 90.0


def test_recovered_trajectory():
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    trace = agent.diagnose_issue("Diagnose access issue", inject_failure_at_step=3, loop_behavior=False)
    assert trace.success is True
    assert any(s.result == "error" for s in trace.steps)

    scores = evaluator.evaluate_trace(trace, optimal_steps=4)
    assert scores["recovery_rate"].score == 1.0
    assert scores["trajectory_score"].score >= 70.0


def test_loop_penalty_detection():
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    trace = agent.diagnose_issue("Diagnose access issue", inject_failure_at_step=3, loop_behavior=True)
    scores = evaluator.evaluate_trace(trace, optimal_steps=4)
    assert scores["loop_penalty"].score > 0.0
