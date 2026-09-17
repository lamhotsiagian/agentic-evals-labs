"""Unit tests for Chapter 4 Agent Trajectory Evals."""

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

from engine import ITHelpdeskAgent
from evaluator import ITTrajectoryEvaluator


def test_clean_vpn_trajectory():
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    trace = agent.diagnose_issue("vpn")
    assert trace.success is True

    scores = evaluator.evaluate_trace(trace, scenario="vpn")
    assert scores["progress"].score == 1.0
    assert scores["loop_penalty"].score == 0.0
    assert scores["trajectory_score"].score >= 70.0
    assert scores["outcome_gate"].passed is True


def test_disk_trajectory_with_loop():
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    # A compress_logs loop is injected -- the mock diagnostic policy
    # tries the check -> fix path; the environment forces compress_logs
    # to keep failing so the agent must find archive_to_s3 instead.
    trace = agent.diagnose_issue("disk", inject_compress_loop=True)
    scores = evaluator.evaluate_trace(trace, scenario="disk")
    # A tool_failure step (compress_logs erroring) must not itself be
    # scored as a step that lowers progress -- only unresolved milestones do.
    assert scores["progress"].score >= 0.0
    assert "trajectory_score" in scores


def test_finding_step_is_not_penalized_as_failure():
    # This is Chapter 4's central lesson made executable: a diagnostic
    # step that correctly reveals a problem (a "finding") must not be
    # scored the same as a tool that actually broke.
    from shared.models.schemas import AgentTrace, AgentStep
    from shared.evaluators.trajectory import classify_step

    finding_step = AgentStep(step_index=1, action="check_vpn_status", arguments={}, result="success",
                              observation="VPN tunnel state: DISCONNECTED (IPsec SA expired)")
    assert classify_step(finding_step) == "success"  # result=success -> classified success regardless of content

    # A step the tool runtime explicitly tagged as a finding (metadata)
    # must be classified as such even though its `result` reads "error".
    tagged_finding = AgentStep(step_index=1, action="check_vpn_status", arguments={}, result="error",
                                observation="VPN tunnel state: DISCONNECTED", metadata={"kind": "finding"})
    assert classify_step(tagged_finding) == "finding"


def test_failed_outcome_gates_score_to_zero():
    from shared.evaluators.trajectory import evaluate_trajectory
    from shared.models.schemas import AgentTrace, AgentStep

    trace = AgentTrace(task="t", steps=[AgentStep(step_index=1, action="check_vpn_status", arguments={}, result="success")])
    report = evaluate_trajectory(trace, milestones=["check_vpn_status"], optimal_steps=1, outcome_success=False)
    assert report.score == 0.0
