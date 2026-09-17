"""Unit tests for Chapter 6 Multi-Agent Systems."""

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

from system import MultiAgentResearchSystem, AgentMessage
from evaluator import MultiAgentEvaluator


def test_clean_multi_agent_flow():
    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()

    res = system.run_research("Agent Evaluation Frameworks", inject_handoff_failure=False)
    assert res["completed"] is True
    assert len(res["messages"]) >= 4

    scores = evaluator.evaluate_system_run(res)
    assert scores["handoff_success_rate"].score == 1.0
    assert scores["first_attempt_handoff_rate"].score == 1.0
    assert scores["role_adherence"].score == 1.0  # mock Analyst sources its number
    assert scores["final_synthesis"].score == 1.0  # no unsourced claim propagates


def test_multi_agent_with_handoff_failure_distinguishes_first_try_from_eventual():
    # This is Chapter 6's central handoff-rate lesson made executable: a
    # retransmission must count against first-attempt success even though
    # the message eventually arrived.
    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()

    res = system.run_research("Agent Evaluation Frameworks", inject_handoff_failure=True)
    assert any(m.handoff_status == "failed" for m in res["messages"])

    scores = evaluator.evaluate_system_run(res)
    assert scores["handoff_success_rate"].score == 1.0  # every edge eventually completed
    assert scores["first_attempt_handoff_rate"].score < 1.0  # but not on the first try
    assert "retransmissions=2" in scores["handoff_success_rate"].reasoning or "retransmissions" in scores["handoff_success_rate"].reasoning


def test_unsourced_claim_propagating_into_synthesis_fails_the_grader():
    # This is Chapter 6's central claim-provenance lesson: an unsourced
    # number from a worker that reaches the executive synthesis unflagged
    # must fail final_synthesis, even though the synthesis is non-empty.
    messages = [
        AgentMessage(message_id=1, sender="Supervisor", recipient="Researcher", content="Investigate."),
        AgentMessage(message_id=2, sender="Supervisor", recipient="Analyst", content="Analyze."),
        AgentMessage(message_id=3, sender="Researcher", recipient="Synthesizer", content="Background context, no numbers here."),
        AgentMessage(message_id=4, sender="Analyst", recipient="Synthesizer",
                     content="Empirical analysis highlights a 34% drop in regression errors."),  # no source tag
        AgentMessage(message_id=5, sender="Synthesizer", recipient="Supervisor",
                     content="Executive Synthesis:\n- Quantitative: a 34% drop in regression errors was observed."),
    ]
    run_result = {
        "messages": messages,
        "researcher_content": messages[2].content,
        "analyst_content": messages[3].content,
        "final_synthesis": messages[4].content,
    }
    scores = MultiAgentEvaluator().evaluate_system_run(run_result)
    assert scores["final_synthesis"].passed is False
    assert "34%" in scores["final_synthesis"].reasoning
