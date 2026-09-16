"""Evaluator for multi-agent collaboration, delegation, and handoffs."""

from __future__ import annotations
from typing import Any, Dict, List
from shared.models.schemas import MetricScore


class MultiAgentEvaluator:
    """Evaluates communication efficiency, handoff reliability, and conflict in multi-agent workflows."""

    def evaluate_system_run(self, run_result: Dict[str, Any]) -> Dict[str, MetricScore]:
        messages = run_result.get("messages", [])
        total_msgs = len(messages)
        if total_msgs == 0:
            return {
                "handoff_success_rate": MetricScore(name="handoff_success_rate", score=0.0, passed=False),
                "role_adherence": MetricScore(name="role_adherence", score=0.0, passed=False),
                "duplicate_work": MetricScore(name="duplicate_work", score=0.0, passed=True),
                "final_synthesis": MetricScore(name="final_synthesis", score=0.0, passed=False),
            }

        # 1. Handoff success
        successful_handoffs = sum(1 for m in messages if m.handoff_status == "success")
        failed_handoffs = sum(1 for m in messages if m.handoff_status == "failed")
        handoff_rate = successful_handoffs / total_msgs

        # 2. Role adherence (check if each agent stays in role)
        roles = {"Supervisor", "Researcher", "Analyst", "Synthesizer"}
        active_senders = {m.sender for m in messages}
        role_score = 1.0 if active_senders.issubset(roles) else 0.8

        # 3. Duplicate work (check if researcher and analyst sent identical text)
        contents = [m.content for m in messages]
        has_duplicates = len(contents) != len(set(contents))
        dup_score = 0.0 if not has_duplicates else 1.0

        # 4. Final synthesis quality
        has_synthesis = bool(run_result.get("final_synthesis"))
        synthesis_score = 1.0 if has_synthesis else 0.0

        return {
            "handoff_success_rate": MetricScore(
                name="handoff_success_rate",
                score=round(handoff_rate, 2),
                passed=(handoff_rate >= 0.8),
                metadata={"successful": successful_handoffs, "failed": failed_handoffs},
            ),
            "role_adherence": MetricScore(
                name="role_adherence",
                score=role_score,
                passed=(role_score == 1.0),
            ),
            "duplicate_work": MetricScore(
                name="duplicate_work",
                score=dup_score,
                passed=(dup_score == 0.0),
            ),
            "final_synthesis": MetricScore(
                name="final_synthesis",
                score=synthesis_score,
                passed=has_synthesis,
            ),
            "total_messages": MetricScore(
                name="total_messages",
                score=float(total_msgs),
                passed=True,
            ),
        }
