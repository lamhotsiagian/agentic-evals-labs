"""Evaluator for multi-agent collaboration, delegation, and handoffs.

Fixed from an earlier version of this lab: handoff success is now scored on logical
edges (a retransmission no longer inflates the count), role adherence
checks CONTENT (does the Analyst source its numbers, does the
Synthesizer avoid inventing new ones), duplicate work uses word-overlap
similarity instead of requiring byte-identical strings, and the
synthesis is graded on claim coverage/faithfulness -- including whether
an unsourced worker claim propagated into it undetected.
"""

from __future__ import annotations
from typing import Any, Dict, List
from shared.models.schemas import MetricScore
from graders import handoff_report, unsupported_quantitative_claims, propagated_claims, jaccard, new_numbers_in_synthesis

DUPLICATE_THRESHOLD = 0.6


class MultiAgentEvaluator:
    """Evaluates communication efficiency, handoff reliability, and conflict in multi-agent workflows."""

    def evaluate_system_run(self, run_result: Dict[str, Any]) -> Dict[str, MetricScore]:
        messages = run_result.get("messages", [])
        researcher_text = run_result.get("researcher_content", "")
        analyst_text = run_result.get("analyst_content", "")
        final_synthesis = run_result.get("final_synthesis", "")

        if not messages:
            return {"handoff_success_rate": MetricScore(name="handoff_success_rate", score=0.0, passed=False)}

        # 1. Handoff success: logical edges, first-attempt rate, retransmission cost.
        report = handoff_report(messages)

        # 2. Role adherence on CONTENT, not just "sender is a known role name".
        analyst_msgs = [m for m in messages if m.sender == "Analyst"]
        unsourced = [c for c in unsupported_quantitative_claims(analyst_msgs)]
        analyst_sources_numbers = len(unsourced) == 0
        synth_invented = new_numbers_in_synthesis(researcher_text, analyst_text, final_synthesis)
        synthesizer_no_new_numbers = len(synth_invented) == 0
        role_score = round((int(analyst_sources_numbers) + int(synthesizer_no_new_numbers)) / 2, 2)

        # 3. Duplicate work: word-overlap similarity, not byte-identical strings.
        worker_pairs = [(researcher_text, analyst_text)] if researcher_text and analyst_text else []
        max_overlap = max((jaccard(a, b) for a, b in worker_pairs), default=0.0)
        has_duplicate_work = max_overlap >= DUPLICATE_THRESHOLD

        # 4. Synthesis: claim provenance -- did an unsourced worker claim
        # reach the final report undetected.
        propagated = propagated_claims(messages, final_synthesis)
        has_synthesis = bool(final_synthesis)
        synthesis_score = 1.0 if (has_synthesis and not propagated) else (0.5 if has_synthesis else 0.0)

        return {
            "handoff_success_rate": MetricScore(
                name="handoff_success_rate",
                score=round(report["handoffs_completed"] / report["handoffs_expected"], 2),
                passed=(report["handoffs_completed"] == report["handoffs_expected"]),
                reasoning=f"missing={report['missing']}, retransmissions={report['retransmissions']}",
            ),
            "first_attempt_handoff_rate": MetricScore(
                name="first_attempt_handoff_rate", score=report["first_attempt_success_rate"],
                passed=(report["first_attempt_success_rate"] >= 0.8),
                reasoning="Distinguishes 'eventually delivered' from 'worked the first time'.",
            ),
            "role_adherence": MetricScore(
                name="role_adherence", score=role_score, passed=(role_score == 1.0),
                reasoning=("Analyst sourced all numbers: %s; Synthesizer introduced no new numbers: %s"
                           % (analyst_sources_numbers, synthesizer_no_new_numbers)),
            ),
            "duplicate_work": MetricScore(
                name="duplicate_work", score=round(max_overlap, 2), passed=(not has_duplicate_work),
                reasoning=f"Jaccard word overlap between worker outputs (threshold {DUPLICATE_THRESHOLD}).",
            ),
            "final_synthesis": MetricScore(
                name="final_synthesis", score=synthesis_score, passed=(synthesis_score == 1.0),
                reasoning=(f"Unsourced claim(s) propagated into synthesis: {propagated}" if propagated else "No unsourced claims propagated."),
            ),
            "total_messages": MetricScore(name="total_messages", score=float(len(messages)), passed=True),
        }
