"""Trajectory evaluation engine for multi-step agent executions."""

from __future__ import annotations
from typing import Any, Dict, List
from shared.models.schemas import AgentTrace, MetricScore


class TrajectoryEvaluator:
    """Evaluates the efficiency, correctness, and recovery of multi-step agent trajectories."""

    def evaluate(self, trace: AgentTrace, optimal_steps: int = 3) -> Dict[str, MetricScore]:
        steps = trace.steps
        total_steps = len(steps)
        if total_steps == 0:
            return {
                "trajectory_score": MetricScore(name="trajectory_score", score=0.0, passed=False, reasoning="Empty trajectory"),
                "step_success_rate": MetricScore(name="step_success_rate", score=0.0, passed=False),
                "loop_penalty": MetricScore(name="loop_penalty", score=0.0, passed=True),
                "recovery_rate": MetricScore(name="recovery_rate", score=1.0, passed=True),
                "efficiency": MetricScore(name="efficiency", score=0.0, passed=False),
            }

        # 1. Step Success Rate
        successful_steps = sum(1 for s in steps if s.result.lower() == "success")
        step_success_rate = successful_steps / total_steps

        # 2. Repeated Action / Loop Detection
        action_signatures: List[str] = [f"{s.action}:{sorted(s.arguments.items())}" for s in steps]
        unique_signatures = set(action_signatures)
        repetition_count = len(action_signatures) - len(unique_signatures)
        loop_penalty = min(1.0, repetition_count * 0.25)

        # 3. Recovery Rate
        error_indices = [i for i, s in enumerate(steps) if s.result.lower() in ("error", "failed", "retry")]
        if not error_indices:
            recovery_rate = 1.0
        else:
            recovered_count = 0
            for err_idx in error_indices:
                # Did any subsequent step succeed or did final output resolve successfully?
                subsequent_success = any(s.result.lower() == "success" for s in steps[err_idx + 1:])
                if subsequent_success or trace.success:
                    recovered_count += 1
            recovery_rate = recovered_count / len(error_indices)

        # 4. Tool / Step Efficiency
        efficiency = min(1.0, optimal_steps / max(optimal_steps, total_steps))

        # 5. Composite Trajectory Score (0 to 100)
        # Weights: Step Success (40%), Efficiency (25%), Recovery (25%), Loop Penalty (-15%)
        raw_score = (
            (step_success_rate * 40.0)
            + (efficiency * 25.0)
            + (recovery_rate * 25.0)
            - (loop_penalty * 15.0)
            + (10.0 if trace.success else 0.0)
        )
        trajectory_score = max(0.0, min(100.0, raw_score))

        return {
            "trajectory_score": MetricScore(
                name="trajectory_score",
                score=round(trajectory_score, 1),
                passed=(trajectory_score >= 70.0),
                reasoning=f"Computed from {total_steps} steps with {successful_steps} successes and {repetition_count} repeats.",
            ),
            "step_success_rate": MetricScore(
                name="step_success_rate",
                score=round(step_success_rate, 2),
                passed=(step_success_rate >= 0.75),
            ),
            "recovery_rate": MetricScore(
                name="recovery_rate",
                score=round(recovery_rate, 2),
                passed=(recovery_rate >= 0.5),
            ),
            "tool_efficiency": MetricScore(
                name="tool_efficiency",
                score=round(efficiency, 2),
                passed=(efficiency >= 0.6),
            ),
            "loop_penalty": MetricScore(
                name="loop_penalty",
                score=round(loop_penalty, 2),
                passed=(loop_penalty < 0.3),
            ),
        }
