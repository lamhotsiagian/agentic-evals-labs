"""Trajectory evaluation engine for multi-step agent executions.

Fixed from an earlier version of this lab: recovery is milestone-based with
cause linkage (not "any later success"), diagnostic findings are
classified separately from tool failures (a probe that reveals a real
problem no longer lowers the score), loop detection only flags a repeat
when the OBSERVATION is unchanged (legitimate polling is not penalized),
and outcome success gates the score instead of adding a flat bonus to it.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
from shared.models.schemas import AgentTrace, AgentStep, MetricScore


def classify_step(step: AgentStep) -> str:
    """Separate tool failures from informative findings.

    Prefer an explicit metadata['kind'] recorded by the tool runtime; the
    heuristic below is only a fallback for legacy traces that predate it.
    """
    if step.metadata.get("kind"):
        return step.metadata["kind"]
    if step.result == "success":
        return "success"
    obs = (step.observation or "").lower()
    if step.action.startswith(("ping", "check", "query", "test", "verify")) and "error:" not in obs:
        return "finding"  # the probe worked; the world is unhealthy
    return "tool_failure"


def wasted_repeats(steps: Sequence[AgentStep]) -> List[int]:
    """Indices of steps that repeat a prior (action, args) AND observed nothing new.

    Legitimate polling (same action, changing observation) is never
    flagged -- only a repeat that learned nothing new is waste.
    """
    seen: Dict[str, set] = {}
    wasted = []
    for i, s in enumerate(steps):
        sig = f"{s.action}:{json.dumps(s.arguments, sort_keys=True)}"
        obs = (s.observation or "").strip()
        if sig in seen and obs in seen[sig]:
            wasted.append(i)
        seen.setdefault(sig, set()).add(obs)
    return wasted


@dataclass
class TrajectoryReport:
    outcome_success: bool
    progress: float
    failures: int
    recovered: int
    wasted: int
    efficiency: float
    score: float


def evaluate_trajectory(
    trace: AgentTrace,
    milestones: List[str],
    optimal_steps: int,
    outcome_success: bool,
) -> TrajectoryReport:
    """Outcome is graded independently (by the caller); process metrics
    never award outcome credit -- a failed outcome cannot score above 0."""
    steps = trace.steps
    kinds = [classify_step(s) for s in steps]

    def reached(window_steps, window_kinds):
        return {m for m in milestones
                if any(s.action == m and k == "success" for s, k in zip(window_steps, window_kinds))}

    progress = len(reached(steps, kinds)) / len(milestones) if milestones else (1.0 if outcome_success else 0.0)
    failures = [i for i, k in enumerate(kinds) if k == "tool_failure"]
    # A failure is recovered only if a LATER step reaches a milestone not reached before it --
    # not merely "some later step succeeded at anything".
    recovered = sum(
        bool(reached(steps[i + 1:], kinds[i + 1:]) - reached(steps[:i], kinds[:i]))
        for i in failures
    )
    wasted = len(wasted_repeats(steps))
    efficiency = min(1.0, optimal_steps / max(1, len(steps))) if steps else 0.0
    score = 0.0 if not outcome_success else round(
        100 * (0.5 * progress + 0.3 * efficiency + 0.2 * (1 - min(1, wasted / 2))), 1
    )
    return TrajectoryReport(outcome_success, round(progress, 2), len(failures), recovered, wasted, round(efficiency, 2), score)


class TrajectoryEvaluator:
    """Evaluates the efficiency, correctness, and recovery of multi-step agent trajectories."""

    def evaluate(
        self,
        trace: AgentTrace,
        optimal_steps: int = 3,
        milestones: List[str] = None,
        outcome_success: bool = None,
    ) -> Dict[str, MetricScore]:
        # outcome_success defaults to trace.success but SHOULD be passed
        # explicitly from an independent outcome grader when one exists
        # (see Chapter 2's IndependentOutcomeGrader for the same pattern).
        if outcome_success is None:
            outcome_success = trace.success
        report = evaluate_trajectory(trace, milestones or [], optimal_steps, outcome_success)

        kinds = [classify_step(s) for s in trace.steps]
        findings = sum(1 for k in kinds if k == "finding")
        tool_failures = sum(1 for k in kinds if k == "tool_failure")

        return {
            "trajectory_score": MetricScore(
                name="trajectory_score", score=report.score, passed=(report.score >= 70.0),
                reasoning=f"{len(trace.steps)} steps: {findings} findings, {tool_failures} tool failures, "
                          f"{report.wasted} wasted repeats, outcome={'success' if outcome_success else 'failed'}.",
            ),
            "progress": MetricScore(name="progress", score=report.progress, passed=(report.progress >= 0.8),
                                     reasoning="Milestone-based: a diagnostic finding is not a failed step."),
            "recovery_rate": MetricScore(
                name="recovery_rate",
                score=round(report.recovered / report.failures, 2) if report.failures else 1.0,
                passed=(report.failures == 0 or report.recovered == report.failures),
                reasoning="Counts only recoveries where a LATER step reached a new milestone, not any later success.",
            ),
            "tool_efficiency": MetricScore(name="tool_efficiency", score=report.efficiency, passed=(report.efficiency >= 0.6)),
            "loop_penalty": MetricScore(
                name="loop_penalty", score=round(min(1.0, report.wasted * 0.25), 2), passed=(report.wasted == 0),
                reasoning="Only repeats with an UNCHANGED observation count; polling with a changing observation is free.",
            ),
            "outcome_gate": MetricScore(
                name="outcome_gate", score=1.0 if outcome_success else 0.0, passed=outcome_success,
                reasoning="A failed outcome gates trajectory_score to 0 -- it can no longer be offset by a bonus.",
            ),
        }
