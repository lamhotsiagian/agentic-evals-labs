"""Evaluator for multi-step agent trajectories."""

from __future__ import annotations
from typing import Any, Dict
from shared.evaluators.trajectory import TrajectoryEvaluator
from shared.models.schemas import AgentTrace, MetricScore


class ITTrajectoryEvaluator:
    """Evaluates IT Helpdesk diagnostic trajectories."""

    def __init__(self):
        self._core_evaluator = TrajectoryEvaluator()

    def evaluate_trace(self, trace: AgentTrace, optimal_steps: int = 4) -> Dict[str, MetricScore]:
        return self._core_evaluator.evaluate(trace, optimal_steps=optimal_steps)
