"""Evaluator for multi-step agent trajectories (Chapter 4)."""

from __future__ import annotations
from typing import Dict, List, Optional
from shared.evaluators.trajectory import TrajectoryEvaluator
from shared.models.schemas import AgentTrace, MetricScore
from tools import SCENARIOS


class ITTrajectoryEvaluator:
    """Evaluates IT Helpdesk diagnostic trajectories."""

    def __init__(self):
        self._core_evaluator = TrajectoryEvaluator()

    def evaluate_trace(
        self,
        trace: AgentTrace,
        optimal_steps: int = 4,
        milestones: Optional[List[str]] = None,
        outcome_success: Optional[bool] = None,
        scenario: Optional[str] = None,
    ) -> Dict[str, MetricScore]:
        if scenario and scenario in SCENARIOS:
            milestones = milestones if milestones is not None else SCENARIOS[scenario]["milestones"]
            optimal_steps = SCENARIOS[scenario]["optimal_steps"]
        return self._core_evaluator.evaluate(
            trace, optimal_steps=optimal_steps, milestones=milestones, outcome_success=outcome_success,
        )
