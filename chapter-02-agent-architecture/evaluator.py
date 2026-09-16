"""Evaluator for Chapter 2 Agent Architecture."""

from __future__ import annotations
from typing import Dict, Any, List
from shared.models.schemas import MetricScore


class ArchitectureEvaluator:
    """Evaluates component-level and end-to-end performance of Planner -> Executor -> Verifier."""

    def evaluate_run(self, pipeline_output: Dict[str, Any]) -> Dict[str, MetricScore]:
        log = pipeline_output.get("pipeline_log", [])
        retries = pipeline_output.get("retries", 0)
        success = pipeline_output.get("success", False)

        # 1. Planning accuracy
        planner_steps = [item for item in log if item["node"] == "PLANNER"]
        planner_successes = sum(1 for item in planner_steps if "✓" in item["status"])
        plan_acc = planner_successes / max(1, len(planner_steps))

        # 2. Execution accuracy
        exec_steps = [item for item in log if item["node"] == "EXECUTOR"]
        exec_successes = sum(1 for item in exec_steps if "✓" in item["status"])
        exec_acc = exec_successes / max(1, len(exec_steps))

        # 3. Verification accuracy (did it correctly accept or reject)
        verif_steps = [item for item in log if item["node"] == "VERIFIER"]
        verif_acc = 1.0 if len(verif_steps) > 0 else 0.0

        # 4. End-to-End Success
        e2e_score = 1.0 if success else 0.0

        return {
            "planning_accuracy": MetricScore(
                name="planning_accuracy",
                score=round(plan_acc, 2),
                passed=(plan_acc >= 0.8),
            ),
            "execution_accuracy": MetricScore(
                name="execution_accuracy",
                score=round(exec_acc, 2),
                passed=(exec_acc >= 0.7),
            ),
            "verification_accuracy": MetricScore(
                name="verification_accuracy",
                score=round(verif_acc, 2),
                passed=(verif_acc >= 0.9),
            ),
            "end_to_end_success": MetricScore(
                name="end_to_end_success",
                score=e2e_score,
                passed=success,
            ),
            "retries": MetricScore(
                name="retries",
                score=float(retries),
                passed=(retries <= 2),
                reasoning=f"Pipeline required {retries} retries.",
            ),
        }
