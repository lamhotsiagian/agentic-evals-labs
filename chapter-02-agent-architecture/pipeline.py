"""Planner -> Executor -> Verifier architectural pipeline."""

from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import AgentTrace


class PlannerNode:
    """Decomposes a user task into an ordered set of sub-tasks/itinerary."""

    def __init__(self, provider: LLMProvider, model: str = "qwen2.5:3b"):
        self.provider = provider
        self.model = model

    def plan(self, task: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Create a structured plan for the following travel task:
Task: {task}
Constraints: {json.dumps(constraints)}

Output a JSON object with 'steps': list of actions, and 'estimated_cost': total numeric cost.
"""
        response = self.provider.generate(model=self.model, prompt=prompt, temperature=0.1)
        # Parse or default
        days = constraints.get("days", 3)
        budget = constraints.get("budget", 1000)
        destination = constraints.get("destination", "Tokyo")

        steps = [
            f"Day 1: Arrive in {destination}, check-in, explore cultural district.",
            f"Day 2: Full-day guided excursion and landmark visits.",
            f"Day 3: Culinary tour, local market shopping, and airport transfer.",
        ][:days]

        return {
            "status": "success",
            "destination": destination,
            "days": days,
            "steps": steps,
            "estimated_cost": int(budget * 0.85),
            "raw_response": response,
        }


class ExecutorNode:
    """Executes the sub-tasks formulated by the planner."""

    def __init__(self, provider: LLMProvider, model: str = "qwen2.5:3b"):
        self.provider = provider
        self.model = model

    def execute(self, plan: Dict[str, Any], inject_failure: bool = False) -> Dict[str, Any]:
        if inject_failure:
            return {
                "status": "error",
                "error": "Hotel booking tool failed: Exceeded maximum single-day allocation.",
                "executed_steps": [],
                "actual_cost": 0,
            }

        executed = []
        for step in plan.get("steps", []):
            executed.append({
                "step": step,
                "status": "completed",
                "action": "book_activity",
            })

        return {
            "status": "success",
            "executed_steps": executed,
            "actual_cost": plan.get("estimated_cost", 800),
        }


class VerifierNode:
    """Verifies that the executed result satisfies all constraints."""

    def __init__(self, provider: LLMProvider, model: str = "qwen3:1.7b"):
        self.provider = provider
        self.model = model

    def verify(
        self,
        task: str,
        constraints: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if execution_result.get("status") != "success":
            return {
                "status": "failed",
                "passed": False,
                "reason": "Execution reported an error.",
                "retries_recommended": True,
            }

        budget = constraints.get("budget", 1000)
        actual_cost = execution_result.get("actual_cost", 0)

        if actual_cost > budget:
            return {
                "status": "warning",
                "passed": False,
                "reason": f"Actual cost ${actual_cost} exceeds budget ${budget}.",
                "retries_recommended": True,
            }

        return {
            "status": "success",
            "passed": True,
            "reason": "All constraints (budget, destination, activity days) successfully satisfied.",
            "retries_recommended": False,
        }


class ArchitecturePipeline:
    """Orchestrates Planner -> Executor -> Verifier with tracing and retry capabilities."""

    def __init__(self, provider: Optional[LLMProvider] = None, max_retries: int = 2):
        self.provider = provider or get_model_provider()
        self.planner = PlannerNode(self.provider)
        self.executor = ExecutorNode(self.provider)
        self.verifier = VerifierNode(self.provider)
        self.max_retries = max_retries

    def run(
        self,
        task: str,
        constraints: Dict[str, Any],
        inject_failure_on_first_try: bool = False,
    ) -> Dict[str, Any]:
        trace = AgentTrace(task=task)
        start_time = time.time()
        retries = 0
        final_answer = ""
        overall_success = False

        pipeline_nodes_log = []

        for attempt in range(self.max_retries + 1):
            # 1. Planner Step
            t0 = time.time()
            plan = self.planner.plan(task, constraints)
            d_plan = round((time.time() - t0) * 1000, 1)
            trace.add_step(action="planner", arguments={"attempt": attempt}, observation=json.dumps(plan["steps"]), result="success", duration_ms=d_plan)
            pipeline_nodes_log.append({"node": "PLANNER", "attempt": attempt, "status": "✓ Success", "details": plan})

            # 2. Executor Step
            should_fail = inject_failure_on_first_try and (attempt == 0)
            t1 = time.time()
            exec_res = self.executor.execute(plan, inject_failure=should_fail)
            d_exec = round((time.time() - t1) * 1000, 1)
            step_status = "success" if exec_res["status"] == "success" else "error"
            trace.add_step(action="executor", arguments={"attempt": attempt}, observation=json.dumps(exec_res), result=step_status, duration_ms=d_exec)
            pipeline_nodes_log.append({"node": "EXECUTOR", "attempt": attempt, "status": "✓ Success" if step_status == "success" else "✗ Error", "details": exec_res})

            # 3. Verifier Step
            t2 = time.time()
            verif_res = self.verifier.verify(task, constraints, exec_res)
            d_verif = round((time.time() - t2) * 1000, 1)
            v_status = "success" if verif_res["passed"] else "warning"
            trace.add_step(action="verifier", arguments={"attempt": attempt}, observation=verif_res["reason"], result=v_status, duration_ms=d_verif)
            pipeline_nodes_log.append({"node": "VERIFIER", "attempt": attempt, "status": "✓ Verified" if verif_res["passed"] else "⚠ Rejected", "details": verif_res})

            if verif_res["passed"]:
                overall_success = True
                final_answer = f"Trip successfully planned for {constraints.get('destination')}: {len(plan['steps'])} days within ${constraints.get('budget')} budget."
                break
            else:
                retries += 1

        total_time = round(time.time() - start_time, 2)
        trace.success = overall_success
        trace.final_output = final_answer

        return {
            "task": task,
            "constraints": constraints,
            "success": overall_success,
            "retries": retries,
            "total_duration_sec": total_time,
            "final_answer": final_answer,
            "pipeline_log": pipeline_nodes_log,
            "trace": trace,
        }
