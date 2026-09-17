"""Planner -> Executor -> Verifier architectural pipeline.

Fixed from an earlier version of this lab: the planner must produce a plan the
model actually generated (parsed and schema-validated, not a template),
the verifier checks every constraint explicitly instead of budget alone,
and a failed verification feeds its violations back into the next
planning attempt instead of repeating the same prompt.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import AgentTrace


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    activities: List[str] = Field(min_length=1)
    cost: float = Field(ge=0)


class TravelPlan(BaseModel):
    destination: str
    days: List[DayPlan]

    @property
    def total_cost(self) -> float:
        return sum(d.cost for d in self.days)


class PlanParseError(ValueError):
    """The planner's response was not a valid TravelPlan -- a planning failure, not a crash."""


def parse_plan(raw: str) -> TravelPlan:
    """Parse the planner model's JSON into a schema-validated TravelPlan."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlanParseError("no JSON object found in planner response")
    try:
        return TravelPlan.model_validate_json(raw[start:end + 1])
    except ValidationError as e:
        raise PlanParseError(str(e)) from e


@dataclass
class Verdict:
    passed: bool
    violations: List[str] = field(default_factory=list)


class ConstraintVerifier:
    """Deterministic verifier: every constraint in the task is checked explicitly.

    Replaces the old VerifierNode, which only checked budget and treated
    a plan with the wrong number of days as "successfully satisfied".
    """

    def verify(self, plan: TravelPlan, c: Dict[str, Any]) -> Verdict:
        v: List[str] = []
        if plan.destination.lower() != str(c.get("destination", "")).lower():
            v.append(f"destination {plan.destination!r} != {c.get('destination')!r}")
        if "days" in c and len(plan.days) != c["days"]:
            v.append(f"plan has {len(plan.days)} days, task requires {c['days']}")
        if sorted(d.day for d in plan.days) != list(range(1, len(plan.days) + 1)):
            v.append("day numbering is not 1..N")
        if "budget" in c and plan.total_cost > c["budget"]:
            v.append(f"total cost {plan.total_cost:.0f} exceeds budget {c['budget']}")
        text = " ".join(a.lower() for d in plan.days for a in d.activities)
        for interest in c.get("interests", []):
            if not any(tok in text for tok in str(interest).lower().split()):
                v.append(f"interest {interest!r} not covered")
        return Verdict(passed=not v, violations=v)


class PlannerNode:
    """Decomposes a user task into a schema-validated day-by-day plan."""

    def __init__(self, provider: LLMProvider, model: str = "qwen2.5:3b"):
        self.provider = provider
        self.model = model

    def plan(self, task: str, constraints: Dict[str, Any], feedback: Optional[List[str]] = None) -> Dict[str, Any]:
        feedback_block = ""
        if feedback:
            feedback_block = (
                "\nYour previous plan violated these constraints -- fix ALL of them:\n"
                + "\n".join(f"- {v}" for v in feedback)
            )
        prompt = f"""Create a day-by-day travel plan for this task:
Task: {task}
Constraints (must all be satisfied exactly): {json.dumps(constraints)}
{feedback_block}

Respond with ONLY a JSON object of this exact shape, no other text:
{{"destination": "<city>", "days": [{{"day": 1, "activities": ["..."], "cost": <number>}}, ...]}}
The number of day objects must equal constraints.days. The sum of every
day's cost must not exceed constraints.budget. Cover every listed interest
in at least one activity's wording.
"""
        response = self.provider.generate(model=self.model, prompt=prompt, temperature=0.1)
        try:
            plan = parse_plan(response)
            return {"status": "success", "plan": plan, "raw_response": response}
        except PlanParseError as e:
            return {"status": "parse_failed", "plan": None, "raw_response": response, "error": str(e)}


class ExecutorNode:
    """Executes the plan's booking steps. actual_cost comes from the parsed plan,
    never from a copy of the planner's own estimate."""

    def __init__(self, provider: LLMProvider, model: str = "qwen2.5:3b"):
        self.provider = provider
        self.model = model

    def execute(self, plan: Optional[TravelPlan], inject_failure: bool = False) -> Dict[str, Any]:
        if inject_failure:
            return {
                "status": "error",
                "error": "Hotel booking tool failed: Exceeded maximum single-day allocation.",
                "executed_steps": [],
                "actual_cost": 0,
            }
        if plan is None:
            return {"status": "error", "error": "no plan to execute", "executed_steps": [], "actual_cost": 0}

        executed = [
            {"step": f"Day {d.day}: {', '.join(d.activities)}", "status": "completed", "action": "book_activity", "cost": d.cost}
            for d in plan.days
        ]
        return {"status": "success", "executed_steps": executed, "actual_cost": plan.total_cost}


class ArchitecturePipeline:
    """Orchestrates Planner -> Executor -> Verifier with tracing and feedback-carrying retries."""

    def __init__(self, provider: Optional[LLMProvider] = None, max_retries: int = 2):
        self.provider = provider or get_model_provider()
        self.planner = PlannerNode(self.provider)
        self.executor = ExecutorNode(self.provider)
        self.verifier = ConstraintVerifier()
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
        feedback: Optional[List[str]] = None

        pipeline_nodes_log: List[Dict[str, Any]] = []
        last_plan: Optional[TravelPlan] = None

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            plan_result = self.planner.plan(task, constraints, feedback=feedback)
            d_plan = round((time.time() - t0) * 1000, 1)
            plan_ok = plan_result["status"] == "success"
            trace.add_step(action="planner", arguments={"attempt": attempt}, observation=plan_result.get("raw_response", "")[:500],
                            result="success" if plan_ok else "error", duration_ms=d_plan)
            pipeline_nodes_log.append({
                "node": "PLANNER", "attempt": attempt,
                "status": "✓ Parsed" if plan_ok else "✗ Parse Failure",
                "details": {"status": plan_result["status"], "plan": plan_result["plan"].model_dump() if plan_ok else None,
                             "error": plan_result.get("error")},
            })

            if not plan_ok:
                retries += 1
                feedback = [f"planner output was not valid JSON: {plan_result.get('error')}"]
                continue

            last_plan = plan_result["plan"]

            should_fail = inject_failure_on_first_try and (attempt == 0)
            t1 = time.time()
            exec_res = self.executor.execute(last_plan, inject_failure=should_fail)
            d_exec = round((time.time() - t1) * 1000, 1)
            step_status = "success" if exec_res["status"] == "success" else "error"
            trace.add_step(action="executor", arguments={"attempt": attempt}, observation=json.dumps(exec_res), result=step_status, duration_ms=d_exec)
            pipeline_nodes_log.append({"node": "EXECUTOR", "attempt": attempt,
                                        "status": "✓ Success" if step_status == "success" else "✗ Error",
                                        "details": exec_res})

            if step_status != "success":
                retries += 1
                feedback = None
                continue

            t2 = time.time()
            verdict = self.verifier.verify(last_plan, constraints)
            d_verif = round((time.time() - t2) * 1000, 1)
            trace.add_step(action="verifier", arguments={"attempt": attempt},
                            observation="; ".join(verdict.violations) or "all constraints satisfied",
                            result="success" if verdict.passed else "warning", duration_ms=d_verif)
            pipeline_nodes_log.append({
                "node": "VERIFIER", "attempt": attempt,
                "status": "✓ Verified" if verdict.passed else "⚠ Rejected",
                "details": {"passed": verdict.passed, "violations": verdict.violations},
            })

            if verdict.passed:
                overall_success = True
                final_answer = f"Trip planned for {last_plan.destination}: {len(last_plan.days)} days, total cost ${last_plan.total_cost:.0f}."
                break
            else:
                retries += 1
                feedback = verdict.violations

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
            "final_plan": last_plan,
            "trace": trace,
        }
