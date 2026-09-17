"""Evaluator for Chapter 2 Agent Architecture.

Fixed from an earlier version of this lab: verification accuracy now measures whether the
verifier's verdict agrees with an independently-computed outcome grader
(not "did it run"), end-to-end success is decided by that independent
grader (not by trusting the pipeline's own self-reported flag), and a
separate mutation-testing harness exists to measure a verifier's true
bad-plan recall against known-good/known-bad plans.
"""

from __future__ import annotations
from itertools import product
from typing import Any, Callable, Dict, List
from shared.models.schemas import MetricScore
from pipeline import TravelPlan, ConstraintVerifier, Verdict


class IndependentOutcomeGrader:
    """Re-derives end-to-end success from ground truth, without trusting the
    pipeline's self-reported `success` flag or the pipeline's own verifier
    instance. Written separately from ConstraintVerifier on purpose: the
    system that decides pass/fail for scoring is not the same object the
    pipeline uses to decide whether to retry."""

    def grade(self, plan: Any, constraints: Dict[str, Any]) -> bool:
        if plan is None:
            return False
        if plan.destination.lower() != str(constraints.get("destination", "")).lower():
            return False
        if "days" in constraints and len(plan.days) != constraints["days"]:
            return False
        if "budget" in constraints and plan.total_cost > constraints["budget"]:
            return False
        text = " ".join(a.lower() for d in plan.days for a in d.activities)
        for interest in constraints.get("interests", []):
            if not any(tok in text for tok in str(interest).lower().split()):
                return False
        return True


class ArchitectureEvaluator:
    """Evaluates component-level and end-to-end performance of Planner -> Executor -> Verifier."""

    def __init__(self):
        self.outcome_grader = IndependentOutcomeGrader()

    def evaluate_run(self, pipeline_output: Dict[str, Any]) -> Dict[str, MetricScore]:
        log = pipeline_output.get("pipeline_log", [])
        retries = pipeline_output.get("retries", 0)
        final_plan = pipeline_output.get("final_plan")
        constraints = pipeline_output.get("constraints", {})

        # 1. Planning accuracy: did the parsed plan actually satisfy the
        # requested day count, not merely "did the planner step run".
        planner_steps = [item for item in log if item["node"] == "PLANNER"]
        planner_successes = sum(
            1 for item in planner_steps
            if item["details"].get("plan") is not None
            and item["details"]["plan"].get("days") is not None
            and ("days" not in constraints or len(item["details"]["plan"]["days"]) == constraints["days"])
        )
        plan_acc = planner_successes / max(1, len(planner_steps))

        # 2. Execution accuracy (unchanged concept, now meaningful because
        # the executor can genuinely fail against a real parsed plan).
        exec_steps = [item for item in log if item["node"] == "EXECUTOR"]
        exec_successes = sum(1 for item in exec_steps if "✓" in item["status"])
        exec_acc = exec_successes / max(1, len(exec_steps))

        # 3. Verification accuracy: does the verifier's verdict AGREE with
        # the independent outcome grader, not "did the verifier run".
        verif_steps = [item for item in log if item["node"] == "VERIFIER"]
        independent_result = self.outcome_grader.grade(final_plan, constraints)
        if verif_steps:
            last_verifier_passed = verif_steps[-1]["details"]["passed"]
            verif_acc = 1.0 if last_verifier_passed == independent_result else 0.0
        else:
            verif_acc = 0.0

        # 4. End-to-End success: graded independently, not read from the
        # pipeline's own `success` flag (the old bug: the system graded
        # itself, so a lenient verifier always looked perfect).
        e2e_score = 1.0 if independent_result else 0.0

        return {
            "planning_accuracy": MetricScore(name="planning_accuracy", score=round(plan_acc, 2), passed=(plan_acc >= 0.8)),
            "execution_accuracy": MetricScore(name="execution_accuracy", score=round(exec_acc, 2), passed=(exec_acc >= 0.7)),
            "verification_accuracy": MetricScore(
                name="verification_accuracy", score=round(verif_acc, 2), passed=(verif_acc >= 0.9),
                reasoning="Agreement between the verifier's verdict and an independent outcome grader, not mere presence.",
            ),
            "end_to_end_success": MetricScore(
                name="end_to_end_success", score=e2e_score, passed=independent_result,
                reasoning="Graded independently of the pipeline's self-reported success flag.",
            ),
            "retries": MetricScore(name="retries", score=float(retries), passed=(retries <= 2),
                                    reasoning=f"Pipeline required {retries} retries."),
        }


# ---- Mutation-testing harness: measures a verifier's TRUE bad-plan recall ----

MUTATIONS: Dict[str, Callable[[TravelPlan, Dict[str, Any]], TravelPlan]] = {
    "over_budget": lambda p, c: p.model_copy(update={"days": [
        d.model_copy(update={"cost": d.cost + c.get("budget", 100)}) if i == 0 else d
        for i, d in enumerate(p.days)]}),
    "missing_day": lambda p, c: p.model_copy(update={"days": p.days[:-1]}) if len(p.days) > 1 else p,
    "wrong_city": lambda p, c: p.model_copy(update={"destination": "Atlantis"}),
    "drop_interest": lambda p, c: p.model_copy(update={"days": [
        d.model_copy(update={"activities": ["hotel rest"]}) for d in p.days]}),
}


def gold_plan(case_metadata: Dict[str, Any]) -> TravelPlan:
    """Build a plan that satisfies every constraint exactly, for use as the
    known-good input to mutation testing."""
    days = case_metadata.get("days", 1)
    budget = case_metadata.get("budget", 100)
    interests = case_metadata.get("interests", [])
    per_day = budget / max(1, days) * 0.8
    activities = interests if interests else ["sightseeing"]
    return TravelPlan(
        destination=case_metadata.get("destination", ""),
        days=[{"day": i + 1, "activities": list(activities), "cost": round(per_day, 2)} for i in range(days)],
    )


def verifier_confusion(verifier, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
    """Treat the verifier as a classifier of BAD plans; estimate recall and false-reject rate."""
    tp = fn = tn = fp = 0
    per_mutation: Dict[str, List[bool]] = {m: [] for m in MUTATIONS}
    for c in tasks:
        good = gold_plan(c)
        if verifier.verify(good, c).passed:
            tn += 1
        else:
            fp += 1
        for name, mutate in MUTATIONS.items():
            mutated = mutate(good, c)
            caught = not verifier.verify(mutated, c).passed
            per_mutation[name].append(caught)
            tp += int(caught)
            fn += int(not caught)
    result = {
        "bad_plan_recall": tp / max(1, (tp + fn)),
        "false_reject_rate": fp / max(1, (fp + tn)),
    }
    result.update({f"recall[{m}]": (sum(x) / len(x) if x else 0.0) for m, x in per_mutation.items()})
    return result
