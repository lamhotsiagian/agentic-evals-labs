"""Evaluator for tool-calling accuracy and failure recovery.

Fixed from an earlier version of this lab: name-level-only grading treats a hallucinated
zip code or a wrong recipient email the same as a correct one and never
penalizes an extra unexpected call. This version validates every call
against its schema, scores selection with multiset precision/recall/F1
(so extra calls hurt precision), checks ordering dependencies, and adds
a groundedness check -- did each argument value actually come from the
user's request or a prior tool observation, or did the model invent it.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from shared.models.schemas import MetricScore
from graders import validate_call, call_prf, order_violations, ungrounded_args, Call


class ToolCallingEvaluator:
    """Evaluates tool selection, parameter validity, sequencing, groundedness, and recovery."""

    DEFAULT_MUST_PRECEDE = [("get_order", "calculate_refund")]

    def evaluate_execution(
        self,
        expected_tools: Optional[List[str]],
        agent_result: Dict[str, Any],
        expect_error_recovery: bool = False,
    ) -> Dict[str, MetricScore]:
        tool_calls = agent_result.get("tool_calls", [])
        call_objs = [Call.of(tc["tool"], **tc.get("args", {})) for tc in tool_calls]
        recovered = agent_result.get("recovered", False)
        prompt = agent_result.get("prompt", "")

        # 1. Tool selection: multiset name-level P/R/F1 against ground truth.
        # Extra or missing calls now change the score; the old version only
        # checked that every expected name appeared somewhere, so an
        # unexpected send_email cost nothing.
        if expected_tools:
            expected_calls = [Call(t, ()) for t in expected_tools]
            prf = call_prf(expected_calls, call_objs, match_args=False)
            selection_score = prf["f1"]
        else:
            selection_score = 1.0 if all(not validate_call(tc["tool"], tc.get("args", {})) for tc in tool_calls) else 0.5

        # 2. Argument accuracy: schema validation against TOOL_SCHEMAS, not
        # "is the value non-empty". A hallucinated but non-empty zip code
        # or email now fails this check if it does not match the schema.
        per_call_errors = [validate_call(tc["tool"], tc.get("args", {})) for tc in tool_calls]
        total_errors = sum(len(e) for e in per_call_errors)
        arg_accuracy = 1.0 if not tool_calls else round(max(0.0, 1.0 - total_errors / len(tool_calls)), 2)

        # 3. Tool order: real dependency check (get_order before calculate_refund).
        violations = order_violations(call_objs, self.DEFAULT_MUST_PRECEDE)
        order_score = 1.0 if not violations else 0.0

        # 4. Result handling
        final_answer = agent_result.get("final_answer", "")
        has_final_answer = bool(final_answer) and final_answer != "STEP_LIMIT_EXCEEDED"

        # 5. Recovery
        recovery_score = (1.0 if recovered else 0.0) if expect_error_recovery else 1.0

        # 6. Groundedness (new): every argument value must be traceable to
        # the user's prompt or a PRIOR tool observation. This is what
        # catches an invented zip code or a fabricated email address that
        # the old grader scored as 100% accurate.
        observations = [str(tc.get("result", "")) for tc in tool_calls]
        ungrounded = ungrounded_args(call_objs, prompt, observations)
        total_args = sum(len(c.args) for c in call_objs) or 1
        groundedness_score = round(max(0.0, 1.0 - len(ungrounded) / total_args), 2)

        return {
            "tool_selection": MetricScore(name="tool_selection", score=round(selection_score, 2), passed=(selection_score >= 0.8)),
            "argument_accuracy": MetricScore(name="argument_accuracy", score=arg_accuracy, passed=(arg_accuracy >= 0.8)),
            "tool_order": MetricScore(name="tool_order", score=order_score, passed=(order_score == 1.0),
                                       reasoning="; ".join(violations) or None),
            "result_handling": MetricScore(name="result_handling", score=1.0 if has_final_answer else 0.0, passed=has_final_answer),
            "agent_recovery": MetricScore(name="agent_recovery", score=recovery_score, passed=(recovery_score == 1.0)),
            "groundedness": MetricScore(name="groundedness", score=groundedness_score, passed=(groundedness_score >= 0.9),
                                         reasoning=("ungrounded: " + ", ".join(ungrounded)) if ungrounded else None),
        }
