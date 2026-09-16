"""Evaluator for tool-calling accuracy and failure recovery."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from shared.models.schemas import MetricScore


class ToolCallingEvaluator:
    """Evaluates tool selection, parameter validity, sequencing, and failure recovery."""

    def evaluate_execution(
        self,
        expected_tools: Optional[List[str]],
        agent_result: Dict[str, Any],
        expect_error_recovery: bool = False,
    ) -> Dict[str, MetricScore]:
        tool_calls = agent_result.get("tool_calls", [])
        actual_tool_names = [tc["tool"] for tc in tool_calls]
        recovered = agent_result.get("recovered", False)

        # 1. Tool Selection Accuracy
        if not expected_tools:
            selection_score = 1.0
        else:
            matches = sum(1 for et in expected_tools if et in actual_tool_names)
            selection_score = matches / len(expected_tools)

        # 2. Argument Accuracy
        valid_args = 0
        total_calls = max(1, len(tool_calls))
        for tc in tool_calls:
            args = tc.get("args", {})
            # Check for non-empty, non-invalid values
            if all(v and str(v).lower() != "invalid_id" for v in args.values()):
                valid_args += 1
        arg_accuracy = valid_args / total_calls

        # 3. Tool Order Correctness
        if expected_tools and len(expected_tools) > 1:
            order_correct = True
            last_idx = -1
            for et in expected_tools:
                if et in actual_tool_names:
                    curr_idx = actual_tool_names.index(et)
                    if curr_idx < last_idx:
                        order_correct = False
                        break
                    last_idx = curr_idx
            order_score = 1.0 if order_correct else 0.0
        else:
            order_score = 1.0

        # 4. Tool Result Handling / Final Answer
        has_final_answer = bool(agent_result.get("final_answer"))
        result_handling_score = 1.0 if has_final_answer else 0.0

        # 5. Recovery Score
        if expect_error_recovery:
            recovery_score = 1.0 if recovered else 0.0
        else:
            recovery_score = 1.0

        return {
            "tool_selection": MetricScore(
                name="tool_selection",
                score=round(selection_score, 2),
                passed=(selection_score >= 0.8),
            ),
            "argument_accuracy": MetricScore(
                name="argument_accuracy",
                score=round(arg_accuracy, 2),
                passed=(arg_accuracy >= 0.8 if not expect_error_recovery else True),
            ),
            "tool_order": MetricScore(
                name="tool_order",
                score=order_score,
                passed=(order_score == 1.0),
            ),
            "result_handling": MetricScore(
                name="result_handling",
                score=result_handling_score,
                passed=has_final_answer,
            ),
            "agent_recovery": MetricScore(
                name="agent_recovery",
                score=recovery_score,
                passed=(recovery_score == 1.0),
            ),
        }
