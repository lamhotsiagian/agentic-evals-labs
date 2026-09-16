"""Base evaluator contract."""

from __future__ import annotations
import abc
from typing import List
from shared.models.schemas import EvaluationCase, EvaluationResult


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluation engines."""

    @abc.abstractmethod
    def evaluate_case(self, case: EvaluationCase, actual_output: str, **kwargs) -> EvaluationResult:
        """Evaluate a single test case."""
        pass

    def evaluate_dataset(
        self, cases: List[EvaluationCase], agent_callable, **kwargs
    ) -> List[EvaluationResult]:
        """Runs the agent on all cases and evaluates them."""
        results: List[EvaluationResult] = []
        for case in cases:
            output = agent_callable(case.input_prompt)
            result = self.evaluate_case(case, output, **kwargs)
            results.append(result)
        return results
