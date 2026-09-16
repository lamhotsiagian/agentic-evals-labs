"""Local LLM-as-a-Judge implementation with multi-judge comparison."""

from __future__ import annotations
from typing import Dict, Optional
from shared.evaluators.llm_judge import LLMJudgeEvaluator, DEFAULT_RUBRICS
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import JudgeEvaluation


class MultiJudgeSystem:
    """Orchestrates primary and alternative judges for consistency analysis."""

    def __init__(
        self,
        primary_model: str = "qwen3:1.7b",
        alt_model: str = "llama3.2:1b",
        provider: Optional[LLMProvider] = None,
    ):
        self.provider = provider or get_model_provider()
        self.primary_judge = LLMJudgeEvaluator(judge_model=primary_model, provider=self.provider)
        self.alt_judge = LLMJudgeEvaluator(judge_model=alt_model, provider=self.provider)

    def evaluate_pair(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, JudgeEvaluation]:
        primary_eval = self.primary_judge.evaluate(prompt, response, context)
        alt_eval = self.alt_judge.evaluate(prompt, response, context)
        return {
            "primary": primary_eval,
            "alternative": alt_eval,
        }
