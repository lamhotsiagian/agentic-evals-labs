"""Local LLM-as-a-Judge implementation with multi-judge comparison.

Fixed from an earlier version of this lab: the alternative judge's verdict used to be
computed and thrown away. This version routes to human review whenever
the two judges disagree on pass/fail, and adds a real, executable
verbosity bias probe -- padding a response with filler and checking
whether the score moves without any change in content.
"""

from __future__ import annotations
from typing import Dict, Optional
from shared.evaluators.llm_judge import LLMJudgeEvaluator, DEFAULT_RUBRICS
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import JudgeEvaluation

VERBOSITY_PADDING = (
    " To elaborate further and provide additional context for completeness, "
    "it is worth noting that this explanation aims to be thorough and detailed "
    "in every relevant respect, covering the topic comprehensively."
)


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
    ) -> Dict[str, object]:
        primary_eval = self.primary_judge.evaluate(prompt, response, context)
        alt_eval = self.alt_judge.evaluate(prompt, response, context)

        # Disagreement routing: previously the alt judge's verdict was
        # computed and never used. A judge that abstained cannot be
        # compared, so that alone routes to review.
        if primary_eval.status != "scored" or alt_eval.status != "scored":
            needs_review, reason = True, "one or both judges abstained"
        elif primary_eval.passed != alt_eval.passed:
            needs_review, reason = True, "primary and alternative judges disagree on pass/fail"
        else:
            needs_review, reason = False, None

        return {
            "primary": primary_eval,
            "alternative": alt_eval,
            "needs_human_review": needs_review,
            "review_reason": reason,
        }

    def probe_verbosity_bias(self, prompt: str, response: str, context: Optional[str] = None) -> Dict[str, object]:
        """Re-score a padded (but not more correct) version of the same
        response. A judge with a verbosity bias will score it higher for
        no substantive reason -- this makes that measurable."""
        base = self.primary_judge.evaluate(prompt, response, context)
        padded = self.primary_judge.evaluate(prompt, response + VERBOSITY_PADDING, context)
        delta = None
        if base.status == "scored" and padded.status == "scored":
            delta = round(padded.overall_score - base.overall_score, 2)
        return {
            "base_score": base.overall_score if base.status == "scored" else None,
            "padded_score": padded.overall_score if padded.status == "scored" else None,
            "delta": delta,
            "biased": bool(delta and delta > 0.3),
        }
