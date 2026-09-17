"""Evaluation pipeline for Chapter 1 Customer Support Agent."""

from __future__ import annotations
from typing import List, Dict, Any
from shared.models.schemas import EvaluationCase, EvaluationResult, MetricScore
from shared.metrics.quality import (
    compute_relevance,
    compute_substring_match,
    compute_similarity,
    detect_hallucination,
)
from shared.metrics.performance import estimate_tokens, estimate_cost_usd
from shared.evaluators.base import BaseEvaluator

REFUND_POLICY_TEXT = (
    "Duplicate or unauthorized charges are refunded to the original payment "
    "method within 3-5 business days once verified. Subscription cancellations "
    "processed before the renewal date are not billed for the next cycle."
)
try:
    from .agent import CustomerSupportAgent
except (ImportError, ValueError):
    from agent import CustomerSupportAgent


class CustomerSupportEvaluator(BaseEvaluator):
    """Evaluates customer support responses against quality, latency, and cost rubrics."""

    def evaluate_case(
        self,
        case: EvaluationCase,
        actual_output: str,
        latency_seconds: float = 0.0,
        model_name: str = "qwen2.5:3b",
    ) -> EvaluationResult:
        tokens_in = estimate_tokens(case.input_prompt)
        tokens_out = estimate_tokens(actual_output)
        total_tokens = tokens_in + tokens_out
        cost = estimate_cost_usd(tokens_in, tokens_out, model=model_name)

        # 1. Relevance
        rel_score = compute_relevance(actual_output, case.input_prompt)

        # 2. Helpfulness: require several distinct resolution signals, not one
        # keyword. A single hit ("account" or "support" alone) used to pass
        # any reply, including one that refuses to help.
        keywords = ["refund", "subscription", "account", "support", "billing", "assist", "instructions", "card", "charge"]
        matched_kw_count = sum(1 for kw in keywords if kw in actual_output.lower())
        helpfulness_score = min(1.0, matched_kw_count / 3.0)

        # 3. Hallucination check -- ground against the prompt, the reference
        # answer, AND policy text together, not the prompt alone. Grounding
        # against the prompt alone flags correct policy numbers ("3-5
        # business days") as hallucinated.
        has_hallucination = detect_hallucination(
            actual_output, case.input_prompt, case.expected_output or "", REFUND_POLICY_TEXT
        )

        # 4. Correctness: coverage against the reference answer when one is
        # provided (dataset cases), blended with the hallucination check.
        # Previously this was a constant (0.4 or 0.95) derived only from the
        # hallucination flag, so expected_output was captured on every case
        # and never once read -- an answer contradicting the reference
        # scored the same as one that matched it.
        if case.expected_output:
            reference_coverage = compute_similarity(actual_output, case.expected_output)
            correctness_score = round(
                0.6 * reference_coverage + 0.4 * (0.0 if has_hallucination else 1.0), 2
            )
        else:
            # Live chat turn with no reference answer: fall back to the
            # hallucination signal alone.
            correctness_score = 0.4 if has_hallucination else 0.95

        # 4. Task completion
        task_completed = (rel_score >= 0.2) and (helpfulness_score >= 0.5) and not has_hallucination

        # Overall pass criteria
        passed = task_completed and (correctness_score >= 0.7)

        metrics = {
            "correctness": MetricScore(name="correctness", score=round(correctness_score, 2), passed=not has_hallucination),
            "relevance": MetricScore(name="relevance", score=round(rel_score, 2), passed=rel_score >= 0.2),
            "helpfulness": MetricScore(name="helpfulness", score=round(helpfulness_score, 2), passed=helpfulness_score >= 0.5),
            "hallucination": MetricScore(name="hallucination", score=1.0 if has_hallucination else 0.0, passed=not has_hallucination),
            "task_completion": MetricScore(name="task_completion", score=1.0 if task_completed else 0.0, passed=task_completed),
        }

        return EvaluationResult(
            case_id=case.id,
            input_prompt=case.input_prompt,
            actual_output=actual_output,
            expected_output=case.expected_output,
            metrics=metrics,
            passed=passed,
            latency_seconds=latency_seconds,
            tokens_used=total_tokens,
            cost_estimate_usd=cost,
        )

    def run_suite(
        self,
        cases: List[EvaluationCase],
        agent: CustomerSupportAgent,
    ) -> List[EvaluationResult]:
        results = []
        for case in cases:
            resp = agent.respond(case.input_prompt)
            eval_res = self.evaluate_case(
                case=case,
                actual_output=resp["response"],
                latency_seconds=resp["latency_seconds"],
                model_name=agent.model_name,
            )
            results.append(eval_res)
        return results
