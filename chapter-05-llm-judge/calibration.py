"""Calibration and meta-evaluation metrics comparing LLM Judges against Human Ground Truth."""

from __future__ import annotations
import math
from typing import Any, Dict, List
from shared.models.schemas import JudgeEvaluation


def compute_calibration_metrics(
    evaluations: List[JudgeEvaluation],
    human_benchmarks: List[Dict[str, Any]],
    pass_threshold: float = 3.5,
) -> Dict[str, Any]:
    """Calculates Agreement, Correlation, False Positives, and False Negatives."""
    n = min(len(evaluations), len(human_benchmarks))
    if n == 0:
        return {"agreement": 0.0, "correlation": 0.0, "false_positives": 0, "false_negatives": 0}

    agreements = 0
    false_positives = 0
    false_negatives = 0

    judge_scores = []
    human_scores = []

    for i in range(n):
        j_eval = evaluations[i]
        h_case = human_benchmarks[i]
        
        j_overall = j_eval.overall_score
        h_overall = sum(h_case["human_scores"].values()) / len(h_case["human_scores"])

        judge_scores.append(j_overall)
        human_scores.append(h_overall)

        j_pass = j_overall >= pass_threshold
        h_pass = h_overall >= pass_threshold

        if j_pass == h_pass:
            agreements += 1
        elif j_pass and not h_pass:
            false_positives += 1
        elif not j_pass and h_pass:
            false_negatives += 1

    agreement_pct = (agreements / n) * 100.0

    # Pearson correlation coefficient
    mean_j = sum(judge_scores) / n
    mean_h = sum(human_scores) / n
    num = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge_scores, human_scores))
    den_j = sum((j - mean_j) ** 2 for j in judge_scores)
    den_h = sum((h - mean_h) ** 2 for h in human_scores)
    denom = math.sqrt(den_j * den_h) if (den_j * den_h) > 0 else 1.0
    correlation = num / denom

    return {
        "samples_evaluated": n,
        "agreement_pct": round(agreement_pct, 1),
        "pearson_correlation": round(max(-1.0, min(1.0, correlation)), 2),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
