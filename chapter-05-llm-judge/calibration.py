"""Calibration and meta-evaluation metrics comparing LLM Judges against Human Ground Truth.

Fixed from an earlier version of this lab: raw agreement on a handful of items rewards a
constant judge (it looks 80% "accurate" just by always saying pass, when
most of the benchmark is good). This version adds Cohen's kappa (which corrects for chance
agreement), a bootstrap confidence interval, false-pass/false-negative
RATES (not raw counts), and an explicit underpowered flag -- and it does
NOT fabricate additional "human" labels to make the numbers look better;
with only 5 real labels, the honest answer is that the interval is wide
and the report says so.
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Sequence
from shared.models.schemas import JudgeEvaluation

MIN_RELIABLE_N = 30  # below this, kappa/CI are reported but flagged as unreliable


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compute_calibration_metrics(
    evaluations: List[JudgeEvaluation],
    human_benchmarks: List[Dict[str, Any]],
    pass_threshold: float = 3.5,
    n_boot: int = 2000,
    seed: int = 7,
) -> Dict[str, Any]:
    """Honest agreement report: kappa, bootstrap CI, and rate-based false
    positive/negative, with an explicit small-sample warning. A judge that
    abstained (status="abstain") is excluded from agreement -- an
    abstention is not a verdict to score against human labels."""
    scored = [(e, h) for e, h in zip(evaluations, human_benchmarks) if e.status == "scored"]
    n = len(scored)
    abstained = len(evaluations) - n
    if n == 0:
        return {"samples_evaluated": 0, "abstained": abstained, "error": "no scored judge outputs to calibrate"}

    judge_scores = [e.overall_score for e, _ in scored]
    human_scores = [sum(h["human_scores"].values()) / len(h["human_scores"]) for _, h in scored]

    jp = [s >= pass_threshold for s in judge_scores]
    hp = [s >= pass_threshold for s in human_scores]

    agreements = sum(j == h for j, h in zip(jp, hp))
    false_positives = sum(j and not h for j, h in zip(jp, hp))
    false_negatives = sum((not j) and h for j, h in zip(jp, hp))
    n_actual_pass = sum(hp) or 1
    n_actual_fail = (n - sum(hp)) or 1

    # Bootstrap CI over items for Cohen's kappa.
    rng = random.Random(seed)
    idx = list(range(n))
    kappas = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        a, b = [jp[i] for i in s], [hp[i] for i in s]
        if len(set(a)) > 1 or len(set(b)) > 1:
            kappas.append(cohen_kappa(a, b))
    kappas.sort()
    kappa_ci = (round(kappas[int(0.025 * len(kappas))], 3), round(kappas[int(0.975 * len(kappas)) - 1], 3)) if kappas else (float("nan"), float("nan"))

    # Pearson correlation on raw scores (kept from the original, still useful alongside kappa).
    mean_j, mean_h = sum(judge_scores) / n, sum(human_scores) / n
    num = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge_scores, human_scores))
    den_j = sum((j - mean_j) ** 2 for j in judge_scores)
    den_h = sum((h - mean_h) ** 2 for h in human_scores)
    denom = math.sqrt(den_j * den_h) if (den_j * den_h) > 0 else 1.0
    correlation = max(-1.0, min(1.0, num / denom))

    underpowered = n < MIN_RELIABLE_N
    return {
        "samples_evaluated": n,
        "abstained": abstained,
        "underpowered": underpowered,
        "warning": (f"Only {n} labeled items -- kappa and its CI are not reliable below "
                    f"~{MIN_RELIABLE_N}. Treat this as a smoke test, not a calibration.") if underpowered else None,
        "agreement_pct": round(agreements / n * 100.0, 1),
        "kappa": round(cohen_kappa(jp, hp), 3),
        "kappa_ci95": kappa_ci,
        "pearson_correlation": round(correlation, 2),
        "false_positive_rate": round(false_positives / n_actual_fail, 3),
        "false_negative_rate": round(false_negatives / n_actual_pass, 3),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
