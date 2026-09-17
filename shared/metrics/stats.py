"""Statistical helpers for honest evaluation reporting.

Small evaluation sets produce pass rates with real uncertainty; a
displayed "80%" on 5 cases is compatible with anything from roughly 38%
to 96%. These helpers make that uncertainty visible instead of hiding it
behind a bare point estimate.
"""

from __future__ import annotations
import math
from typing import Dict, Iterable, List, Tuple


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (Wilson, 1927)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test p-value for paired binary outcomes.

    `b` is the number of paired cases that regressed (baseline passed,
    candidate failed) and `c` the number that got fixed (baseline failed,
    candidate passed). Only the discordant pairs carry information about a
    directional shift -- cases that agree in both runs are silent. This is
    the exact binomial form (not the chi-squared approximation), which
    stays valid at the small case counts typical of an eval suite.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def sliced_report(rows: Iterable[dict], slice_key: str = "category", min_n: int = 10) -> List[dict]:
    """Pass rate per slice with a Wilson interval and an under-powered flag."""
    buckets: Dict[str, List[bool]] = {}
    for r in rows:
        buckets.setdefault(r[slice_key], []).append(bool(r["passed"]))
    report = []
    for name, outcomes in sorted(buckets.items()):
        k, n = sum(outcomes), len(outcomes)
        lo, hi = wilson_interval(k, n)
        report.append({
            "slice": name, "n": n, "pass_rate": round(k / n, 3) if n else 0.0,
            "ci_low": round(lo, 3), "ci_high": round(hi, 3), "underpowered": n < min_n,
        })
    return report
