"""Evaluator for chaos experiments: outcome classification with disclosed
freshness, plus the latency and call-amplification cost of each recovery
mechanism -- not a single success/fail rate that hides staleness."""

from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional

from chaos import ChaosInjector, CircuitBreaker, FaultSpec, VirtualClock
from resilient_agent import CACHE, BaselineAgent, ResilientAgent, seed_cache

FAULT_KINDS = ["latency", "timeout", "http_500_transient", "outage", "malformed_success"]
OUTCOME_KINDS = ["correct", "degraded_stale", "honest_failure", "crash", "silent_wrong"]


def _build_faults(fault_rates: Dict[str, float]) -> List[FaultSpec]:
    return [FaultSpec(kind=kind, rate=fault_rates[kind]) for kind in FAULT_KINDS if fault_rates.get(kind, 0.0) > 0]


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return round(s[idx], 3)


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results) or 1
    counts = Counter(r["outcome"] for r in results)
    return {
        "outcome_rate": {k: round(counts.get(k, 0) / n, 3) for k in OUTCOME_KINDS},
        "p95_latency_s": _p95([r["latency"] for r in results]),
        "calls_per_request": round(sum(r["calls"] for r in results) / n, 2),
        "n": len(results),
    }


class ChaosExperimentEvaluator:
    """Runs the naive and resilient agents against the same seeded fault
    sequence and workload, and reports honest, disclosed outcomes for each."""

    def run_experiment(
        self, fault_rates: Dict[str, float], test_cases: Optional[List[str]] = None, seed: int = 0,
    ) -> Dict[str, Any]:
        cases = test_cases or [f"ORD-{i:04d}" for i in range(1001, 1021)]
        faults = _build_faults(fault_rates)

        CACHE.clear()
        seed_cache(cases[: len(cases) // 2])   # half the workload pre-cached, per the book's experiment setup

        naive_clock = VirtualClock()
        naive_chaos = ChaosInjector(faults=faults, clock=naive_clock, seed=seed)
        naive_agent = BaselineAgent(naive_chaos)
        naive_results = [naive_agent.process(cid) for cid in cases]
        naive_summary = _summarize(naive_results)

        # Fresh clock/injector/cache state so the resilient run starts from
        # the exact same conditions the naive run did, not wherever the
        # naive run's calls/cache-writes left things.
        CACHE.clear()
        seed_cache(cases[: len(cases) // 2])
        res_clock = VirtualClock()
        res_chaos = ChaosInjector(faults=faults, clock=res_clock, seed=seed)
        breaker = CircuitBreaker(res_clock)
        resilient_agent = ResilientAgent(res_chaos, breaker=breaker)
        resilient_results = [resilient_agent.process(cid) for cid in cases]
        resilient_summary = _summarize(resilient_results)

        # Backward-compatible top-level fields, now computed honestly:
        # "success" means a truthful correct answer, and a disclosed stale
        # fallback is reported separately -- it is never folded into "success".
        normal_faults_active = bool(faults)
        return {
            "total_trials": len(cases),
            "active_faults": fault_rates,
            "naive": naive_summary,
            "resilient": resilient_summary,
            "naive_correct_rate": round(naive_summary["outcome_rate"]["correct"] * 100.0, 1),
            "resilient_correct_rate": round(resilient_summary["outcome_rate"]["correct"] * 100.0, 1),
            "resilient_degraded_rate": round(resilient_summary["outcome_rate"]["degraded_stale"] * 100.0, 1),
            "naive_silent_wrong_rate": round(naive_summary["outcome_rate"]["silent_wrong"] * 100.0, 1),
            "resilient_silent_wrong_rate": round(resilient_summary["outcome_rate"]["silent_wrong"] * 100.0, 1),
            "circuit_breaker_state": breaker.state,
            "faults_active": normal_faults_active,
            "sample_naive_run": naive_results[0] if naive_results else None,
            "sample_resilient_run": resilient_results[0] if resilient_results else None,
        }
