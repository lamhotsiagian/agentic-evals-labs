"""Baseline (naive) vs resilient order-lookup agents under chaos.

The old ResilientAgent's "recovery" was pure theater: instant retries against
a fault that could never succeed, then an unconditional fallback that was
scored identical to a real answer -- "Alice (Cached)" counted as success with
no staleness ever disclosed. This version validates every response (a
malformed "success" is not trusted), backs off with jitter between retries,
respects a circuit breaker and a deadline, and reports one of five honest
outcomes instead of a single success/fail boolean.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional

from chaos import ChaosInjector, CircuitBreaker

CACHE: Dict[str, Dict[str, Any]] = {}


def primary_db(order_id: str) -> Dict[str, Any]:
    return {"status": "success", "order_id": order_id, "customer": "Alice", "amount": 120.0}


def cache_lookup(order_id: str) -> Optional[Dict[str, Any]]:
    return CACHE.get(order_id)


def seed_cache(order_ids: List[str]) -> None:
    """Pre-populates the cache for a subset of the workload -- orders a
    customer looked up recently, so a fallback has somewhere real to read
    from instead of the lookup being fabricated."""
    for oid in order_ids:
        CACHE[oid] = {"status": "success", "order_id": oid, "customer": "Alice", "amount": 120.0, "as_of": "cached"}


def valid_order(r: Dict[str, Any]) -> bool:
    """A response is a real success only if its fields are actually usable --
    this is what catches a malformed-success fault the old agents trusted blindly."""
    return r.get("status") == "success" and isinstance(r.get("order_id"), str) \
        and isinstance(r.get("amount"), (int, float))


class BaselineAgent:
    """Fragile agent: one attempt, no backoff, no breaker, and -- critically
    -- trusts any status=="success" payload without checking its fields.
    That last point is what turns a malformed-success fault into a SILENT
    wrong answer instead of a visible crash."""

    def __init__(self, chaos: ChaosInjector):
        self.chaos = chaos

    def process(self, order_id: str) -> Dict[str, Any]:
        start = self.chaos.clock.now()
        r = self.chaos.call("primary_db", primary_db, request_key=order_id, order_id=order_id)
        latency = self.chaos.clock.now() - start
        if r.get("status") == "success":
            if valid_order(r):
                return {"outcome": "correct", "calls": 1, "latency": latency, "order_data": r}
            return {"outcome": "silent_wrong", "calls": 1, "latency": latency, "order_data": r}
        return {"outcome": "crash", "calls": 1, "latency": latency, "error_type": r.get("error_type")}


class ResilientAgent:
    """Hardened agent: validated responses, backoff with full jitter, a
    circuit breaker, and a deadline -- reports correct / degraded_stale /
    honest_failure, and can never report silent_wrong (it validates) or
    crash (it always resolves to a disclosed outcome)."""

    def __init__(
        self, chaos: ChaosInjector, breaker: Optional[CircuitBreaker] = None,
        max_attempts: int = 3, base_backoff: float = 0.2, deadline_s: float = 3.5, seed: int = 1,
    ):
        self.chaos = chaos
        self.breaker = breaker or CircuitBreaker(self.chaos.clock)
        self.max_attempts, self.base, self.deadline = max_attempts, base_backoff, deadline_s
        self.rng = random.Random(seed)

    def process(self, order_id: str) -> Dict[str, Any]:
        start, calls = self.chaos.clock.now(), 0
        for attempt in range(self.max_attempts):
            if not self.breaker.allow() or self.chaos.clock.now() - start > self.deadline:
                break
            calls += 1
            r = self.chaos.call("primary_db", primary_db, request_key=order_id, order_id=order_id)
            ok = valid_order(r)
            self.breaker.record(ok)
            if ok:
                CACHE[order_id] = {**r, "as_of": "fresh"}
                return {"outcome": "correct", "calls": calls, "latency": self.chaos.clock.now() - start, "order_data": r}
            self.chaos.clock.sleep(self.rng.uniform(0, self.base * 2 ** attempt))   # backoff, full jitter
        latency = self.chaos.clock.now() - start
        cached = cache_lookup(order_id)
        if cached:
            return {"outcome": "degraded_stale", "calls": calls, "latency": latency, "order_data": cached}
        return {"outcome": "honest_failure", "calls": calls, "latency": latency}
