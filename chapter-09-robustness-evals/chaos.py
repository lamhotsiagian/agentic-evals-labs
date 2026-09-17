"""Chaos engineering fault injection: seeded, probabilistic faults with a
virtual clock, including the dangerous "success with garbage" case.

The old injector's faults were always-on boolean switches: once
"tool_timeout" was True, EVERY call failed, so a retry could never succeed
and "recovery" always meant "fell back to cache." This version fires each
fault with its own per-call probability from a seeded RNG, so retries have
somewhere real to land -- exactly like a real transient dependency.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class VirtualClock:
    """A clock that only advances when `sleep()` is called, so simulated
    backoff, timeouts, and SLA/deadline checks run at test speed while
    latency numbers (p95, deadlines) stay meaningful."""

    def __init__(self):
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, seconds)


@dataclass
class FaultSpec:
    kind: str              # latency | timeout | http_500_transient | outage | malformed_success
    rate: float = 1.0      # probability per call that this fault fires
    transient_failures: int = 1   # http_500_transient: attempts (per request_key) before it starts succeeding
    latency_s: float = 2.0


class ChaosInjector:
    """Seeded, probabilistic fault injection engine."""

    def __init__(self, faults: Optional[List[FaultSpec]] = None, clock: Optional[VirtualClock] = None, seed: int = 0):
        self.faults = faults or []
        self.clock = clock or VirtualClock()
        self.rng = random.Random(seed)
        self.attempts: Dict[str, int] = {}

    def call(self, name: str, fn: Callable[..., Dict[str, Any]], request_key: str, **kw) -> Dict[str, Any]:
        n = self.attempts[request_key] = self.attempts.get(request_key, 0) + 1
        for f in self.faults:
            if self.rng.random() >= f.rate:
                continue
            if f.kind == "latency":
                self.clock.sleep(f.latency_s)
            elif f.kind == "timeout":
                self.clock.sleep(5.0)
                return {"status": "error", "error_type": "Timeout"}
            elif f.kind == "http_500_transient" and n <= f.transient_failures:
                self.clock.sleep(0.05)
                return {"status": "error", "error_type": "HTTP500"}
            elif f.kind == "outage":
                self.clock.sleep(0.05)
                return {"status": "error", "error_type": "HTTP503"}
            elif f.kind == "malformed_success":
                # The dangerous case: a 200-shaped payload with garbage
                # fields, injected as a SUCCESS, not an error -- an agent
                # that trusts status=="success" without validating fields
                # will silently serve this as a real answer.
                self.clock.sleep(0.05)
                return {"status": "success", "order_id": None, "amount": "NaN"}
        self.clock.sleep(0.08)
        return fn(**kw)


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures and stops
    letting calls through until `reset_after_s` has elapsed, then allows one
    probe (half-open) before fully closing again."""

    def __init__(self, clock: VirtualClock, failure_threshold: int = 5, reset_after_s: float = 30.0):
        self.clock, self.threshold, self.reset_after = clock, failure_threshold, reset_after_s
        self.failures, self.opened_at, self.state = 0, None, "closed"

    def allow(self) -> bool:
        if self.state == "open" and self.opened_at is not None and self.clock.now() - self.opened_at >= self.reset_after:
            self.state = "half_open"          # let one probe through
        return self.state != "open"

    def record(self, ok: bool) -> None:
        if ok:
            self.failures, self.state = 0, "closed"
        else:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                self.state, self.opened_at = "open", self.clock.now()
