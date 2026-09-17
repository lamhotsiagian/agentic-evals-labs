"""Unit tests for Chapter 9 Chaos Testing and Reliability.

Several tests pin down bugs an earlier version of this lab had, against the
now-fixed code: always-on fault
switches, ungraded staleness, unvalidated "successes", and a circuit breaker
that never existed.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "graders", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
    sys.modules.pop(m, None)

from chaos import CircuitBreaker, ChaosInjector, FaultSpec, VirtualClock
from resilient_agent import CACHE, BaselineAgent, ResilientAgent, primary_db, seed_cache, valid_order
from evaluator import ChaosExperimentEvaluator


def test_faults_are_probabilistic_not_always_on():
    # Central lesson: the old injector's faults were boolean switches -- once
    # active, EVERY call failed, so a retry against the SAME dependency could
    # never succeed. A seeded 40% rate must let SOME calls through clean.
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="outage", rate=0.4)], clock=clock, seed=7)
    outcomes = [chaos.call("primary_db", primary_db, request_key=f"ORD-{i}", order_id=f"ORD-{i}")["status"]
                for i in range(50)]
    n_success = sum(1 for o in outcomes if o == "success")
    assert 0 < n_success < 50, "a 40% fault rate must not behave like an always-on switch"


def test_malformed_success_is_caught_by_validation_not_by_error_type():
    # Central lesson: a malformed-success fault used to be untestable because
    # the old injector could only inject ERRORS. It must arrive as a 200-
    # shaped payload with garbage fields, and only response validation catches it.
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="malformed_success", rate=1.0)], clock=clock, seed=0)
    r = chaos.call("primary_db", primary_db, request_key="ORD-1", order_id="ORD-1")
    assert r["status"] == "success"          # arrives as a success...
    assert not valid_order(r)                # ...but fails validation


def test_baseline_agent_silently_serves_malformed_success():
    # The dangerous case the old suite never tested: the naive agent trusts
    # any status=="success" payload, so a malformed one becomes a SILENT
    # wrong answer -- not a visible crash, and not "recovered".
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="malformed_success", rate=1.0)], clock=clock, seed=0)
    agent = BaselineAgent(chaos)
    res = agent.process("ORD-1")
    assert res["outcome"] == "silent_wrong"


def test_resilient_agent_never_serves_a_malformed_success():
    # The fix: the resilient agent validates every response, so a malformed
    # success is treated exactly like an error -- it retries, and (with
    # nothing valid to fall back to) reports an honest outcome, never silent_wrong.
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="malformed_success", rate=1.0)], clock=clock, seed=0)
    breaker = CircuitBreaker(clock)
    agent = ResilientAgent(chaos, breaker=breaker)
    res = agent.process("ORD-99-no-cache")
    assert res["outcome"] != "silent_wrong"
    assert res["outcome"] == "honest_failure"   # no cache entry exists for this order


def test_stale_fallback_is_disclosed_not_scored_as_a_fresh_success():
    # Central lesson: "Alice (Cached)" used to be scored identically to a
    # fresh answer. A cache-served fallback must report its own outcome class.
    CACHE.clear()
    seed_cache(["ORD-CACHED"])
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="outage", rate=1.0)], clock=clock, seed=0)
    breaker = CircuitBreaker(clock)
    agent = ResilientAgent(chaos, breaker=breaker)
    res = agent.process("ORD-CACHED")
    assert res["outcome"] == "degraded_stale"
    assert res["order_data"]["as_of"] == "cached"


def test_circuit_breaker_opens_and_stops_hammering_a_dead_dependency():
    # No breaker existed before. Under a full, persistent outage, the breaker
    # must open after `failure_threshold` failures and the resilient agent
    # must stop attempting calls well short of max_attempts * n_requests.
    CACHE.clear()
    cases = [f"ORD-{i:04d}" for i in range(1001, 1021)]  # 20 requests, nothing cached
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="outage", rate=1.0)], clock=clock, seed=0)
    breaker = CircuitBreaker(clock, failure_threshold=5)
    agent = ResilientAgent(chaos, breaker=breaker, max_attempts=3)
    results = [agent.process(cid) for cid in cases]

    assert breaker.state == "open"
    total_calls = sum(r["calls"] for r in results)
    # Without a breaker this would be up to 3 * 20 = 60 calls; the breaker
    # must cut this off far short of that once it opens.
    assert total_calls < 30, f"breaker should have curtailed calls, got {total_calls}"
    assert all(r["outcome"] in ("honest_failure", "degraded_stale") for r in results)
    assert all(r["outcome"] != "crash" for r in results)


def test_timeout_exceeding_the_deadline_is_a_latency_problem_not_a_retry_problem():
    # Fourth lesson from the book: resilience improves honesty under
    # timeouts, but a single 5s tool timeout already exceeds a 3.5s deadline,
    # so retries cannot fix latency -- p95 stays high regardless of the agent.
    CACHE.clear()
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="timeout", rate=1.0)], clock=clock, seed=0)
    breaker = CircuitBreaker(clock)
    agent = ResilientAgent(chaos, breaker=breaker, deadline_s=3.5)
    res = agent.process("ORD-TIMEOUT")
    assert res["latency"] >= 5.0            # the single timeout alone blew the deadline
    assert res["calls"] == 1                # the deadline check stopped a second attempt
    assert res["outcome"] == "honest_failure"


def test_transient_500_recovers_via_backoff_retry():
    # A dependency that fails once then heals -- the whole point of a retry.
    clock = VirtualClock()
    chaos = ChaosInjector(faults=[FaultSpec(kind="http_500_transient", rate=1.0, transient_failures=1)], clock=clock, seed=0)
    breaker = CircuitBreaker(clock)
    agent = ResilientAgent(chaos, breaker=breaker)
    res = agent.process("ORD-RETRY")
    assert res["outcome"] == "correct"
    assert res["calls"] == 2                # first attempt failed, second succeeded


def test_evaluate_suite_reports_disclosed_outcome_rates_not_a_single_success_bool():
    evaluator = ChaosExperimentEvaluator()
    results = evaluator.run_experiment({"http_500_transient": 0.4}, seed=3)
    assert results["total_trials"] == 20
    for summary_key in ("naive", "resilient"):
        summary = results[summary_key]
        assert set(summary["outcome_rate"]) == {"correct", "degraded_stale", "honest_failure", "crash", "silent_wrong"}
        assert abs(sum(summary["outcome_rate"].values()) - 1.0) < 1e-6
        assert summary["p95_latency_s"] >= 0.0
        assert summary["calls_per_request"] >= 1.0


def test_no_faults_means_everyone_succeeds_and_resilient_costs_nothing_extra():
    evaluator = ChaosExperimentEvaluator()
    results = evaluator.run_experiment({}, seed=0)
    assert results["naive_correct_rate"] == 100.0
    assert results["resilient_correct_rate"] == 100.0
    assert results["naive"]["calls_per_request"] == 1.0
    assert results["resilient"]["calls_per_request"] == 1.0
