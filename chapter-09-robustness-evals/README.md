# Chapter 9 -- Robustness & Chaos Evals
### Lab: Chaos Testing for AI Agents

> Companion to Chapter 9 of *Agentic Evals System Design*.

A seeded, probabilistic fault injector with a virtual clock, a naive baseline agent and a
resilient agent (response validation, backoff with full jitter, circuit breaker, deadline), and an
evaluator that reports five disclosed outcome classes with p95 latency and calls per request.
This chapter makes no model calls: it evaluates infrastructure resilience mechanics.

## Files

```text
chapter-09-robustness-evals/
├── chaos.py             # VirtualClock, FaultSpec, ChaosInjector (per-call probability), CircuitBreaker
├── resilient_agent.py   # BaselineAgent, ResilientAgent, valid_order, seed_cache
├── evaluator.py         # ChaosExperimentEvaluator: correct / degraded_stale / honest_failure / crash / silent_wrong
└── tests/
    └── test_robustness.py   # 10 tests
```

UI page: `pages/9_Ch9_Robustness_Evals.py`.

## Fault kinds

`latency`, `timeout` (5 s), `http_500_transient`, `outage`, and `malformed_success` -- a
success-shaped payload with garbage fields, which only response validation catches.

A stale cache fallback is reported as `degraded_stale`, never folded into `correct`.

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 9" in the sidebar
```

Set each fault's per-call probability, the request count, and the seed in the sidebar.
**Run Experiment** runs both agents against the same seeded faults; **Circuit Breaker** runs a 100%
outage against the resilient agent and shows the breaker opening; **The Four Lessons** summarises
what each fault scenario teaches.

## Tests

```bash
MOCK_LLM=1 pytest chapter-09-robustness-evals/tests/test_robustness.py -v
```
