"""Chapter 9: Robustness & Chaos Evals -- naive vs resilient order-lookup
agent under seeded, probabilistic faults, with honest disclosed outcomes."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-09-robustness-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.ui.components import chapter_page_header, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from chaos import ChaosInjector, CircuitBreaker, FaultSpec, VirtualClock
from resilient_agent import CACHE, BaselineAgent, ResilientAgent, seed_cache
from evaluator import ChaosExperimentEvaluator

chapter_page_header(
    9, "Robustness & Chaos Evals",
    "Naive vs resilient order-lookup agent under seeded, probabilistic faults -- "
    "no model calls here, this chapter evaluates infrastructure resilience.",
)

st.caption(
    "This chapter's agent doesn't call an LLM -- it evaluates retry/backoff/circuit-breaker "
    "mechanics against a simulated dependency, which is why there's no chat box here. Every "
    "run below executes real (seeded, deterministic) code against the fault injector."
)

st.sidebar.header("Fault rates (probability per call, not an on/off switch)")
fault_rates = {}
for kind, default in [("latency", 0.0), ("timeout", 0.0), ("http_500_transient", 0.4),
                       ("outage", 0.0), ("malformed_success", 0.1)]:
    fault_rates[kind] = st.sidebar.slider(kind, 0.0, 1.0, default, 0.05, key=f"ch9_{kind}")
n_requests = st.sidebar.slider("Number of requests", 5, 200, 20, key="ch9_n")
seed = st.sidebar.number_input("Random seed", value=0, step=1, key="ch9_seed")

evaluator = ChaosExperimentEvaluator()

tab_run, tab_breaker, tab_lessons = st.tabs(
    ["Run Experiment", "Circuit Breaker", "The Four Lessons"]
)

with tab_run:
    if st.button("Run naive vs resilient", type="primary", key="ch9_run"):
        cases = [f"ORD-{i:04d}" for i in range(1001, 1001 + n_requests)]
        with st.spinner(f"Running {n_requests} requests through both agents..."):
            res = evaluator.run_experiment(fault_rates, test_cases=cases, seed=int(seed))
        st.session_state.ch9_res = res

    if "ch9_res" in st.session_state:
        res = st.session_state.ch9_res
        st.subheader("Naive agent")
        st.caption("One attempt, no backoff, no validation -- trusts any status=='success' payload.")
        metrics_row({
            "correct": res["naive"]["outcome_rate"]["correct"],
            "silent_wrong": res["naive"]["outcome_rate"]["silent_wrong"],
            "crash": res["naive"]["outcome_rate"]["crash"],
            "p95_latency_s": res["naive"]["p95_latency_s"],
            "calls/req": res["naive"]["calls_per_request"],
        })
        if res["naive"]["outcome_rate"]["silent_wrong"] > 0:
            st.warning(f"{res['naive']['outcome_rate']['silent_wrong']*100:.0f}% of naive responses "
                       "are silently WRONG -- served as confident answers with garbage data.")

        st.subheader("Resilient agent")
        st.caption("Validates responses, backs off with jitter, respects a circuit breaker and deadline.")
        metrics_row({
            "correct": res["resilient"]["outcome_rate"]["correct"],
            "degraded_stale": res["resilient"]["outcome_rate"]["degraded_stale"],
            "honest_failure": res["resilient"]["outcome_rate"]["honest_failure"],
            "p95_latency_s": res["resilient"]["p95_latency_s"],
            "calls/req": res["resilient"]["calls_per_request"],
        })
        st.info(f"Circuit breaker ended in state: **{res['circuit_breaker_state']}**")

        compare_df = pd.DataFrame([
            {"outcome": k, "naive": res["naive"]["outcome_rate"][k], "resilient": res["resilient"]["outcome_rate"][k]}
            for k in ("correct", "degraded_stale", "honest_failure", "crash", "silent_wrong")
        ])
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.caption("A stale fallback (`degraded_stale`) is never counted as `correct` -- staleness "
                   "is disclosed, not hidden inside a single success rate.")

with tab_breaker:
    st.caption("Watch the breaker open under a full outage and stop hammering a dead dependency, "
               "instead of retrying every request at full cost.")
    if st.button("Run a 100% outage against the resilient agent", key="ch9_run_outage"):
        CACHE.clear()
        cases = [f"ORD-OUT-{i:04d}" for i in range(n_requests)]
        clock = VirtualClock()
        chaos = ChaosInjector(faults=[FaultSpec(kind="outage", rate=1.0)], clock=clock, seed=int(seed))
        breaker = CircuitBreaker(clock, failure_threshold=5)
        agent = ResilientAgent(chaos, breaker=breaker)
        rows = []
        for cid in cases:
            r = agent.process(cid)
            rows.append({"order_id": cid, "outcome": r["outcome"], "calls": r["calls"], "breaker_state": breaker.state})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        total_calls = sum(r["calls"] for r in rows)
        st.metric("Total calls made", total_calls, f"of up to {3*len(cases)} without a breaker")

with tab_lessons:
    st.markdown(
        "**Four lessons from seeded, realistic fault injection** (see Chapter 9):\n\n"
        "1. Under transient errors, backoff with jitter turns a real crash rate into full "
        "correctness, at a modest extra-calls cost.\n"
        "2. Under malformed successes, only response validation stops a silent wrong answer -- "
        "an error-only fault model can never test this.\n"
        "3. Under a full outage, the breaker stops the agent from hammering a dead dependency; "
        "every request ends as disclosed stale data or an honest failure, never a crash.\n"
        "4. Under timeouts, resilience improves honesty but not latency -- a tool timeout that "
        "already exceeds the deadline needs a tighter client-side timeout, not more retries."
    )
    st.caption("Use the sidebar sliders to reproduce each of these by setting the matching "
               "fault rate to a high value and the others to 0.")
