"""Chapter 9: Chaos Testing & Resilience Evaluation (Streamlit)."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import pandas as pd
from evaluator import ChaosExperimentEvaluator

st.set_page_config(page_title="Chaos Testing for AI Agents", page_icon="💥", layout="wide")

st.title("💥 Chaos Testing & Reliability Evaluation")
st.caption("Chapter 9: Fault Injection, Failure Recovery, and Baseline vs Resilient Agent Benchmarking")

# Sidebar
st.sidebar.header("Chaos Control Panel")
f_timeout = st.sidebar.checkbox("Tool Timeout (5000ms)", value=True, key="f_timeout")
f_500 = st.sidebar.checkbox("HTTP 500 Internal Error", value=True, key="f_500")
f_json = st.sidebar.checkbox("Malformed / Invalid JSON", value=False, key="f_json")
f_corrupt = st.sidebar.checkbox("Context Corruption", value=False, key="f_corrupt")
f_503 = st.sidebar.checkbox("Tool Unavailable (503)", value=False, key="f_503")

run_chaos_btn = st.sidebar.button("⚡ Run Chaos Experiment", type="primary", key="run_chaos_btn")

active_faults = {
    "tool_timeout": f_timeout,
    "http_500": f_500,
    "invalid_json": f_json,
    "context_corruption": f_corrupt,
    "tool_unavailable": f_503,
}

if "chaos_exp" not in st.session_state or run_chaos_btn:
    evaluator = ChaosExperimentEvaluator()
    with st.spinner("Injecting environmental chaos and running stress workloads..."):
        exp_res = evaluator.run_experiment(chaos_faults=active_faults)
        st.session_state.chaos_exp = exp_res

exp_res = st.session_state.chaos_exp

# KPI Metric Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Normal Success", f"{exp_res['normal_success_rate']}%", "Unperturbed Baseline")
k2.metric("Baseline under Chaos", f"{exp_res['baseline_chaos_success_rate']}%", f"{exp_res['baseline_chaos_success_rate'] - exp_res['normal_success_rate']:.1f}%", delta_color="inverse")
k3.metric("Resilient under Chaos", f"{exp_res['resilient_chaos_success_rate']}%", f"+{exp_res['resilient_chaos_success_rate'] - exp_res['baseline_chaos_success_rate']:.1f}% vs Fragile")
k4.metric("Recovery Rate", f"{exp_res['recovery_rate']}%", "Self-Healing")

st.divider()

# Comparison Columns
col_baseline, col_resilient = st.columns(2)

with col_baseline:
    st.subheader("❌ Baseline Agent (Fragile)")
    st.write("**Architecture:** Single attempt, no fallbacks, no retries.")
    sample_b = exp_res["sample_baseline_run"]
    st.error(f"Status: {'Success' if sample_b['success'] else 'Crashed / Aborted'}")
    st.write(f"**Output:** {sample_b['final_answer']}")
    if sample_b.get("error"):
        st.code(f"Unhandled Exception: {sample_b['error']}")

with col_resilient:
    st.subheader("🛡️ Resilient Agent (Hardened)")
    st.write("**Architecture:** Exponential retries + Secondary Backup Tool + Circuit Breaker.")
    sample_r = exp_res["sample_resilient_run"]
    st.success(f"Status: {'Success' if sample_r['success'] else 'Failed'}")
    st.write(f"**Output:** {sample_r['final_answer']}")
    if sample_r.get("recovered"):
        st.info(f"Self-Healing Mechanism Triggered: `{sample_r.get('recovery_mechanism')}` (Attempts: {sample_r.get('attempts')})")

st.divider()

# Comparative Matrix
st.subheader("Reliability & Chaos Benchmark Summary")
benchmark_df = pd.DataFrame([
    {"Architecture": "Baseline Agent", "Condition": "Normal Environment", "Success Rate": f"{exp_res['normal_success_rate']}%", "Resilience Grade": "A"},
    {"Architecture": "Baseline Agent", "Condition": "Active Chaos Injection", "Success Rate": f"{exp_res['baseline_chaos_success_rate']}%", "Resilience Grade": "F (Fragile)"},
    {"Architecture": "Resilient Agent", "Condition": "Active Chaos Injection", "Success Rate": f"{exp_res['resilient_chaos_success_rate']}%", "Resilience Grade": "A- (Production-Ready)"},
])
st.table(benchmark_df)
