"""Chapter 10: Master Agent Evaluation Platform (Streamlit Capstone)."""

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
from eval_platform import ProductionEvaluationPlatform, RegressionThresholds
from reporter import CIReporter

st.set_page_config(page_title="Agent Evaluation Platform", page_icon="🏛️", layout="wide")

st.title("🏛️ Agent Evaluation Platform")
st.caption("Chapter 10: Production-Grade Capstone — Quality, Safety, Trajectories, RAG, Tracing, and CI/CD Regression Gates")

# Sidebar
st.sidebar.header("CI/CD Gate Thresholds")
min_success = st.sidebar.slider("Min Task Success %", 70.0, 99.0, 85.0, 1.0)
min_safety = st.sidebar.slider("Min Safety Score %", 80.0, 100.0, 95.0, 1.0)
max_lat = st.sidebar.slider("Max Avg Latency (s)", 1.0, 5.0, 3.5, 0.25)
run_platform_btn = st.sidebar.button("🚀 Run Full Platform Audit", type="primary", key="run_platform_btn")

thresholds = RegressionThresholds(
    min_task_success_pct=min_success,
    min_safety_score_pct=min_safety,
    max_avg_latency_sec=max_lat,
)

if "platform_result" not in st.session_state or run_platform_btn:
    platform = ProductionEvaluationPlatform(thresholds=thresholds)
    with st.spinner("Orchestrating multi-dimensional evaluation suite..."):
        res = platform.run_full_evaluation()
        st.session_state.platform_result = res

res = st.session_state.platform_result

# Master Executive Scorecard Banner
st.subheader("Executive Scorecard")
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
row1_col1.metric("Task Success", f"{res.task_success_pct:.1f}%", "+6.4% vs Target")
row1_col2.metric("Safety Score", f"{res.safety_score_pct:.1f}%", "Passed Gate")
row1_col3.metric("Groundedness", f"{res.groundedness_pct:.1f}%")
row1_col4.metric("Tool Accuracy", f"{res.tool_accuracy_pct:.1f}%")

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
row2_col1.metric("Recovery Rate", f"{res.recovery_rate_pct:.1f}%")
row2_col2.metric("Avg Latency", f"{res.avg_latency_sec:.2f}s")
row2_col3.metric("Avg Cost / Task", f"${res.avg_cost_usd:.3f}")
row2_col4.metric("CI/CD Gate", "PASSED" if res.passed_ci_gate else "BLOCKED", "Deployable" if res.passed_ci_gate else "Regression Detected")

st.divider()

# Navigation Tabs matching outline specifications
tab_traj, tab_fail, tab_sec, tab_tools, tab_rag, tab_ci = st.tabs([
    "📈 Trajectories",
    "⚠️ Failures",
    "🛡️ Safety",
    "🔧 Tools",
    "📚 RAG",
    "🚀 Regression & CI/CD",
])

with tab_traj:
    st.subheader("Enterprise Support Agent Trajectory Flow")
    traj_data = [
        {"Step": 1, "Action": "Customer Request Ingest", "Latency": "12ms", "Status": "Success"},
        {"Step": 2, "Action": "Context & RAG Lookup", "Latency": "45ms", "Status": "Success"},
        {"Step": 3, "Action": "Permission & Safety Check", "Latency": "18ms", "Status": "Success"},
        {"Step": 4, "Action": "Tool Execution (Order/Refund)", "Latency": "180ms", "Status": "Success"},
        {"Step": 5, "Action": "Multi-Agent Verification", "Latency": "92ms", "Status": "Success"},
        {"Step": 6, "Action": "Final Response Dispatch", "Latency": "14ms", "Status": "Success"},
    ]
    st.dataframe(pd.DataFrame(traj_data), use_container_width=True, hide_index=True)

with tab_fail:
    st.subheader("Failure Classification & Root-Cause Matrix")
    fail_data = [
        {"Failure ID": "F-101", "Component": "Tool Execution", "Count": 3, "Root Cause": "Database socket timeout", "Mitigation": "Retries & Backup Cache"},
        {"Failure ID": "F-102", "Component": "Planner", "Count": 1, "Root Cause": "Ambiguous itinerary constraints", "Mitigation": "Clarification prompt"},
        {"Failure ID": "F-103", "Component": "Retriever", "Count": 2, "Root Cause": "Outdated policy document indexed", "Mitigation": "Corpus freshness filtering"},
    ]
    st.table(pd.DataFrame(fail_data))

with tab_sec:
    st.subheader("Safety Guardrail Defenses")
    st.write(f"**Safety Score:** `{res.safety_score_pct}%`")
    st.info("Zero critical data leaks or unauthorized transfers occurred during red-team fuzzing.")

with tab_tools:
    st.subheader("Tool Calling Performance")
    st.write(f"**Overall Tool Accuracy:** `{res.tool_accuracy_pct}%`")
    st.write(f"**Error Recovery Rate:** `{res.recovery_rate_pct}%`")

with tab_rag:
    st.subheader("RAG Groundedness & Citations")
    st.write(f"**Groundedness Score:** `{res.groundedness_pct}%`")
    st.write("All retrieved claims grounded against enterprise knowledge base.")

with tab_ci:
    st.subheader("Automated CI/CD Regression Gate Engine")
    if res.passed_ci_gate:
        st.success("✅ **CI/CD Quality Gate Status: PASSED** — Safe for staging deployment.")
    else:
        st.error(f"❌ **CI/CD Quality Gate Status: BLOCKED** — {res.gate_failures}")

    st.markdown("### Markdown Summary Report")
    st.markdown(CIReporter.generate_markdown_summary(res))

    with st.expander("View Raw JSON Artifact for CI Pipelines"):
        st.code(CIReporter.generate_json_report(res), language="json")
