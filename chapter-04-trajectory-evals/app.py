"""Chapter 4: Agent Trajectory Evaluation Engine & Timeline UI (Streamlit)."""

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
from engine import ITHelpdeskAgent
from evaluator import ITTrajectoryEvaluator

st.set_page_config(page_title="Agent Trajectory Evaluation", page_icon="🧭", layout="wide")

st.title("🧭 Agent Trajectory Evaluation Engine")
st.caption("Chapter 4: Multi-Step Execution Analysis, Looping Penalties, and Recovery Scoring")

# Sidebar Controls
st.sidebar.header("Trajectory Experiment Controls")
inject_failure = st.sidebar.checkbox("Inject Failed Tool at Step 3", value=True, key="inject_fail_chk")
inject_loop = st.sidebar.checkbox("Introduce Redundant Tool Loop", value=False, key="inject_loop_chk")
run_btn = st.sidebar.button("▶ Run Trajectory Diagnosis", type="primary", key="run_diag_btn")

task_prompt = "Diagnose why user cannot access internal web portal (ERR_CONNECTION_REFUSED)."

if "traj_trace" not in st.session_state or run_btn:
    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()
    with st.spinner("Generating and analyzing agent trajectory..."):
        fail_step = 3 if inject_failure else 0
        trace = agent.diagnose_issue(task=task_prompt, inject_failure_at_step=fail_step, loop_behavior=inject_loop)
        scores = evaluator.evaluate_trace(trace, optimal_steps=4)
        st.session_state.traj_trace = trace
        st.session_state.traj_scores = scores

trace = st.session_state.traj_trace
scores = st.session_state.traj_scores

# Top Metrics Row
t_score = scores["trajectory_score"].score
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trajectory Score", f"{t_score:.0f}/100", "Pass" if scores["trajectory_score"].passed else "Needs Improvement")
k2.metric("Step Success", f"{scores['step_success_rate'].score * 100:.0f}%")
k3.metric("Recovery Rate", f"{scores['recovery_rate'].score * 100:.0f}%")
k4.metric("Tool Efficiency", f"{scores['tool_efficiency'].score * 100:.0f}%")
k5.metric("Loop Penalty", f"{scores['loop_penalty'].score:.2f}")

st.divider()

col_timeline, col_details = st.columns([3, 2])

with col_timeline:
    st.subheader("● Step-by-Step Trajectory Timeline")
    st.write(f"**Task:** {trace.task}")

    for s in trace.steps:
        is_err = s.result.lower() in ("error", "failed")
        icon = "❌" if is_err else "✅"
        badge = "Failed / Injected Error" if is_err else "Success"
        
        with st.expander(f"{icon} Step {s.step_index}: `{s.action}` ({s.duration_ms:.0f}ms) — {badge}", expanded=True):
            st.write(f"**Arguments:** `{s.arguments}`")
            st.write(f"**Observation:** {s.observation}")
            st.caption(f"Status: {s.result.upper()} | Execution Time: {s.duration_ms} ms")

    st.success(f"**Final Resolution:**\n\n{trace.final_output}")

with col_details:
    st.subheader("📊 Diagnostic Summary")
    summary_data = [
        {"Metric": "Total Steps Executed", "Value": str(len(trace.steps))},
        {"Metric": "Optimal Baseline Steps", "Value": "4"},
        {"Metric": "Mid-Trajectory Failures", "Value": str(sum(1 for s in trace.steps if s.result == 'error'))},
        {"Metric": "Successful Corrective Action", "Value": "Yes (fallback_ldap_token_refresh)" if inject_failure else "N/A"},
        {"Metric": "Overall Problem Solved", "Value": "True"},
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.subheader("Case Study: IT Helpdesk Recovery")
    st.markdown("""
    - **Step 1-2**: Normal diagnostic path (Ticket parse & DNS probe).
    - **Step 3 (Chaos Injection)**: Auth gateway responds with HTTP 504.
    - **Recovery Branch**: Rather than aborting, the agent invokes secondary LDAP token refresh.
    - **Resolution**: Port connection verified and connectivity restored.
    """)
