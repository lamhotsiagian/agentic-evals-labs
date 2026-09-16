"""Chapter 2: Agent Architecture Evaluation (Planner -> Executor -> Verifier)."""

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
from pipeline import ArchitecturePipeline
from evaluator import ArchitectureEvaluator
from shared.datasets.loader import load_travel_planner_cases

st.set_page_config(page_title="Agent Architecture Evaluation", page_icon="🏗️", layout="wide")

st.title("🏗️ Agent Architecture Evaluation")
st.caption("Chapter 2: Evaluating Planner → Executor → Verifier Triad with Local Models")

cases = load_travel_planner_cases()
case_options = {f"{c.id}: {c.input_prompt[:45]}...": c for c in cases}

# Sidebar controls
st.sidebar.header("Pipeline Configuration")
selected_case_key = st.sidebar.selectbox("Select Travel Scenario", list(case_options.keys()))
selected_case = case_options[selected_case_key]

inject_fault = st.sidebar.checkbox("Inject Executor Failure (1st Attempt)", value=False, key="inject_fault_cb")
run_pipeline_btn = st.sidebar.button("⚡ Run Pipeline", type="primary", key="run_pipeline_btn")

if "pipeline_result" not in st.session_state or run_pipeline_btn:
    pipeline = ArchitecturePipeline()
    evaluator = ArchitectureEvaluator()
    with st.spinner("Executing Planner -> Executor -> Verifier..."):
        res = pipeline.run(
            task=selected_case.input_prompt,
            constraints=selected_case.metadata,
            inject_failure_on_first_try=inject_fault,
        )
        scores = evaluator.evaluate_run(res)
        st.session_state.pipeline_result = res
        st.session_state.pipeline_scores = scores

res = st.session_state.pipeline_result
scores = st.session_state.pipeline_scores

# Top metrics
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Planning Accuracy", f"{scores['planning_accuracy'].score * 100:.0f}%")
mcol2.metric("Execution Accuracy", f"{scores['execution_accuracy'].score * 100:.0f}%")
mcol3.metric("Verification Accuracy", f"{scores['verification_accuracy'].score * 100:.0f}%")
mcol4.metric("E2E Success", "Passed" if res["success"] else "Failed")
mcol5.metric("Retries Needed", f"{res['retries']}")

st.divider()

# Visual Agent Pipeline Representation
st.subheader("Visual Agent Pipeline Status")
nodes = [item for item in res["pipeline_log"]]

p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    st.info("**1. USER TASK**\n\n" + res["task"])

with p_col2:
    p_stat = [n["status"] for n in nodes if n["node"] == "PLANNER"][-1]
    st.success(f"**2. [PLANNER]**\n\nStatus: {p_stat}\n\nModel: Qwen2.5:3B")

with p_col3:
    e_stat = [n["status"] for n in nodes if n["node"] == "EXECUTOR"][-1]
    if "Error" in e_stat:
        st.error(f"**3. [EXECUTOR]**\n\nStatus: {e_stat}\n\nTool Call Error")
    else:
        st.success(f"**3. [EXECUTOR]**\n\nStatus: {e_stat}\n\nAll tools executed")

with p_col4:
    v_stat = [n["status"] for n in nodes if n["node"] == "VERIFIER"][-1]
    if "Rejected" in v_stat:
        st.warning(f"**4. [VERIFIER]**\n\nStatus: {v_stat}\n\nConstraint violation")
    else:
        st.success(f"**4. [VERIFIER]**\n\nStatus: {v_stat}\n\nConstraints met")

st.divider()

# Trace Inspector
st.subheader("Node Trace Inspector")
tab_planner, tab_executor, tab_verifier, tab_trace = st.tabs(["Planner Details", "Executor Details", "Verifier Details", "Full OpenTelemetry Trace"])

with tab_planner:
    planner_logs = [n for n in nodes if n["node"] == "PLANNER"]
    for pl in planner_logs:
        st.write(f"**Attempt {pl['attempt']} — {pl['status']}**")
        st.json(pl["details"])

with tab_executor:
    exec_logs = [n for n in nodes if n["node"] == "EXECUTOR"]
    for el in exec_logs:
        st.write(f"**Attempt {el['attempt']} — {el['status']}**")
        st.json(el["details"])

with tab_verifier:
    verif_logs = [n for n in nodes if n["node"] == "VERIFIER"]
    for vl in verif_logs:
        st.write(f"**Attempt {vl['attempt']} — {vl['status']}**")
        st.json(vl["details"])

with tab_trace:
    st.write(f"**Total Duration**: {res['total_duration_sec']}s")
    trace_steps = [s.model_dump() for s in res["trace"].steps]
    st.dataframe(pd.DataFrame(trace_steps), use_container_width=True)
