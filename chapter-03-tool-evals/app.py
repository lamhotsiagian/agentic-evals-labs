"""Chapter 3: Tool-Calling Evaluation Harness & Trace Viewer (Streamlit)."""

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
from agent import EcommerceCustomerAgent
from evaluator import ToolCallingEvaluator
from shared.datasets.loader import load_ecommerce_tool_cases

st.set_page_config(page_title="Tool Calling Evaluation Harness", page_icon="🔧", layout="wide")

st.title("🔧 Tool Calling Evaluation Harness")
st.caption("Chapter 3: Evaluating Tool Selection, Parameter Accuracy, Sequencing, and Error Recovery")

cases = load_ecommerce_tool_cases()
case_map = {f"{c.id} — {c.input_prompt}": c for c in cases}

st.sidebar.header("Evaluation Controls")
selected_key = st.sidebar.selectbox("Select Customer Task", list(case_map.keys()))
selected_case = case_map[selected_key]

inject_error = st.sidebar.checkbox("Inject Invalid Order / Parameter", value=selected_case.metadata.get("injected_failure", False))
run_task_btn = st.sidebar.button("⚙️ Execute Tool Task", type="primary", key="run_tool_btn")

if "tool_result" not in st.session_state or run_task_btn:
    agent = EcommerceCustomerAgent()
    evaluator = ToolCallingEvaluator()
    with st.spinner("Executing agent tool harness..."):
        res = agent.execute_task(selected_case.input_prompt, inject_bad_arg=inject_error)
        scores = evaluator.evaluate_execution(
            expected_tools=selected_case.expected_tools,
            agent_result=res,
            expect_error_recovery=inject_error,
        )
        st.session_state.tool_result = res
        st.session_state.tool_scores = scores

res = st.session_state.tool_result
scores = st.session_state.tool_scores

# Top metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tool Selection", f"{scores['tool_selection'].score * 100:.0f}%")
m2.metric("Argument Accuracy", f"{scores['argument_accuracy'].score * 100:.0f}%")
m3.metric("Tool Order", "Correct" if scores['tool_order'].passed else "Incorrect")
m4.metric("Result Handling", "Pass" if scores['result_handling'].passed else "Fail")
m5.metric("Recovery Rate", f"{scores['agent_recovery'].score * 100:.0f}%")

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🌲 Tool Execution Trace Viewer")
    st.write(f"**Task Prompt:** {res['prompt']}")
    
    # Render hierarchical trace
    for i, call in enumerate(res.get("tool_calls", [])):
        is_success = call["status"] == "✓"
        symbol = "✓" if is_success else "✗"
        card_class = "success" if is_success else "error"
        
        with st.container():
            st.markdown(f"**Step {i+1}: `{call['tool']}({call['args']})`** — `{symbol}`")
            st.code(f"Result: {call['result']}")
            
    st.info(f"**Agent Final Answer:**\n\n{res['final_answer']}")

with col_right:
    st.subheader("⚠️ Tool Failure & Recovery Panel")
    failed_calls = [tc for tc in res.get("tool_calls", []) if tc["status"] != "✓"]
    if failed_calls:
        for fc in failed_calls:
            st.error(f"Failed Tool: `{fc['tool']}` with args {fc['args']}")
            st.write(f"**Reported Error:** {fc['result'].get('error')}")
        if res.get("recovered"):
            st.success("✅ **Agent Successfully Recovered**: Gracefully handled missing/invalid data without crash.")
        else:
            st.warning("Agent did not initiate recovery procedure.")
    else:
        st.success("No tool failures observed in this trace. All tools returned HTTP 200 / Success.")

st.divider()

# Case Study & Matrix
st.subheader("Case Study: Fault Injection Matrix (E-commerce Agent)")
case_study_df = pd.DataFrame([
    {"Fault Type": "Invalid Order Number", "Expected Action": "get_order fails", "Recovery": "Prompt for re-entry", "Pass Rate": "95%"},
    {"Fault Type": "Missing Arguments", "Expected Action": "Validation error", "Recovery": "Query fallback database", "Pass Rate": "88%"},
    {"Fault Type": "Tool 500 Failure", "Expected Action": "Network timeout", "Recovery": "Exponential retry backoff", "Pass Rate": "91%"},
])
st.table(case_study_df)
