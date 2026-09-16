"""Chapter 1: Agent Evaluation Dashboard (Streamlit)."""

import sys
import os

# Add root and current dir to path for resilient imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import pandas as pd
from agent import CustomerSupportAgent
from evaluator import CustomerSupportEvaluator
from shared.datasets.loader import load_customer_support_cases

st.set_page_config(page_title="Agent Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Agent Evaluation Dashboard")
st.caption("Chapter 1: Agent Evaluation Fundamentals — Evaluating Customer Support Agents")

# Sidebar
st.sidebar.header("Evaluation Settings")
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"],
    index=0,
    key="model_selector",
)
test_count = st.sidebar.slider("Number of Test Cases", min_value=3, max_value=50, value=3, step=1, key="test_count_slider")
run_eval_btn = st.sidebar.button("🚀 Run Evaluation", type="primary", key="run_eval_btn")

# Initialize state
if "eval_results" not in st.session_state or run_eval_btn:
    agent = CustomerSupportAgent(model_name=selected_model)
    evaluator = CustomerSupportEvaluator()
    cases = load_customer_support_cases(count=test_count)
    with st.spinner(f"Evaluating {test_count} cases on {selected_model}..."):
        st.session_state.eval_results = evaluator.run_suite(cases, agent)
        st.session_state.current_model = selected_model

results = st.session_state.eval_results
total_tasks = len(results)
passed_tasks = sum(1 for r in results if r.passed)
failed_tasks = total_tasks - passed_tasks
success_rate = (passed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
avg_latency = sum(r.latency_seconds for r in results) / total_tasks if total_tasks > 0 else 0.0
total_cost = sum(r.cost_estimate_usd for r in results)

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tasks Evaluated", f"{total_tasks}")
col2.metric("Success Rate", f"{success_rate:.1f}%", f"{'+' if success_rate >= 80 else ''}{success_rate - 80:.1f}% vs Target")
col3.metric("Avg Latency", f"{avg_latency:.2f}s")
col4.metric("Avg Cost", f"${total_cost:.4f}*")

st.divider()

# Success vs Failure Distribution
st.subheader("Evaluation Breakdown")
c_left, c_right = st.columns([2, 1])

with c_left:
    dist_df = pd.DataFrame({
        "Status": ["Success", "Failure"],
        "Count": [passed_tasks, failed_tasks],
        "Percentage": [f"{success_rate:.1f}%", f"{100 - success_rate:.1f}%"],
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

with c_right:
    st.progress(success_rate / 100.0)
    st.write(f"**Passed**: {passed_tasks} / {total_tasks} ({success_rate:.1f}%)")
    st.caption("*Local Ollama model cost is $0.00.")

st.divider()

# Tabs for inspection
tab_all, tab_failures, tab_comparison = st.tabs(["📋 View Test Cases", "⚠️ View Failures", "⚖️ Model Comparison"])

with tab_all:
    records = []
    for r in results:
        records.append({
            "ID": r.case_id,
            "Prompt": r.input_prompt,
            "Response": r.actual_output,
            "Passed": "✅ Yes" if r.passed else "❌ No",
            "Latency (s)": r.latency_seconds,
            "Tokens": r.tokens_used,
        })
    st.dataframe(pd.DataFrame(records), use_container_width=True, key="all_cases_table")

with tab_failures:
    failures = [r for r in results if not r.passed]
    if failures:
        for f in failures:
            with st.expander(f"Case {f.case_id} — Failed"):
                st.write(f"**Prompt:** {f.input_prompt}")
                st.write(f"**Output:** {f.actual_output}")
                st.json({k: v.model_dump() for k, v in f.metrics.items()})
    else:
        st.success("🎉 All test cases passed! No regressions or hallucinations detected.")

with tab_comparison:
    st.write("### Model Quality & Latency Tradeoff (Case Study)")
    benchmark_data = pd.DataFrame({
        "Model": ["Qwen2.5:3B", "Qwen3:1.7B", "Llama3.2:1B"],
        "Success Rate": ["92.0%", "86.0%", "78.0%"],
        "Avg Latency": ["2.8s", "1.4s", "0.9s"],
        "Correctness": [4.7, 4.2, 3.8],
        "Best Use Case": ["Primary Agent & Complex Tooling", "Lightweight Critic & Routing", "Fast Regression Tests"]
    })
    st.table(benchmark_data)
