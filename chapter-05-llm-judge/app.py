"""Chapter 5: LLM-as-a-Judge Evaluation & Calibration Dashboard (Streamlit)."""

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
from judge import MultiJudgeSystem
from calibration import compute_calibration_metrics
from shared.datasets.loader import load_judge_benchmark_cases

st.set_page_config(page_title="LLM-as-a-Judge Evaluation", page_icon="⚖️", layout="wide")

st.title("⚖️ LLM-as-a-Judge Evaluation & Meta-Calibration")
st.caption("Chapter 5: Multi-Dimensional Rubric Judging and Evaluating the Evaluator")

benchmarks = load_judge_benchmark_cases()
bench_options = {f"{b['id']}: {b['prompt'][:40]}...": b for b in benchmarks}

st.sidebar.header("Judge Controls")
selected_sample_key = st.sidebar.selectbox("Select Test Scenario", list(bench_options.keys()))
selected_sample = bench_options[selected_sample_key]

judge_model = st.sidebar.selectbox("Primary Judge Model", ["qwen3:1.7b", "qwen2.5:3b", "llama3.2:1b"], index=0)
run_judge_btn = st.sidebar.button("🧑‍⚖️ Evaluate with LLM Judge", type="primary", key="run_judge_btn")

if "current_eval" not in st.session_state or run_judge_btn:
    system = MultiJudgeSystem(primary_model=judge_model, alt_model="llama3.2:1b")
    with st.spinner("Invoking LLM Judge..."):
        eval_pair = system.evaluate_pair(
            prompt=selected_sample["prompt"],
            response=selected_sample["response"],
        )
        st.session_state.current_eval = eval_pair["primary"]
        st.session_state.alt_eval = eval_pair["alternative"]

eval_res = st.session_state.current_eval

# Main interface layout
tab_eval, tab_calibration = st.tabs(["📋 Judge Scorecard", "🔬 Meta-Evaluation (Judge Calibration)"])

with tab_eval:
    col_input, col_score = st.columns([1, 1])

    with col_input:
        st.subheader("Agent Conversation")
        st.markdown(f"**User Prompt:**\n\n> {selected_sample['prompt']}")
        st.markdown(f"**Agent Response:**\n\n{selected_sample['response']}")
        st.caption(f"Evaluated by Judge Model: `{eval_res.judge_model}`")

    with col_score:
        st.subheader("Judge Scorecard")
        st.metric("Overall Score", f"{eval_res.overall_score:.2f} / 5.0", "Pass" if eval_res.passed else "Fail")

        score_rows = []
        for dim, score in eval_res.dimension_scores.items():
            score_rows.append({"Dimension": dim.replace("_", " ").title(), "Score (0-5)": f"{score:.1f}"})
        st.table(pd.DataFrame(score_rows))

    # Collapsible Judge Reasoning
    with st.expander("🔍 View Judge Reasoning & Evidence Citations", expanded=True):
        st.markdown(f"**Judge Reasoning:**\n\n{eval_res.reasoning}")
        if eval_res.evidence:
            st.markdown("**Evidence Quotes:**")
            for ev in eval_res.evidence:
                st.markdown(f"- *\"{ev}\"*")

with tab_calibration:
    st.subheader("Evaluating the Evaluator (Calibration vs Human Ground Truth)")
    st.info("A core lesson in AI engineering: **A judge itself requires rigorous evaluation.**")

    # Run calibration across benchmark suite (cached in session state for performance)
    if "suite_evals" not in st.session_state or run_judge_btn:
        calib_system = MultiJudgeSystem(primary_model=judge_model)
        st.session_state.suite_evals = [calib_system.primary_judge.evaluate(b["prompt"], b["response"]) for b in benchmarks]
        st.session_state.calib_metrics = compute_calibration_metrics(st.session_state.suite_evals, benchmarks)

    suite_evals = st.session_state.suite_evals
    metrics = st.session_state.calib_metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Human Agreement", f"{metrics['agreement_pct']}%")
    c2.metric("Pearson Correlation", f"{metrics['pearson_correlation']:.2f}")
    c3.metric("False Positives", f"{metrics['false_positives']}")
    c4.metric("False Negatives", f"{metrics['false_negatives']}")

    st.write("### Benchmark Comparison Matrix")
    matrix_rows = []
    for i, b in enumerate(benchmarks):
        h_avg = sum(b["human_scores"].values()) / len(b["human_scores"])
        j_score = suite_evals[i].overall_score
        matrix_rows.append({
            "Case ID": b["id"],
            "Task": b["prompt"][:35] + "...",
            "Human Golden Score": f"{h_avg:.2f}",
            "LLM Judge Score": f"{j_score:.2f}",
            "Delta": f"{abs(h_avg - j_score):.2f}",
            "Verdict Match": "✅ Match" if (h_avg >= 3.5) == (j_score >= 3.5) else "❌ Mismatch",
        })
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
