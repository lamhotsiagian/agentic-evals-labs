"""Chapter 1: Evaluation Foundations -- live chat + honest regression suite."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-01-evaluation-foundations")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.metrics.stats import sliced_report
from shared.ui.components import chapter_page_header, provider_status_badge, render_message_history, metrics_row
from shared.datasets.loader import load_customer_support_cases

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from agent import CustomerSupportAgent
from evaluator import CustomerSupportEvaluator

chapter_page_header(1, "Evaluation Foundations", "Customer support agent -- type a question, get a live-scored answer.")

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], key="ch1_model")
provider = get_model_provider()
is_live = provider_status_badge(provider)

evaluator = CustomerSupportEvaluator()

tab_chat, tab_suite = st.tabs(["Live Chat", "Regression Suite"])

with tab_chat:
    st.caption("This box calls your local Ollama model directly -- nothing here is canned.")
    if "ch1_history" not in st.session_state:
        st.session_state.ch1_history = []

    render_message_history(st.session_state.ch1_history)

    user_msg = st.chat_input("Ask the support agent a billing/account question...", key="ch1_chat_input")
    if user_msg:
        st.session_state.ch1_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                require_live_provider(provider)
                agent = CustomerSupportAgent(model_name=model, provider=provider)
                with st.spinner(f"{model} is responding..."):
                    resp = agent.respond(user_msg)
                st.markdown(resp["response"])

                from shared.models.schemas import EvaluationCase
                case = EvaluationCase(id=f"live-{int(time.time())}", input_prompt=user_msg)
                result = evaluator.evaluate_case(case, resp["response"], latency_seconds=resp["latency_seconds"], model_name=model)
                scores = {k: v.score for k, v in result.metrics.items()}
                metrics_row(scores, passed=result.passed)
                st.caption("No reference answer exists for a live typed question, so correctness here "
                           "falls back to the hallucination check alone. Run the Regression Suite tab "
                           "for reference-graded scoring with uncertainty intervals.")

                st.session_state.ch1_history.append({
                    "role": "assistant", "content": resp["response"],
                })
            except ProviderUnavailableError as e:
                st.error(str(e))

with tab_suite:
    st.caption("Runs the fixed grader against the labeled dataset (real reference answers), "
               "with Wilson-interval uncertainty per category -- replaces the old static "
               "'Model Comparison' table with numbers computed from this run.")
    compare_models = st.multiselect(
        "Models to run (select 2+ to compare head-to-head)",
        ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], default=[model], key="ch1_compare_models",
    )
    n_cases = st.slider("Number of cases", 3, 50, 10, key="ch1_n_cases")
    if st.button("Run suite", type="primary", key="ch1_run_suite"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            cases = load_customer_support_cases(count=n_cases)
            comparison_rows = []
            for m in compare_models:
                agent = CustomerSupportAgent(model_name=m, provider=provider)
                with st.spinner(f"Evaluating {n_cases} cases on {m}..."):
                    results = evaluator.run_suite(cases, agent)
                total = len(results)
                passed = sum(1 for r in results if r.passed)
                avg_latency = sum(r.latency_seconds for r in results) / total if total else 0.0
                comparison_rows.append({
                    "Model": m, "Pass Rate": f"{passed/total*100:.1f}%" if total else "n/a",
                    "Avg Latency (s)": round(avg_latency, 2), "n": total,
                })
                st.session_state[f"ch1_last_results_{m}"] = results

            st.subheader("Head-to-head (this run, not a fixed table)")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            for m in compare_models:
                results = st.session_state[f"ch1_last_results_{m}"]
                category_by_id = {c.id: c.category for c in cases}
                rows = [{"category": category_by_id.get(r.case_id, "general"), "passed": r.passed} for r in results]
                with st.expander(f"{m}: sliced pass rate with 95% Wilson interval"):
                    st.dataframe(pd.DataFrame(sliced_report(rows, slice_key="category", min_n=10)),
                                 use_container_width=True, hide_index=True)
