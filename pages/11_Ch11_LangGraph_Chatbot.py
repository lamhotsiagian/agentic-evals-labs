"""Chapter 11: Evaluating an Agentic Chatbot using DeepEval, Ragas, Langsmith, and Trulens.

Demonstrates multi-dimensional agent evaluation across four pillars:
1. DeepEval: Behavioral assertions and G-Eval criteria scoring.
2. Ragas: Context Precision, Recall, and Faithfulness triad.
3. LangSmith: Hierarchical run-tree inspection and trajectory milestone auditing.
4. TruLens: Groundedness triad feedback functions and tool call correctness.
"""

import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-11-langgraph-chatbot")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.models.provider import get_model_provider
from shared.ui.components import chapter_page_header, metrics_row, provider_status_badge, render_message_history
from shared.ui.components import load_chapter_modules
from shared.datasets.loader import load_langgraph_support_cases

load_chapter_modules(CHAPTER_DIR)

from graph import CustomerSupportGraph
from evaluators import DeepEvalEngine, LangSmithTraceEngine, RagasEngine, TruLensFeedbackEngine

chapter_page_header(
    11, "Evaluating an Agentic Chatbot using DeepEval, Ragas, Langsmith, and Trulens",
    "Customer support agent orchestrating StateGraph routing, RAG retrieval, and operational tools. "
    "Evaluated across four pillars: DeepEval (unit tests), Ragas (RAG triad), LangSmith (run trees), and TruLens (groundedness).",
)

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], key="ch11_model")
provider = get_model_provider()
is_live = provider_status_badge(provider)

st.sidebar.header("Evaluation SLA Thresholds")
max_tool_lat = st.sidebar.slider("Max Tool Span Latency (ms)", 500, 5000, 2500, 250, key="ch11_tool_lat")
max_llm_lat = st.sidebar.slider("Max LLM Span Latency (ms)", 1000, 10000, 6000, 500, key="ch11_llm_lat")

graph = CustomerSupportGraph(provider=provider, model=model)
langsmith_engine = LangSmithTraceEngine(max_tool_latency_ms=max_tool_lat, max_llm_latency_ms=max_llm_lat)
trulens_engine = TruLensFeedbackEngine()
deepeval_engine = DeepEvalEngine()
ragas_engine = RagasEngine()

tab_chat, tab_benchmark, tab_diagnostics = st.tabs(
    ["Live Agent & Run Tree", "DeepEval & Ragas Benchmark", "LangSmith & TruLens Diagnostics"]
)

# ---------------------------------------------------------------------
# Tab 1: Live Agent & LangSmith Run Tree
# ---------------------------------------------------------------------
with tab_chat:
    st.caption("Chat live with the LangGraph state machine. Every turn records LangSmith run trees and TruLens feedback scores.")
    if "ch11_history" not in st.session_state:
        st.session_state.ch11_history = []
    if "ch11_last_state" not in st.session_state:
        st.session_state.ch11_last_state = None

    render_message_history(st.session_state.ch11_history)

    user_msg = st.chat_input(
        "Ask about billing, refunds, rate limits, tracking, or security policies...",
        key="ch11_chat_input",
    )

    if user_msg:
        st.session_state.ch11_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("LangGraph agent executing state nodes and emitting run spans..."):
                state = graph.invoke(user_msg)
            st.markdown(state["final_answer"])
            st.session_state.ch11_last_state = state
            st.session_state.ch11_history.append({"role": "assistant", "content": state["final_answer"]})

    if st.session_state.ch11_last_state:
        st.divider()
        last_s = st.session_state.ch11_last_state
        ls_eval = langsmith_engine.evaluate_run_tree(last_s["trace_spans"])
        tl_eval = trulens_engine.evaluate_feedback(
            query=last_s["user_message"],
            response=last_s["final_answer"],
            retrieved_contexts=[d["content"] for d in last_s.get("retrieved_docs", [])],
            tool_results=last_s.get("tool_results", []),
        )
        de_eval = deepeval_engine.evaluate(last_s["user_message"], last_s["final_answer"])

        st.subheader("Turn Evaluation Summary")
        metrics_row(
            {
                "langsmith_status": ls_eval["verdict"],
                "trulens_groundedness_%": tl_eval["trulens_groundedness"].score,
                "deepeval_relevancy_%": de_eval["deepeval_relevancy"].score,
                "total_run_spans": len(last_s["trace_spans"]),
            },
            passed=ls_eval["verdict"] == "PASSED" and tl_eval["trulens_groundedness"].passed,
        )

        with st.expander("LangSmith Hierarchical Run Tree", expanded=True):
            spans_df = pd.DataFrame([
                {
                    "span_id": s.get("span_id", ""),
                    "parent_id": s.get("parent_span_id") or "[ROOT]",
                    "name": s.get("name", ""),
                    "duration_ms": s.get("duration_ms", 0.0),
                    "status": s.get("status", "OK"),
                    "key_attributes": str({k: v for k, v in s.get("attributes", {}).items() if not k.startswith("retrieval.query")})[:80],
                }
                for s in last_s["trace_spans"]
            ])
            st.dataframe(spans_df, use_container_width=True, hide_index=True)

        with st.expander("TruLens Groundedness & Feedback Functions", expanded=False):
            fb_df = pd.DataFrame([
                {"feedback_function": m.name, "score": f"{m.score}%", "passed": m.passed, "details": m.reasoning}
                for m in tl_eval.values()
            ])
            st.dataframe(fb_df, use_container_width=True, hide_index=True)

        if last_s["tool_results"]:
            with st.expander(f"Executed Tools ({len(last_s['tool_results'])})"):
                st.json(last_s["tool_results"])

        if last_s["retrieved_docs"]:
            with st.expander(f"Retrieved Context ({len(last_s['retrieved_docs'])} documents)"):
                for d in last_s["retrieved_docs"]:
                    st.markdown(f"**[{d['doc_id']}] {d['title']}** ({d['status']})")
                    st.caption(d["content"][:250] + "...")


# ---------------------------------------------------------------------
# Tab 2: DeepEval & Ragas Benchmark Suite
# ---------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Customer Support Golden Benchmark (12 Test Cases)")
    st.caption("Evaluates the agent against shared/datasets/data/langgraph_customer_support.jsonl using all four frameworks.")

    benchmark_cases = load_langgraph_support_cases()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_suite_btn = st.button("Run Full Benchmark Suite", type="primary", key="ch11_run_suite")

    if run_suite_btn or "ch11_suite_results" not in st.session_state:
        results = []
        with st.spinner("Running LangGraph agent on golden test cases..."):
            for case in benchmark_cases:
                state = graph.invoke(case["input_prompt"])
                ans = state["final_answer"]

                de_res = deepeval_engine.evaluate(
                    prompt=case["input_prompt"],
                    actual_output=ans,
                    expected_output=case.get("expected_output"),
                    retrieved_context=case.get("reference_context"),
                )
                rag_res = ragas_engine.evaluate(
                    question=case["input_prompt"],
                    answer=ans,
                    retrieved_contexts=[d["content"] for d in state["retrieved_docs"]] or [case.get("reference_context", "")],
                    reference_context=case.get("reference_context"),
                )
                ls_res = langsmith_engine.evaluate_run_tree(state["trace_spans"], expected_intent=case.get("expected_intent"))
                tl_res = trulens_engine.evaluate_feedback(
                    query=case["input_prompt"],
                    response=ans,
                    retrieved_contexts=[d["content"] for d in state["retrieved_docs"]] or [case.get("reference_context", "")],
                    tool_results=state.get("tool_results", []),
                )

                results.append({
                    "case_id": case["id"],
                    "category": case["category"],
                    "expected_intent": case.get("expected_intent", ""),
                    "predicted_intent": state["intent"],
                    "deepeval_correctness_%": de_res["deepeval_correctness"].score,
                    "deepeval_relevancy_%": de_res["deepeval_relevancy"].score,
                    "ragas_faithfulness_%": rag_res["ragas_faithfulness"].score,
                    "ragas_precision_%": rag_res["ragas_context_precision"].score,
                    "trulens_groundedness_%": tl_res["trulens_groundedness"].score,
                    "langsmith_verdict": ls_res["verdict"],
                })
        st.session_state.ch11_suite_results = results

    res_df = pd.DataFrame(st.session_state.ch11_suite_results)

    avg_de_corr = round(res_df["deepeval_correctness_%"].mean(), 1)
    avg_de_rel = round(res_df["deepeval_relevancy_%"].mean(), 1)
    avg_rag_faith = round(res_df["ragas_faithfulness_%"].mean(), 1)
    avg_tl_ground = round(res_df["trulens_groundedness_%"].mean(), 1)
    pass_rate = round((res_df["langsmith_verdict"] == "PASSED").mean() * 100.0, 1)

    metrics_row(
        {
            "deepeval_correctness_%": avg_de_corr,
            "deepeval_relevancy_%": avg_de_rel,
            "ragas_faithfulness_%": avg_rag_faith,
            "trulens_groundedness_%": avg_tl_ground,
            "langsmith_pass_rate_%": pass_rate,
        },
        passed=pass_rate >= 80.0,
    )

    st.dataframe(res_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Tab 3: LangSmith & TruLens Diagnostics
# ---------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("LangSmith & TruLens Architecture Comparison")
    st.markdown(
        """
This tab contrasts the complementary operational roles of **LangSmith** and **TruLens**:

1. **LangSmith (Run Trees & Trajectories)**:
   - Evaluates the structural execution DAG of the LangGraph state machine.
   - Validates node transitions, ensures loop cycle thresholds ($N \\le 3$) are respected, and verifies tool argument schemas.

2. **TruLens (Feedback Functions & Groundedness Triad)**:
   - Evaluates continuous programmatic feedback functions over live agent turns.
   - Measures the **Groundedness Triad**: Context Relevance, Groundedness (source support), and Answer Relevance.
   - Tracks intermediate tool correctness and detects unauthorized side effects.
        """
    )

    if st.session_state.get("ch11_last_state"):
        last_s = st.session_state.ch11_last_state
        ls_audit = langsmith_engine.evaluate_run_tree(last_s["trace_spans"])
        tl_audit = trulens_engine.evaluate_feedback(
            query=last_s["user_message"],
            response=last_s["final_answer"],
            retrieved_contexts=[d["content"] for d in last_s.get("retrieved_docs", [])],
            tool_results=last_s.get("tool_results", []),
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### LangSmith Trajectory Diagnostics")
            st.markdown(f"**Trajectory Verdict:** `{ls_audit['verdict']}` (Compliance: {ls_audit['compliance_score']}%)")
            for diag in ls_audit["diagnostics"]:
                status_color = "green" if diag["status"] == "PASS" else "red"
                st.markdown(f"- **{diag['rule']}**: :{status_color}[{diag['status']}] -- {diag['message']}")

        with col_right:
            st.markdown("#### TruLens Groundedness Radar")
            st.markdown(f"**Overall TruLens Score:** `{tl_audit['trulens_overall'].score}%`")
            for k, m in tl_audit.items():
                if k != "trulens_overall":
                    status_color = "green" if m.passed else "red"
                    st.markdown(f"- **{m.name}**: :{status_color}[{m.score}%] -- {m.reasoning}")
    else:
        st.info("Execute a chat message in Tab 1 or run the benchmark in Tab 2 to inspect LangSmith and TruLens diagnostics.")
