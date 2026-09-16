"""Chapter 7: RAG Evaluation Explorer (Streamlit)."""

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
from pipeline import RAGAgentPipeline
from evaluator import RAGEvaluator
from shared.datasets.loader import load_rag_enterprise_corpus

st.set_page_config(page_title="RAG Evaluation Explorer", page_icon="📚", layout="wide")

st.title("📚 RAG Evaluation Explorer")
st.caption("Chapter 7: Evaluating Dense Retrieval, Faithfulness, Groundedness, and Citations")

corpus = load_rag_enterprise_corpus()

queries = {
    "What is our customer refund policy?": {"expected": ["DOC-FIN-303"]},
    "What is the annual home office and internet stipend?": {"expected": ["DOC-HR-101"]},
    "What are the minimum password length and MFA requirements?": {"expected": ["DOC-SEC-202"]},
    "What is the page escalation SLA for Severity 1 outages?": {"expected": ["DOC-ENG-404"]},
}

# Sidebar
st.sidebar.header("RAG Settings")
selected_q = st.sidebar.selectbox("Select Sample Question", list(queries.keys()))
expected_ids = queries[selected_q]["expected"]

inject_noise = st.sidebar.checkbox("Inject Distractor & Outdated Docs", value=False, key="noise_chk")
top_k_val = st.sidebar.slider("Top K Chunks", min_value=1, max_value=4, value=3)
run_rag_btn = st.sidebar.button("🔍 Run RAG Pipeline", type="primary", key="run_rag_btn")

if "rag_res" not in st.session_state or run_rag_btn:
    pipeline = RAGAgentPipeline()
    evaluator = RAGEvaluator()

    full_corpus = list(corpus)
    if inject_noise:
        full_corpus.append({
            "doc_id": "DOC-NOISE-999",
            "department": "Legacy",
            "title": "Deprecated 2018 Policy",
            "content": "All refunds are strictly non-refundable under any conditions.",
        })

    pipeline.retriever.index_documents(full_corpus)

    with st.spinner("Retrieving semantic contexts & generating answer..."):
        res = pipeline.query(selected_q, top_k=top_k_val)
        scores = evaluator.evaluate_rag_output(
            question=selected_q,
            retrieved_docs=res["retrieved_docs"],
            answer=res["answer"],
            expected_doc_ids=expected_ids,
        )
        st.session_state.rag_res = res
        st.session_state.rag_scores = scores

res = st.session_state.rag_res
scores = st.session_state.rag_scores

# Metric Row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Retrieval Precision", f"{scores['retrieval_precision'].score * 100:.0f}%")
m2.metric("Retrieval Recall", f"{scores['retrieval_recall'].score * 100:.0f}%")
m3.metric("Faithfulness", f"{scores['faithfulness'].score * 100:.0f}%")
m4.metric("Groundedness", f"{scores['groundedness'].score * 100:.0f}%")
m5.metric("Citation Correctness", f"{scores['citation_correctness'].score * 100:.0f}%")

st.divider()

col_query, col_docs = st.columns([1, 1])

with col_query:
    st.subheader("Question & Augmented Answer")
    st.markdown(f"**Question:**\n> {res['question']}")
    st.markdown(f"**Generated Answer:**\n\n{res['answer']}")

    st.subheader("RAG Metric Explanations")
    st.markdown("""
    - **Retrieval Precision**: Ratio of retrieved documents that contain golden ground-truth facts.
    - **Faithfulness**: Proportion of generated assertions strictly supported by retrieved context.
    - **Groundedness**: Composite score measuring hallucination absence and reference alignment.
    - **Citation Correctness**: Verification that referenced document IDs exist in the retrieved set.
    """)

with col_docs:
    st.subheader("Retrieved Documents")
    for doc in res["retrieved_docs"]:
        is_golden = doc["doc_id"] in expected_ids
        icon = "✓" if is_golden else "✗"
        badge = "Golden Evidence" if is_golden else "Distractor / Secondary Context"
        
        with st.container():
            st.markdown(f"**`[{doc['doc_id']}]` {doc['title']}** — `{icon} {badge}` (Score: {doc['similarity_score']})")
            st.caption(f"Department: {doc['department']}")
            st.write(doc["content"])
            st.divider()
