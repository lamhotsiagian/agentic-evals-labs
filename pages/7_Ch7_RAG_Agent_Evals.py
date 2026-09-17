"""Chapter 7: RAG Agent Evals -- live chunk-retrieval chat + claim-level audit."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-07-rag-agent-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, render_message_history, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from pipeline import RAGAgentPipeline
from evaluator import RAGEvaluator
from graders import Chunk, claim_audit, load_chunks

chapter_page_header(
    7, "RAG Agent Evals",
    "Enterprise knowledge assistant -- ask a real question, see the retrieved chunks, "
    "citations, and a claim-level audit (not whole-document lexical overlap).",
)

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Generator model", ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], key="ch7_model")
top_k = st.sidebar.slider("Chunks to retrieve (top_k)", 1, 5, 3, key="ch7_top_k")
provider = get_model_provider()
is_live = provider_status_badge(provider)

evaluator = RAGEvaluator()


@st.cache_resource(show_spinner="Indexing the enterprise knowledge base...")
def _get_pipeline(_provider, generator_model: str, provider_identity: str):
    chunks = load_chunks()
    pipeline = RAGAgentPipeline(provider=_provider, generator_model=generator_model)
    pipeline.index(chunks)
    return pipeline, chunks


from shared.models.provider import provider_identity as _provider_identity_fn
pipeline, all_chunks = _get_pipeline(provider, model, _provider_identity_fn(provider))

with st.sidebar.expander(f"Knowledge base: {len(all_chunks)} indexed chunks"):
    kb_df = pd.DataFrame([
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "status": c.status, "effective": c.effective}
        for c in all_chunks
    ])
    st.dataframe(kb_df, use_container_width=True, hide_index=True)
    st.caption("Every markdown file in the knowledge base is indexed, including the "
               "obsolete legacy distractor -- it used to be skipped entirely.")

tab_chat, tab_suite, tab_known = st.tabs(
    ["\U0001F4AC Live Chat", "\U0001F4CA Regression Suite", "\U0001F50D Known-Failure Checks"]
)

with tab_chat:
    st.caption("This box calls your local Ollama model directly -- nothing here is canned.")
    if "ch7_history" not in st.session_state:
        st.session_state.ch7_history = []

    render_message_history(st.session_state.ch7_history)

    user_msg = st.chat_input("Ask about HR, security, finance, or engineering policy...", key="ch7_chat_input")
    if user_msg:
        st.session_state.ch7_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                require_live_provider(provider)
                with st.spinner(f"{model} is retrieving and answering..."):
                    res = pipeline.query(user_msg, top_k=top_k)
                st.markdown(res["answer"])

                st.markdown("**Retrieved chunks**")
                retrieved_df = pd.DataFrame([
                    {
                        "chunk_id": d["chunk_id"], "status": d["status"],
                        "similarity": d["similarity_score"], "preview": d["content"][:90] + "...",
                    }
                    for d in res["retrieved_docs"]
                ])
                st.dataframe(retrieved_df, use_container_width=True, hide_index=True)
                if any(d["status"] == "obsolete" for d in res["retrieved_docs"]):
                    st.warning("An obsolete/superseded document was retrieved into context. "
                               "The claim audit below fails any sentence that cites it.")

                chunk_lookup = {
                    d["chunk_id"]: Chunk(d["chunk_id"], d["doc_id"], d["title"], d["content"], d["status"], d.get("effective", ""))
                    for d in res["retrieved_docs"]
                }
                audit = claim_audit(res["answer"], chunk_lookup)
                scores = evaluator.evaluate_rag_output(
                    question=user_msg, retrieved_docs=res["retrieved_docs"], answer=res["answer"],
                )
                metrics_row({
                    "citation_coverage": scores["citation_coverage"].score,
                    "supported_claim_rate": scores["supported_claim_rate"].score,
                    "groundedness": scores["groundedness"].score,
                    "context_relevance": scores["context_relevance"].score,
                })
                with st.expander("Per-sentence claim audit"):
                    st.dataframe(pd.DataFrame(audit["rows"]), use_container_width=True, hide_index=True)

                st.session_state.ch7_history.append({"role": "assistant", "content": res["answer"]})
            except ProviderUnavailableError as e:
                st.error(str(e))

with tab_suite:
    st.caption("Runs a fixed set of real questions against the indexed knowledge base and "
               "reports chunk-level retrieval ranking metrics (recall@k, precision@k relative "
               "to the number of relevant chunks, MRR) plus claim-audit groundedness -- computed "
               "from this run, not a static table.")
    eval_questions = [
        {"question": "What is our customer refund policy?", "expected_doc_ids": ["DOC-FIN-303"]},
        {"question": "What are the password and MFA requirements?", "expected_doc_ids": ["DOC-SEC-202"]},
        {"question": "What is the remote work equipment stipend?", "expected_doc_ids": ["DOC-HR-101"]},
        {"question": "What is the Sev-1 incident paging SLA?", "expected_doc_ids": ["DOC-ENG-404"]},
        {"question": "What is the CEO's approved private jet travel budget?", "expected_doc_ids": []},
    ]
    if st.button("Run suite", type="primary", key="ch7_run_suite"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            rows = []
            for case in eval_questions:
                with st.spinner(f"Evaluating: {case['question']}"):
                    res = pipeline.query(case["question"], top_k=top_k)
                    scores = evaluator.evaluate_rag_output(
                        question=case["question"], retrieved_docs=res["retrieved_docs"],
                        answer=res["answer"], expected_doc_ids=case["expected_doc_ids"] or None,
                        k=top_k,
                    )
                is_unanswerable = not case["expected_doc_ids"]
                abstained = "don't know" in res["answer"].lower() or "cannot" in res["answer"].lower()
                rows.append({
                    "question": case["question"],
                    "answer": res["answer"][:80] + ("..." if len(res["answer"]) > 80 else ""),
                    "recall@k": scores["retrieval_recall"].score,
                    "precision@k": scores["retrieval_precision"].score,
                    "mrr": scores["retrieval_mrr"].score,
                    "groundedness": scores["groundedness"].score,
                    "cites_obsolete": scores["cites_obsolete_evidence"].score == 1.0,
                    "correctly_abstained" if is_unanswerable else "passed": (
                        abstained if is_unanswerable else scores["groundedness"].passed
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("The last row has no relevant document in the corpus -- a live model should "
                       "abstain rather than guess (an earlier version of this lab never tested abstention).")

with tab_known:
    st.caption(
        "Replays known failure cases (a 90-day claim against a 30-day policy, a citation to "
        "the obsolete document, an uncited answer) directly against the fixed evaluator -- deterministic, no model "
        "call needed, so this always runs even with Ollama offline."
    )
    finance_chunks = [c for c in all_chunks if c.doc_id == "DOC-FIN-303"]
    noise_chunks = [c for c in all_chunks if c.doc_id == "DOC-NOISE-999"]
    retrieved_for_probe = {c.chunk_id: c for c in finance_chunks + noise_chunks}

    probes = [
        ("correct + cited", "New subscriptions canceled within 30 calendar days qualify for a full money-back guarantee [DOC-FIN-303]."),
        ("wrong + uncited (90-day claim)", "Refunds are available within 90 calendar days to the original payment method."),
        ("stale-cited (cites the 2018 obsolete policy)", "All subscriptions are non-refundable [DOC-NOISE-999]."),
    ]
    probe_rows = []
    for label, answer in probes:
        audit = claim_audit(answer, retrieved_for_probe)
        probe_rows.append({
            "probe": label, "answer": answer,
            "citation_coverage": audit["citation_coverage"],
            "supported_claim_rate": audit["supported_claim_rate"],
            "verdict": "PASS" if audit["supported_claim_rate"] >= 0.8 else "FAIL (correctly caught)",
        })
    st.dataframe(pd.DataFrame(probe_rows), use_container_width=True, hide_index=True)
    st.caption("Under the old whole-context lexical scoring, the wrong 90-day answer scored "
               "faithfulness 1.0 / groundedness 0.92, and citing the obsolete distractor was "
               "rewarded as a valid citation. Both now fail the claim audit.")
