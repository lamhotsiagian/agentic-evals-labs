"""Chapter 6: Multi-Agent Evals -- type a research topic, watch the handoff chain and claim provenance."""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-06-multi-agent-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from system import MultiAgentResearchSystem
from evaluator import MultiAgentEvaluator

chapter_page_header(6, "Multi-Agent Evals", "Type a research topic -- watch handoffs, claim provenance, and synthesis fidelity.", icon="\U0001F91D")

provider = get_model_provider()
provider_status_badge(provider)
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b"], key="ch6_model")
inject_fault = st.sidebar.checkbox("Inject a dropped handoff (Analyst -> Synthesizer)", key="ch6_inject")

topic = st.text_input("Research topic", value="Evolution of LLM-as-a-Judge Techniques (2024-2026)", key="ch6_topic")

if st.button("\U0001F680 Run the research team", type="primary", key="ch6_run_btn"):
    try:
        require_live_provider(provider)
    except ProviderUnavailableError as e:
        st.error(str(e))
    else:
        system = MultiAgentResearchSystem(provider=provider, model=model)
        evaluator = MultiAgentEvaluator()
        with st.spinner("Supervisor is delegating to Researcher and Analyst..."):
            run = system.run_research(topic, inject_handoff_failure=inject_fault)
            scores = evaluator.evaluate_system_run(run)
        st.session_state.ch6_run = run
        st.session_state.ch6_scores = scores

if "ch6_run" in st.session_state:
    run, scores = st.session_state.ch6_run, st.session_state.ch6_scores
    metrics_row({k: v.score for k, v in scores.items() if k != "total_messages"})
    for name, m in scores.items():
        if m.reasoning:
            st.caption(f"**{name}**: {m.reasoning}")

    st.subheader("Message trace")
    for m in run["messages"]:
        icon = "✅" if m.handoff_status == "success" else "❌"
        with st.chat_message("assistant" if m.sender != "Supervisor" else "user"):
            st.markdown(f"{icon} **{m.sender} → {m.recipient}**")
            st.write(m.content)

    st.info(f"**Final synthesis:**\n\n{run['final_synthesis']}")
