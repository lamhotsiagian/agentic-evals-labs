"""Chapter 3: Tool-Call Evals -- type a request, watch the model choose real tools."""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-03-tool-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, render_message_history, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from agent import EcommerceCustomerAgent
from evaluator import ToolCallingEvaluator

chapter_page_header(3, "Tool-Call Evals", "E-commerce support agent -- native tool calling, schema-aware grading.", icon="\U0001F6E0️")

provider = get_model_provider()
provider_status_badge(provider)
model = st.sidebar.selectbox("Model (needs tool-calling support)", ["qwen2.5:3b", "qwen3:1.7b"], key="ch3_model")
st.sidebar.caption("Try: order lookups (#1234 is real, #99999 is not), weather for a zip, or 'refund order 1234 for customer C-103' (owner mismatch).")

evaluator = ToolCallingEvaluator()

if "ch3_history" not in st.session_state:
    st.session_state.ch3_history = []

render_message_history(st.session_state.ch3_history)

user_msg = st.chat_input("Ask about an order, a refund, weather, or a customer lookup...", key="ch3_chat_input")
if user_msg:
    st.session_state.ch3_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        try:
            require_live_provider(provider)
            agent = EcommerceCustomerAgent(provider=provider, model=model)
            with st.spinner(f"{model} is deciding which tools to call..."):
                result = agent.execute_task(user_msg)

            st.markdown(result["final_answer"])

            if result["tool_calls"]:
                with st.expander(f"{len(result['tool_calls'])} tool call(s) -- exact args as executed"):
                    for c in result["tool_calls"]:
                        st.code(f"{c['tool']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})\n-> {c['result']}", language="python")

            expected = st.session_state.get("ch3_expected_tools")
            scores = evaluator.evaluate_execution(expected, result)
            metrics_row({k: v.score for k, v in scores.items()})
            reasons = [v.reasoning for v in scores.values() if v.reasoning]
            if reasons:
                st.caption(" | ".join(reasons))

            st.session_state.ch3_history.append({"role": "assistant", "content": result["final_answer"]})
        except ProviderUnavailableError as e:
            st.error(str(e))
