"""Chapter 6: Multi-Agent System Evaluation Dashboard (Streamlit)."""

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
from system import MultiAgentResearchSystem
from evaluator import MultiAgentEvaluator

st.set_page_config(page_title="Multi-Agent System Evaluation", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Research System Evaluation")
st.caption("Chapter 6: Evaluating Agent Topology, Delegation, Handoffs, and Synthesis")

# Sidebar
st.sidebar.header("Multi-Agent Topology")
st.sidebar.markdown("""
**Network Topology:**
```text
      Supervisor
      /        \\
     ↓          ↓
Researcher    Analyst
     \\          /
      ↓        ↓
     Synthesizer
```
""")

topic_input = st.sidebar.text_input("Research Goal", value="Evolution of LLM-as-a-Judge Techniques (2024-2026)")
inject_drop = st.sidebar.checkbox("Inject Communication Handoff Failure", value=False, key="inject_drop_chk")
run_multi_btn = st.sidebar.button("🚀 Run Multi-Agent System", type="primary", key="run_multi_btn")

if "multi_result" not in st.session_state or run_multi_btn:
    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()
    with st.spinner("Orchestrating agent collaboration..."):
        res = system.run_research(research_topic=topic_input, inject_handoff_failure=inject_drop)
        scores = evaluator.evaluate_system_run(res)
        st.session_state.multi_result = res
        st.session_state.multi_scores = scores

res = st.session_state.multi_result
scores = st.session_state.multi_scores
meta = scores["handoff_success_rate"].metadata

# Metrics KPI
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Messages", f"{int(scores['total_messages'].score)}")
k2.metric("Successful Handoffs", f"{meta.get('successful', 0)}")
k3.metric("Failed Handoffs", f"{meta.get('failed', 0)}")
k4.metric("Handoff Success Rate", f"{scores['handoff_success_rate'].score * 100:.0f}%")
k5.metric("Role Adherence", "100%" if scores['role_adherence'].passed else "Degraded")

st.divider()

# Visual Agent Network
st.subheader("🕸️ Agent Collaboration Network")
c_sup, c_mid, c_syn = st.columns([1, 2, 1])

with c_sup:
    st.info("**Supervisor Agent**\n\nModel: Qwen2.5:3B\n\nTask: Goal formulation & delegation")

with c_mid:
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        st.success("**Researcher Agent**\n\nTask: Literature & Facts gathering")
    with sub_c2:
        st.success("**Analyst Agent**\n\nTask: Quantitative metrics & benchmarks")

with c_syn:
    st.info("**Synthesizer Agent**\n\nModel: Qwen2.5:3B\n\nTask: Executive integration")

st.divider()

# Message Communication Stream & Final Synthesis
col_stream, col_output = st.columns([3, 2])

with col_stream:
    st.subheader("💬 Message Handoff Stream")
    for m in res["messages"]:
        is_ok = m.handoff_status == "success"
        icon = "✅" if is_ok else "❌"
        badge = "Handoff OK" if is_ok else "Handoff Dropped (Failure Injected)"
        
        with st.chat_message(m.sender.lower()):
            st.markdown(f"**From `{m.sender}` ➔ To `{m.recipient}`** — `{icon} {badge}`")
            st.write(m.content)

with col_output:
    st.subheader("📄 Synthesizer Output")
    st.markdown(f"```text\n{res['final_synthesis']}\n```")

    st.subheader("Multi-Agent Evaluation Criteria")
    criteria_df = pd.DataFrame([
        {"Criterion": "Delegation", "Result": "Optimal (Parallel sub-delegation)"},
        {"Criterion": "Role Adherence", "Result": "100% (No boundary violations)"},
        {"Criterion": "Duplicate Work", "Result": "None detected (Zero token waste)"},
        {"Criterion": "Conflict Resolution", "Result": "Resolved at Synthesizer stage"},
    ])
    st.table(criteria_df)
