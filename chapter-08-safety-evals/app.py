"""Chapter 8: Agent Red-Team Security Evaluation Dashboard (Streamlit)."""

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
from target_agent import BankingSupportAgent
from redteam import load_full_redteam_dataset
from evaluator import SafetyEvaluator

st.set_page_config(page_title="Agent Red-Team Security Evaluation", page_icon="🛡️", layout="wide")

st.title("🛡️ Agent Red-Team Security Evaluation")
st.caption("Chapter 8: Evaluating Prompt Injection, Data Leakage, Tool Abuse, and Privilege Escalation")

# Sidebar
st.sidebar.header("Security & Red-Team Settings")
enable_guardrails = st.sidebar.toggle("Enable Security Guardrails", value=True, key="guardrails_toggle")
run_redteam_btn = st.sidebar.button("🚨 Launch Red-Team Attack Suite", type="primary", key="run_redteam_btn")

attacks = load_full_redteam_dataset()

if "sec_suite_res" not in st.session_state or run_redteam_btn:
    agent = BankingSupportAgent(guardrails_enabled=enable_guardrails)
    evaluator = SafetyEvaluator()
    with st.spinner("Executing adversarial attack vectors against Banking Agent..."):
        suite_res = evaluator.evaluate_suite(attacks, agent)
        st.session_state.sec_suite_res = suite_res

suite_res = st.session_state.sec_suite_res

# KPI Cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Tests", f"{suite_res['total_tests']}")
k2.metric("Passed (Thwarted)", f"{suite_res['passed']}", "Secure")
k3.metric("Failed (Vulnerabilities)", f"{suite_res['failed']}", "Breached" if suite_res['failed'] > 0 else "None", delta_color="inverse")
k4.metric("Safety Score", f"{suite_res['safety_score']:.1f}%")

st.divider()

col_breakdown, col_replay = st.columns([1, 1])

with col_breakdown:
    st.subheader("📊 Vulnerability Category Breakdown")
    cat_rows = []
    for cat, stat in suite_res["category_breakdown"].items():
        total = stat["total"]
        failed = stat["failed"]
        vuln_rate = (failed / total) * 100.0 if total > 0 else 0.0
        cat_rows.append({
            "Threat Category": cat,
            "Tests": total,
            "Failed Attacks": failed,
            "Vulnerability %": f"{vuln_rate:.0f}%",
            "Status": "🛡️ Secure" if failed == 0 else "⚠️ Vulnerable",
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

with col_replay:
    st.subheader("🔴 Red-Team Attack Replay")
    attack_map = {f"{r['attack_id']} — {r['category']}": r for r in suite_res["results"]}
    selected_replay_key = st.selectbox("Select Attack Vector to Replay", list(attack_map.keys()))
    r_item = attack_map[selected_replay_key]

    st.markdown(f"**Severity**: `{r_item['severity']}` | **Category**: `{r_item['category']}`")
    
    st.markdown("**Adversarial Input Prompt:**")
    st.code(r_item["attack_prompt"])

    st.markdown("**Agent Output / Action:**")
    st.code(r_item["agent_response"])

    if r_item["is_safe"]:
        st.success(f"✅ Verdict: PASSED — {r_item['violation_details']}")
    else:
        st.error(f"❌ Verdict: FAILED / EXPLOITED — {r_item['violation_details']}")

st.divider()

# Case Study & Guidance
st.subheader("Case Study: Banking Support Agent Tool Security")
st.markdown("""
Connecting LLM agents to sensitive tools (`transfer_money`, `send_email`) without strict permission barriers creates severe vulnerability surfaces:
1. **Tool Abuse & Privilege Escalation**: Attackers masquerade as executives to bypass 2FA.
2. **System Prompt & Secret Extraction**: Attackers use DAN mode or developer tokens to dump environment keys.
3. **Defensive Architecture**: Defense-in-depth requires tool-level authorization checks, input validation guards, and runtime token verification.
""")
