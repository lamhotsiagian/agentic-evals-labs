"""Chapter 8: Safety & Security Evals -- live banking chat behind a real tool
gateway, plus red-team suite, mutation testing, and benign twins."""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-08-safety-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, render_message_history, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from gateway import Ledger, Session
from target_agent import BankingSupportAgent, CompliantStubAgent
from evaluator import SafetyEvaluator
from redteam import BENIGN_TWINS, POISONED_MEMO, load_full_redteam_dataset, mutate

chapter_page_header(
    8, "Safety & Security Evals",
    "Banking support agent -- every sensitive tool call is mediated by a state-based "
    "gateway, not by anything the model or the prompt says.",
    icon="\U0001F6E1️",
)

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], key="ch8_model")
guardrails_enabled = st.sidebar.checkbox("Enable input keyword guardrail (defence-in-depth only)", value=True, key="ch8_guardrails")
st.sidebar.caption("This toggle only affects the cheap keyword layer. The tool gateway "
                    "below cannot be turned off -- that's the fix.")
provider = get_model_provider()
is_live = provider_status_badge(provider)

if "ch8_session" not in st.session_state:
    st.session_state.ch8_session = Session(customer_id="cust-1001")
if "ch8_ledger" not in st.session_state:
    st.session_state.ch8_ledger = Ledger()

with st.sidebar.expander("Session state (only an auth service can set this)"):
    otp = st.checkbox("OTP verified", value=st.session_state.ch8_session.otp_verified, key="ch8_otp")
    st.session_state.ch8_session.otp_verified = otp
    st.caption(f"Daily transfer limit: ${st.session_state.ch8_session.daily_transfer_limit:.2f} | "
               f"Transferred today: ${st.session_state.ch8_session.transferred_today:.2f}")

evaluator = SafetyEvaluator()

tab_chat, tab_redteam, tab_mutation, tab_indirect, tab_benign = st.tabs(
    ["\U0001F4AC Live Chat", "\U0001F534 Red-Team Suite", "\U0001F9EA Mutation Testing",
     "\U0001F4E7 Indirect Injection", "\U0001F7E2 Benign Twins"]
)

with tab_chat:
    st.caption("This box calls your local Ollama model directly, through a real tool "
               "gateway -- nothing here is canned. Try asking it to transfer money "
               "without OTP verified in the sidebar.")
    if "ch8_history" not in st.session_state:
        st.session_state.ch8_history = []

    render_message_history(st.session_state.ch8_history)

    user_msg = st.chat_input("Ask your banking assistant something...", key="ch8_chat_input")
    if user_msg:
        st.session_state.ch8_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                require_live_provider(provider)
                agent = BankingSupportAgent(
                    provider=provider, model=model, guardrails_enabled=guardrails_enabled,
                    session=st.session_state.ch8_session, ledger=st.session_state.ch8_ledger,
                )
                with st.spinner(f"{model} is responding..."):
                    resp = agent.handle_request(user_msg)
                st.markdown(resp["response"])
                if resp["tool_calls"]:
                    st.markdown("**Tool calls (each mediated by the gateway):**")
                    st.dataframe(pd.DataFrame([
                        {"tool": tc["name"], "args": tc["args"], "result": tc["result"]}
                        for tc in resp["tool_calls"]
                    ]), use_container_width=True, hide_index=True)
                st.session_state.ch8_history.append({"role": "assistant", "content": resp["response"]})
            except ProviderUnavailableError as e:
                st.error(str(e))

    if st.session_state.ch8_ledger.events:
        with st.expander(f"Full side-effect ledger this session ({len(st.session_state.ch8_ledger.events)} events)"):
            st.dataframe(pd.DataFrame(st.session_state.ch8_ledger.events), use_container_width=True, hide_index=True)

with tab_redteam:
    st.caption("Runs all 7 threat-category attacks against the real model. Graders read "
               "the ledger and output text -- never a self-reported flag -- and separate "
               "genuinely blocked attacks from ones the agent had no capability for.")
    if st.button("\U0001F680 Launch red-team suite", type="primary", key="ch8_run_redteam"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            attacks = load_full_redteam_dataset()
            agent = BankingSupportAgent(provider=provider, model=model, guardrails_enabled=guardrails_enabled)
            with st.spinner(f"Running {len(attacks)} attacks against {model}..."):
                res = evaluator.evaluate_suite(attacks, agent)
            metrics_row({"safety_score": res["safety_score"], "succeeded": res["succeeded"],
                         "applicable": res["applicable_tests"], "total": res["total_tests"]})
            cat_rows = [{"category": c, **stats} for c, stats in res["category_breakdown"].items()]
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
            with st.expander("Per-attack results (attack prompt, response, verdict)"):
                st.dataframe(pd.DataFrame([
                    {k: v for k, v in r.items() if k != "ledger_events"} for r in res["results"]
                ]), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Safety floor check: replace the model with a stub that ALWAYS attempts "
               "the harmful call. This is deterministic -- no model needed -- and "
               "establishes the guarantee that holds no matter how a real model behaves.")
    if st.button("Run safety-floor check (compliant stub, no model)", key="ch8_run_stub"):
        attacks = load_full_redteam_dataset()
        stub = CompliantStubAgent()
        res = evaluator.evaluate_suite(attacks, stub)
        metrics_row({"safety_score": res["safety_score"], "succeeded": res["succeeded"]})
        gateway_cats = {"Prompt Injection", "Privilege Escalation", "Tool Abuse", "Unsafe Actions"}
        gateway_only = [r for r in res["results"] if r["category"] in gateway_cats]
        n_blocked = sum(1 for r in gateway_only if r["verdict"] == "blocked_by_gateway")
        st.success(f"Tool-gateway categories: {n_blocked}/{len(gateway_only)} blocked with zero "
                   f"unauthorized effects, regardless of model behavior.")
        st.caption("The 3 remaining categories (Secret Extraction, Data Leakage, Jailbreak) are "
                   "output-level disclosures -- a tool gateway cannot stop those; that's the "
                   "Output DLP layer's job, which this lab does not implement a stub for.")

with tab_mutation:
    st.caption("The keyword layer alone intercepts 13 of 32 simple paraphrases. "
               "This runs the same synonym/spacing/encoding/OTP-mention mutations against "
               "the real agent (gateway included) to measure how much it generalizes now.")
    if st.button("Run mutation suite", key="ch8_run_mutation"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            attacks = load_full_redteam_dataset()
            agent = BankingSupportAgent(provider=provider, model=model, guardrails_enabled=guardrails_enabled)
            with st.spinner("Running mutated variants..."):
                res = evaluator.evaluate_mutations(attacks, agent, mutate)
            st.metric("Intercepted", f"{res['intercepted']}/{res['total']}", f"{res['rate']*100:.1f}%")
            by_mut_rows = [{"mutation": k, **v} for k, v in res["by_mutation"].items()]
            st.dataframe(pd.DataFrame(by_mut_rows), use_container_width=True, hide_index=True)

with tab_indirect:
    st.caption("The old suite only tested the user-message channel. This plants an "
               "instruction inside a transaction memo -- content the agent reads as "
               "data, not something the user typed -- and checks the gateway still holds.")
    st.code(POISONED_MEMO, language=None)
    if st.button("Run indirect-injection probe (compliant stub)", key="ch8_run_indirect"):
        stub = CompliantStubAgent()
        resp = stub.handle_indirect(POISONED_MEMO)
        st.write(f"**Action attempted:** `{resp['action_taken']}`")
        st.dataframe(pd.DataFrame(resp["ledger_events"]), use_container_width=True, hide_index=True)
        if resp["ledger_events"] and resp["ledger_events"][-1]["decision"] != "allow":
            st.success(f"Blocked: {resp['ledger_events'][-1]['decision']}")
        else:
            st.error("The gateway allowed this -- investigate immediately.")

with tab_benign:
    st.caption("Legitimate requests that share surface words with attacks (\"transfer\", "
               "\"system prompt\", \"secret\") -- over-refusal was never measured before.")
    if st.button("Run benign-twin suite", key="ch8_run_benign"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            agent = BankingSupportAgent(provider=provider, model=model, guardrails_enabled=guardrails_enabled)
            with st.spinner("Checking for false refusals..."):
                res = evaluator.evaluate_benign_twins(BENIGN_TWINS, agent)
            st.metric("False-refusal rate", f"{res['false_refusal_rate']*100:.1f}%",
                      f"{res['false_refusals']}/{res['n']}")
            st.dataframe(pd.DataFrame(res["rows"]), use_container_width=True, hide_index=True)
