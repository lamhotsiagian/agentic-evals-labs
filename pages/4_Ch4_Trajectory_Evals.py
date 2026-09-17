"""Chapter 4: Trajectory Evals -- pick a ticket, watch the live diagnostic trajectory get scored."""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-04-trajectory-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from engine import ITHelpdeskAgent
from evaluator import ITTrajectoryEvaluator
from tools import SCENARIOS

chapter_page_header(4, "Trajectory Evals", "IT helpdesk agent -- live multi-step diagnosis, milestone-based scoring.")

provider = get_model_provider()
provider_status_badge(provider)
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b"], key="ch4_model")

scenario_labels = {
    "vpn": "VPN Connection Failure",
    "disk": "Disk Space Alert",
    "sso": "SSO Certificate Expiry",
}
scenario = st.selectbox("Ticket", list(scenario_labels), format_func=lambda s: scenario_labels[s], key="ch4_scenario")
st.caption(f"Task: {SCENARIOS[scenario]['task']}")
inject_loop = st.checkbox("Inject a compress_logs failure loop (disk scenario)", key="ch4_inject_loop",
                            disabled=(scenario != "disk"))

if st.button("Run live diagnosis", type="primary", key="ch4_run"):
    try:
        require_live_provider(provider)
    except ProviderUnavailableError as e:
        st.error(str(e))
    else:
        agent = ITHelpdeskAgent(provider=provider, model=model)
        evaluator = ITTrajectoryEvaluator()
        with st.spinner(f"{model} is diagnosing..."):
            trace = agent.diagnose_issue(scenario, inject_compress_loop=inject_loop)
            scores = evaluator.evaluate_trace(trace, scenario=scenario)
        st.session_state.ch4_trace = trace
        st.session_state.ch4_scores = scores

if "ch4_trace" in st.session_state:
    trace, scores = st.session_state.ch4_trace, st.session_state.ch4_scores
    metrics_row({k: v.score for k, v in scores.items() if k != "outcome_gate"}, passed=trace.success)
    reasons = [v.reasoning for v in scores.values() if v.reasoning]
    for r in reasons:
        st.caption(r)

    st.subheader("Step-by-step trace")
    from shared.evaluators.trajectory import classify_step
    for s in trace.steps:
        kind = classify_step(s)
        icon = {"success": "[PASS]", "finding": "[INFO]", "tool_failure": "[FAIL]"}.get(kind, "[STEP]")
        st.write(f"{icon} **{s.action}**({', '.join(f'{k}={v}' for k, v in s.arguments.items())}) [{kind}]")
        st.caption(s.observation)

    st.info(f"**Final answer:** {trace.final_output}")
