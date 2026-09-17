"""Chapter 2: Agent Architecture -- type a trip request, watch Planner -> Executor -> Verifier live."""

import sys, os, re
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-02-agent-architecture")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, metrics_row

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from pipeline import ArchitecturePipeline
from evaluator import ArchitectureEvaluator, verifier_confusion, gold_plan

chapter_page_header(2, "Agent Architecture", "Planner -> Executor -> Verifier -- type a trip and watch every node.", icon="\U0001F3D7️")

provider = get_model_provider()
is_live = provider_status_badge(provider)

tab_chat, tab_diag = st.tabs(["\U0001F4AC Plan a Trip", "\U0001F52C Verifier Diagnostics"])

with tab_chat:
    st.caption("Describe a trip in your own words, then set the constraints the plan must satisfy exactly.")
    task_text = st.text_input("Trip request", value="Plan a trip to Kyoto focused on temples and ramen.", key="ch2_task")

    c1, c2, c3 = st.columns(3)
    destination = c1.text_input("Destination", value=re.search(r"to ([A-Z][a-zA-Z ]+)", task_text).group(1).strip() if re.search(r"to ([A-Z][a-zA-Z ]+)", task_text) else "Kyoto", key="ch2_dest")
    days = c2.number_input("Days", min_value=1, max_value=10, value=3, key="ch2_days")
    budget = c3.number_input("Budget ($)", min_value=50, max_value=10000, value=900, step=50, key="ch2_budget")
    interests_text = st.text_input("Interests (comma-separated)", value="temples, ramen", key="ch2_interests")
    interests = [i.strip() for i in interests_text.split(",") if i.strip()]

    inject_fault = st.checkbox("Inject an executor failure on the first attempt (tests retry)", key="ch2_inject")

    if st.button("⚡ Run Planner → Executor → Verifier", type="primary", key="ch2_run"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            constraints = {"destination": destination, "days": int(days), "budget": float(budget), "interests": interests}
            pipeline = ArchitecturePipeline(provider=provider)
            evaluator = ArchitectureEvaluator()
            with st.spinner("Running the live pipeline..."):
                res = pipeline.run(task=task_text, constraints=constraints, inject_failure_on_first_try=inject_fault)
                scores = evaluator.evaluate_run(res)
            st.session_state.ch2_result = res
            st.session_state.ch2_scores = scores

    if "ch2_result" in st.session_state:
        res, scores = st.session_state.ch2_result, st.session_state.ch2_scores
        metrics_row({k: v.score for k, v in scores.items() if k != "retries"}, passed=res["success"])
        st.caption(f"Retries: {res['retries']} | Total time: {res['total_duration_sec']}s")

        st.subheader("Pipeline trace")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        nodes = res["pipeline_log"]
        p_col1.info(f"**1. TASK**\n\n{res['task']}")
        p_stat = [n["status"] for n in nodes if n["node"] == "PLANNER"][-1]
        p_col2.success(f"**2. PLANNER**\n\n{p_stat}") if "✓" in p_stat else p_col2.error(f"**2. PLANNER**\n\n{p_stat}")
        e_nodes = [n["status"] for n in nodes if n["node"] == "EXECUTOR"]
        if e_nodes:
            e_stat = e_nodes[-1]
            (p_col3.success if "✓" in e_stat else p_col3.error)(f"**3. EXECUTOR**\n\n{e_stat}")
        else:
            p_col3.warning("**3. EXECUTOR**\n\n(not reached)")
        v_nodes = [n["status"] for n in nodes if n["node"] == "VERIFIER"]
        if v_nodes:
            v_stat = v_nodes[-1]
            (p_col4.success if "✓" in v_stat else p_col4.warning)(f"**4. VERIFIER**\n\n{v_stat}")
        else:
            p_col4.warning("**4. VERIFIER**\n\n(not reached)")

        with st.expander("Full node-by-node log (every attempt)"):
            for n in nodes:
                st.write(f"**{n['node']} (attempt {n['attempt']})** -- {n['status']}")
                st.json(n["details"])

with tab_diag:
    st.caption("Mutation testing: is the verifier actually catching bad plans, or just approving "
               "everything? Each of the four scenarios below is checked against a known-good plan "
               "and four specific mutations of it (over budget, missing a day, wrong city, dropped interest).")
    if st.button("Run mutation-testing diagnostic", key="ch2_diag_run"):
        import json
        tasks_path = os.path.join(ROOT_DIR, "shared", "datasets", "data", "travel_planner_tasks.json")
        tasks_meta = [t["metadata"] for t in json.load(open(tasks_path))]
        from pipeline import ConstraintVerifier
        confusion = verifier_confusion(ConstraintVerifier(), tasks_meta)
        st.dataframe(pd.DataFrame([confusion]).T.rename(columns={0: "value"}), use_container_width=True)
        st.success(f"Bad-plan recall: {confusion['bad_plan_recall']*100:.0f}% "
                   f"(false-reject rate: {confusion['false_reject_rate']*100:.0f}%)")
