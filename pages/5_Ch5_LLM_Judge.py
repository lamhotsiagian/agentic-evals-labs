"""Chapter 5: LLM-as-a-Judge -- type a prompt/response pair, get a fail-closed verdict."""

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-05-llm-judge")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider, require_live_provider, ProviderUnavailableError
from shared.ui.components import chapter_page_header, provider_status_badge, metrics_row
from shared.datasets.loader import load_judge_benchmark_cases

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from judge import MultiJudgeSystem
from calibration import compute_calibration_metrics

chapter_page_header(5, "LLM-as-a-Judge", "Type a prompt and a response -- get a fail-closed, anchor-grounded verdict.")

provider = get_model_provider()
provider_status_badge(provider)
primary_model = st.sidebar.selectbox("Primary judge", ["qwen3:1.7b", "qwen2.5:3b"], key="ch5_primary")
alt_model = st.sidebar.selectbox("Alternative judge", ["llama3.2:1b", "qwen3:1.7b"], index=0, key="ch5_alt")

tab_judge, tab_calib = st.tabs(["Judge a Response", "Calibration Report"])

with tab_judge:
    with st.form("ch5_judge_form"):
        prompt = st.text_area("User prompt", value="What is the capital of Australia?", key="ch5_prompt")
        response = st.text_area("Response to judge", value="Sydney", key="ch5_response")
        context = st.text_area("Reference context (optional)", value="", key="ch5_context")
        run_probe = st.checkbox("Also run the verbosity-bias probe", key="ch5_probe")
        submitted = st.form_submit_button("Judge it", type="primary")

    if submitted:
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            system = MultiJudgeSystem(primary_model=primary_model, alt_model=alt_model, provider=provider)
            with st.spinner("Both judges are evaluating..."):
                result = system.evaluate_pair(prompt, response, context or None)

            for label in ("primary", "alternative"):
                verdict = result[label]
                st.subheader(f"{label.title()} judge ({verdict.judge_model})")
                if verdict.status == "abstain":
                    st.warning(f"ABSTAINED -- {verdict.reasoning}")
                else:
                    metrics_row(verdict.dimension_scores, passed=verdict.passed)
                    st.caption(f"Overall (computed, weighted): {verdict.overall_score} | {verdict.reasoning}")

            if result["needs_human_review"]:
                st.error(f"Needs human review: {result['review_reason']}")
            else:
                st.success("Primary and alternative judges agree.")

            if run_probe:
                with st.spinner("Running verbosity-bias probe..."):
                    probe = system.probe_verbosity_bias(prompt, response, context or None)
                st.write("**Verbosity bias probe** (same content, padded with filler):")
                st.json(probe)
                if probe["biased"]:
                    st.warning("Score moved by more than 0.3 points for no substantive reason -- possible verbosity bias.")

with tab_calib:
    st.caption("Real human-labeled benchmark (5 items) -- honestly reported: kappa, bootstrap CI, "
               "and an explicit underpowered warning instead of presenting raw agreement as trustworthy.")
    if st.button("Run calibration against human labels", key="ch5_calib_run"):
        try:
            require_live_provider(provider)
        except ProviderUnavailableError as e:
            st.error(str(e))
        else:
            benchmarks = load_judge_benchmark_cases()
            system = MultiJudgeSystem(primary_model=primary_model, provider=provider)
            with st.spinner(f"Judging {len(benchmarks)} benchmark items..."):
                evals = [system.primary_judge.evaluate(b["prompt"], b["response"]) for b in benchmarks]
            metrics = compute_calibration_metrics(evals, benchmarks)
            if metrics.get("warning"):
                st.warning(metrics["warning"])
            st.json(metrics)
