"""Chapter 10: Production Evaluation Platform -- runs the real Chapter
1/3/7/8/9 suites through suites.py, gates them with the paired Wilson-CI +
McNemar non-inferiority test in gate.py, and renders the dashboard from that
one stored run result. Nothing here is a static table: every number comes
from `st.session_state.ch10_result`, produced by an actual
`ProductionEvaluationPlatform.run_full_evaluation()` call."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(ROOT_DIR, "chapter-10-production-evals")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from shared.models.provider import get_model_provider
from shared.ui.components import chapter_page_header, metrics_row, provider_status_badge

from shared.ui.components import load_chapter_modules
load_chapter_modules(CHAPTER_DIR)

from eval_platform import BaselineStore, ProductionEvaluationPlatform, RegressionThresholds
from reporter import CIReporter
from gate import to_otel_span

chapter_page_header(
    10, "Production Evaluation Platform",
    "Capstone: orchestrates the already-fixed Chapter 1/3/7/8/9 suites and gates the "
    "run with a paired Wilson-CI + McNemar non-inferiority test -- the report can never "
    "disagree with the gate, because it is rendered from the gate's own decision.",
    icon="\U0001F3DB️",
)

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["qwen2.5:3b", "qwen3:1.7b", "llama3.2:1b"], key="ch10_model")
provider = get_model_provider()
is_live = provider_status_badge(provider)
st.sidebar.caption(
    "4 of the 5 suites below (task success, tool accuracy, groundedness, safety) call this "
    "model; the recovery suite (Chapter 9) makes no model calls at all."
)

st.sidebar.header("CI/CD Gate Thresholds")
min_success = st.sidebar.slider("Min Task Success %", 0.0, 100.0, 85.0, 1.0, key="ch10_min_success")
min_safety = st.sidebar.slider("Min Safety Score %", 0.0, 100.0, 95.0, 1.0, key="ch10_min_safety")
min_grounded = st.sidebar.slider("Min Groundedness %", 0.0, 100.0, 85.0, 1.0, key="ch10_min_grounded")
min_tool = st.sidebar.slider("Min Tool Accuracy %", 0.0, 100.0, 85.0, 1.0, key="ch10_min_tool")
min_recovery = st.sidebar.slider("Min Recovery Rate %", 0.0, 100.0, 75.0, 1.0, key="ch10_min_recovery")
max_lat = st.sidebar.slider("Max Avg Latency (s)", 0.5, 10.0, 3.5, 0.25, key="ch10_max_lat")
promote = st.sidebar.checkbox(
    "Promote this run as the new baseline if it passes the gate", value=False, key="ch10_promote",
    help="Only a passing run is ever saved as a baseline -- a blocked run can never poison "
         "future paired comparisons.",
)

thresholds = RegressionThresholds(
    min_task_success_pct=min_success,
    min_safety_score_pct=min_safety,
    min_groundedness_pct=min_grounded,
    min_tool_accuracy_pct=min_tool,
    min_recovery_rate_pct=min_recovery,
    max_avg_latency_sec=max_lat,
)

run_btn = st.sidebar.button("\U0001F680 Run Full Platform Audit", type="primary", key="ch10_run_btn")

baseline_store = BaselineStore()
if st.sidebar.button("\U0001F5D1️ Clear stored baseline", key="ch10_clear_baseline"):
    if os.path.exists(baseline_store.path):
        os.remove(baseline_store.path)
    st.session_state.pop("ch10_result", None)
    st.sidebar.success("Baseline cleared.")

if run_btn or "ch10_result" not in st.session_state:
    platform = ProductionEvaluationPlatform(thresholds=thresholds, baseline_store=baseline_store)
    with st.spinner(
        "Running the real Chapter 1/3/7/8/9 suites"
        + (" against the live model..." if is_live else " (mock mode -- offline, deterministic)...")
    ):
        res = platform.run_full_evaluation(provider=provider, model=model, promote_if_passed=promote)
    st.session_state.ch10_result = res
    st.session_state.ch10_tracer = platform.tracer

res = st.session_state.ch10_result

st.subheader("Executive scorecard")
metrics_row(
    {
        "task_success_%": res.task_success_pct,
        "safety_%": res.safety_score_pct,
        "groundedness_%": res.groundedness_pct,
        "tool_accuracy_%": res.tool_accuracy_pct,
        "recovery_%": res.recovery_rate_pct,
    },
    passed=res.passed_ci_gate,
)
col1, col2, col3 = st.columns(3)
col1.metric("Avg latency", f"{res.avg_latency_sec:.3f}s")
col2.metric("Avg cost / task", f"${res.avg_cost_usd:.4f}")
col3.metric("Baseline on file", "yes" if res.baseline_present else "no")

if res.passed_ci_gate:
    st.success("CI/CD gate: PASSED -- deployment approved.")
else:
    st.error("CI/CD gate: BLOCKED -- deployment blocked.")
    for f in res.gate_failures:
        st.write(f"- {f}")

st.divider()

tab_report, tab_suites, tab_trace, tab_lessons = st.tabs(
    ["\U0001F4CB Gate Report", "\U0001F9EA Suite Details", "\U0001F50D Trace / OTEL", "\U0001F4D6 The Fix"]
)

with tab_report:
    st.caption(
        "This markdown is `CIReporter.generate_markdown_summary(res)` verbatim -- the core "
        "metrics table below is `gate.markdown_report()`'s own output, copied in by "
        "`eval_platform.py`. There are no independent thresholds here to disagree with the gate."
    )
    st.markdown(CIReporter.generate_markdown_summary(res))
    st.download_button(
        "Download JSON report", CIReporter.generate_json_report(res),
        file_name="ch10_platform_result.json", mime="application/json", key="ch10_dl_json",
    )

with tab_suites:
    st.caption("Per-case pass/fail for the suite run that produced the scorecard above.")
    for metric, detail in res.suite_details.items():
        with st.expander(f"{metric} -- {detail['suite']} (rate {detail['rate']:.3f}, n={len(detail['case_ids'])})"):
            df = pd.DataFrame({"case_id": detail["case_ids"], "passed": detail["passed"]})
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Gate checks (paired comparison detail)")
    checks_df = pd.DataFrame(res.gate_checks)
    if not checks_df.empty:
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

with tab_trace:
    st.caption(
        "Each suite ran inside its own tracer span. `to_otel_span()` maps this lab's own "
        "Span/Tracer objects to an OTLP-style record so any OpenTelemetry backend can store "
        "evaluation runs next to production traces."
    )
    tracer = st.session_state.get("ch10_tracer")
    if tracer is not None and tracer.spans:
        otel_spans = [to_otel_span(s, service="production-eval-platform") for s in tracer.spans]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "name": o["name"],
                        "spanId": o["spanId"],
                        "status": o["status"]["code"],
                        "duration_ms": round((o["endTimeUnixNano"] - o["startTimeUnixNano"]) / 1e6, 2),
                        "rate": o["attributes"].get("eval.rate"),
                        "n": o["attributes"].get("eval.n"),
                    }
                    for o in otel_spans
                ]
            ),
            use_container_width=True, hide_index=True,
        )
        with st.expander("Raw OTLP-style record (first span)"):
            st.json(otel_spans[0])
    else:
        st.info("Run the audit to populate trace spans.")

with tab_lessons:
    st.markdown(
        "**What changed from the first version of this lab:**\n\n"
        "- `run_full_evaluation()` used to return six hardcoded constants "
        "(`task_success = 91.4`, ...). It now actually executes "
        "`suites.py`'s five runners against the already-fixed Chapter 1/3/7/8/9 "
        "agents and evaluators.\n"
        "- The old gate compared point estimates to absolute floors only, so a real drop "
        "from 97% to 90% still cleared an 85% floor. The gate now uses the lower bound of "
        "a 95% Wilson confidence interval, plus a paired McNemar non-inferiority test "
        "against a persisted baseline (`BaselineStore`).\n"
        "- `CIReporter.generate_markdown_summary()` used to hardcode its own threshold "
        "strings, independent of the gate -- the exact bug where a blocked gate could "
        "render a report showing every row green. It now renders `gate.markdown_report()`'s "
        "own output verbatim, so the report cannot disagree with the gate.\n"
        "- A baseline is only ever promoted from a run that itself passed the gate, so a "
        "bad run can never poison future paired comparisons.\n"
    )
