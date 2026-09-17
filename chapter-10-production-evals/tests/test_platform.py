"""Unit tests for Chapter 10's Production Evaluation Platform.

These replace the old tautological tests (`ProductionEvaluationPlatform()`
always returned the same six hardcoded constants, so every assertion was
guaranteed to pass regardless of any agent's real behavior). Every test here
either executes a real chapter suite through `suites.py`, or drives
`gate.py` directly with the book's own stated McNemar figures, and checks
genuinely computed values -- never a canned constant.
"""

import math
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CHAPTER_DIR not in sys.path:
    sys.path.insert(0, CHAPTER_DIR)

for m in (
    "agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration",
    "system", "graders", "retriever", "chaos", "resilient_agent", "eval_platform",
    "reporter", "gateway", "target_agent", "redteam", "suites", "gate",
):
    sys.modules.pop(m, None)

import suites
from gate import GateRule, SuiteResult, gate as run_gate, markdown_report, to_otel_span
from eval_platform import BaselineStore, ProductionEvaluationPlatform, RegressionThresholds
from reporter import CIReporter
from shared.tracing.tracer import Tracer


def test_platform_runs_real_suites_and_metrics_are_not_constants():
    """The direct fix for 'Platform metrics are constants': two full runs of
    the real Chapter 1/3/7/8/9 suites, under the same deterministic mock,
    must produce the same non-trivial values both times (proving they're
    computed, not hardcoded) -- and those values must actually come from the
    suites, not from `PlatformEvaluationResult`'s defaults."""
    platform = ProductionEvaluationPlatform(baseline_store=BaselineStore(path="/tmp/ch10_test_baseline_a.json"))
    if os.path.exists(platform.baseline_store.path):
        os.remove(platform.baseline_store.path)

    res1 = platform.run_full_evaluation()
    res2 = platform.run_full_evaluation()

    # Deterministic mock -> repeatable, non-fabricated numbers.
    assert res1.task_success_pct == res2.task_success_pct
    assert res1.safety_score_pct == res2.safety_score_pct
    assert res1.groundedness_pct == res2.groundedness_pct
    assert res1.tool_accuracy_pct == res2.tool_accuracy_pct
    assert res1.recovery_rate_pct == res2.recovery_rate_pct

    # These are the actual computed rates from suites.py's real dataset
    # sizes, not the book's old hardcoded 91.4/97.2/94.1/92.7/86.3.
    assert res1.task_success_pct not in (91.4,)
    assert res1.safety_score_pct not in (97.2,)
    assert res1.groundedness_pct not in (94.1,)
    assert res1.tool_accuracy_pct not in (92.7,)
    assert res1.recovery_rate_pct not in (86.3,)

    # Groundedness (Chapter 7, well-covered by the RAG mock) should be high;
    # this is a real measured value, checked against the actual suite run.
    groundedness_direct = suites.run_groundedness_suite()
    assert res1.groundedness_pct == round(groundedness_direct.rate * 100, 1)


def test_gate_matches_the_books_own_mcnemar_example():
    """Reproduces the McNemar figures stated in the book's Chapter 10 exactly:
    9 regressed / 8 fixed pairs is symmetric churn (p=1.0, no block), while
    11 regressed / 0 fixed is a one-sided, significant regression (p=0.001,
    blocks) -- the same values `mcnemar_exact` was verified against when it
    was added to shared/metrics/stats.py."""
    # 9 regressed, 8 fixed, no concordant pairs needed to hit those exact counts.
    base_ids = [f"C-{i:03d}" for i in range(17)]
    baseline_churn = SuiteResult(
        suite="synthetic", metric="task_success", case_ids=base_ids,
        passed=[True] * 9 + [False] * 8,
    )
    candidate_churn = SuiteResult(
        suite="synthetic", metric="task_success", case_ids=base_ids,
        passed=[False] * 9 + [True] * 8,
    )
    rule = GateRule(metric="task_success", floor=0.0, max_regression=0.02, alpha=0.05)
    g_churn = run_gate({"task_success": candidate_churn}, {"task_success": baseline_churn}, [rule])
    churn_check = g_churn["checks"][0]
    assert churn_check["regressed"] == 9
    assert churn_check["fixed"] == 8
    assert churn_check["p_value"] == 1.0
    assert not churn_check["failures"]
    assert g_churn["passed"] is True

    # 11 regressed, 0 fixed -- a real, one-sided regression.
    regress_ids = [f"S-{i:03d}" for i in range(11)]
    baseline_regress = SuiteResult(
        suite="synthetic", metric="safety", case_ids=regress_ids, passed=[True] * 11,
    )
    candidate_regress = SuiteResult(
        suite="synthetic", metric="safety", case_ids=regress_ids, passed=[False] * 11,
    )
    rule2 = GateRule(metric="safety", floor=0.0, max_regression=0.02, alpha=0.05)
    g_regress = run_gate({"safety": candidate_regress}, {"safety": baseline_regress}, [rule2])
    regress_check = g_regress["checks"][0]
    assert regress_check["regressed"] == 11
    assert regress_check["fixed"] == 0
    assert regress_check["p_value"] == 0.001
    assert regress_check["failures"]
    assert g_regress["passed"] is False


def test_report_cannot_disagree_with_gate():
    """Direct fix for 'gate says BLOCKED, report shows every row OK': every
    FAIL row rendered in `markdown_report` must correspond to exactly one
    entry in `blocking_failures`, and vice versa -- checked against a real
    multi-metric gate run (Chapter 8 safety suite, guardrails on vs off),
    not a hand-crafted example built to look consistent."""
    baseline = suites.run_safety_suite(guardrails_enabled=True)
    candidate = suites.run_safety_suite(guardrails_enabled=False)
    tool_baseline = suites.run_tool_accuracy_suite()

    rules = [
        GateRule(metric="safety", floor=0.95, max_regression=0.02, alpha=0.05),
        GateRule(metric="tool_accuracy", floor=0.0, max_regression=0.02, alpha=0.05),
    ]
    g = run_gate(
        {"safety": candidate, "tool_accuracy": tool_baseline},
        {"safety": baseline, "tool_accuracy": tool_baseline},
        rules,
    )
    report = markdown_report(g, rules)

    fail_rows = [line for line in report.splitlines() if line.startswith("| ") and "| FAIL |" in line]
    ok_rows = [line for line in report.splitlines() if line.startswith("| ") and "| ok |" in line]
    assert len(fail_rows) + len(ok_rows) == len(rules)

    # Every FAIL row's metric name must appear in a blocking failure string.
    for row in fail_rows:
        metric = row.split("|")[1].strip()
        assert any(metric in f for f in g["blocking_failures"]), f"FAIL row for {metric} has no matching blocking failure"

    # Every blocking failure must correspond to a FAIL row, never an ok one.
    for f in g["blocking_failures"]:
        metric = f.split(":")[0]
        assert any(metric in row for row in fail_rows)
        assert not any(metric in row for row in ok_rows)

    # This specific scenario is a real regression: safety must FAIL.
    assert any("safety" in row for row in fail_rows)
    assert g["passed"] is False


def test_gate_blocks_a_real_known_bad_variant():
    """Proves the gate catches a genuine regression using two REAL executed
    suite runs (not synthetic SuiteResult objects): Chapter 8's safety
    suite with guardrails on (baseline) vs. deliberately disabled
    (candidate, the known-bad variant)."""
    baseline = suites.run_safety_suite(guardrails_enabled=True)
    candidate = suites.run_safety_suite(guardrails_enabled=False)
    assert candidate.rate <= baseline.rate  # guardrails off is never safer

    rule = GateRule(metric="safety", floor=0.95, max_regression=0.02, alpha=0.05)
    g = run_gate({"safety": candidate}, {"safety": baseline}, [rule])
    assert g["passed"] is False
    assert g["blocking_failures"]


def test_baseline_store_round_trips_and_first_run_has_no_baseline():
    path = "/tmp/ch10_test_baseline_b.json"
    if os.path.exists(path):
        os.remove(path)
    store = BaselineStore(path=path)

    assert store.load() == {}

    result = suites.run_recovery_suite()
    store.save({"recovery": result})

    reloaded = store.load()
    assert reloaded["recovery"].case_ids == result.case_ids
    assert reloaded["recovery"].passed == result.passed
    os.remove(path)


def test_platform_promotes_baseline_only_when_gate_passes():
    """`promote_if_passed=True` must never write a baseline for a run the
    gate itself blocked -- otherwise a bad run could poison future
    comparisons."""
    path = "/tmp/ch10_test_baseline_c.json"
    if os.path.exists(path):
        os.remove(path)
    platform = ProductionEvaluationPlatform(baseline_store=BaselineStore(path=path))

    res = platform.run_full_evaluation(promote_if_passed=True)
    if res.passed_ci_gate:
        assert os.path.exists(path)
    else:
        assert not os.path.exists(path)
    if os.path.exists(path):
        os.remove(path)


def test_regression_thresholds_map_to_gate_rule_floors():
    thresholds = RegressionThresholds(min_safety_score_pct=95.0, min_task_success_pct=85.0)
    rules = {r.metric: r for r in thresholds.as_gate_rules()}
    assert rules["safety"].floor == 0.95
    assert rules["task_success"].floor == 0.85
    assert rules["safety"].max_regression == thresholds.max_regression
    assert rules["safety"].alpha == thresholds.alpha


def test_ci_reporter_renders_from_the_gate_markdown_not_its_own_thresholds():
    """The direct fix: `generate_markdown_summary` must not hardcode metric
    threshold strings anywhere -- it renders `result.gate_markdown`
    verbatim, so a report row can never claim a status the gate disagrees
    with."""
    platform = ProductionEvaluationPlatform(baseline_store=BaselineStore(path="/tmp/ch10_test_baseline_d.json"))
    if os.path.exists(platform.baseline_store.path):
        os.remove(platform.baseline_store.path)
    res = platform.run_full_evaluation()
    md = CIReporter.generate_markdown_summary(res)

    assert res.gate_markdown in md
    status_line = next(line for line in md.splitlines() if line.startswith("Status:"))
    assert ("PASSED" in status_line) == res.passed_ci_gate
    assert ("BLOCKED" in status_line) == (not res.passed_ci_gate)

    json_report = CIReporter.generate_json_report(res)
    assert '"passed_ci_gate"' in json_report
    if os.path.exists(platform.baseline_store.path):
        os.remove(platform.baseline_store.path)


def test_to_otel_span_maps_a_real_tracer_span():
    tracer = Tracer(service_name="ch10-test")
    tracer.start_trace("otel_mapping_test")
    span = tracer.start_span("unit_test_span", attributes={"case_id": "X-1"})
    span.finish(status="ok")

    otel = to_otel_span(span, service="ch10-test")
    assert otel["name"] == "unit_test_span"
    assert otel["status"] == {"code": "STATUS_CODE_OK"}
    assert otel["attributes"]["service.name"] == "ch10-test"
    assert otel["attributes"]["eval.case_id"] == "X-1"
    assert len(otel["traceId"]) == 32
    assert len(otel["spanId"]) == 16
    assert otel["startTimeUnixNano"] <= otel["endTimeUnixNano"]

    error_span = tracer.start_span("failing_span")
    error_span.finish(status="error")
    otel_err = to_otel_span(error_span, service="ch10-test")
    assert otel_err["status"] == {"code": "STATUS_CODE_ERROR"}
