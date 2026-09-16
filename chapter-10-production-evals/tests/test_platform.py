"""Unit tests for Chapter 10 Production Evaluation Platform."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
    sys.modules.pop(m, None)

from eval_platform import ProductionEvaluationPlatform, RegressionThresholds
from reporter import CIReporter


def test_platform_evaluation_passes_gate():
    platform = ProductionEvaluationPlatform()
    res = platform.run_full_evaluation()

    assert res.task_success_pct >= 90.0
    assert res.safety_score_pct >= 95.0
    assert res.passed_ci_gate is True
    assert len(res.gate_failures) == 0


def test_platform_evaluation_blocks_on_strict_gate():
    # Require 99% task success (should block)
    strict_thresholds = RegressionThresholds(min_task_success_pct=99.0)
    platform = ProductionEvaluationPlatform(thresholds=strict_thresholds)
    res = platform.run_full_evaluation()

    assert res.passed_ci_gate is False
    assert len(res.gate_failures) > 0


def test_ci_reporter_artifacts():
    platform = ProductionEvaluationPlatform()
    res = platform.run_full_evaluation()

    json_report = CIReporter.generate_json_report(res)
    assert "task_success_pct" in json_report
    assert res.trace_id in json_report

    md_summary = CIReporter.generate_markdown_summary(res)
    assert "CI Regression Gate Report" in md_summary
    assert "PASSED" in md_summary
