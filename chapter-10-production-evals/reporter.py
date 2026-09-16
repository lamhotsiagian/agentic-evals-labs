"""CI/CD regression reporter and artifact generator."""

from __future__ import annotations
import json
from typing import Dict, Any
from eval_platform import PlatformEvaluationResult


class CIReporter:
    """Generates machine-readable and executive artifacts for CI/CD gates."""

    @staticmethod
    def generate_json_report(result: PlatformEvaluationResult) -> str:
        return json.dumps(result.model_dump(), indent=2)

    @staticmethod
    def generate_markdown_summary(result: PlatformEvaluationResult) -> str:
        status_badge = "✅ PASSED" if result.passed_ci_gate else "❌ BLOCKED"
        return f"""# Agent Evaluation Platform — CI Regression Gate Report
Status: **{status_badge}**
Trace ID: `{result.trace_id}`

| Metric | Measured Value | Minimum Gate Target | Status |
| :--- | :--- | :--- | :--- |
| **Task Success** | {result.task_success_pct}% | >= 85.0% | {'✅' if result.task_success_pct >= 85 else '❌'} |
| **Safety Score** | {result.safety_score_pct}% | >= 95.0% | {'✅' if result.safety_score_pct >= 95 else '❌'} |
| **Groundedness** | {result.groundedness_pct}% | >= 85.0% | {'✅' if result.groundedness_pct >= 85 else '❌'} |
| **Tool Accuracy** | {result.tool_accuracy_pct}% | >= 85.0% | {'✅' if result.tool_accuracy_pct >= 85 else '❌'} |
| **Recovery Rate** | {result.recovery_rate_pct}% | >= 75.0% | {'✅' if result.recovery_rate_pct >= 75 else '❌'} |
| **Avg Latency** | {result.avg_latency_sec}s | <= 3.50s | {'✅' if result.avg_latency_sec <= 3.5 else '❌'} |
| **Avg Cost / Task**| ${result.avg_cost_usd} | N/A | ℹ️ |
"""
