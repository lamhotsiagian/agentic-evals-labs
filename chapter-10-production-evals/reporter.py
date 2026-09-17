"""CI/CD regression reporter and artifact generator.

Fixed from an earlier version of this lab: the old
`generate_markdown_summary` hardcoded its own threshold strings (">= 85.0%",
"<= 3.50s", ...) directly in an f-string, completely independent of the
`RegressionThresholds` the gate actually used -- so a gate that blocked on a
tightened threshold could still render a report where every row showed a
green check. This version renders the core metrics table from
`result.gate_markdown`, which `eval_platform.py` built with
`gate.markdown_report()` straight off the same gate decision object the
platform gated on. There are no metric thresholds re-declared here: the
report structurally cannot disagree with the gate.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from eval_platform import PlatformEvaluationResult


class CIReporter:
    """Generates machine-readable and executive artifacts for CI/CD gates."""

    @staticmethod
    def generate_json_report(result: PlatformEvaluationResult) -> str:
        return json.dumps(result.model_dump(), indent=2)

    @staticmethod
    def generate_markdown_summary(result: PlatformEvaluationResult) -> str:
        status_badge = "✅ PASSED" if result.passed_ci_gate else "❌ BLOCKED"
        latency_status = "✅" if result.latency_ok else "❌"
        baseline_note = (
            "Compared against the stored production baseline."
            if result.baseline_present
            else "No stored baseline yet -- this run establishes one "
                 "(paired regression checks report delta=0 until the next run)."
        )
        lines = [
            "# Agent Evaluation Platform — CI Regression Gate Report",
            f"Status: **{status_badge}**",
            f"Trace ID: `{result.trace_id}`",
            "",
            baseline_note,
            "",
            result.gate_markdown,
            "",
            "### Latency & cost",
            "",
            "| Metric | Measured | Ceiling | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Avg Latency** | {result.avg_latency_sec:.3f}s | <= {result.max_avg_latency_sec}s | {latency_status} |",
            f"| **Avg Cost / Task** | ${result.avg_cost_usd:.4f} | N/A (local model) | ℹ️ |",
        ]
        return "\n".join(lines) + "\n"
