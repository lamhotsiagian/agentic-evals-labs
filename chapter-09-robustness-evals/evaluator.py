"""Evaluator for chaos experiments and agent resilience benchmarking."""

from __future__ import annotations
from typing import Any, Dict, List
from shared.models.schemas import MetricScore
from chaos import ChaosInjector
from resilient_agent import BaselineAgent, ResilientAgent


class ChaosExperimentEvaluator:
    """Runs comparative resilience stress tests under active chaos."""

    def run_experiment(
        self,
        chaos_faults: Dict[str, bool],
        test_cases: List[str] = None,
    ) -> Dict[str, Any]:
        if not test_cases:
            import json
            import os
            w_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared", "datasets", "data", "chaos_workload.jsonl")
            if os.path.exists(w_file):
                with open(w_file, "r", encoding="utf-8") as f:
                    cases = [json.loads(line)["order_id"] for line in f if line.strip()]
            else:
                cases = [f"ORD-{i:04d}" for i in range(1001, 1021)]
        else:
            cases = test_cases

        # 1. Normal Baseline (No Chaos)
        clean_injector = ChaosInjector(active_faults={k: False for k in chaos_faults})
        clean_agent = BaselineAgent(clean_injector)
        normal_runs = [clean_agent.process_order_request(cid) for cid in cases]
        normal_success = sum(1 for r in normal_runs if r["success"]) / len(cases)

        # 2. Baseline under Chaos
        active_injector = ChaosInjector(active_faults=chaos_faults)
        baseline_chaos_agent = BaselineAgent(active_injector)
        baseline_chaos_runs = [baseline_chaos_agent.process_order_request(cid) for cid in cases]
        baseline_chaos_success = sum(1 for r in baseline_chaos_runs if r["success"]) / len(cases)

        # 3. Resilient Agent under Chaos
        resilient_agent = ResilientAgent(active_injector)
        resilient_runs = [resilient_agent.process_order_request(cid) for cid in cases]
        resilient_success = sum(1 for r in resilient_runs if r["success"]) / len(cases)
        recovered_count = sum(1 for r in resilient_runs if r.get("recovered"))
        recovery_rate = recovered_count / len(cases)

        return {
            "total_trials": len(cases),
            "normal_success_rate": round(normal_success * 100.0, 1),
            "baseline_chaos_success_rate": round(baseline_chaos_success * 100.0, 1),
            "resilient_chaos_success_rate": round(resilient_success * 100.0, 1),
            "recovery_rate": round(recovery_rate * 100.0, 1),
            "active_faults": chaos_faults,
            "sample_baseline_run": baseline_chaos_runs[0],
            "sample_resilient_run": resilient_runs[0],
        }
