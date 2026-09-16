"""Baseline vs Resilient agent implementations under chaos conditions."""

from __future__ import annotations
from typing import Any, Dict, List
from chaos import ChaosInjector


def primary_database_query(order_id: str) -> Dict[str, Any]:
    return {"status": "success", "order_id": order_id, "customer": "Alice", "amount": 120.0}


def secondary_backup_cache(order_id: str) -> Dict[str, Any]:
    return {"status": "success", "order_id": order_id, "customer": "Alice (Cached)", "amount": 120.0, "source": "backup_cache"}


class BaselineAgent:
    """Fragile agent that fails immediately on tool error or timeout."""

    def __init__(self, chaos_injector: ChaosInjector):
        self.chaos = chaos_injector

    def process_order_request(self, order_id: str) -> Dict[str, Any]:
        res = self.chaos.execute_tool_with_chaos("primary_db", primary_database_query, order_id=order_id)
        if res.get("status") != "success":
            return {
                "success": False,
                "agent_type": "Baseline (Fragile)",
                "error": res.get("error"),
                "error_type": res.get("error_type"),
                "recovered": False,
                "final_answer": "System Error: Transaction aborted due to upstream tool failure.",
            }

        return {
            "success": True,
            "agent_type": "Baseline (Fragile)",
            "order_data": res,
            "recovered": False,
            "final_answer": f"Order #{order_id} retrieved successfully for {res.get('customer')}.",
        }


class ResilientAgent:
    """Hardened agent with retries, secondary fallback tools, and circuit breaker recovery."""

    def __init__(self, chaos_injector: ChaosInjector, max_retries: int = 2):
        self.chaos = chaos_injector
        self.max_retries = max_retries

    def process_order_request(self, order_id: str) -> Dict[str, Any]:
        # Attempt primary with retry
        attempts = 0
        last_error = None
        for attempt in range(self.max_retries):
            attempts += 1
            res = self.chaos.execute_tool_with_chaos("primary_db", primary_database_query, order_id=order_id)
            if res.get("status") == "success":
                return {
                    "success": True,
                    "agent_type": "Resilient",
                    "order_data": res,
                    "recovered": False,
                    "attempts": attempts,
                    "final_answer": f"Order #{order_id} retrieved successfully.",
                }
            last_error = res.get("error")

        # Graceful fallback to secondary backup cache
        fallback_res = secondary_backup_cache(order_id)
        return {
            "success": True,
            "agent_type": "Resilient",
            "order_data": fallback_res,
            "recovered": True,
            "attempts": attempts,
            "recovery_mechanism": "secondary_backup_cache",
            "final_answer": f"Order #{order_id} retrieved from fallback cache after primary failure: {last_error}",
        }
