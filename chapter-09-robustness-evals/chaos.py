"""Chaos engineering fault injection engine for agent environments."""

from __future__ import annotations
import random
from typing import Any, Dict, List


class ChaosInjector:
    """Injects synthetic environmental faults into tools and LLM contexts."""

    def __init__(self, active_faults: Dict[str, bool] = None):
        self.active_faults = active_faults or {
            "tool_timeout": False,
            "http_500": False,
            "invalid_json": False,
            "context_corruption": False,
            "tool_unavailable": False,
        }

    def is_active(self, fault_name: str) -> bool:
        return self.active_faults.get(fault_name, False)

    def execute_tool_with_chaos(self, tool_name: str, tool_callable, *args, **kwargs) -> Dict[str, Any]:
        """Wraps tool invocation with configured chaos faults."""
        if self.is_active("tool_timeout"):
            return {
                "status": "error",
                "error_type": "TimeoutError",
                "error": f"Tool '{tool_name}' timed out after 5000ms (chaos injected).",
            }

        if self.is_active("http_500"):
            return {
                "status": "error",
                "error_type": "HTTP500",
                "error": f"Internal Server Error 500 from upstream service for '{tool_name}'.",
            }

        if self.is_active("invalid_json"):
            return {
                "status": "error",
                "error_type": "JSONDecodeError",
                "error": f"Malformed payload from '{tool_name}': Expecting value: line 1 column 1 (char 0).",
                "raw_corrupted": "{status: success, order: 1234 invalid}",
            }

        if self.is_active("tool_unavailable"):
            return {
                "status": "error",
                "error_type": "ServiceUnavailable503",
                "error": f"Service 503: '{tool_name}' is currently unavailable.",
            }

        return tool_callable(*args, **kwargs)

    def corrupt_context(self, context: str) -> str:
        """Injects random context corruption noise."""
        if not self.is_active("context_corruption"):
            return context
        noise = " \n[CORRUPTED_SYSTEM_HEADER_0x8F91A #!&? NULL_PTR_EXCEPTION]\n "
        return context + noise
