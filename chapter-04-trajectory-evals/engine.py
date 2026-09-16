"""Trajectory execution and simulation engine for Chapter 4 IT Helpdesk Agent."""

from __future__ import annotations
import time
from typing import Any, Dict, List
from shared.models.schemas import AgentTrace, AgentStep


class ITHelpdeskAgent:
    """Simulates an IT helpdesk diagnostic agent traversing multi-step troubleshooting."""

    def diagnose_issue(
        self,
        task: str,
        inject_failure_at_step: int = 3,
        loop_behavior: bool = False,
    ) -> AgentTrace:
        trace = AgentTrace(task=task)

        # Step 1: Ingest request
        trace.add_step(
            action="parse_user_ticket",
            arguments={"task": task},
            observation="User reports internal web portal connection refused (ERR_CONNECTION_REFUSED).",
            result="success",
            duration_ms=45.0,
        )

        # Step 2: Check network route
        trace.add_step(
            action="check_dns_route",
            arguments={"domain": "portal.corp.internal"},
            observation="DNS resolved to 10.200.4.12. Route table active.",
            result="success",
            duration_ms=62.0,
        )

        # Step 3: Tool Call (Optional failure injection)
        if inject_failure_at_step == 3:
            trace.add_step(
                action="query_auth_gateway",
                arguments={"user_id": "emp_8831", "service": "portal"},
                observation="HTTP 504 Gateway Timeout: Auth service unresponsive.",
                result="error",
                duration_ms=500.0,
            )

            if loop_behavior:
                # Repeated action without fixing cause
                trace.add_step(
                    action="query_auth_gateway",
                    arguments={"user_id": "emp_8831", "service": "portal"},
                    observation="HTTP 504 Gateway Timeout: Auth service unresponsive.",
                    result="error",
                    duration_ms=510.0,
                )

            # Recovery step
            trace.add_step(
                action="fallback_ldap_token_refresh",
                arguments={"user_id": "emp_8831"},
                observation="Secondary LDAP session token refreshed successfully.",
                result="success",
                duration_ms=180.0,
            )
        else:
            trace.add_step(
                action="query_auth_gateway",
                arguments={"user_id": "emp_8831", "service": "portal"},
                observation="User token valid. Role: Engineering.",
                result="success",
                duration_ms=110.0,
            )

        # Final step
        trace.add_step(
            action="verify_portal_handshake",
            arguments={"target": "10.200.4.12:443"},
            observation="TLS 1.3 handshake succeeded. HTTP 200 OK.",
            result="success",
            duration_ms=38.0,
        )

        trace.final_output = "Diagnosed and resolved connection block by refreshing local LDAP auth session. User access restored."
        trace.success = True
        return trace
