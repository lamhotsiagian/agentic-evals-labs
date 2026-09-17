"""Simulated IT environment for Chapter 4 -- one function per diagnostic
action the agent can take. Responses are scenario-scripted (the same way
Chapter 3's order/weather API is a scripted stand-in for a real backend);
what makes this chapter's evaluation real is that the MODEL decides which
of these to call and in what order, not a hard-coded script.
"""

from __future__ import annotations
from typing import Any, Dict

# Each scenario is a small state machine: repeating a diagnostic action
# before its fix returns the SAME observation (so wasted_repeats can
# detect it); after the relevant fix step, downstream checks succeed.

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "vpn": {
        "task": "User reports they cannot access the internal database server.",
        "milestones": ["reset_vpn_session", "verify_connectivity"],  # check_vpn_status is the diagnostic finding, not a resolution milestone
        "optimal_steps": 3,
    },
    "disk": {
        "task": "prod-worker-01 is alerting on log partition disk space.",
        # check_disk always returns a finding (the partition IS full), so it can
        # never be a resolution milestone -- listing it capped progress at 0.5.
        "milestones": ["archive_to_s3"],
        "optimal_steps": 2,
    },
    "sso": {
        "task": "A user cannot sign in to the Okta SAML application.",
        "milestones": ["rotate_signing_cert", "test_saml_assertion"],
        "optimal_steps": 3,  # query_idp_metadata (finding) -> rotate -> test, same shape as vpn
    },
}


class ITEnvironment:
    """Holds mutable state for one diagnostic session so tool responses
    change realistically as the agent takes fixing actions."""

    def __init__(self, scenario: str, inject_compress_loop: bool = False):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        self.scenario = scenario
        self.inject_compress_loop = inject_compress_loop
        self.vpn_reset = False
        self.cert_rotated = False
        self.compress_attempts = 0

    def ping_host(self, host: str) -> Dict[str, Any]:
        if self.scenario == "vpn" and not self.vpn_reset:
            return {"status": "error", "observation": "Timeout: 100% packet loss", "kind": "tool_failure"}
        return {"status": "success", "observation": f"Reply from {host}: 3.8ms", "kind": "success"}

    def check_vpn_status(self, user_id: str) -> Dict[str, Any]:
        if self.scenario != "vpn":
            return {"status": "success", "observation": "VPN tunnel state: CONNECTED", "kind": "success"}
        if not self.vpn_reset:
            return {"status": "success", "observation": "VPN tunnel state: DISCONNECTED (IPsec SA expired)", "kind": "finding"}
        return {"status": "success", "observation": "VPN tunnel state: CONNECTED", "kind": "success"}

    def reset_vpn_session(self, user_id: str) -> Dict[str, Any]:
        self.vpn_reset = True
        return {"status": "success", "observation": "IKEv2 session re-negotiated. Assigned virtual IP: 10.240.12.88", "kind": "success"}

    def verify_connectivity(self, host: str) -> Dict[str, Any]:
        if self.scenario == "vpn" and not self.vpn_reset:
            return {"status": "error", "observation": "Connection refused", "kind": "tool_failure"}
        return {"status": "success", "observation": f"TCP connection established to {host}:5432. Latency: 3.8ms", "kind": "success"}

    def check_disk(self, host: str) -> Dict[str, Any]:
        return {"status": "success", "observation": "Partition /var/log is 98% full (196GB / 200GB)", "kind": "finding"}

    def compress_logs(self, older_than_days: int) -> Dict[str, Any]:
        if self.scenario == "disk" and self.inject_compress_loop:
            self.compress_attempts += 1
            return {"status": "error", "observation": "Error: Insufficient scratch disk space to create gzip archive.", "kind": "tool_failure"}
        return {"status": "success", "observation": "Archive created: 14GB", "kind": "success"}

    def archive_to_s3(self, path: str) -> Dict[str, Any]:
        return {"status": "success", "observation": f"Direct stream {path} to s3://corp-logs-cold/ completed. 14GB reclaimed.", "kind": "success"}

    def query_idp_metadata(self, app_id: str) -> Dict[str, Any]:
        if self.scenario == "sso" and not self.cert_rotated:
            return {"status": "error", "observation": "X.509 Signing Certificate expired at 2026-03-01 00:00:00 UTC", "kind": "finding"}
        return {"status": "success", "observation": "Signing certificate valid.", "kind": "success"}

    def rotate_signing_cert(self, app_id: str, key_id: str) -> Dict[str, Any]:
        self.cert_rotated = True
        return {"status": "success", "observation": "New certificate installed and published to IdP metadata endpoint.", "kind": "success"}

    def test_saml_assertion(self, user_id: str) -> Dict[str, Any]:
        if self.scenario == "sso" and not self.cert_rotated:
            return {"status": "error", "observation": "SAMLResponse signature invalid.", "kind": "tool_failure"}
        return {"status": "success", "observation": "SAMLResponse signed and validated. HTTP 302 to app dashboard.", "kind": "success"}

    def registry(self) -> Dict[str, Any]:
        return {
            "ping_host": self.ping_host, "check_vpn_status": self.check_vpn_status,
            "reset_vpn_session": self.reset_vpn_session, "verify_connectivity": self.verify_connectivity,
            "check_disk": self.check_disk, "compress_logs": self.compress_logs, "archive_to_s3": self.archive_to_s3,
            "query_idp_metadata": self.query_idp_metadata, "rotate_signing_cert": self.rotate_signing_cert,
            "test_saml_assertion": self.test_saml_assertion,
        }


TOOL_SCHEMAS = {
    "ping_host": {"host": "string"}, "check_vpn_status": {"user_id": "string"},
    "reset_vpn_session": {"user_id": "string"}, "verify_connectivity": {"host": "string"},
    "check_disk": {"host": "string"}, "compress_logs": {"older_than_days": "integer"},
    "archive_to_s3": {"path": "string"}, "query_idp_metadata": {"app_id": "string"},
    "rotate_signing_cert": {"app_id": "string", "key_id": "string"}, "test_saml_assertion": {"user_id": "string"},
}
