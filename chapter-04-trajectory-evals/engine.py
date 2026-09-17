"""Live IT helpdesk diagnostic agent for Chapter 4.

Replaces the fully scripted ITHelpdeskAgent (which hard-coded every step
regardless of what any model decided) with a real tool-calling loop: the
model chooses which diagnostic action to take next, sees the (simulated
but scenario-consistent) observation, and decides what to do about it.
Every step is tagged with its real kind (finding / tool_failure /
success) by the tool itself, feeding directly into the fixed
classify_step() logic in shared/evaluators/trajectory.py.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import AgentTrace
from tools import ITEnvironment, TOOL_SCHEMAS, SCENARIOS

DIAGNOSTIC_POLICY = """You are an IT helpdesk diagnostic agent. Investigate
the ticket by calling diagnostic tools one at a time. When a check reveals
a problem (an expired session, a full disk, an expired certificate),
call the matching fix tool, then re-verify with a check tool. Stop and
give a final answer once you have confirmed the issue is resolved, or
after you have tried every relevant tool without success."""


def _tools_schema() -> List[Dict[str, Any]]:
    out = []
    for name, props in TOOL_SCHEMAS.items():
        out.append({
            "type": "function",
            "function": {
                "name": name, "description": name.replace("_", " "),
                "parameters": {"type": "object", "required": list(props),
                                "properties": {k: {"type": "string"} for k in props}},
            },
        })
    return out


class ITHelpdeskAgent:
    """Diagnoses tickets live: the model picks tools, the environment answers."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b", max_steps: int = 8):
        self.provider = provider or get_model_provider()
        self.model = model
        self.max_steps = max_steps

    def diagnose_issue(self, scenario: str, inject_compress_loop: bool = False) -> AgentTrace:
        env = ITEnvironment(scenario, inject_compress_loop=inject_compress_loop)
        registry = env.registry()
        task = SCENARIOS[scenario]["task"]
        trace = AgentTrace(task=task)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": DIAGNOSTIC_POLICY},
            {"role": "user", "content": f"Ticket: {task}"},
        ]

        for _ in range(self.max_steps):
            reply = self.provider.chat(model=self.model, messages=messages, tools=_tools_schema(), temperature=0.0)
            messages.append(reply)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                trace.final_output = reply.get("content", "")
                break
            for tc in tool_calls:
                fn = tc["function"]
                name, args = fn["name"], fn.get("arguments", {})
                if name not in registry:
                    obs, kind, result = f"unknown tool {name}", "tool_failure", "error"
                else:
                    out = registry[name](**args)
                    obs, kind, result = out["observation"], out["kind"], ("success" if out["status"] == "success" else "error")
                trace.add_step(action=name, arguments=args, observation=obs, result=result, metadata={"kind": kind})
                messages.append({"role": "tool", "name": name, "content": obs})
        else:
            trace.final_output = "STEP_LIMIT_EXCEEDED"

        # Outcome is graded independently of the agent's own narration: did
        # every milestone for this scenario actually get reached.
        reached_actions = {s.action for s in trace.steps if s.result == "success"}
        trace.success = set(SCENARIOS[scenario]["milestones"]).issubset(reached_actions)
        return trace
