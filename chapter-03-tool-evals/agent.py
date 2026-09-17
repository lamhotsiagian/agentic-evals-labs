"""E-commerce customer service agent with NATIVE tool-calling.

Replaces the old keyword-routing script (which never let the model choose
a tool or build an argument) with a real Ollama /api/chat tool-calling
loop. The ToolRuntime, not the agent, executes and records every call
with its full arguments -- fixing the old bug where the trace logged a
different (truncated) argument set than what was actually executed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from shared.models.provider import LLMProvider, get_model_provider
from shared.models.schemas import AgentTrace
from tools import TOOL_REGISTRY
from graders import TOOL_SCHEMAS, validate_call, Call

SUPPORT_POLICY = """You are an e-commerce customer support agent. Use the
provided tools to look up orders, calculate refunds, search customers,
check weather, or send email notifications. Only call a tool when you
have real values for its arguments from the user's message or a prior
tool result -- never invent an order ID, zip code, or email address.
Look up an order with get_order before calling calculate_refund on it.
If the customer who owns the order does not match who is asking, stop
and ask them to confirm their account instead of proceeding."""


def _json_schema_for(tool: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    required = spec.get("required", {})
    optional = spec.get("optional", {})
    props = {k: {"type": "string", "pattern": p} for k, p in {**required, **optional}.items()}
    return {
        "type": "function",
        "function": {
            "name": tool,
            "description": TOOL_REGISTRY[tool].__doc__ or tool,
            "parameters": {"type": "object", "required": list(required), "properties": props},
        },
    }


TOOLS = [_json_schema_for(name, spec) for name, spec in TOOL_SCHEMAS.items()]


class ToolRuntime:
    """Validates, executes, and records calls exactly as executed -- never
    what the agent merely claims it called."""

    def __init__(self, registry: Dict[str, Any] = TOOL_REGISTRY, injector=None):
        self.registry = registry
        self.injector = injector
        self.calls: List[Dict[str, Any]] = []

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        errors = validate_call(name, args)
        if errors:
            result: Dict[str, Any] = {"status": "error", "error_type": "SchemaValidation", "error": "; ".join(errors)}
        elif name not in self.registry:
            result = {"status": "error", "error_type": "UnknownTool", "error": f"no such tool: {name}"}
        elif self.injector:
            result = self.injector.execute_tool_with_chaos(name, self.registry[name], **args)
        else:
            try:
                result = self.registry[name](**args)
            except TypeError as e:
                result = {"status": "error", "error_type": "ArgumentError", "error": str(e)}
        self.calls.append({"tool": name, "args": dict(args), "result": result})
        return result

    def as_call_objects(self) -> List[Call]:
        return [Call.of(c["tool"], **c["args"]) for c in self.calls]


class EcommerceCustomerAgent:
    """Agent that resolves customer requests by letting the model choose tools."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b", max_steps: int = 6):
        self.provider = provider or get_model_provider()
        self.model = model
        self.max_steps = max_steps

    def execute_task(self, prompt: str, injector=None) -> Dict[str, Any]:
        trace = AgentTrace(task=prompt)
        runtime = ToolRuntime(injector=injector)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SUPPORT_POLICY},
            {"role": "user", "content": prompt},
        ]
        final_answer = ""
        recovered = False

        for _ in range(self.max_steps):
            reply = self.provider.chat(model=self.model, messages=messages, tools=TOOLS, temperature=0.0)
            messages.append(reply)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:
                final_answer = reply.get("content", "")
                break
            for tc in tool_calls:
                fn = tc["function"]
                name, args = fn["name"], fn.get("arguments", {})
                result = runtime.execute(name, args)
                ok = result.get("status") == "success"
                trace.add_step(action=name, arguments=args, observation=str(result), result="success" if ok else "error")
                if not ok:
                    recovered = True
                messages.append({"role": "tool", "name": name, "content": str(result)})
        else:
            final_answer = "STEP_LIMIT_EXCEEDED"

        trace.final_output = final_answer
        trace.success = all(c["result"].get("status") == "success" for c in runtime.calls) or recovered or not runtime.calls

        return {
            "prompt": prompt,
            "tool_calls": runtime.calls,
            "final_answer": final_answer,
            "recovered": recovered,
            "trace": trace,
        }
