"""Grounded, schema-aware call grading for Chapter 3.

Replaces name-level-only grading (which scores a hallucinated zip code or
a wrong recipient email the same as a correct one) with schema validation,
multiset precision/recall/F1 over calls, ordering checks, and provenance
(is each argument value actually grounded in the user's request or a
prior tool observation, or did the model invent it).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "get_weather":      {"required": {"zip_code": r"^\d{5}$"}, "side_effect": False},
    "search_customer":  {"required": {}, "optional": {"customer_id": r"^C-\d{3}$",
                                                        "name": r"^[A-Za-z .'-]{2,}$"}, "side_effect": False},
    "get_order":        {"required": {"order_id": r"^\d{4,5}$"}, "side_effect": False},
    "calculate_refund": {"required": {"order_id": r"^\d{4,5}$"}, "side_effect": False},
    "send_email":       {"required": {"to": r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", "subject": r".+",
                                       "body": r".+"}, "side_effect": True},
}


def validate_call(tool: str, args: Dict[str, Any]) -> List[str]:
    """Schema errors for one call: unknown tool, missing/extra/ill-formatted arguments."""
    if tool not in TOOL_SCHEMAS:
        return [f"unknown tool {tool}"]
    s = TOOL_SCHEMAS[tool]
    allowed = {**s.get("required", {}), **s.get("optional", {})}
    errs = [f"missing {k}" for k in s.get("required", {}) if k not in args]
    errs += [f"unexpected {k}" for k in args if k not in allowed]
    errs += [f"bad {k}={v!r}" for k, v in args.items()
             if k in allowed and not re.fullmatch(allowed[k], str(v))]
    return errs


@dataclass(frozen=True)
class Call:
    tool: str
    args: Tuple[Tuple[str, str], ...]

    @staticmethod
    def of(tool: str, **args) -> "Call":
        return Call(tool, tuple(sorted((k, str(v)) for k, v in args.items())))


def call_prf(expected: List[Call], actual: List[Call], match_args: bool = True) -> Dict[str, float]:
    """Multiset precision/recall/F1 over calls. Extra calls hurt precision."""
    key = (lambda c: c) if match_args else (lambda c: c.tool)
    remaining, tp = [key(c) for c in expected], 0
    for c in actual:
        if key(c) in remaining:
            remaining.remove(key(c))
            tp += 1
    p = tp / len(actual) if actual else (1.0 if not expected else 0.0)
    r = tp / len(expected) if expected else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 2), "recall": round(r, 2), "f1": round(f, 2)}


def order_violations(actual: List[Call], must_precede: List[Tuple[str, str]]) -> List[str]:
    """Partial-order check: only dependencies that matter (get_order before calculate_refund)."""
    first: Dict[str, int] = {}
    for i, c in enumerate(actual):
        first.setdefault(c.tool, i)
    return [f"{a} must precede {b}" for a, b in must_precede
            if a in first and b in first and first[a] > first[b]]


def ungrounded_args(actual: List[Call], user_prompt: str, observations: List[str]) -> List[str]:
    """Argument values that appear neither in the user's request nor in any PRIOR tool output."""
    out, seen = [], user_prompt.lower()
    for c, obs in zip(actual, observations + [""] * len(actual)):
        for k, v in c.args:
            if v.lower() not in seen:
                out.append(f"{c.tool}.{k}={v}")
        seen += " " + obs.lower()  # this call's output grounds only LATER calls
    return out
