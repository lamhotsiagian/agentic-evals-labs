"""Paired regression gate: floors on confidence bounds plus non-inferiority
testing against a baseline, with a report rendered FROM the gate decision.

In an earlier version of this lab, the gate compared point estimates to absolute thresholds
only -- a drop from 97% to 90% still passed an 85% floor -- and the report
hard-coded its own thresholds separately from the gate's, so they could (and
did) disagree. Here, `markdown_report()` reads its numbers only from the
`gate()` decision and the same `GateRule` objects the gate used, so a report
that says PASSED cannot coexist with a gate that blocked.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shared.metrics.stats import mcnemar_exact, wilson_interval


@dataclass
class SuiteResult:
    suite: str
    metric: str
    case_ids: List[str]
    passed: List[bool]

    @property
    def rate(self) -> float:
        return sum(self.passed) / max(1, len(self.passed))


@dataclass
class GateRule:
    metric: str
    floor: Optional[float] = None            # absolute minimum on the LOWER 95% bound
    max_regression: float = 0.02             # non-inferiority margin vs baseline
    alpha: float = 0.05
    blocking: bool = True


def compare(candidate: SuiteResult, baseline: Optional[SuiteResult], rule: GateRule) -> Dict[str, object]:
    lo, hi = wilson_interval(sum(candidate.passed), len(candidate.passed))
    reasons: List[str] = []
    if rule.floor is not None and lo < rule.floor:
        reasons.append(f"{rule.metric}: lower 95% bound {lo:.3f} < floor {rule.floor}")

    paired_n = regressed = fixed = 0
    delta = 0.0
    p_value = 1.0
    if baseline is not None:
        base = dict(zip(baseline.case_ids, baseline.passed))
        pairs = [(base[c], p) for c, p in zip(candidate.case_ids, candidate.passed) if c in base]
        paired_n = len(pairs)
        regressed = sum(1 for old, new in pairs if old and not new)
        fixed = sum(1 for old, new in pairs if not old and new)
        p_value = mcnemar_exact(regressed, fixed)
        delta = (sum(n for _, n in pairs) - sum(o for o, _ in pairs)) / max(1, paired_n)
        if delta < -rule.max_regression and p_value < rule.alpha:
            reasons.append(
                f"{rule.metric}: significant regression {delta:+.3f} "
                f"(p={p_value:.3f}, {regressed} regressed / {fixed} fixed)"
            )

    return {
        "metric": rule.metric, "rate": round(candidate.rate, 3), "ci95": (round(lo, 3), round(hi, 3)),
        "paired_n": paired_n, "regressed": regressed, "fixed": fixed, "delta": round(delta, 3),
        "p_value": round(p_value, 4), "blocking": rule.blocking, "failures": reasons,
    }


def gate(results: Dict[str, SuiteResult], baselines: Dict[str, Optional[SuiteResult]], rules: List[GateRule]) -> Dict[str, object]:
    checks = [compare(results[r.metric], baselines.get(r.metric), r) for r in rules]
    blocking_failures = [f for ch in checks if ch["blocking"] for f in ch["failures"]]
    return {"passed": not blocking_failures, "blocking_failures": blocking_failures, "checks": checks}


def markdown_report(g: Dict[str, object], rules: List[GateRule]) -> str:
    """Rendered FROM the gate decision and its rules: the report cannot
    disagree with the gate, because it has no numbers of its own."""
    rule_by = {r.metric: r for r in rules}
    lines = [
        f"## Evaluation gate: {'PASSED' if g['passed'] else 'BLOCKED'}", "",
        "| Metric | Rate [95% CI] | Floor | Paired delta | Regressed / fixed | p | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for ch in g["checks"]:
        r = rule_by[ch["metric"]]
        floor_str = r.floor if r.floor is not None else "-"
        status = "FAIL" if ch["failures"] else "ok"
        lines.append(
            f"| {ch['metric']} | {ch['rate']:.3f} [{ch['ci95'][0]:.3f}, {ch['ci95'][1]:.3f}] | "
            f"{floor_str} | {ch['delta']:+.3f} | {ch['regressed']}/{ch['fixed']} | "
            f"{ch['p_value']:.3f} | {status} |"
        )
    if g["blocking_failures"]:
        lines += ["", "**Blocking failures:**"] + [f"- {f}" for f in g["blocking_failures"]]
    return "\n".join(lines)


def to_otel_span(span, service: str) -> Dict[str, object]:
    """Map a lab Tracer span to an OTLP-style record, so any OpenTelemetry
    backend can store and query evaluation runs next to production traces."""
    attrs = {"service.name": service, **{f"eval.{k}": v for k, v in span.attributes.items()}}
    return {
        "traceId": span.trace_id.replace("-", "").ljust(32, "0")[:32],
        "spanId": span.span_id.replace("-", "").ljust(16, "0")[:16],
        "parentSpanId": span.parent_span_id,
        "name": span.name,
        "startTimeUnixNano": int(span.start_time * 1e9),
        "endTimeUnixNano": int((span.end_time or span.start_time) * 1e9),
        "status": {"code": "STATUS_CODE_OK" if span.status == "ok" else "STATUS_CODE_ERROR"},
        "attributes": attrs,
    }
