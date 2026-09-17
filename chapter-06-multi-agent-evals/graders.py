"""Grounded multi-agent collaboration graders for Chapter 6.

Replaces message-counting metrics (a retransmission counted as a
successful handoff; any non-empty synthesis scored 1.0) with logical
edge-level handoff accounting, claim-provenance tracking (does an
unsourced number propagate from a worker into the executive synthesis),
and content-based role contracts.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

EXPECTED_HANDOFFS: Set[Tuple[str, str]] = {
    ("Supervisor", "Researcher"), ("Supervisor", "Analyst"),
    ("Researcher", "Synthesizer"), ("Analyst", "Synthesizer"), ("Synthesizer", "Supervisor"),
}

_claim = re.compile(r"\b\d+(?:\.\d+)?%|\b\d{2,}(?:\.\d+)?\b")
_source = re.compile(r"\[(?:src|source|doc)[:\-][^\]]+\]", re.I)


def handoff_report(messages) -> Dict[str, Any]:
    """Evaluate LOGICAL handoffs (edges), not raw message counts."""
    delivered: Counter = Counter()
    failed: Counter = Counter()
    for m in messages:
        (delivered if m.handoff_status == "success" else failed)[(m.sender, m.recipient)] += 1
    missing = sorted(e for e in EXPECTED_HANDOFFS if delivered[e] == 0)
    unexpected = sorted(e for e in (set(delivered) | set(failed)) if e not in EXPECTED_HANDOFFS)
    retransmissions = sum(max(0, delivered[e] - 1) + failed[e] for e in EXPECTED_HANDOFFS)
    first_try = sum(1 for e in EXPECTED_HANDOFFS if delivered[e] >= 1 and failed[e] == 0)
    return {
        "handoffs_expected": len(EXPECTED_HANDOFFS),
        "handoffs_completed": len(EXPECTED_HANDOFFS) - len(missing),
        "first_attempt_success_rate": round(first_try / len(EXPECTED_HANDOFFS), 2),
        "missing": missing, "unexpected_edges": unexpected, "retransmissions": retransmissions,
    }


def unsupported_quantitative_claims(messages) -> List[Dict[str, str]]:
    """Numbers asserted by any agent without a source tag in the same sentence."""
    out = []
    for m in messages:
        for sent in re.split(r"(?<=[.!?])\s+", m.content):
            nums = [n for n in _claim.findall(sent) if not re.fullmatch(r"20\d\d", n)]  # ignore years
            if nums and not _source.search(sent):
                out.append({"agent": m.sender, "claim": sent.strip()[:120], "numbers": ",".join(nums)})
    return out


def propagated_claims(messages, final_text: str) -> List[str]:
    """Unsupported worker claims that reached the final synthesis."""
    worker_claims = [c for c in unsupported_quantitative_claims(messages) if c["agent"] in {"Researcher", "Analyst"}]
    return [c["numbers"] for c in worker_claims if all(n in final_text for n in c["numbers"].split(","))]


def jaccard(a: str, b: str) -> float:
    """Cheap word-overlap proxy for duplicate-work detection; swap for
    embedding cosine similarity (nomic-embed-text) in production."""
    ta = set(re.findall(r"[a-z0-9%]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9%]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def new_numbers_in_synthesis(researcher_text: str, analyst_text: str, synthesis_text: str) -> List[str]:
    """Numbers in the synthesis that appear in neither worker's text --
    a fabricated figure the Synthesizer introduced on its own."""
    source_nums = set(_claim.findall(researcher_text)) | set(_claim.findall(analyst_text))
    synth_nums = set(_claim.findall(synthesis_text))
    return sorted(n for n in synth_nums if n not in source_nums)
