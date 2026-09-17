"""Chunk-level retrieval indexing, ranking metrics, and claim-level groundedness auditing.

Fixed from an earlier version of this lab: section-level chunks (not whole documents) carry status/effective-date
metadata, ranking is reported with recall@k / precision@k / MRR / nDCG@k instead of a
single unreachable precision bar, and a per-sentence claim audit checks citation
presence, numeric support inside the CITED text specifically, and staleness -- instead
of whole-context lexical word overlap.
"""

from __future__ import annotations
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from shared.datasets.loader import DATA_DIR

DEFAULT_KB_DIR = os.path.join(DATA_DIR, "enterprise_knowledge_base")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    status: str = "active"          # active | obsolete
    effective: str = ""


def load_chunks(kb_dir: str = DEFAULT_KB_DIR) -> List[Chunk]:
    """Section-level chunks for EVERY markdown file in kb_dir, including obsolete
    ones, with status/effective-date metadata parsed from each doc's own text."""
    if not os.path.exists(kb_dir):
        raise FileNotFoundError(f"Knowledge base directory not found at: {kb_dir}")
    chunks: List[Chunk] = []
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
            text = f.read()
        doc_id_match = re.search(r"Document ID\*\*:\s*(\S+)", text)
        doc_id = doc_id_match.group(1) if doc_id_match else fname
        status = "obsolete" if re.search(r"OBSOLETE|SUPERSEDED", text) else "active"
        eff_match = re.search(r"Effective Date\*\*:\s*([^\n]+)", text)
        eff = eff_match.group(1).strip() if eff_match else ""
        title_match = re.search(r"^#\s+(.*)", text, re.M)
        title = title_match.group(1).strip() if title_match else fname
        sections = re.split(r"\n(?=## )", text)
        body_sections = sections[1:] or sections
        for i, sec in enumerate(body_sections, start=1):
            sec = sec.strip()
            if sec:
                chunks.append(Chunk(f"{doc_id}#s{i}", doc_id, title, sec, status, eff))
    return chunks


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def mrr(ranked: List[str], relevant: Set[str]) -> float:
    return next((1 / (i + 1) for i, c in enumerate(ranked) if c in relevant), 0.0)


def ndcg_at_k(ranked: List[str], gains: Dict[str, int], k: int) -> float:
    dcg = sum(gains.get(c, 0) / math.log2(i + 2) for i, c in enumerate(ranked[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains.values(), reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


_CITE_RE = re.compile(r"\[([A-Z]+-[A-Z]+-\d+(?:#s\d+)?)\]")
_NUM_RE = re.compile(r"\$?\d+(?:[.,]\d+)?%?")


def claim_audit(answer: str, retrieved: Dict[str, Chunk]) -> Dict[str, object]:
    """Per-sentence citation presence, validity, numeric support in the CITED text,
    and staleness. `retrieved` maps chunk_id -> Chunk for every chunk actually shown
    to the generator, so an unsupported or stale citation can be distinguished from
    a citation the model invented outright."""
    rows = []
    for sent in [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 12]:
        cites = _CITE_RE.findall(sent)
        cited_text = " ".join(
            c.text for cid in cites for c in retrieved.values()
            if c.chunk_id == cid or c.doc_id == cid
        )
        valid = bool(cites) and bool(cited_text)
        nums = [n.strip("$%,.") for n in _NUM_RE.findall(_CITE_RE.sub("", sent))]
        unsupported = [
            n for n in nums
            if n and not re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", cited_text)
        ]
        obsolete = any(
            c.status == "obsolete" for c in retrieved.values()
            if c.doc_id in {x.split("#")[0] for x in cites}
        )
        rows.append({
            "sentence": sent[:70],
            "cited": bool(cites),
            "valid": valid,
            "unsupported_numbers": unsupported,
            "cites_obsolete": obsolete,
            "supported": valid and not unsupported and not obsolete,
        })
    n = len(rows) or 1
    return {
        "citation_coverage": round(sum(r["cited"] for r in rows) / n, 2),
        "supported_claim_rate": round(sum(r["supported"] for r in rows) / n, 2),
        "rows": rows,
    }
