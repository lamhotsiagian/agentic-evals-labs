"""Evaluator for RAG systems: chunk-ranking metrics and claim-level groundedness.

Replaces the earlier whole-context lexical word-overlap scoring (which passed a
90-day refund claim against a 30-day policy at faithfulness 1.0) with the claim
audit from graders.py, and replaces a single unreachable retrieval-precision bar
with recall@k / precision@k / MRR reported against the number of relevant chunks
that actually exist.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set

from shared.models.schemas import MetricScore
from shared.metrics.quality import compute_relevance
from graders import Chunk, claim_audit, mrr, ndcg_at_k, precision_at_k, recall_at_k


class RAGEvaluator:
    """Computes retrieval-ranking and claim-level groundedness metrics for the RAG agent."""

    def evaluate_rag_output(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        expected_doc_ids: Optional[List[str]] = None,
        expected_chunk_ids: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, MetricScore]:
        ranked = [d.get("chunk_id", d["doc_id"]) for d in retrieved_docs]
        eval_k = k or len(retrieved_docs) or 1

        if expected_chunk_ids:
            relevant: Set[str] = set(expected_chunk_ids)
        elif expected_doc_ids:
            # Doc-level relevance labels: judge against whichever chunk(s) from
            # an expected document were actually retrieved (so recall/precision
            # are computed at the same granularity as `ranked`). Only fall back
            # to the bare doc id -- which can never appear in `ranked` -- when
            # NO chunk from that document was retrieved at all, so a missed
            # document still counts against recall exactly once.
            relevant = set()
            for expected_id in expected_doc_ids:
                matched = {d.get("chunk_id", d["doc_id"]) for d in retrieved_docs if d["doc_id"] == expected_id}
                relevant |= matched if matched else {expected_id}
        else:
            relevant = set()

        if relevant:
            retrieval_recall = recall_at_k(ranked, relevant, eval_k)
            retrieval_prec = precision_at_k(ranked, relevant, eval_k)
            retrieval_mrr = mrr(ranked, relevant)
            # Precision judged against what's actually achievable: with fewer
            # relevant chunks than k, perfect retrieval cannot reach precision 1.0.
            achievable_prec = min(len(relevant), eval_k) / eval_k if eval_k else 1.0
            precision_passed = retrieval_prec >= achievable_prec - 1e-9
            recall_passed = retrieval_recall >= 0.5
            mrr_passed = retrieval_mrr > 0
        else:
            retrieval_recall = retrieval_prec = retrieval_mrr = 1.0
            precision_passed = recall_passed = mrr_passed = True

        all_context = " ".join(d.get("content", "") for d in retrieved_docs)
        context_rel = compute_relevance(all_context, question)

        chunk_lookup: Dict[str, Chunk] = {}
        for d in retrieved_docs:
            cid = d.get("chunk_id", d["doc_id"])
            chunk_lookup[cid] = Chunk(
                chunk_id=cid,
                doc_id=d["doc_id"],
                title=d.get("title", ""),
                text=d.get("content", ""),
                status=d.get("status", "active"),
                effective=d.get("effective", ""),
            )

        audit = claim_audit(answer, chunk_lookup)
        citation_coverage = audit["citation_coverage"]
        supported_claim_rate = audit["supported_claim_rate"]
        cites_obsolete = any(r["cites_obsolete"] for r in audit["rows"])

        # An uncited answer scores zero groundedness -- no generous 0.8 default.
        groundedness = supported_claim_rate if citation_coverage > 0 else 0.0

        return {
            "retrieval_recall": MetricScore(
                name="retrieval_recall", score=round(retrieval_recall, 2), passed=recall_passed,
            ),
            "retrieval_precision": MetricScore(
                name="retrieval_precision", score=round(retrieval_prec, 2), passed=precision_passed,
                metadata={"achievable_at_k": round(min(len(relevant), eval_k) / eval_k, 2) if relevant and eval_k else 1.0},
            ),
            "retrieval_mrr": MetricScore(
                name="retrieval_mrr", score=round(retrieval_mrr, 2), passed=mrr_passed,
            ),
            "context_relevance": MetricScore(
                name="context_relevance", score=round(context_rel, 2), passed=(context_rel >= 0.25),
            ),
            "citation_coverage": MetricScore(
                name="citation_coverage", score=citation_coverage, passed=(citation_coverage >= 0.8),
            ),
            "supported_claim_rate": MetricScore(
                name="supported_claim_rate", score=supported_claim_rate, passed=(supported_claim_rate >= 0.8),
            ),
            "groundedness": MetricScore(
                name="groundedness", score=round(groundedness, 2), passed=(groundedness >= 0.7),
            ),
            "cites_obsolete_evidence": MetricScore(
                name="cites_obsolete_evidence", score=(1.0 if cites_obsolete else 0.0), passed=not cites_obsolete,
            ),
        }
