"""Evaluator for RAG systems: Retrieval, Faithfulness, Groundedness, Citations."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from shared.models.schemas import MetricScore
from shared.metrics.quality import compute_faithfulness, compute_relevance


class RAGEvaluator:
    """Computes RAGAS/DeepEval style metrics for RAG agents."""

    def evaluate_rag_output(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        expected_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, MetricScore]:
        # 1. Retrieval Precision
        retrieved_ids = [d["doc_id"] for d in retrieved_docs]
        if expected_doc_ids:
            hits = sum(1 for ed in expected_doc_ids if ed in retrieved_ids)
            retrieval_prec = hits / max(1, len(retrieved_docs))
            retrieval_rec = hits / max(1, len(expected_doc_ids))
        else:
            retrieval_prec = 1.0
            retrieval_rec = 1.0

        # 2. Context Relevance
        all_context = " ".join([d["content"] for d in retrieved_docs])
        context_rel = compute_relevance(all_context, question)

        # 3. Faithfulness
        faithfulness = compute_faithfulness(answer, all_context)

        # 4. Citation Correctness
        cited_ids = set(re.findall(r"\[([A-Z0-9_-]+)\]", answer))
        if cited_ids:
            valid_citations = sum(1 for cid in cited_ids if cid in retrieved_ids)
            citation_score = valid_citations / len(cited_ids)
        else:
            # Default citation score if doc id referenced in text
            doc_mentioned = any(d["doc_id"] in answer for d in retrieved_docs)
            citation_score = 0.9 if doc_mentioned else 0.8

        # 5. Groundedness (composite of faithfulness and citation)
        groundedness = (faithfulness * 0.6) + (citation_score * 0.4)

        return {
            "retrieval_precision": MetricScore(
                name="retrieval_precision",
                score=round(retrieval_prec, 2),
                passed=(retrieval_prec >= 0.5),
            ),
            "retrieval_recall": MetricScore(
                name="retrieval_recall",
                score=round(retrieval_rec, 2),
                passed=(retrieval_rec >= 0.5),
            ),
            "context_relevance": MetricScore(
                name="context_relevance",
                score=round(context_rel, 2),
                passed=(context_rel >= 0.25),
            ),
            "faithfulness": MetricScore(
                name="faithfulness",
                score=round(faithfulness, 2),
                passed=(faithfulness >= 0.7),
            ),
            "groundedness": MetricScore(
                name="groundedness",
                score=round(groundedness, 2),
                passed=(groundedness >= 0.7),
            ),
            "citation_correctness": MetricScore(
                name="citation_correctness",
                score=round(citation_score, 2),
                passed=(citation_score >= 0.8),
            ),
        }
