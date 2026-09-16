"""Unit tests for Chapter 7 RAG Agent Evals."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
    sys.modules.pop(m, None)

from retriever import SemanticRetriever
from pipeline import RAGAgentPipeline
from evaluator import RAGEvaluator
from shared.datasets.loader import load_rag_enterprise_corpus


def test_semantic_retriever():
    corpus = load_rag_enterprise_corpus()
    retriever = SemanticRetriever()
    retriever.index_documents(corpus)

    results = retriever.retrieve("What is our customer refund policy?", top_k=2)
    assert len(results) == 2
    doc_ids = [r["doc_id"] for r in results]
    assert "DOC-FIN-303" in doc_ids


def test_rag_pipeline_and_evaluator():
    corpus = load_rag_enterprise_corpus()
    pipeline = RAGAgentPipeline()
    pipeline.retriever.index_documents(corpus)
    evaluator = RAGEvaluator()

    question = "What is our customer refund policy?"
    res = pipeline.query(question, top_k=2)
    assert "answer" in res
    assert len(res["retrieved_docs"]) == 2

    scores = evaluator.evaluate_rag_output(
        question=question,
        retrieved_docs=res["retrieved_docs"],
        answer=res["answer"],
        expected_doc_ids=["DOC-FIN-303"],
    )

    assert scores["retrieval_precision"].score > 0.0
    assert scores["faithfulness"].score >= 0.7
    assert scores["groundedness"].score >= 0.7
