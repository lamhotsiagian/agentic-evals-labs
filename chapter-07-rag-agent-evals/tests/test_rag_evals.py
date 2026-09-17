"""Unit tests for Chapter 7 RAG Agent Evals.

Several tests below pin down bugs an earlier version of this lab had, against
the now-fixed code: a 90-day claim
against a 30-day policy, a citation to the obsolete distractor document, an
uncited answer defaulting to high groundedness, and a retrieval-precision bar
that perfect retrieval could never meet.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "graders", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter"):
    sys.modules.pop(m, None)

from retriever import SemanticRetriever
from pipeline import RAGAgentPipeline
from evaluator import RAGEvaluator
from graders import Chunk, claim_audit, load_chunks, mrr, ndcg_at_k, precision_at_k, recall_at_k
from shared.datasets.loader import load_rag_enterprise_corpus

os.environ.setdefault("MOCK_LLM", "1")


def test_semantic_retriever():
    chunks = load_chunks()
    retriever = SemanticRetriever()
    retriever.index_chunks(chunks)

    results = retriever.retrieve("What is our customer refund policy?", top_k=2)
    assert len(results) == 2
    doc_ids = [r["doc_id"] for r in results]
    assert "DOC-FIN-303" in doc_ids


def test_load_chunks_indexes_every_file_including_the_legacy_distractor():
    # Central lesson: the physical legacy_distractor_v1.md file was never loaded
    # by the old whole-document loader. The chunk loader must index ALL markdown
    # files in the knowledge base directory, tagging the obsolete one correctly.
    chunks = load_chunks()
    doc_ids = {c.doc_id for c in chunks}
    assert "DOC-NOISE-999" in doc_ids, "legacy distractor file must be indexed, not silently skipped"

    noise_chunks = [c for c in chunks if c.doc_id == "DOC-NOISE-999"]
    assert noise_chunks and all(c.status == "obsolete" for c in noise_chunks)

    active_chunks = [c for c in chunks if c.doc_id == "DOC-FIN-303"]
    assert active_chunks and all(c.status == "active" for c in active_chunks)
    # Section-level, not whole-document: the finance policy has more than one chunk.
    assert len(active_chunks) >= 2


def test_claim_audit_catches_the_wrong_uncited_number():
    # Exact reproduction of the book's probe: a 90-day claim against a 30-day
    # policy used to score faithfulness 1.0 and groundedness 0.92 under lexical
    # word-overlap scoring, because every other word in the sentence overlaps
    # with the real policy text and only the number "90" is wrong.
    finance_text = (
        "## 1. 30-Day Money-Back Guarantee\n"
        "New subscriptions canceled within 30 calendar days of initial purchase "
        "qualify for a 100% full money-back guarantee. Refunds are credited back "
        "to the original payment method within 3 to 5 business days."
    )
    retrieved = {"DOC-FIN-303#s1": Chunk("DOC-FIN-303#s1", "DOC-FIN-303", "Customer Refund & Subscription Terms", finance_text)}

    correct = "New subscriptions canceled within 30 calendar days qualify for a full money-back guarantee [DOC-FIN-303]."
    wrong = "Refunds are available within 90 calendar days to the original payment method."

    good_audit = claim_audit(correct, retrieved)
    bad_audit = claim_audit(wrong, retrieved)

    assert good_audit["citation_coverage"] == 1.0
    assert good_audit["supported_claim_rate"] == 1.0

    assert bad_audit["citation_coverage"] == 0.0, "the wrong answer is not even cited"
    assert bad_audit["supported_claim_rate"] == 0.0, "an uncited claim must never be scored as supported"


def test_claim_audit_flags_citation_of_obsolete_evidence():
    # Central lesson: citing the injected non-refundable 2018 distractor used to
    # be scored as a "correct" citation just because the ID was in the retrieved
    # set. A citation to an obsolete/superseded document must fail the audit even
    # when the citation itself is syntactically well-formed.
    retrieved = {
        "DOC-NOISE-999#s1": Chunk(
            "DOC-NOISE-999#s1", "DOC-NOISE-999", "Legacy Corporate Terms & Archive",
            "Under the legacy 2018 corporate guidelines, all software sales were strictly non-refundable.",
            status="obsolete", effective="",
        ),
    }
    stale_cited = "All subscriptions are non-refundable [DOC-NOISE-999]."
    audit = claim_audit(stale_cited, retrieved)
    assert audit["citation_coverage"] == 1.0, "the citation itself is present and well-formed"
    assert audit["supported_claim_rate"] == 0.0, "citing obsolete evidence must not count as supported"
    assert audit["rows"][0]["cites_obsolete"] is True


def test_uncited_answer_scores_zero_groundedness_not_a_generous_default():
    # Central lesson: answers with zero citations used to receive a 0.8 default
    # citation_correctness score just because a doc_id string happened to appear
    # somewhere in the answer text. Zero citations must mean zero credit.
    evaluator = RAGEvaluator()
    retrieved_docs = [{
        "chunk_id": "DOC-FIN-303#s1", "doc_id": "DOC-FIN-303", "title": "Customer Refund & Subscription Terms",
        "content": "New subscriptions canceled within 30 calendar days qualify for a full money-back guarantee.",
        "status": "active", "effective": "January 1, 2026",
    }]
    scores = evaluator.evaluate_rag_output(
        question="What is our customer refund policy?",
        retrieved_docs=retrieved_docs,
        answer="Refunds are available within 90 calendar days to the original payment method.",
        expected_doc_ids=["DOC-FIN-303"],
    )
    assert scores["citation_coverage"].score == 0.0
    assert scores["groundedness"].score == 0.0
    assert scores["groundedness"].passed is False


def test_retrieval_precision_is_judged_against_what_is_achievable():
    # Central lesson: with only one relevant document at k=3, precision@3 is
    # capped at 0.33, so a fixed 0.5 bar fails PERFECT retrieval. Precision must
    # instead be judged against the number of relevant chunks that actually exist.
    evaluator = RAGEvaluator()
    retrieved_docs = [
        {"chunk_id": "DOC-FIN-303#s1", "doc_id": "DOC-FIN-303", "title": "t", "content": "the golden section", "status": "active"},
        {"chunk_id": "DOC-HR-101#s1", "doc_id": "DOC-HR-101", "title": "t", "content": "unrelated", "status": "active"},
        {"chunk_id": "DOC-SEC-202#s1", "doc_id": "DOC-SEC-202", "title": "t", "content": "unrelated", "status": "active"},
    ]
    scores = evaluator.evaluate_rag_output(
        question="q",
        retrieved_docs=retrieved_docs,
        answer="Answer [DOC-FIN-303].",
        expected_doc_ids=["DOC-FIN-303"],
        k=3,
    )
    assert scores["retrieval_recall"].score == 1.0
    assert scores["retrieval_precision"].score == round(1 / 3, 2)
    assert scores["retrieval_precision"].passed is True, "perfect retrieval of the one relevant chunk must pass"


def test_ranking_metrics_match_the_book_probe():
    # Reproduces the book's own probe ranking: the golden finance section is
    # retrieved first, but the obsolete distractor slips into second place.
    ranked = ["DOC-FIN-303#s1", "DOC-NOISE-999#s1", "DOC-FIN-303#s2", "DOC-HR-101#s1"]
    relevant = {"DOC-FIN-303#s1", "DOC-FIN-303#s2"}

    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert round(precision_at_k(ranked, relevant, 3), 2) == 0.67
    assert mrr(ranked, relevant) == 1.0


def test_ndcg_at_k_matches_a_hand_computed_example():
    # Independent, hand-verified check of the nDCG@k formula itself (the book's
    # console listing shows a result computed from gains it does not print, so
    # rather than guess those gains, verify the implementation directly):
    # ranked = [A, B, C], gains = {A: 2, B: 0, C: 1}. At k=3:
    #   DCG   = 2/log2(2) + 0/log2(3) + 1/log2(4) = 2 + 0 + 0.5           = 2.5
    #   IDCG  = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2 + 0.6309... + 0     = 2.6309...
    #   nDCG  = DCG / IDCG                                                = 0.9502...
    ranked = ["A", "B", "C"]
    gains = {"A": 2, "B": 0, "C": 1}
    import math
    expected_dcg = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    expected_idcg = 2 / math.log2(2) + 1 / math.log2(3) + 0 / math.log2(4)
    assert ndcg_at_k(ranked, gains, 3) == expected_dcg / expected_idcg
    assert round(ndcg_at_k(ranked, gains, 3), 2) == 0.95
    # Perfect ranking (highest gain first) always scores a perfect 1.0.
    assert ndcg_at_k(["A", "C", "B"], gains, 3) == 1.0


def test_rag_pipeline_and_evaluator():
    chunks = load_chunks()
    pipeline = RAGAgentPipeline()
    pipeline.index(chunks)
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

    assert scores["retrieval_recall"].score > 0.0
    assert scores["groundedness"].score >= 0.7
    assert scores["citation_coverage"].score >= 0.8


def test_unanswerable_question_triggers_abstention_not_a_guess():
    # Low-severity finding: abstention was never exercised. A question the
    # corpus cannot answer should not be forced into a confident-sounding guess.
    chunks = load_chunks()
    pipeline = RAGAgentPipeline()
    pipeline.index(chunks)

    res = pipeline.query("What is the CEO's approved private jet travel budget?", top_k=3)
    assert "don't know" in res["answer"].lower() or "cannot" in res["answer"].lower()


def test_whole_document_loader_still_available_for_generic_dataset_smoke_tests():
    # load_rag_enterprise_corpus() remains a generic whole-document loader used
    # elsewhere (shared/tests/test_shared.py); Chapter 7 itself now uses the
    # chunk-level load_chunks() defined in graders.py instead.
    docs = load_rag_enterprise_corpus()
    assert len(docs) >= 4
