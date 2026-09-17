"""Unit tests for Chapter 11 LangGraph chatbot and multi-framework evaluation."""

import os
import sys
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CHAPTER_DIR not in sys.path:
    sys.path.insert(0, CHAPTER_DIR)

from shared.datasets.loader import load_langgraph_support_cases
from graph import CustomerSupportGraph
from evaluators import LangSmithTraceEngine, TruLensFeedbackEngine, DeepEvalEngine, RagasEngine


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def test_dataset_loads_all_twelve_cases():
    cases = load_langgraph_support_cases()
    assert len(cases) == 12
    for c in cases:
        assert "id" in c
        assert "input_prompt" in c
        assert "expected_intent" in c
        assert "category" in c


def test_langgraph_customer_support_graph_execution():
    graph = CustomerSupportGraph()
    state = graph.invoke("I was charged twice for invoice #INV-9281. Please refund $79.00.")

    assert state["intent"] == "billing"
    assert len(state["tool_calls"]) >= 1
    assert len(state["trace_spans"]) >= 4
    assert len(state["final_answer"]) > 10

    root_span = state["trace_spans"][0]
    assert root_span["name"] == "agent.langgraph.run"
    assert root_span["status"] == "OK"
    assert root_span["duration_ms"] >= 0


def test_langsmith_trace_engine_passes_valid_trajectory():
    graph = CustomerSupportGraph()
    state = graph.invoke("What is the data retention policy under GDPR [DOC-SEC-202]?")

    ls_engine = LangSmithTraceEngine()
    eval_res = ls_engine.evaluate_run_tree(state["trace_spans"], expected_intent="policy_rag")

    assert eval_res["verdict"] == "PASSED"
    assert eval_res["compliance_score"] == 100.0
    assert eval_res["passed_rules"] == 4
    assert len(eval_res["diagnostics"]) == 4


def test_langsmith_detects_broken_trajectory():
    ls_engine = LangSmithTraceEngine()
    # Provide a malformed trace with missing router and generator
    bad_spans = [
        {"name": "tool.execution", "span_id": "s1", "status": "OK", "duration_ms": 10.0, "attributes": {}},
    ]
    eval_res = ls_engine.evaluate_run_tree(bad_spans)

    assert eval_res["verdict"] == "BLOCKED"
    assert eval_res["compliance_score"] < 100.0
    diag_map = {d["rule"]: d["status"] for d in eval_res["diagnostics"]}
    assert diag_map["LangSmithTrajectoryValidator"] == "FAIL"


def test_trulens_feedback_engine_evaluates_triad_and_tools():
    tl_engine = TruLensFeedbackEngine()
    query = "What is the data retention policy under GDPR [DOC-SEC-202]?"
    response = "Under GDPR document DOC-SEC-202, user transaction logs are retained for 7 years."
    contexts = ["DOC-SEC-202: User transaction logs and billing records must be retained for 7 years."]
    tool_results = [{"result": {"status": "success"}}]

    fb_res = tl_engine.evaluate_feedback(
        query=query,
        response=response,
        retrieved_contexts=contexts,
        tool_results=tool_results,
    )

    assert "trulens_groundedness" in fb_res
    assert "trulens_context_relevance" in fb_res
    assert "trulens_answer_relevance" in fb_res
    assert "trulens_tool_correctness" in fb_res
    assert fb_res["trulens_groundedness"].score >= 70.0
    assert fb_res["trulens_tool_correctness"].score == 100.0


def test_deepeval_and_ragas_evaluation_engines():
    de_engine = DeepEvalEngine()
    rag_engine = RagasEngine()

    prompt = "Where is my shipment #ORD-44910?"
    actual_output = "Order #ORD-44910 is currently out for delivery via FedEx Priority."
    expected_output = "Order #ORD-44910 is out for delivery with FedEx."
    ref_context = "Order #ORD-44910 shipped via FedEx Priority. Out for delivery today."

    de_scores = de_engine.evaluate(
        prompt=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieved_context=ref_context,
    )
    assert de_scores["deepeval_correctness"].score >= 70.0
    assert de_scores["deepeval_relevancy"].score >= 70.0

    rag_scores = rag_engine.evaluate(
        question=prompt,
        answer=actual_output,
        retrieved_contexts=[ref_context],
        reference_context=ref_context,
    )
    assert rag_scores["ragas_faithfulness"].score >= 70.0
    assert rag_scores["ragas_answer_relevance"].score >= 70.0

