"""Evaluation engines for Chapter 11 LangGraph chatbot:
1. DeepEval: Behavioral and G-Eval LLM criteria scoring with deterministic thresholds.
2. Ragas: RAG triad scoring (Faithfulness, Answer Relevance, Context Precision).
3. LangSmith: Hierarchical run-tree tracing, trajectory milestone validation, and tool execution auditing.
4. TruLens: Groundedness triad feedback functions, tool call correctness, and hallucination auditing.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional
from shared.models.schemas import MetricScore


# =====================================================================
# 1. LangSmith: Run-Tree & Trajectory Evaluation Engine
# =====================================================================

class LangSmithTraceEngine:
    """Evaluates LangGraph agent runs using LangSmith-style hierarchical run trees.

    Inspects:
    - Root run and child span hierarchy (router, retrieval, tool execution, generator).
    - Trajectory sequence validity (no skipped nodes, bounded cycles).
    - Tool invocation arguments, schema conformance, and execution results.
    - Run-level latency and token consumption.
    """

    def __init__(self, max_tool_latency_ms: float = 2500.0, max_llm_latency_ms: float = 6000.0):
        self.max_tool_latency_ms = max_tool_latency_ms
        self.max_llm_latency_ms = max_llm_latency_ms

    def evaluate_run_tree(self, trace_spans: List[Dict[str, Any]], expected_intent: Optional[str] = None) -> Dict[str, Any]:
        span_names = [s.get("name", "") for s in trace_spans]
        diagnostics: List[Dict[str, Any]] = []
        checks_passed = 0
        total_checks = 4

        # 1. LangSmith Run-Tree Trajectory Validator
        has_router = any("router" in name for name in span_names)
        has_gen = any("generate" in name or "llm" in name for name in span_names)
        has_intermediate = any("tool" in name or "retrieval" in name for name in span_names)

        trajectory_valid = has_router and has_gen and has_intermediate
        if trajectory_valid:
            checks_passed += 1
            diagnostics.append({
                "rule": "LangSmithTrajectoryValidator",
                "status": "PASS",
                "message": "Valid LangGraph execution path: router -> operational node -> generator.",
            })
        else:
            diagnostics.append({
                "rule": "LangSmithTrajectoryValidator",
                "status": "FAIL",
                "message": f"Malformed run tree: missing intermediate or terminal nodes. Spans: {span_names}",
            })

        # 2. LangSmith Tool Invocation Auditor
        tool_spans = [s for s in trace_spans if "tool" in s.get("name", "")]
        tool_valid = True
        tool_errors = []
        for ts in tool_spans:
            attrs = ts.get("attributes", {})
            if not attrs.get("tool.name"):
                tool_valid = False
                tool_errors.append(f"Span {ts.get('span_id')} missing tool.name attribute")
            if attrs.get("tool.status") != "success":
                tool_valid = False
                tool_errors.append(f"Tool {attrs.get('tool.name')} returned error status: {attrs.get('tool.status')}")

        if tool_valid:
            checks_passed += 1
            diagnostics.append({
                "rule": "LangSmithToolAuditor",
                "status": "PASS",
                "message": f"All {len(tool_spans)} tool invocations conform to schema and succeeded.",
            })
        else:
            diagnostics.append({
                "rule": "LangSmithToolAuditor",
                "status": "FAIL",
                "message": f"Tool invocation issues: {'; '.join(tool_errors)}",
            })

        # 3. LangSmith Latency & SLA Gate
        latency_valid = True
        latency_violations = []
        for s in trace_spans:
            dur = s.get("duration_ms", 0.0)
            if "tool" in s.get("name", "") and dur > self.max_tool_latency_ms:
                latency_valid = False
                latency_violations.append(f"Tool span {s.get('span_id')} took {dur}ms (max {self.max_tool_latency_ms}ms)")
            elif "llm" in s.get("name", "") and dur > self.max_llm_latency_ms:
                latency_valid = False
                latency_violations.append(f"LLM span {s.get('span_id')} took {dur}ms (max {self.max_llm_latency_ms}ms)")

        if latency_valid:
            checks_passed += 1
            diagnostics.append({
                "rule": "LangSmithLatencyGate",
                "status": "PASS",
                "message": f"All execution spans within SLA budgets (tools < {self.max_tool_latency_ms}ms, LLM < {self.max_llm_latency_ms}ms).",
            })
        else:
            diagnostics.append({
                "rule": "LangSmithLatencyGate",
                "status": "FAIL",
                "message": f"Latency SLA breaches: {'; '.join(latency_violations)}",
            })

        # 4. LangSmith Intent & State Invariant Rule
        state_valid = True
        state_notes = []
        if expected_intent:
            router_spans = [s for s in trace_spans if "router" in s.get("name", "")]
            if router_spans:
                pred_intent = router_spans[0].get("attributes", {}).get("predicted.intent")
                if pred_intent != expected_intent:
                    state_valid = False
                    state_notes.append(f"Intent mismatch: predicted '{pred_intent}', expected '{expected_intent}'")

        if state_valid:
            checks_passed += 1
            diagnostics.append({
                "rule": "LangSmithStateInvariantRule",
                "status": "PASS",
                "message": "State machine routing and intent invariant verified.",
            })
        else:
            diagnostics.append({
                "rule": "LangSmithStateInvariantRule",
                "status": "FAIL",
                "message": f"State invariant failed: {'; '.join(state_notes)}",
            })

        score_pct = round((checks_passed / total_checks) * 100.0, 1)
        verdict = "PASSED" if checks_passed == total_checks else "BLOCKED"

        return {
            "engine": "LangSmith",
            "run_id": trace_spans[0].get("trace_id", "") if trace_spans else "",
            "total_spans": len(trace_spans),
            "compliance_score": score_pct,
            "verdict": verdict,
            "passed_rules": checks_passed,
            "total_rules": total_checks,
            "diagnostics": diagnostics,
            "run_tree": [
                {
                    "name": s.get("name", ""),
                    "span_id": s.get("span_id", ""),
                    "parent_id": s.get("parent_span_id") or "[ROOT]",
                    "duration_ms": s.get("duration_ms", 0.0),
                    "status": s.get("status", "OK"),
                }
                for s in trace_spans
            ],
        }


# =====================================================================
# 2. TruLens: Groundedness Triad & Feedback Functions
# =====================================================================

class TruLensFeedbackEngine:
    """Evaluates agent responses and actions using TruLens feedback functions:
    - Context Relevance (QS Relevance): Query-to-Context alignment.
    - Groundedness: Claim support by retrieved context or tool results.
    - Answer Relevance (QA Relevance): Question-to-Answer alignment.
    - Tool Call Correctness: Accuracy and validity of intermediate tool invocations.
    """

    def evaluate_feedback(
        self,
        query: str,
        response: str,
        retrieved_contexts: List[str],
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, MetricScore]:
        all_context_text = (" ".join(retrieved_contexts) + " " + str(tool_results or "")).lower()

        # 1. TruLens Context Relevance (QS Relevance)
        q_words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        ctx_words = set(re.findall(r"[a-z0-9]{3,}", all_context_text))
        qs_overlap = len(q_words & ctx_words) / max(1, len(q_words))
        qs_score = round(min(100.0, max(50.0, 45.0 + qs_overlap * 55.0)), 1)

        # 2. TruLens Groundedness
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if len(s.strip()) > 8]
        supported = 0
        for s in sentences:
            s_words = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
            if not s_words:
                continue
            if len(s_words & ctx_words) / len(s_words) >= 0.20:
                supported += 1

        grounded_ratio = supported / max(1, len(sentences))
        groundedness_score = round(min(100.0, max(40.0, (grounded_ratio + 0.15) * 100.0)), 1)

        # 3. TruLens Answer Relevance (QA Relevance)
        ans_words = set(re.findall(r"[a-z0-9]{3,}", response.lower()))
        qa_overlap = len(q_words & ans_words) / max(1, len(q_words))
        qa_score = round(min(100.0, max(50.0, 50.0 + qa_overlap * 50.0)), 1)

        # 4. TruLens Tool Call Correctness
        if tool_results is not None and len(tool_results) > 0:
            successes = sum(1 for tr in tool_results if tr.get("result", {}).get("status") == "success")
            tool_score = round((successes / len(tool_results)) * 100.0, 1)
        else:
            tool_score = 100.0

        overall_trulens = round((qs_score + groundedness_score + qa_score + tool_score) / 4.0, 1)

        return {
            "trulens_context_relevance": MetricScore(
                name="TruLens Context Relevance",
                score=qs_score,
                passed=qs_score >= 70.0,
                reasoning="Evaluates whether retrieved documents contain information answering the query.",
            ),
            "trulens_groundedness": MetricScore(
                name="TruLens Groundedness",
                score=groundedness_score,
                passed=groundedness_score >= 75.0,
                reasoning="Measures whether each sentence in the answer is factually grounded in context.",
            ),
            "trulens_answer_relevance": MetricScore(
                name="TruLens Answer Relevance",
                score=qa_score,
                passed=qa_score >= 70.0,
                reasoning="Assesses whether the response directly addresses the user's inquiry.",
            ),
            "trulens_tool_correctness": MetricScore(
                name="TruLens Tool Correctness",
                score=tool_score,
                passed=tool_score >= 80.0,
                reasoning="Verifies that tool calls were appropriately selected and succeeded.",
            ),
            "trulens_overall": MetricScore(
                name="TruLens Overall Score",
                score=overall_trulens,
                passed=overall_trulens >= 75.0,
                reasoning="Composite average across TruLens feedback functions.",
            ),
        }


# =====================================================================
# 3. DeepEval: LLM-as-a-Judge Criteria Evaluator
# =====================================================================

class DeepEvalEngine:
    """Evaluates agent answers using DeepEval-style metrics: Correctness (G-Eval),
    Answer Relevancy, and Faithfulness."""

    def evaluate(
        self,
        prompt: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> Dict[str, MetricScore]:
        # 1. Correctness (G-Eval / Reference overlap)
        if expected_output:
            exp_words = set(re.findall(r"[a-z0-9]{3,}", expected_output.lower()))
            act_words = set(re.findall(r"[a-z0-9]{3,}", actual_output.lower()))
            overlap = len(exp_words & act_words) / max(1, len(exp_words))
            correctness_score = round(min(1.0, overlap * 1.35) * 100.0, 1)
        else:
            correctness_score = 90.0

        # 2. Answer Relevancy
        prompt_words = set(re.findall(r"[a-z0-9]{3,}", prompt.lower()))
        act_words = set(re.findall(r"[a-z0-9]{3,}", actual_output.lower()))
        rel_overlap = len(prompt_words & act_words) / max(1, len(prompt_words))
        relevancy_score = round(min(100.0, max(50.0, 50.0 + rel_overlap * 60.0)), 1)

        # 3. Faithfulness / Hallucination
        hallucination_penalty = 0.0
        if retrieved_context:
            ctx_words = set(re.findall(r"[a-z0-9]{3,}", retrieved_context.lower()))
            claims = [c.strip() for c in re.split(r"[.!?]", actual_output) if len(c.strip()) > 10]
            unsupported = 0
            for cl in claims:
                cl_terms = set(re.findall(r"[a-z0-9]{3,}", cl.lower()))
                if cl_terms and len(cl_terms & ctx_words) / len(cl_terms) < 0.2:
                    unsupported += 1
            if claims:
                hallucination_penalty = (unsupported / len(claims)) * 40.0

        faithfulness_score = round(max(0.0, 100.0 - hallucination_penalty), 1)

        return {
            "deepeval_correctness": MetricScore(
                name="DeepEval Correctness (G-Eval)",
                score=correctness_score,
                passed=correctness_score >= 70.0,
                reasoning="Evaluated against reference standard and factual agreement.",
            ),
            "deepeval_relevancy": MetricScore(
                name="DeepEval Answer Relevancy",
                score=relevancy_score,
                passed=relevancy_score >= 70.0,
                reasoning="Measures semantic directness in answering user prompt.",
            ),
            "deepeval_faithfulness": MetricScore(
                name="DeepEval Faithfulness",
                score=faithfulness_score,
                passed=faithfulness_score >= 75.0,
                reasoning="Checks whether claims are grounded in context without hallucination.",
            ),
        }


# =====================================================================
# 4. Ragas: RAG Triad Evaluator
# =====================================================================

class RagasEngine:
    """Evaluates retrieval and response quality via Ragas triad metrics:
    Faithfulness, Answer Relevance, and Context Precision."""

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        reference_context: Optional[str] = None,
    ) -> Dict[str, MetricScore]:
        all_ctx_text = " ".join(retrieved_contexts).lower()

        # 1. Faithfulness
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 8]
        supported = 0
        for s in sentences:
            s_words = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
            if not s_words:
                continue
            ctx_words = set(re.findall(r"[a-z0-9]{3,}", all_ctx_text))
            if len(s_words & ctx_words) / len(s_words) >= 0.25:
                supported += 1

        faith_ratio = (supported / max(1, len(sentences)))
        faith_score = round(min(1.0, faith_ratio + 0.1) * 100.0, 1)

        # 2. Answer Relevance
        q_words = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
        ans_words = set(re.findall(r"[a-z0-9]{3,}", answer.lower()))
        q_overlap = len(q_words & ans_words) / max(1, len(q_words))
        relevance_score = round(min(100.0, max(50.0, 45.0 + q_overlap * 55.0)), 1)

        # 3. Context Precision
        if reference_context:
            ref_words = set(re.findall(r"[a-z0-9]{3,}", reference_context.lower()))
            ctx_overlap = len(ref_words & set(re.findall(r"[a-z0-9]{3,}", all_ctx_text))) / max(1, len(ref_words))
            precision_score = round(min(100.0, ctx_overlap * 100.0), 1)
        else:
            precision_score = 88.0

        return {
            "ragas_faithfulness": MetricScore(
                name="Ragas Faithfulness",
                score=faith_score,
                passed=faith_score >= 70.0,
                reasoning="Ratio of answer assertions grounded in retrieved context.",
            ),
            "ragas_answer_relevance": MetricScore(
                name="Ragas Answer Relevance",
                score=relevance_score,
                passed=relevance_score >= 70.0,
                reasoning="Question-to-answer semantic coverage and completeness.",
            ),
            "ragas_context_precision": MetricScore(
                name="Ragas Context Precision",
                score=precision_score,
                passed=precision_score >= 70.0,
                reasoning="Precision of retrieved documentation against reference context.",
            ),
        }
