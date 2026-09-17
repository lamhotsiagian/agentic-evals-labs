"""Real suite runners: each returns a gate.SuiteResult built from genuine
per-case pass/fail outcomes produced by that chapter's own fixed evaluator --
never a constant. This is the direct fix for "Platform metrics are
constants": `run_full_evaluation` used to return fixed values that no change
to any agent could ever move; these runners actually execute the agent and
grader they claim to.
"""

from __future__ import annotations
import os
import sys
from typing import List, Optional

from shared.models.provider import LLMProvider, get_model_provider
from shared.ui.components import load_chapter_modules
from gate import SuiteResult

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_task_success_suite(provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b", n: int = 8) -> SuiteResult:
    """Chapter 1's refined evaluator: correctness/faithfulness against real
    reference answers, not a hardcoded task_success_pct."""
    ch_dir = os.path.join(ROOT_DIR, "chapter-01-evaluation-foundations")
    load_chapter_modules(ch_dir)
    from agent import CustomerSupportAgent
    from evaluator import CustomerSupportEvaluator
    from shared.datasets.loader import load_customer_support_cases

    provider = provider or get_model_provider()
    cases = load_customer_support_cases(count=n)
    agent = CustomerSupportAgent(model_name=model, provider=provider)
    evaluator = CustomerSupportEvaluator()
    results = evaluator.run_suite(cases, agent)
    return SuiteResult(
        suite="chapter1_task_success", metric="task_success",
        case_ids=[r.case_id for r in results], passed=[bool(r.passed) for r in results],
    )


def run_tool_accuracy_suite(provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b") -> SuiteResult:
    """Chapter 3's call graders: schema-validated, multiset-scored tool
    calling, not name-only matching."""
    ch_dir = os.path.join(ROOT_DIR, "chapter-03-tool-evals")
    load_chapter_modules(ch_dir)
    from agent import EcommerceCustomerAgent
    from evaluator import ToolCallingEvaluator
    from shared.datasets.loader import load_ecommerce_tool_cases

    provider = provider or get_model_provider()
    cases = load_ecommerce_tool_cases()
    agent = EcommerceCustomerAgent(model=model, provider=provider)
    evaluator = ToolCallingEvaluator()
    case_ids, passed = [], []
    for case in cases:
        res = agent.execute_task(case.input_prompt)
        scores = evaluator.evaluate_execution(case.expected_tools, res)
        case_ids.append(case.id)
        passed.append(all(m.passed for m in scores.values()))
    return SuiteResult(suite="chapter3_tool_accuracy", metric="tool_accuracy", case_ids=case_ids, passed=passed)


def run_groundedness_suite(provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b") -> SuiteResult:
    """Chapter 7's claim audit: per-sentence citation and numeric support,
    not whole-context lexical overlap."""
    ch_dir = os.path.join(ROOT_DIR, "chapter-07-rag-agent-evals")
    load_chapter_modules(ch_dir)
    from pipeline import RAGAgentPipeline
    from evaluator import RAGEvaluator
    from graders import load_chunks

    provider = provider or get_model_provider()
    questions = [
        ("What is our customer refund policy?", ["DOC-FIN-303"]),
        ("What are the password and MFA requirements?", ["DOC-SEC-202"]),
        ("What is the remote work equipment stipend?", ["DOC-HR-101"]),
        ("What is the Sev-1 incident paging SLA?", ["DOC-ENG-404"]),
    ]
    chunks = load_chunks()
    pipeline = RAGAgentPipeline(provider=provider, generator_model=model)
    pipeline.index(chunks)
    evaluator = RAGEvaluator()
    case_ids, passed = [], []
    for i, (q, expected) in enumerate(questions):
        res = pipeline.query(q, top_k=2)
        scores = evaluator.evaluate_rag_output(question=q, retrieved_docs=res["retrieved_docs"], answer=res["answer"], expected_doc_ids=expected)
        case_ids.append(f"RAG-{i:03d}")
        passed.append(bool(scores["groundedness"].passed))
    return SuiteResult(suite="chapter7_groundedness", metric="groundedness", case_ids=case_ids, passed=passed)


def run_safety_suite(provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b", guardrails_enabled: bool = True) -> SuiteResult:
    """Chapter 8's ledger-based attack success: graded from the tool
    gateway's side-effect ledger, never a self-reported flag."""
    ch_dir = os.path.join(ROOT_DIR, "chapter-08-safety-evals")
    load_chapter_modules(ch_dir)
    from target_agent import BankingSupportAgent
    from evaluator import SafetyEvaluator
    from redteam import load_full_redteam_dataset

    provider = provider or get_model_provider()
    attacks = load_full_redteam_dataset()
    agent = BankingSupportAgent(provider=provider, model=model, guardrails_enabled=guardrails_enabled)
    evaluator = SafetyEvaluator()
    res = evaluator.evaluate_suite(attacks, agent)
    return SuiteResult(
        suite="chapter8_safety", metric="safety",
        case_ids=[r["attack_id"] for r in res["results"]], passed=[bool(r["is_safe"]) for r in res["results"]],
    )


def run_recovery_suite(fault_rates: Optional[dict] = None, n: int = 20, seed: int = 0) -> SuiteResult:
    """Chapter 9's outcome classifier: a resilient run "recovers" only if it
    resolves the request (correct or a disclosed stale fallback), never if
    it crashes or silently serves a malformed response. No model call --
    this suite always runs, even with Ollama offline."""
    ch_dir = os.path.join(ROOT_DIR, "chapter-09-robustness-evals")
    load_chapter_modules(ch_dir)
    from chaos import ChaosInjector, CircuitBreaker, VirtualClock
    from resilient_agent import CACHE, ResilientAgent, seed_cache
    from evaluator import _build_faults

    fault_rates = fault_rates if fault_rates is not None else {"http_500_transient": 0.4, "outage": 0.1}
    cases = [f"ORD-{i:04d}" for i in range(1001, 1001 + n)]
    CACHE.clear()
    seed_cache(cases[: n // 2])
    clock = VirtualClock()
    chaos = ChaosInjector(faults=_build_faults(fault_rates), clock=clock, seed=seed)
    breaker = CircuitBreaker(clock)
    agent = ResilientAgent(chaos, breaker=breaker)
    case_ids, passed = [], []
    for cid in cases:
        r = agent.process(cid)
        case_ids.append(cid)
        passed.append(r["outcome"] in ("correct", "degraded_stale"))
    return SuiteResult(suite="chapter9_recovery", metric="recovery", case_ids=case_ids, passed=passed)


ALL_SUITES = {
    "task_success": run_task_success_suite,
    "tool_accuracy": run_tool_accuracy_suite,
    "groundedness": run_groundedness_suite,
    "safety": run_safety_suite,
    "recovery": run_recovery_suite,
}
