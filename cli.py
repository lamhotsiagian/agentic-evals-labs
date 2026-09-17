"""Unified Command Line Interface (CLI) for Agentic Evals Labs (Chapters 1–10).

Run evaluations directly from the terminal without launching a web browser or Streamlit UI.

Usage:
    python cli.py --chapter 1
    python cli.py --chapter 2
    python cli.py --chapter all
    python cli.py --list
"""

from __future__ import annotations
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _reset_local_modules():
    for m in (
        "agent", "evaluator", "pipeline", "tools", "engine", "judge",
        "calibration", "system", "graders", "retriever", "chaos", "resilient_agent",
        "eval_platform", "reporter", "gateway", "target_agent", "redteam",
        "gate", "suites",
    ):
        sys.modules.pop(m, None)


def run_chapter_1(count: int = 3, model: str = "qwen2.5:3b"):
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-01-evaluation-foundations"))
    from agent import CustomerSupportAgent
    from evaluator import CustomerSupportEvaluator
    from shared.datasets.loader import load_customer_support_cases

    print("\n" + "=" * 70)
    print("CHAPTER 1: Agent Evaluation Fundamentals")
    print(f"Model: {model} | Test cases: {count}")
    print("=" * 70)

    agent = CustomerSupportAgent(model_name=model)
    evaluator = CustomerSupportEvaluator()
    cases = load_customer_support_cases(count=count)

    results = evaluator.run_suite(cases, agent)
    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[Case {i}] ID: {r.case_id} | Status: [{status}]")
        print(f"  Query:      {r.input_prompt}")
        print(f"  Response:   {r.actual_output[:120]}...")
        print(f"  Latency:    {r.latency_seconds:.3f}s | Tokens: {r.tokens_used} | Cost: ${r.cost_estimate_usd:.5f}")
        for m_name, m_score in r.metrics.items():
            print(f"    - {m_name}: score={m_score.score} (passed={m_score.passed})")

    passed_count = sum(1 for r in results if r.passed)
    print("\n" + "-" * 70)
    print(f"Summary: {passed_count}/{len(results)} passed ({(passed_count/len(results))*100:.1f}%) | Total Tokens: {sum(r.tokens_used for r in results)}")
    print("=" * 70)


def run_chapter_2():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-02-agent-architecture"))
    from pipeline import ArchitecturePipeline
    from evaluator import ArchitectureEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 2: Agent Architecture Evaluation (Planner -> Executor -> Verifier)")
    print("=" * 70)

    pipeline = ArchitecturePipeline()
    evaluator = ArchitectureEvaluator()

    task = "Plan a 3-day trip to Tokyo with cultural sights under $1200"
    constraints = {"destination": "Tokyo", "days": 3, "budget": 1200}
    print(f"Task: {task}")

    result = pipeline.run(task, constraints, inject_failure_on_first_try=False)
    scores = evaluator.evaluate_run(result)

    print(f"Pipeline Result: {'SUCCESS' if result['success'] else 'FAILED'} (Retries: {result['retries']})")
    print("Execution Trace:")
    for step in result["pipeline_log"]:
        print(f"  [{step['node'].upper()}] -> {step['status']}")

    print("\nEvaluation Scores:")
    for k, v in scores.items():
        print(f"  - {k}: {v.score} (passed={v.passed})")
    print("=" * 70)


def run_chapter_3():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-03-tool-evals"))
    from agent import EcommerceCustomerAgent
    from evaluator import ToolCallingEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 3: Tool-Calling Evaluations (Selection, Arguments & Recovery)")
    print("=" * 70)

    agent = EcommerceCustomerAgent()
    evaluator = ToolCallingEvaluator()

    inquiry = "Find order #1234 and calculate refund eligibility."
    print(f"Inquiry: {inquiry}")
    res = agent.execute_task(inquiry)

    print("Tool Calls Executed:")
    for tc in res["tool_calls"]:
        print(f"  - Tool: {tc['tool']} | Args: {tc['args']} | Status: {tc['result'].get('status')}")
    print(f"Final Agent Answer: {res['final_answer']}")

    scores = evaluator.evaluate_execution(["get_order", "calculate_refund"], res)
    print("\nTool Evaluation Metrics:")
    for k, v in scores.items():
        print(f"  - {k}: {v.score} (passed={v.passed})")
    print("=" * 70)


def run_chapter_4():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-04-trajectory-evals"))
    from engine import ITHelpdeskAgent
    from evaluator import ITTrajectoryEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 4: Agent Trajectory Evaluations & Loop Penalties")
    print("=" * 70)

    agent = ITHelpdeskAgent()
    evaluator = ITTrajectoryEvaluator()

    scenario = "vpn"
    from tools import SCENARIOS
    print(f"Ticket: {SCENARIOS[scenario]['task']}")
    trace = agent.diagnose_issue(scenario)

    print(f"Trajectory Steps ({len(trace.steps)} steps):")
    for i, step in enumerate(trace.steps, 1):
        print(f"  Step {i}: [{step.action}] args={step.arguments} -> {step.result} ({step.metadata.get('kind')})")
    print(f"Outcome (all milestones reached): {trace.success}")

    scores = evaluator.evaluate_trace(trace, scenario=scenario)
    print("\nTrajectory Scorecard:")
    for k, v in scores.items():
        print(f"  - {k}: {v.score}")
    print("=" * 70)


def run_chapter_5():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-05-llm-judge"))
    from judge import MultiJudgeSystem
    from calibration import compute_calibration_metrics
    from shared.datasets.loader import load_judge_benchmark_cases

    print("\n" + "=" * 70)
    print("CHAPTER 5: LLM-as-a-Judge & Meta-Calibration vs Human Benchmarks")
    print("=" * 70)

    system = MultiJudgeSystem()
    benchmarks = load_judge_benchmark_cases()[:3]

    print(f"Evaluating {len(benchmarks)} benchmark samples with Dual Judges (Primary + Alternative)...")
    evals = []
    for b in benchmarks:
        ev = system.primary_judge.evaluate(b["prompt"], b["response"])
        evals.append(ev)
        print(f"\n[Case: {b.get('id', 'N/A')}] Prompt: {b['prompt'][:60]}...")
        print(f"  Judge Overall Score: {ev.overall_score}/5.0")
        print(f"  Human Expert Score:  {sum(b['human_scores'].values()) / len(b['human_scores']):.1f}/5.0")
        print(f"  Reasoning: {ev.reasoning[:100]}...")

    cal = compute_calibration_metrics(evals, benchmarks)
    print("\n" + "-" * 70)
    print("Meta-Calibration Summary:")
    print(f"  Human Agreement Rate: {cal['agreement_pct']}%")
    print(f"  Pearson Correlation:  {cal['pearson_correlation']}")
    print(f"  False Positives:      {cal['false_positives']}")
    print(f"  False Negatives:      {cal['false_negatives']}")
    print("=" * 70)


def run_chapter_6():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-06-multi-agent-evals"))
    from system import MultiAgentResearchSystem
    from evaluator import MultiAgentEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 6: Multi-Agent Collaboration Network & Handoffs")
    print("=" * 70)

    system = MultiAgentResearchSystem()
    evaluator = MultiAgentEvaluator()

    task = "State of Local Open-Weights Agentic Evaluation in 2026"
    print(f"Task: {task}")
    res = system.run_research(task, inject_handoff_failure=False)

    print("\nAgent Message Stream:")
    for msg in res["messages"]:
        print(f"  [{msg.sender.upper()}] -> [{msg.recipient.upper()}]: status={msg.handoff_status}")
        print(f"    Content: {msg.content[:80]}...")

    scores = evaluator.evaluate_system_run(res)
    print("\nMulti-Agent Metrics:")
    for k, v in scores.items():
        print(f"  - {k}: {v.score} (passed={v.passed})")
    print("=" * 70)


def run_chapter_7():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-07-rag-agent-evals"))
    from pipeline import RAGAgentPipeline
    from evaluator import RAGEvaluator
    from graders import load_chunks

    print("\n" + "=" * 70)
    print("CHAPTER 7: RAG Agent Evaluation (Chunk Retrieval, Claim Audit & Citations)")
    print("=" * 70)

    chunks = load_chunks()
    pipeline = RAGAgentPipeline()
    pipeline.index(chunks)
    evaluator = RAGEvaluator()

    question = "What is our customer refund policy?"
    print(f"Question: {question}")
    res = pipeline.query(question, top_k=2)

    print(f"\nRetrieved {len(res['retrieved_docs'])} chunks (of {len(chunks)} indexed, including obsolete):")
    for d in res["retrieved_docs"]:
        print(f"  - [{d['chunk_id']}] {d['title']} status={d['status']} (score: {d.get('similarity_score', 0):.3f})")

    print(f"\nGenerated Answer: {res['answer']}")

    scores = evaluator.evaluate_rag_output(
        question=question,
        retrieved_docs=res["retrieved_docs"],
        answer=res["answer"],
        expected_doc_ids=["DOC-FIN-303"],
    )
    print("\nRAG Evaluation Metrics:")
    for k, v in scores.items():
        print(f"  - {k}: {v.score} (passed={v.passed})")
    print("=" * 70)


def run_chapter_8():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-08-safety-evals"))
    from target_agent import BankingSupportAgent
    from redteam import load_full_redteam_dataset
    from evaluator import SafetyEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 8: Safety & Red-Team Adversarial Suite (7 Threat Vectors)")
    print("=" * 70)

    attacks = load_full_redteam_dataset()
    agent = BankingSupportAgent(guardrails_enabled=True)
    evaluator = SafetyEvaluator()

    print(f"Executing {len(attacks)} red-team attacks against Banking Support Agent (Guardrails: ON)...")
    print("Every sensitive tool call is mediated by the ToolGateway (gateway.py); ")
    print("graders read the side-effect ledger and outputs, never a self-reported flag.")
    res = evaluator.evaluate_suite(attacks, agent)

    print("\nAttack Results by Category (applicable cases only, capability gaps shown separately):")
    for cat, data in res["category_breakdown"].items():
        applicable = data["applicable"]
        succeeded = data["succeeded"]
        blocked = applicable - succeeded
        pct = (blocked / applicable) * 100 if applicable > 0 else 0
        na = f", {data['not_applicable']} not_applicable" if data["not_applicable"] else ""
        print(f"  - {cat:<24}: {blocked}/{applicable} blocked ({pct:.1f}%){na}")

    print("\n" + "-" * 70)
    print(f"Safety Score: {res['safety_score']}% | Succeeded: {res['succeeded']} | "
          f"Applicable: {res['applicable_tests']}/{res['total_tests']}")
    print("=" * 70)


def run_chapter_9():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-09-robustness-evals"))
    from evaluator import ChaosExperimentEvaluator

    print("\n" + "=" * 70)
    print("CHAPTER 9: Robustness & Chaos Engineering (Fragile vs Resilient Agent)")
    print("=" * 70)

    evaluator = ChaosExperimentEvaluator()
    fault_rates = {"http_500_transient": 0.4, "malformed_success": 0.1}
    print(f"Active fault rates (seeded, probabilistic -- not always-on switches): {fault_rates}")

    res = evaluator.run_experiment(fault_rates)
    print(f"\nExperiment Results ({res['total_trials']} requests, half the workload pre-cached):")
    print(f"  Naive agent      : correct={res['naive']['outcome_rate']['correct']*100:.1f}%  "
          f"silent_wrong={res['naive']['outcome_rate']['silent_wrong']*100:.1f}%  "
          f"crash={res['naive']['outcome_rate']['crash']*100:.1f}%  "
          f"p95={res['naive']['p95_latency_s']}s  calls/req={res['naive']['calls_per_request']}")
    print(f"  Resilient agent  : correct={res['resilient']['outcome_rate']['correct']*100:.1f}%  "
          f"degraded_stale={res['resilient']['outcome_rate']['degraded_stale']*100:.1f}%  "
          f"honest_failure={res['resilient']['outcome_rate']['honest_failure']*100:.1f}%  "
          f"silent_wrong={res['resilient']['outcome_rate']['silent_wrong']*100:.1f}%  "
          f"p95={res['resilient']['p95_latency_s']}s  calls/req={res['resilient']['calls_per_request']}")
    print(f"  Circuit breaker final state: {res['circuit_breaker_state']}")
    print("=" * 70)


def run_chapter_10():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-10-production-evals"))
    from eval_platform import ProductionEvaluationPlatform
    from reporter import CIReporter

    print("\n" + "=" * 70)
    print("CHAPTER 10: Production Evaluation Platform & CI/CD Quality Gate")
    print("=" * 70)
    print("Executing the real Chapter 1/3/7/8/9 suites (task success, tool")
    print("accuracy, groundedness, safety, recovery) through suites.py, then")
    print("gating them with the paired Wilson-CI + McNemar test in gate.py --")
    print("no metric here is a hardcoded constant.")

    platform = ProductionEvaluationPlatform()
    res = platform.run_full_evaluation(promote_if_passed=True)
    t = platform.thresholds

    gate_str = "PASS (Deployment Approved)" if res.passed_ci_gate else "BLOCK (Deployment Blocked)"
    print(f"\nCI/CD Quality Gate Status: [{gate_str}]")
    print(f"  - Task Success Rate:     {res.task_success_pct:.1f}% (Threshold >= {t.min_task_success_pct:.1f}%)")
    print(f"  - Safety Score:          {res.safety_score_pct:.1f}% (Threshold >= {t.min_safety_score_pct:.1f}%)")
    print(f"  - Groundedness / RAG:    {res.groundedness_pct:.1f}% (Threshold >= {t.min_groundedness_pct:.1f}%)")
    print(f"  - Tool Calling Accuracy: {res.tool_accuracy_pct:.1f}% (Threshold >= {t.min_tool_accuracy_pct:.1f}%)")
    print(f"  - Fault Recovery Rate:   {res.recovery_rate_pct:.1f}% (Threshold >= {t.min_recovery_rate_pct:.1f}%)")
    print(f"  - Average Latency:       {res.avg_latency_sec:.3f}s (Max <= {t.max_avg_latency_sec:.2f}s)")
    print(f"  - Estimated Cost/Run:    ${res.avg_cost_usd:.5f} (local Ollama model -- no per-token billing)")
    print(f"  - Baseline available:    {'yes' if res.baseline_present else 'no (this run establishes one)'}")

    if res.gate_failures:
        print("\nGate Block Reasons:")
        for f in res.gate_failures:
            print(f"  x {f}")

    reporter = CIReporter()
    report_md = reporter.generate_markdown_summary(res)
    print(f"\nGenerated GitHub Actions / CI Markdown Artifact ({len(report_md)} bytes).")
    print("=" * 70)


def run_chapter_11():
    _reset_local_modules()
    sys.path.insert(0, os.path.join(ROOT_DIR, "chapter-11-langgraph-chatbot"))
    from graph import CustomerSupportGraph
    from evaluators import DeepEvalEngine, LangSmithTraceEngine, RagasEngine, TruLensFeedbackEngine
    from shared.datasets.loader import load_langgraph_support_cases

    print("\n" + "=" * 70)
    print("CHAPTER 11: Evaluating an Agentic Chatbot (DeepEval, Ragas, LangSmith, TruLens)")
    print("=" * 70)

    graph = CustomerSupportGraph()
    langsmith_engine = LangSmithTraceEngine()
    trulens_engine = TruLensFeedbackEngine()
    deepeval_engine = DeepEvalEngine()
    ragas_engine = RagasEngine()

    cases = load_langgraph_support_cases()
    print(f"Running evaluation on {len(cases)} golden test cases across 4 frameworks...\n")

    passed_count = 0
    for case in cases:
        state = graph.invoke(case["input_prompt"])
        ls = langsmith_engine.evaluate_run_tree(state["trace_spans"], expected_intent=case.get("expected_intent"))
        tl = trulens_engine.evaluate_feedback(
            case["input_prompt"],
            state["final_answer"],
            retrieved_contexts=[d["content"] for d in state["retrieved_docs"]] or [case.get("reference_context", "")],
            tool_results=state.get("tool_results", []),
        )
        de = deepeval_engine.evaluate(
            case["input_prompt"],
            state["final_answer"],
            expected_output=case.get("expected_output"),
            retrieved_context=case.get("reference_context"),
        )
        rag = ragas_engine.evaluate(
            case["input_prompt"],
            state["final_answer"],
            retrieved_contexts=[d["content"] for d in state["retrieved_docs"]] or [case.get("reference_context", "")],
            reference_context=case.get("reference_context"),
        )

        status = "PASS" if ls["verdict"] == "PASSED" else "FAIL"
        if ls["verdict"] == "PASSED":
            passed_count += 1

        print(f"[{case['id']}] Category: {case['category']:<25} | Intent: {state['intent']:<15} | Status: [{status}]")
        print(f"  Query:               {case['input_prompt'][:80]}...")
        print(f"  LangSmith Run Tree:  {ls['compliance_score']:.1f}% ({ls['verdict']}) | {len(state['trace_spans'])} spans")
        print(f"  TruLens Grounded:    {tl['trulens_groundedness'].score:.1f}% | Context Rel: {tl['trulens_context_relevance'].score:.1f}%")
        print(f"  DeepEval Relevancy:  {de['deepeval_relevancy'].score:.1f}% | Correctness: {de['deepeval_correctness'].score:.1f}%")
        print(f"  Ragas Faithfulness:  {rag['ragas_faithfulness'].score:.1f}% | Precision: {rag['ragas_context_precision'].score:.1f}%")

    print("\n" + "-" * 70)
    print(f"SUMMARY: {passed_count}/{len(cases)} cases passed LangSmith trajectory gating ({(passed_count/len(cases))*100:.1f}%)")
    print("=" * 70)


CHAPTER_RUNNERS = {
    1: run_chapter_1,
    2: run_chapter_2,
    3: run_chapter_3,
    4: run_chapter_4,
    5: run_chapter_5,
    6: run_chapter_6,
    7: run_chapter_7,
    8: run_chapter_8,
    9: run_chapter_9,
    10: run_chapter_10,
    11: run_chapter_11,
}

CHAPTER_NAMES = {
    1: "Agent Evaluation Fundamentals",
    2: "Agent Architecture Evaluation (Planner-Executor-Verifier)",
    3: "Tool-Calling Evaluations",
    4: "Agent Trajectory Evaluations & Loop Penalties",
    5: "LLM-as-a-Judge & Meta-Calibration",
    6: "Multi-Agent Collaboration Evaluations",
    7: "RAG Agent Evaluations (Dense Retrieval & Groundedness)",
    8: "Safety & Adversarial Red-Teaming (7 Attack Categories)",
    9: "Robustness & Chaos Engineering (Self-Healing Backoff)",
    10: "Production Evaluation Platform & CI/CD Quality Gates",
    11: "Evaluating an Agentic Chatbot using DeepEval, Ragas, Langsmith, and Trulens",
}


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Evals Labs - Terminal Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py --chapter 1
  python cli.py --chapter 11
  python cli.py --all
  python cli.py --list
""",
    )
    parser.add_argument(
        "--chapter",
        "-c",
        type=str,
        default=None,
        help="Chapter number to run (1-11) or 'all'",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all 11 chapters in sequence",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available chapter evaluation labs",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of test samples to evaluate (for Chapter 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b",
        help="Model name to evaluate (default: qwen2.5:3b)",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Agentic Evals Labs:")
        print("-" * 65)
        for num, name in CHAPTER_NAMES.items():
            print(f"  Chapter {num:2d}: {name}")
        print("-" * 65)
        print("Run with: python cli.py --chapter <number>\n")
        return

    if args.all or args.chapter == "all":
        start_t = time.time()
        print("\nRunning Full Agentic Evals Suite (Chapters 1–11)...")
        for num in range(1, 12):
            CHAPTER_RUNNERS[num]()
        print(f"\nAll 11 chapters completed in {time.time() - start_t:.2f}s!")
        return

    if args.chapter is not None:
        try:
            ch_num = int(args.chapter)
            if ch_num not in CHAPTER_RUNNERS:
                print(f"Error: Chapter {ch_num} not recognized. Choose 1–11.")
                sys.exit(1)
            if ch_num == 1:
                CHAPTER_RUNNERS[1](count=args.count, model=args.model)
            else:
                CHAPTER_RUNNERS[ch_num]()
        except ValueError:
            print(f"Error: Invalid chapter '{args.chapter}'. Specify a number from 1 to 11 or 'all'.")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
