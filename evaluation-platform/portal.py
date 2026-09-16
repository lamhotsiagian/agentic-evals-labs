"""Central Portal Launcher for Agentic Evals 2026 Labs."""

import os
import sys
import subprocess
import streamlit as st

st.set_page_config(page_title="Agentic Evals Portal", page_icon="🧭", layout="wide")

st.title("🧭 Agentic Evals 2026 — Master Labs Portal")
st.caption("A Comprehensive, Production-Grade Hands-on Evaluation Platform for Autonomous AI Agents")

st.markdown("""
Welcome to the companion lab repository for **Agentic Evals 2026**. Each chapter provides a production-grade evaluation lab with local Ollama model integration, interactive Streamlit dashboards, and end-to-end automated Playwright UI test suites.
""")

LABS = [
    {
        "chapter": "Chapter 1",
        "title": "Agent Evaluation Fundamentals",
        "description": "Customer support agent evaluation, synthetic test sets, quality/latency/cost metrics, and model tradeoffs (Qwen2.5:3B vs Qwen3:1.7B vs Llama3.2:1B).",
        "file": "chapter-01-evaluation-foundations/app.py",
    },
    {
        "chapter": "Chapter 2",
        "title": "Agent Architecture Evaluation",
        "description": "Planner → Executor → Verifier triad with component-level accuracy, retries, and travel planning case study.",
        "file": "chapter-02-agent-architecture/app.py",
    },
    {
        "chapter": "Chapter 3",
        "title": "Tool-Calling Evals",
        "description": "Tool selection, parameter validation, sequencing, failure recovery, and tool trace hierarchy viewer.",
        "file": "chapter-03-tool-evals/app.py",
    },
    {
        "chapter": "Chapter 4",
        "title": "Agent Trajectory Evals",
        "description": "Step timeline analysis, loop detection penalties, recovery rate, and IT helpdesk diagnosis scoring.",
        "file": "chapter-04-trajectory-evals/app.py",
    },
    {
        "chapter": "Chapter 5",
        "title": "LLM-as-a-Judge",
        "description": "Multi-dimensional rubrics, judge scorecards, reasoning citations, and meta-calibration vs human benchmarks.",
        "file": "chapter-05-llm-judge/app.py",
    },
    {
        "chapter": "Chapter 6",
        "title": "Multi-Agent Evals",
        "description": "Supervisor → Researcher / Analyst → Synthesizer topology, delegation, communication stream, and handoffs.",
        "file": "chapter-06-multi-agent-evals/app.py",
    },
    {
        "chapter": "Chapter 7",
        "title": "RAG Agent Evals",
        "description": "Semantic dense retrieval (nomic-embed-text), faithfulness, groundedness, citations, and noise resistance.",
        "file": "chapter-07-rag-agent-evals/app.py",
    },
    {
        "chapter": "Chapter 8",
        "title": "Safety & Security Evals",
        "description": "Red-team attack suite (injections, jailbreaks, data leaks, privilege escalation) and banking agent defenses.",
        "file": "chapter-08-safety-evals/app.py",
    },
    {
        "chapter": "Chapter 9",
        "title": "Robustness & Reliability Evals",
        "description": "Chaos engineering (timeouts, 500 errors, corrupt JSON), and Baseline vs Resilient agent comparison.",
        "file": "chapter-09-robustness-evals/app.py",
    },
    {
        "chapter": "Chapter 10",
        "title": "Production Agent Evals (Capstone)",
        "description": "Consolidated executive platform combining Quality, Safety, Trajectories, Tools, RAG, and CI/CD regression gates.",
        "file": "chapter-10-production-evals/app.py",
    },
]

st.subheader("Available Labs & Interactive Dashboards")

for lab in LABS:
    with st.expander(f"**{lab['chapter']}: {lab['title']}**", expanded=True):
        st.write(lab["description"])
        st.code(f"streamlit run {lab['file']}")
