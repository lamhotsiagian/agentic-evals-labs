# Agentic Evals Labs (2026)
### Hands-on evaluation labs for autonomous AI agents, on local models

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Tests-92%2F92%20passing%20(MOCK__LLM%3D1)-brightgreen?logo=pytest&logoColor=white" alt="Test Status" />
  <img src="https://img.shields.io/badge/Inference-Local%20Ollama-purple?logo=ollama&logoColor=white" alt="Local Ollama" />
  <img src="https://img.shields.io/badge/UI-One%20Streamlit%20app-red?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

> The code companion for the ebook *Agentic Evals System Design using DeepEval, Ragas, Langsmith, and Trulens*. Every chapter's
> "Hands-On Lab" section quotes the code in this repository.

Eleven labs, one Streamlit app. Each chapter page is interactive: you type a question, a ticket, a
topic, or a prompt and response to judge, the agent answers with your local Ollama model, and that
chapter's evaluator scores the result in the open. Chapter 9 is the exception by design: it evaluates
retry, backoff, and circuit-breaker mechanics against a simulated dependency, so it makes no model calls.


## Quickstart

```bash
# 1. Models
ollama serve
ollama pull qwen2.5:3b && ollama pull qwen3:1.7b && ollama pull llama3.2:1b && ollama pull nomic-embed-text

# 2. Environment
cd agentic-evals-labs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. One app for all eleven chapters -- pick a chapter in the sidebar
streamlit run Home.py
```

Headless runs without the UI:

```bash
python cli.py --list
python cli.py --chapter 11    # or 1-11, or all
```

## The labs

| Ch | Directory | Page | What you do on the page | Tests |
| :-- | :-- | :-- | :-- | :-- |
| 1 | [`chapter-01-evaluation-foundations`](chapter-01-evaluation-foundations/) | `pages/1_Ch1_Evaluation_Foundations.py` | Chat with a support agent and see each answer scored; run a multi-model regression suite with Wilson intervals | 3 |
| 2 | [`chapter-02-agent-architecture`](chapter-02-agent-architecture/) | `pages/2_Ch2_Agent_Architecture.py` | Plan a trip through planner, executor, verifier with feedback retries; mutation-test the verifier | 2 |
| 3 | [`chapter-03-tool-evals`](chapter-03-tool-evals/) | `pages/3_Ch3_Tool_Evals.py` | Ask an e-commerce agent that picks tools natively; see every call as executed and six graded scores | 4 |
| 4 | [`chapter-04-trajectory-evals`](chapter-04-trajectory-evals/) | `pages/4_Ch4_Trajectory_Evals.py` | Run a live helpdesk diagnosis and read the milestone-based trajectory score step by step | 4 |
| 5 | [`chapter-05-llm-judge`](chapter-05-llm-judge/) | `pages/5_Ch5_LLM_Judge.py` | Judge any prompt/response pair with two fail-closed judges; probe verbosity bias; calibrate against human labels | 4 |
| 6 | [`chapter-06-multi-agent-evals`](chapter-06-multi-agent-evals/) | `pages/6_Ch6_Multi_Agent_Evals.py` | Give a research team a topic; grade handoff edges, role contracts, and claim provenance | 3 |
| 7 | [`chapter-07-rag-agent-evals`](chapter-07-rag-agent-evals/) | `pages/7_Ch7_RAG_Agent_Evals.py` | Ask policy questions over a chunked knowledge base; see retrieval metrics and the per-sentence claim audit | 11 |
| 8 | [`chapter-08-safety-evals`](chapter-08-safety-evals/) | `pages/8_Ch8_Safety_Evals.py` | Chat with a banking agent behind a state-based tool gateway; run red-team, mutation, indirect-injection, and benign-twin suites | 12 |
| 9 | [`chapter-09-robustness-evals`](chapter-09-robustness-evals/) | `pages/9_Ch9_Robustness_Evals.py` | Set fault probabilities and compare a naive and a resilient agent on five disclosed outcome classes | 10 |
| 10 | [`chapter-10-production-evals`](chapter-10-production-evals/) | `pages/10_Ch10_Production_Evals.py` | Run the Ch1/3/7/8/9 suites through a paired Wilson-CI + McNemar regression gate; read the gate report and OTLP-style spans | 9 |
| 11 | [`chapter-11-langgraph-chatbot`](chapter-11-langgraph-chatbot/) | `pages/11_Ch11_LangGraph_Chatbot.py` | StateGraph customer support agent evaluated with DeepEval (G-Eval), Ragas (retrieval triad), LangSmith (run tree), and TruLens (groundedness) | 5 |

Each chapter directory has its own README with the file list, how grading works, and the exact test command.

## Models

| Model | Used for | Where |
| :-- | :-- | :-- |
| `qwen2.5:3b` | Default agent and generator | Ch 1, 2, 3, 4, 6, 7, 8, 10 |
| `qwen3:1.7b` | Default primary judge; alternative agent model | Ch 5; selectable in Ch 1, 3, 4, 6, 7, 8, 10 |
| `llama3.2:1b` | Default alternative judge; small-model comparison | Ch 5; selectable in Ch 1, 7, 8, 10 |
| `nomic-embed-text` | Chunk embeddings for retrieval | Ch 7, 10 |

## Tests

```bash
# Offline and deterministic (explicit mock provider): 85 tests
MOCK_LLM=1 pytest

# The same tests against your local Ollama models (slower, needs `ollama serve`)
pytest
```

The suite is 70 unit and evaluation tests across the ten chapters and `shared/`, plus
[`tests/test_ui_pages.py`](tests/test_ui_pages.py) (15 tests): Streamlit `AppTest` boots `Home.py` and
every chapter page, walks all ten pages forward and backward to catch module collisions between chapters,
and drives a Chapter 1 chat turn and a Chapter 9 experiment. No browser is needed.

Under `MOCK_LLM=1` the mock returns generic text, so model-dependent pass rates (for example Chapter 1
task success) are low by design. Mock-mode tests check evaluator logic, not model quality.

## Repository layout

```text
agentic-evals-labs/
├── Home.py                     # the single Streamlit entry point
├── pages/                      # one page per chapter
├── cli.py                      # headless runner for every chapter
├── chapter-01-evaluation-foundations/ ... chapter-10-production-evals/
├── tests/test_ui_pages.py      # AppTest smoke tests for Home.py and all pages
└── shared/
    ├── models/                 # schemas.py, provider.py (fail-closed OllamaClient, explicit mock)
    ├── evaluators/             # base.py, llm_judge.py, trajectory.py
    ├── metrics/                # quality.py, performance.py, security.py, stats.py (Wilson, McNemar, sliced reports)
    ├── tracing/tracer.py       # OpenTelemetry-compatible spans
    ├── ui/components.py        # page header, provider badge, metrics row, load_chapter_modules
    ├── datasets/               # loader.py and data/ (support cases, travel tasks, e-commerce DB,
    │                           #   judge benchmark, knowledge base, red-team suite, ...)
    └── tests/test_shared.py
```
