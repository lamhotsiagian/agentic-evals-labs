# Shared Infrastructure (`shared/`)
### Core Primitives, Data Schemas, Metric Engines, and Local Model Abstraction

The `shared/` package forms the foundational evaluation backbone across all 10 labs in the **Agentic Evals (2026)** curriculum. It guarantees architectural consistency, strict typing via **Pydantic v2**, lightweight **OpenTelemetry-compatible tracing**, and standardized local model inference via **Ollama**.

---

## 🏛️ Directory Architecture

```text
shared/
├── models/
│   ├── schemas.py           # Pydantic v2 data models (EvaluationCase, AgentTrace, MetricScore, JudgeRubric)
│   └── provider.py          # LLM Provider abstraction: OllamaClient (live localhost:11434) + fallback
├── metrics/
│   ├── quality.py           # Exact match, F1 token overlap, semantic similarity, hallucination detection
│   ├── performance.py       # Token counting, latency percentiles (p50, p95), operational cost estimation
│   └── security.py          # Prompt injection heuristics, secret & credential leakage detection
├── evaluators/
│   ├── base.py              # BaseEvaluator abstract base class
│   ├── llm_judge.py         # LLMJudgeEvaluator with 5 standardized evaluation rubrics
│   └── trajectory.py        # TrajectoryEvaluator with step efficiency and loop penalty scoring
├── tracing/
│   └── tracer.py            # OpenTelemetry-compatible span-based telemetry and execution tracing
├── datasets/
│   ├── loader.py            # Physical disk loaders for JSON, JSONL, and Markdown corpora
│   └── data/                # Ground-truth physical dataset files stored on disk
│       ├── customer_support_50.jsonl        # 50 real support cases
│       ├── travel_planner_tasks.json        # Real constraint scenarios
│       ├── ecommerce_db.json                # Relational customers/orders
│       ├── helpdesk_trajectories.jsonl      # Multi-step IT diagnostic traces
│       ├── human_benchmark_judge.jsonl      # Human golden scores across 5 dimensions
│       ├── enterprise_knowledge_base/       # Markdown corpora (HR, Sec, Finance, Runbook)
│       ├── redteam_adversarial_suite.jsonl  # 7-category adversarial red-team suite
│       └── chaos_workload.jsonl             # Batch stress workload
└── testing_utils.py         # StreamlitServerRunner & ephemeral port manager for Playwright E2E testing
```

---

## 🧩 Core Modules Explained

### 1. Data Models (`shared/models/schemas.py`)
Provides strictly-typed Pydantic v2 models:
* `EvaluationCase`: Ground-truth test case definition containing `id`, `input_prompt`, `expected_output`, `category`, and `metadata`.
* `AgentTraceStep`: Single execution step containing `step_index`, `action`, `arguments`, `observation`, `result` (`success` | `error`), and `duration_ms`.
* `AgentTrace`: Complete agent trajectory containing `task`, sequence of `steps`, `final_output`, and `success` flag.
* `MetricScore`: Atomic metric result with `name`, normalized `score` (0.0–1.0 or 0–100), `passed` boolean, and optional diagnostic `metadata`.
* `EvaluationResult`: Aggregated case outcome containing all computed metric scores, latency, tokens, and cost.
* `JudgeRubric` & `JudgeEvaluation`: Definition of evaluation dimensions and structured outputs produced by LLM judges.

### 2. Model Provider (`shared/models/provider.py`)
Dual-engine abstraction decoupled from proprietary cloud APIs:
* **`OllamaClient`**: Direct HTTP client connecting to local Ollama daemon (`http://127.0.0.1:11434`). Supports text generation (`/api/generate`) and dense embeddings (`/api/embeddings`) for `qwen2.5:3b`, `qwen3:1.7b`, `llama3.2:1b`, and `nomic-embed-text`.
* **`DeterministicMockProvider`**: Fast pseudo-random fallback engine used only when Ollama is unreachable, guaranteeing deterministic behavior in restricted test runners.
* **`get_model_provider()`**: Factory function prioritizing live local Ollama if available.

### 3. Metric Engines (`shared/metrics/`)
* **Quality (`quality.py`)**:
  - `compute_exact_match(actual, expected)`: Case-insensitive exact match.
  - `compute_substring_match(actual, expected_keywords)`: Ratio of essential domain keywords present.
  - `compute_similarity(actual, expected)`: Normalized character/token sequence similarity.
  - `detect_hallucination(actual, source_context)`: Entity extraction and consistency verification.
* **Performance & Cost (`performance.py`)**:
  - `estimate_tokens(text)`: Accurate token estimation based on character and subword heuristics.
  - `estimate_cost_usd(in_tokens, out_tokens, model)`: Financial modeling (local models compute as $0.00).
  - `compute_latency_stats(latencies)`: Mean, min, max, p50, and p95 latency aggregations.
* **Security (`security.py`)**:
  - `detect_prompt_injection(prompt, response)`: Detection of instruction override attempts and DAN markers.
  - `detect_data_leakage(text)`: Regular expression detection of API keys, bearer tokens, and internal secrets.

### 4. Evaluators (`shared/evaluators/`)
* **`LLMJudgeEvaluator`**: Implements 5 standardized rubrics (Correctness, Relevance, Groundedness, Safety, Task Completion on a 0.0–5.0 scale) and extracts structured JSON citations.
* **`TrajectoryEvaluator`**: Grades multi-turn agent paths using step transitions, optimal step ratios, and repeated action loop penalties.

### 5. Datasets & Loaders (`shared/datasets/`)
All datasets are stored as permanent physical files on disk under `shared/datasets/data/`. `loader.py` validates schemas upon reading:
* `load_customer_support_cases(count=50)` ➔ `List[EvaluationCase]`
* `load_travel_planner_cases()` ➔ `List[EvaluationCase]`
* `load_ecommerce_tool_cases()` ➔ `List[Dict[str, Any]]`
* `load_it_helpdesk_trajectories()` ➔ `List[AgentTrace]`
* `load_judge_benchmark_cases()` ➔ `List[Dict[str, Any]]`
* `load_rag_enterprise_corpus()` ➔ `List[Dict[str, str]]`
* `load_redteam_attack_cases()` ➔ `List[Dict[str, Any]]`
* `load_chaos_scenarios()` ➔ `List[Dict[str, Any]]`

### 6. Playwright Test Runner (`shared/testing_utils.py`)
* `StreamlitServerRunner`: Spawns dedicated Streamlit app instances on dynamically allocated ephemeral TCP ports.
* `run_streamlit_app(script_path)`: Context manager yielding base URLs to Playwright tests and guaranteeing clean process termination.

---

## 🧪 Unit Tests

Run the shared infrastructure unit tests:
```bash
.venv/bin/pytest shared/tests/test_shared.py -v
```
All 8 test suites validate schemas, metric formulas, trajectory scoring, and live Ollama embeddings.
