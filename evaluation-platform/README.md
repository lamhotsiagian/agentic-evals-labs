# Master Evaluation Platform Portal (`evaluation-platform/`)
### Central Navigation & Lab Launcher

The `evaluation-platform/` module provides a unified Streamlit portal for launching, previewing, and navigating across all 10 hands-on evaluation labs in the **Agentic Evals (2026)** curriculum.

---

## Features
* **One-Click Lab Launcher**: Instant access to any chapter dashboard from a single interface.
* **Curriculum Roadmap**: Visual progression tracking across Foundations, Architecture, Tools, Trajectories, Judges, Multi-Agent, RAG, Safety, Chaos, and Production.
* **Local Model Status**: Verifies local Ollama daemon connectivity and model availability (`qwen2.5:3b`, `qwen3:1.7b`, `llama3.2:1b`, `nomic-embed-text`).

---

## Running the Portal

Launch the central portal:
```bash
streamlit run evaluation-platform/portal.py
```
Access in your browser at `http://localhost:8501`.
