"""Agentic Evals Labs -- single entry point for all 11 chapter labs.

Run once:  streamlit run Home.py
Then use the sidebar to open any chapter. Every chapter page is a live
chat you type into; the agent's reply comes from your local Ollama
server unless you explicitly opt into MOCK_LLM=1 for offline dev/tests.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from shared.models.provider import get_model_provider
from shared.ui.components import provider_status_badge

st.set_page_config(page_title="Agentic Evals Labs", layout="wide")

st.title("Agentic Evals Labs")
st.caption("Companion labs for *Agentic Evals System Design using DeepEval, Ragas, Langsmith, and Trulens* -- one app, eleven chapters, all live.")

provider = get_model_provider()
provider_status_badge(provider)

st.markdown(
    """
Use the sidebar to open a chapter. Each page is a live chat: type a message,
the agent answers using your local Ollama model, and that chapter's
evaluator scores the turn in the open -- no fixed canned dataset, no
static "run and forget" button.

If the sidebar shows **UNREACHABLE**, start Ollama first:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull qwen3:1.7b
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

| Chapter | Lab |
|---|---|
| 1 | Evaluation Foundations -- customer support agent dashboard |
| 2 | Agent Architecture -- planner/executor/verifier pipeline |
| 3 | Tool-Call Evals -- order-lookup tool agent |
| 4 | Trajectory Evals -- multi-step task agent |
| 5 | LLM-as-Judge -- judge calibration |
| 6 | Multi-Agent Evals -- handoff and claim provenance |
| 7 | RAG Agent Evals -- retrieval + faithfulness |
| 8 | Safety Evals -- red-team / guardrails |
| 9 | Robustness Evals -- chaos and resilience |
| 10 | Production Evals -- regression gate platform |
| 11 | Evaluating an Agentic Chatbot using DeepEval, Ragas, Langsmith, and Trulens |
"""
)
