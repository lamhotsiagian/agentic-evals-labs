# Chapter 1 -- Evaluation Foundations
### Lab: Build Your First Agent Evaluation Dashboard

> Companion to Chapter 1 of *Agentic Evals System Design*.

A customer support agent for a fictional SaaS company, an evaluator that grades its answers
against real reference answers and policy text, and a Streamlit page where you type a question,
get a live answer from your local model, and see it scored immediately.

## Files

```text
chapter-01-evaluation-foundations/
├── agent.py        # CustomerSupportAgent: system prompt + provider.generate(), measures latency
├── evaluator.py    # CustomerSupportEvaluator: correctness vs reference, relevance, helpfulness,
│                   # grounded hallucination check, task completion, tokens, cost
└── tests/
    └── test_evaluator.py   # 3 tests, incl. "faithful answer passes, policy-contradicting answer fails"
```

Related shared code: `shared/models/provider.py` (fail-closed provider),
`shared/metrics/stats.py` (`wilson_interval`, `sliced_report`), UI page
`pages/1_Ch1_Evaluation_Foundations.py`, dataset `shared/datasets/data/customer_support_50.jsonl`.

## How grading works

| Metric | How it is computed | Pass |
| :--- | :--- | :--- |
| Correctness | `0.6 * similarity(answer, reference) + 0.4 * (no hallucination)`; live chat turns with no reference fall back to the hallucination signal | >= 0.7 |
| Relevance | Stemmed token overlap with the prompt | >= 0.2 |
| Helpfulness | Distinct support-action keywords / 3 | >= 0.5 |
| Hallucination | Numbers in the answer absent from the prompt, reference, **and** policy text | none flagged |
| Task completion | relevance and helpfulness pass, no hallucination | true |

A case passes when task completion holds and correctness >= 0.7. Suite pass rates per category
carry a 95% Wilson interval and an under-powered flag.

## Running the lab

```bash
streamlit run Home.py   # from the repo root; open "Chapter 1" in the sidebar
```

* **Live Chat** -- type a billing or account question; the selected model answers and the five
  scores for that answer appear underneath.
* **Regression Suite** -- pick one or more models and a case count, click **Run suite**: a
  head-to-head table computed from that run, plus per-category pass rates with Wilson intervals.

If Ollama is not reachable, both tabs show a `ProviderUnavailableError` banner; nothing is
silently replaced with mock output.

## Tests

```bash
MOCK_LLM=1 pytest chapter-01-evaluation-foundations/tests/test_evaluator.py -v
```
