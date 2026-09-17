# Chapter 3 -- Tool Evals
### Lab: Tool Calling Evaluation Harness

> Companion to Chapter 3 of *Agentic Evals System Design*.

An e-commerce support agent that lets the model choose tools through Ollama's native tool-calling
API, a `ToolRuntime` that validates, executes, and records every call exactly as issued, and an
evaluator that grades selection, arguments, ordering, groundedness, result handling, and recovery.

## Files

```text
chapter-03-tool-evals/
├── tools.py       # get_order, calculate_refund, search_customer, get_weather, send_email (JSON DB-backed)
├── graders.py     # TOOL_SCHEMAS, validate_call, call_prf, order_violations, ungrounded_args
├── agent.py       # EcommerceCustomerAgent (native tool-calling loop) + ToolRuntime
├── evaluator.py   # ToolCallingEvaluator
└── tests/
    └── test_tool_evals.py   # 4 tests
```

UI page: `pages/3_Ch3_Tool_Evals.py`. Dataset: `shared/datasets/data/ecommerce_db.json`.

## How grading works

| Metric | What it checks |
| :--- | :--- |
| Tool selection | Multiset precision/recall/F1 over calls, so extra calls are penalised |
| Argument accuracy | Regex schema per tool: missing, unexpected, or ill-formatted arguments are named errors |
| Tool order | Dependencies such as `get_order` before `calculate_refund` |
| Groundedness | Argument values must appear in the user's request or an **earlier** tool observation |
| Result handling / recovery | Errors are recorded as observations and the loop continues |

The runtime, not the agent, owns the call record, so the trace always holds the full arguments that were executed.

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 3" in the sidebar
```

Type a request in the chat box (an order lookup, a refund check, a weather check). Each turn shows
an expander with every tool call and its raw result, and six evaluator scores with the graders'
reasoning (schema errors, order violations, ungrounded arguments).

## Tests

```bash
MOCK_LLM=1 pytest chapter-03-tool-evals/tests/test_tool_evals.py -v
```
