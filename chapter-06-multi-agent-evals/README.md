# Chapter 6 -- Multi-Agent Evals
### Lab: Multi-Agent Research System Evaluation

> Companion to Chapter 6 of *Agentic Evals System Design*.

A supervisor delegates a research topic to a researcher and an analyst; both workers and the
synthesizer generate their text live through the local model, and every message is a typed
`AgentMessage`. Graders evaluate handoff edges, role contracts on content, duplicate work, and
whether an unsourced number reached the executive synthesis.

## Files

```text
chapter-06-multi-agent-evals/
├── system.py      # AgentMessage, role prompts, MultiAgentResearchSystem (optional dropped-handoff injection)
├── graders.py     # EXPECTED_HANDOFFS, handoff_report, unsupported_quantitative_claims,
│                  # propagated_claims, new_numbers_in_synthesis, jaccard
├── evaluator.py   # MultiAgentEvaluator
└── tests/
    └── test_multi_agent.py   # 3 tests
```

UI page: `pages/6_Ch6_Multi_Agent_Evals.py`.

## How grading works

| Metric | What it checks |
| :--- | :--- |
| Handoff success | Share of the 5 required sender -> recipient edges eventually delivered |
| First-attempt handoff rate | Edges delivered with no failed attempt; retransmissions are counted |
| Role adherence | Analyst tagged every number with `[source: ...]`; synthesizer added no number absent from both workers |
| Duplicate work | Jaccard word overlap between worker outputs (threshold 0.6; swap for embeddings in production) |
| Final synthesis | 1.0 only if no unsourced worker number propagated into it |

## Running the lab

```bash
streamlit run Home.py   # open "Chapter 6" in the sidebar
```

Type a research topic, optionally tick **Inject a dropped handoff**, and click **Run the research
team**. The page shows the scores with each grader's reasoning, the full message trace, and the synthesis.

## Tests

```bash
MOCK_LLM=1 pytest chapter-06-multi-agent-evals/tests/test_multi_agent.py -v
```
