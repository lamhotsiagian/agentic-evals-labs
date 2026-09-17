# Chapter 7 -- RAG Agent Evals
### Lab: Build a RAG Evaluation Lab

> Companion to Chapter 7 of *Agentic Evals System Design*.

An enterprise knowledge base indexed at section-chunk granularity with local embeddings, a local
generator that must cite chunk-level document IDs and abstain when the context lacks the answer,
and live scoring of retrieval ranking and claim-level groundedness.

## Files

```text
chapter-07-rag-agent-evals/
├── graders.py     # Chunk, load_chunks (status / effective date), recall@k, precision@k, MRR, nDCG@k, claim_audit
├── retriever.py   # SemanticRetriever over nomic-embed-text chunk embeddings + lexical bonus
├── pipeline.py    # RAGAgentPipeline: retrieve top-k -> qwen2.5:3b, cite IDs, never cite obsolete docs as current, abstain
├── evaluator.py   # RAGEvaluator: ranking vs achievable precision, claim-level groundedness
└── tests/
    └── test_rag_evals.py   # 11 tests
```

UI page: `pages/7_Ch7_RAG_Agent_Evals.py`. Corpus: `shared/datasets/data/enterprise_knowledge_base/`
(including the obsolete `legacy_distractor_v1.md`, which is indexed on purpose).

## Running the lab

```bash
ollama pull nomic-embed-text && ollama pull qwen2.5:3b
streamlit run Home.py   # open "Chapter 7" in the sidebar
```

* **Live Chat** -- ask a policy question; see the answer, retrieved chunks (obsolete ones flagged),
  ranking and groundedness metrics, and the per-sentence claim audit.
* **Regression Suite** -- five questions, including one with no answer in the corpus, to check abstention.
* **Known-Failure Checks** -- replays a correct cited answer, a wrong uncited 90-day claim, and an
  answer citing the obsolete 2018 policy against the claim audit, with no model call.

## Tests

```bash
MOCK_LLM=1 pytest chapter-07-rag-agent-evals/tests/test_rag_evals.py -v
```
