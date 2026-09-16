"""End-to-end RAG agent pipeline."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from shared.models.provider import LLMProvider, get_model_provider
from retriever import SemanticRetriever


class RAGAgentPipeline:
    """Combines semantic retrieval with augmented generation."""

    def __init__(self, provider: Optional[LLMProvider] = None, generator_model: str = "qwen2.5:3b"):
        self.provider = provider or get_model_provider()
        self.generator_model = generator_model
        self.retriever = SemanticRetriever(provider=self.provider)

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        context_str = "\n\n".join([
            f"[{d['doc_id']}] {d['title']}: {d['content']}" for d in retrieved_docs
        ])

        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the question using ONLY the provided context. "
            "Always cite the document ID (e.g. [DOC-FIN-303]) for your claims. Do not make up facts."
        )

        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer with document citations:"
        answer = self.provider.generate(
            model=self.generator_model,
            prompt=prompt,
            system=system_prompt,
            temperature=0.1,
        )

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "context_str": context_str,
            "answer": answer,
        }
