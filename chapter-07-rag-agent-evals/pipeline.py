"""End-to-end RAG agent pipeline: chunk-level retrieval + citation-grounded generation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from shared.models.provider import LLMProvider, get_model_provider
from retriever import SemanticRetriever
from graders import Chunk, load_chunks


class RAGAgentPipeline:
    """Combines section-level chunk retrieval with citation-grounded generation."""

    def __init__(self, provider: Optional[LLMProvider] = None, generator_model: str = "qwen2.5:3b"):
        self.provider = provider or get_model_provider()
        self.generator_model = generator_model
        self.retriever = SemanticRetriever(provider=self.provider)

    def index(self, chunks: Optional[List[Chunk]] = None) -> None:
        """Index the knowledge base. Defaults to every markdown file in the real
        enterprise_knowledge_base directory, including obsolete/superseded ones --
        the pipeline no longer silently skips the legacy distractor file."""
        self.retriever.index_chunks(chunks if chunks is not None else load_chunks())

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)

        context_str = "\n\n".join([
            f"[{c['doc_id']}] {c['title']} ({c['status']}): {c['content']}"
            for c in retrieved_chunks
        ])

        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the question using ONLY the "
            "provided context. Always cite the document ID in brackets, e.g. [DOC-FIN-303], "
            "for every factual claim. Never cite or rely on a document marked 'obsolete' as "
            "current policy -- call out if only obsolete evidence is available. If the context "
            "does not contain the answer, say you don't know instead of guessing. Do not make up facts."
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
            "retrieved_docs": retrieved_chunks,
            "context_str": context_str,
            "answer": answer,
        }
