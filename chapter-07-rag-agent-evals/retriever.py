"""Semantic vector retrieval engine with nomic-embed-text embeddings, indexed at
section-chunk granularity (see graders.py:load_chunks) instead of whole documents,
so the answer-bearing section can be distinguished from the rest of the document."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

from shared.models.provider import LLMProvider, get_model_provider
from graders import Chunk


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0


class SemanticRetriever:
    """In-memory vector store over section-level chunks using nomic-embed-text."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: str = "nomic-embed-text"):
        self.provider = provider or get_model_provider()
        self.model = model
        self.corpus: List[Chunk] = []
        self._embeddings: Dict[str, List[float]] = {}

    def index_chunks(self, chunks: List[Chunk]) -> None:
        self.corpus = list(chunks)
        self._embeddings = {}
        for chunk in self.corpus:
            text_to_embed = f"{chunk.title} {chunk.text}"
            self._embeddings[chunk.chunk_id] = self.provider.embed(self.model, text_to_embed)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.corpus:
            return []
        query_vec = self.provider.embed(self.model, query)
        scored: List[Dict[str, Any]] = []
        for chunk in self.corpus:
            sim = _cosine_similarity(query_vec, self._embeddings.get(chunk.chunk_id, []))
            # Add lexical keyword bonus for realistic retrieval. Match on the
            # same title+text surface used for embedding (index_chunks), since
            # the section body alone drops the document-level title terms a
            # query often echoes (e.g. "refund" appears in the finance doc's
            # title but not literally inside the 30-day guarantee section).
            words_in_query = set(query.lower().split())
            words_in_chunk = set(f"{chunk.title} {chunk.text}".lower().split())
            lexical_overlap = len(words_in_query.intersection(words_in_chunk)) / max(1, len(words_in_query))
            combined_score = sim * 0.6 + lexical_overlap * 0.4

            scored.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "content": chunk.text,
                "status": chunk.status,
                "effective": chunk.effective,
                "similarity_score": round(combined_score, 3),
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]
