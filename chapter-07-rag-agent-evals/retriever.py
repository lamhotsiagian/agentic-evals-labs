"""Semantic vector retrieval engine with nomic-embed-text embeddings."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from shared.models.provider import LLMProvider, get_model_provider


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0


class SemanticRetriever:
    """In-memory vector store using nomic-embed-text."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: str = "nomic-embed-text"):
        self.provider = provider or get_model_provider()
        self.model = model
        self.corpus: List[Dict[str, Any]] = []

    def index_documents(self, documents: List[Dict[str, str]]):
        self.corpus.clear()
        for doc in documents:
            text_to_embed = f"{doc.get('title', '')} {doc.get('content', '')}"
            vec = self.provider.embed(self.model, text_to_embed)
            self.corpus.append({
                **doc,
                "embedding": vec,
            })

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.provider.embed(self.model, query)
        scored_docs = []
        for doc in self.corpus:
            sim = _cosine_similarity(query_vec, doc["embedding"])
            # Add lexical keyword bonus for realistic retrieval
            words_in_query = set(query.lower().split())
            words_in_doc = set(doc["content"].lower().split())
            lexical_overlap = len(words_in_query.intersection(words_in_doc)) / max(1, len(words_in_query))
            combined_score = sim * 0.6 + lexical_overlap * 0.4

            scored_docs.append({
                "doc_id": doc["doc_id"],
                "department": doc["department"],
                "title": doc["title"],
                "content": doc["content"],
                "similarity_score": round(combined_score, 3),
            })

        scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_docs[:top_k]
