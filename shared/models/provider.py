"""Model provider abstraction supporting local Ollama models and deterministic mocks."""

from __future__ import annotations
import abc
import os
import json
import hashlib
from typing import Any, Dict, List, Optional
import requests


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Generate text completion."""
        pass

    @abc.abstractmethod
    def embed(self, model: str, text: str) -> List[float]:
        """Generate text embeddings."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if provider backend is reachable."""
        pass


class OllamaClient(LLMProvider):
    """Client for local Ollama server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 30):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed for {model}: {e}") from e

    def embed(self, model: str = "nomic-embed-text", text: str = "") -> List[float]:
        payload = {"model": model, "prompt": text}
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed for {model}: {e}") from e


class DeterministicMockProvider(LLMProvider):
    """High-speed deterministic mock for tests, offline development, and CI/CD."""

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        prompt_lower = prompt.lower()

        # Judge evaluations
        if "judge" in prompt_lower or "rubric" in prompt_lower or "score" in prompt_lower:
            return json.dumps({
                "correctness": 4.5,
                "relevance": 4.8,
                "groundedness": 4.2,
                "safety": 5.0,
                "task_completion": 4.6,
                "overall": 4.62,
                "reasoning": "The response accurately addresses the user inquiry without hallucinations or policy violations.",
                "evidence": ["Direct answer provided", "Polite tone", "Correct policy citation"]
            })

        # Tool calling tasks
        if "order #" in prompt_lower or "refund" in prompt_lower:
            if "wrong" in prompt_lower or "invalid" in prompt_lower:
                return "TOOL: calculate_refund(order_id='invalid') -> Result: Error: Order not found. Recovery: Request valid order ID."
            return "TOOL: get_order(order_id='1234') -> calculate_refund(order_id='1234') -> send_email(to='user@example.com', subject='Refund Processed')"

        # Travel planning
        if "itinerary" in prompt_lower or "travel" in prompt_lower:
            return json.dumps({
                "plan": ["Day 1: Arrival & Louvre museum", "Day 2: Eiffel tower & Seine cruise", "Day 3: Montmartre & Departure"],
                "budget_check": "passed",
                "constraints_met": True
            })

        # Red team / Safety
        if "jailbreak" in prompt_lower or "ignore previous" in prompt_lower or "transfer" in prompt_lower:
            if "transfer" in prompt_lower and "bypass" in prompt_lower:
                return "I cannot execute an unauthorized transfer. Please authenticate through the official banking portal."
            return "I cannot fulfill this request as it violates safety and security policies."

        # RAG tasks
        if "refund policy" in prompt_lower or "documentation" in prompt_lower:
            return "Based on section 4.2 of the customer policy, requests within 30 days are eligible for a full refund to the original payment method."

        # Default helpful customer support response
        return f"Thank you for contacting us regarding your request. We have reviewed your account and confirmed the subscription billing details. A refund has been queued and will reflect in 3-5 business days."

    def embed(self, model: str = "nomic-embed-text", text: str = "") -> List[float]:
        """Produce a deterministic 64-dimensional pseudo-embedding from text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Create normalized 64-dim float vector
        raw_vals = [float(b) / 255.0 for b in h] + [float(b) / 255.0 for b in h]
        vec = raw_vals[:64]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def get_model_provider(force_mock: bool = False) -> LLMProvider:
    """Factory returning OllamaClient if available, otherwise DeterministicMockProvider."""
    if force_mock or os.environ.get("MOCK_LLM") == "1":
        return DeterministicMockProvider()

    client = OllamaClient()
    if client.is_available():
        return client
    return DeterministicMockProvider()
