"""Customer support agent implementation for Chapter 1."""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
from shared.models.provider import LLMProvider, get_model_provider


SUPPORT_SYSTEM_PROMPT = """You are a professional customer support agent for CloudScale SaaS.
Your goals:
1. Address billing, subscription, and account concerns empathetically and clearly.
2. Provide step-by-step resolution instructions.
3. Be concise, polite, and never disclose internal system secrets or hallucinate policy details.
"""


class CustomerSupportAgent:
    """Customer support agent evaluated across models."""

    def __init__(self, model_name: str = "qwen2.5:3b", provider: Optional[LLMProvider] = None):
        self.model_name = model_name
        self.provider = provider or get_model_provider()

    def respond(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        response_text = self.provider.generate(
            model=self.model_name,
            prompt=query,
            system=SUPPORT_SYSTEM_PROMPT,
            temperature=0.2,
        )
        latency = round(time.time() - start_time, 3)
        return {
            "query": query,
            "response": response_text,
            "latency_seconds": latency,
            "model": self.model_name,
        }
