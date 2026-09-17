"""Model provider abstraction supporting local Ollama models and deterministic mocks."""

from __future__ import annotations
import abc
import os
import json
import re
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

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Native chat with optional tool-calling. Returns an Ollama-shaped
        {'role': 'assistant', 'content': str, 'tool_calls': [...] } message.
        Subclasses that cannot do real tool calling must still implement
        this (see DeterministicMockProvider) so calling code never branches
        on provider type."""
        raise NotImplementedError


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

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Real native tool calling via Ollama's /api/chat endpoint. The
        model chooses which tool to call and with what arguments -- this
        is what makes tool-selection evaluation measure the model instead
        of a keyword-routing script."""
        payload: Dict[str, Any] = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("message", {"role": "assistant", "content": "", "tool_calls": []})
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed for {model}: {e}") from e


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
        system_text = system or ""

        # Chapter 6 multi-agent roles: content keyed off the SYSTEM prompt
        # (the user prompt is just the research topic), each role staying
        # in its lane -- the Analyst always sources its number here so the
        # mock represents a compliant agent; the unsourced-claim detector
        # itself is exercised directly with a hand-crafted bad message in
        # tests, the same pattern used for Chapters 1/3/5's central bugs.
        if "You are the Researcher" in system_text:
            return (f"Literature review for '{prompt}' identifies three evolutionary phases: "
                    "initial heuristics, supervised fine-tuning, and agentic self-verification.")
        if "You are the Analyst" in system_text:
            return ("Empirical analysis highlights a 34% drop in regression errors when combining "
                    "LLM judges with rubric calibration [source: internal benchmark suite].")
        if "You are the Synthesizer" in system_text:
            r_match = re.search(r"Researcher findings:\s*(.*?)\n\nAnalyst findings:", prompt, re.DOTALL)
            a_match = re.search(r"Analyst findings:\s*(.*)", prompt, re.DOTALL)
            r_text = r_match.group(1).strip() if r_match else ""
            a_text = a_match.group(1).strip() if a_match else ""
            return f"Executive Synthesis:\n- Qualitative: {r_text}\n- Quantitative: {a_text}"

        # RAG tasks (Chapter 7): read the real retrieved context out of the prompt
        # (parsed the same way the real pipeline builds it -- "[doc_id] title (status):
        # content") rather than returning one fixed, uncited canned line, so offline/CI
        # runs actually exercise the claim-audit's citation, numeric-support, and
        # staleness checks against real values instead of a scripted answer.
        if "Context:\n" in prompt and "Question:" in prompt:
            ctx_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
            q_match = re.search(r"Question:\s*(.*?)\n\nAnswer", prompt, re.DOTALL)
            ctx_text = ctx_match.group(1) if ctx_match else ""
            question_text = (q_match.group(1) if q_match else prompt).lower()
            entries = re.findall(
                r"\[([A-Z0-9#-]+)\]\s+(.*?)\s+\((active|obsolete)\):\s+(.*?)(?=\n\n\[|\Z)",
                ctx_text, re.DOTALL,
            )
            q_words = set(re.findall(r"[a-z]{4,}", question_text))
            best = None
            best_overlap = 0
            for doc_id, title, status, content in entries:
                if status != "active":
                    continue
                # Match against title + content -- the section body alone can
                # omit the very terms (e.g. "refund", "customer") that only
                # appear in the document's own title.
                c_words = set(re.findall(r"[a-z]{4,}", f"{title} {content}".lower()))
                overlap = len(q_words & c_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (doc_id, content)
            if not best or best_overlap < 2:
                return ("I don't know based on the provided context -- none of the retrieved "
                        "documents address this question.")
            doc_id, content = best
            plain_content = content.replace("**", "")
            day_m = re.search(r"(\d+)\s*calendar\s*days", plain_content)
            pct_m = re.search(r"(\d+)%", plain_content)
            if day_m:
                days = day_m.group(1)
                pct = pct_m.group(1) if pct_m else "100"
                return (f"New subscriptions canceled within {days} calendar days qualify for a "
                        f"{pct}% money-back guarantee [{doc_id}].")
            body = re.sub(r"^##?\s*.*\n", "", plain_content.strip())  # drop the leading section heading
            body = re.sub(r"^\*\s*", "", body.strip())                    # drop a leading list bullet
            first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0] if body.strip() else plain_content.strip()
            # Keep the citation inside the same sentence (before the final
            # period), not as a trailing fragment split off by sentence tokenization.
            return f"{first_sentence.rstrip('.').strip()} [{doc_id}]."

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

        # Travel planning: build a plan that satisfies the embedded
        # constraints JSON exactly, so offline/CI runs exercise the same
        # schema-validated TravelPlan a real model is asked to produce
        # (Chapter 2), rather than a shape the real prompt never requested.
        if "itinerary" in prompt_lower or "travel" in prompt_lower or '"destination"' in prompt:
            m = re.search(r"Constraints[^:]*:\s*(\{.*?\})", prompt, re.DOTALL)
            constraints = {}
            if m:
                try:
                    constraints = json.loads(m.group(1))
                except json.JSONDecodeError:
                    constraints = {}
            destination = constraints.get("destination", "Paris")
            days = constraints.get("days", 3)
            budget = constraints.get("budget", 900)
            interests = constraints.get("interests") or ["sightseeing"]
            per_day_cost = round((budget * 0.8) / max(1, days), 2)
            day_plans = [
                {"day": i + 1, "activities": [f"{interests[i % len(interests)]} in {destination}"], "cost": per_day_cost}
                for i in range(days)
            ]
            return json.dumps({"destination": destination, "days": day_plans})

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

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Deterministic tool-call simulation for offline dev/tests only.

        Unlike the old keyword-routing agent this REPLACED, this reads the
        actual values out of the user's message (zip code, email, order id)
        instead of hard-coding them, and tracks which tools already ran in
        this episode from the message history, so a multi-step tool loop
        still terminates sensibly. It is still a mock: never present its
        output as a captured live-model transcript.
        """
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        text = (user_msgs[0] if user_msgs else "").lower()
        done_tools = [m.get("name") or self._infer_tool_from_tool_msg(m) for m in messages if m.get("role") == "tool"]

        def call(name, args):
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": name, "arguments": args}}
            ]}

        def finish(content):
            return {"role": "assistant", "content": content, "tool_calls": []}

        zip_match = re.search(r"\b(\d{5})\b", text)
        order_match = re.search(r"#?\b(\d{4,5})\b", text)
        email_match = re.search(r"[^\s@]+@[^\s@]+\.[a-z]{2,}", text)

        if "weather" in text:
            if "get_weather" not in done_tools:
                return call("get_weather", {"zip_code": zip_match.group(1) if zip_match else "00000"})
            if "email" in text and "send_email" not in done_tools:
                return call("send_email", {
                    "to": email_match.group(0) if email_match else "unknown@example.com",
                    "subject": "Weather Update", "body": "Here is the forecast you asked about.",
                })
            return finish("Weather checked and notification handled.")

        if "order" in text or "refund" in text:
            if "get_order" not in done_tools:
                if not order_match:
                    return finish("I need your order number to look that up -- could you share it?")
                return call("get_order", {"order_id": order_match.group(1)})
            if "calculate_refund" not in done_tools:
                return call("calculate_refund", {"order_id": order_match.group(1) if order_match else ""})
            return finish("I've checked your order and refund eligibility above.")

        if "customer" in text or "search" in text:
            if "search_customer" not in done_tools:
                name_match = re.search(r"(?:for|customer)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text, re.IGNORECASE)
                return call("search_customer", {"name": name_match.group(1) if name_match else ""})
            return finish("Customer profile located above.")

        # Generic diagnostic-style fallback (e.g. Chapter 4's IT helpdesk
        # tools), driven entirely by the tool schemas actually offered and
        # the observations seen so far -- no chapter-specific keywords.
        tool_names = [t["function"]["name"] for t in (tools or [])]
        if tool_names:
            last_obs = ""
            for m in reversed(messages):
                if m.get("role") == "tool":
                    last_obs = (m.get("content") or "").lower()
                    break

            def first_unused(prefixes):
                return next((n for n in tool_names if n not in done_tools and n.startswith(prefixes)), None)

            if "expired" in last_obs or "disconnected" in last_obs:
                fix = next((n for n in tool_names if n not in done_tools and n.startswith(("reset_", "rotate_", "fix_"))), None)
                if fix:
                    return call(fix, self._guess_args(fix))
            if "full" in last_obs or "insufficient" in last_obs:
                fix = next((n for n in tool_names if n not in done_tools and n.startswith(("archive_", "compress_"))), None)
                if fix:
                    return call(fix, self._guess_args(fix))
            probe = first_unused(("check_", "query_", "ping_"))
            if probe:
                return call(probe, self._guess_args(probe))
            verify = first_unused(("verify_", "test_"))
            if verify:
                return call(verify, self._guess_args(verify))
            if tool_names and not done_tools:
                return call(tool_names[0], self._guess_args(tool_names[0]))
            return finish("Diagnosis complete based on the checks performed above.")

        return finish("I can help with orders, refunds, weather lookups, or customer search -- what do you need?")

    @staticmethod
    def _guess_args(tool_name: str) -> Dict[str, str]:
        """Plausible placeholder args for the generic diagnostic fallback --
        offline/test mode only; a real model fills these from context."""
        if "host" in tool_name or tool_name in ("ping_host", "verify_connectivity", "check_disk"):
            return {"host": "target-host"}
        if "user" in tool_name or tool_name in ("check_vpn_status", "reset_vpn_session", "test_saml_assertion"):
            return {"user_id": "u000"}
        if tool_name == "compress_logs":
            return {"older_than_days": "7"}
        if tool_name == "archive_to_s3":
            return {"path": "/var/log/app.log"}
        if tool_name == "query_idp_metadata":
            return {"app_id": "app_saml_009"}
        if tool_name == "rotate_signing_cert":
            return {"app_id": "app_saml_009", "key_id": "cert_primary"}
        return {}

    @staticmethod
    def _infer_tool_from_tool_msg(m: Dict[str, Any]) -> str:
        return m.get("name", "")


class ProviderUnavailableError(RuntimeError):
    """Raised when a live run cannot reach its configured model backend.

    This is the fail-closed replacement for the old behavior of silently
    swapping a down Ollama server for the deterministic mock. A caller
    that hits this should surface it to the user (a UI banner, a failed
    CI job) rather than catch-and-continue with fabricated output.
    """


def get_model_provider(force_mock: bool = False) -> LLMProvider:
    """Factory for the active LLM backend.

    Fails closed: this NEVER silently swaps a live run to the mock
    provider. The mock is returned only when explicitly requested
    (``force_mock=True`` or ``MOCK_LLM=1``) -- the correct switch for
    unit tests, CI, and offline development. Every other call returns a
    real ``OllamaClient``; if the server is unreachable, ``generate()``
    raises rather than returning canned text, so a dashboard can never
    show green KPIs for a model that never ran.
    """
    if force_mock or os.environ.get("MOCK_LLM") == "1":
        return DeterministicMockProvider()
    return OllamaClient()


def provider_identity(provider: LLMProvider) -> str:
    """Human-readable identity string for run manifests and UI badges."""
    if isinstance(provider, DeterministicMockProvider):
        return "mock:deterministic"
    if isinstance(provider, OllamaClient):
        return f"ollama:{provider.base_url}"
    return provider.__class__.__name__


def require_live_provider(provider: LLMProvider) -> None:
    """Fail closed before a live run if `provider` cannot actually serve it.

    Call this once at the start of a chat turn or an evaluation suite.
    An explicit mock (opted into via MOCK_LLM=1 / force_mock=True) always
    passes; a real provider that fails its own availability check raises
    ProviderUnavailableError with an actionable message instead of being
    silently substituted.
    """
    if isinstance(provider, DeterministicMockProvider):
        return
    if not provider.is_available():
        raise ProviderUnavailableError(
            f"Model backend unreachable: {provider_identity(provider)}. "
            "Start Ollama (`ollama serve`) and pull the required models "
            "(see README), or set MOCK_LLM=1 for offline/dev/test mode."
        )
