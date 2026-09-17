"""Multi-agent research system topology: Supervisor -> Researcher / Analyst -> Synthesizer.

Fixed from an earlier version of this lab: the worker agents now generate their content
live through the model instead of returning a fixed script -- the
provider was previously constructed and never called. Delegation
messages from the Supervisor stay templated (they are routing, not
claims that need evaluating); the Researcher, Analyst, and Synthesizer
outputs are real model output the evaluator has to actually check.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from shared.models.provider import LLMProvider, get_model_provider


class AgentMessage(BaseModel):
    message_id: int
    sender: str
    recipient: str
    content: str
    handoff_status: str = "success"  # success, failed
    timestamp: float = Field(default_factory=time.time)


RESEARCHER_PROMPT = """You are the Researcher on a strategy research team.
Write 2-3 sentences of foundational/historical context for the topic
below. Do not include statistics or percentages -- that is the
Analyst's job."""

ANALYST_PROMPT = """You are the Analyst on a strategy research team.
Write 1-2 sentences with a quantitative finding (a percentage or metric)
relevant to the topic below. Every number you state MUST be followed by
a source tag in the exact form [source: <short description>] in the
same sentence. Do not state a number without a source tag."""

SYNTHESIZER_PROMPT = """You are the Synthesizer on a strategy research
team. Write a short executive synthesis combining the Researcher's and
Analyst's findings below. Only restate facts and numbers that already
appear in their text -- never introduce a new number, and never claim
a specific degree of agreement (like "zero role overlap") that neither
of them stated."""


class MultiAgentResearchSystem:
    """Orchestrates multi-agent collaboration with explicit, live-generated message passing."""

    def __init__(self, provider: Optional[LLMProvider] = None, model: str = "qwen2.5:3b"):
        self.provider = provider or get_model_provider()
        self.model = model

    def run_research(
        self,
        research_topic: str,
        inject_handoff_failure: bool = False,
    ) -> Dict[str, Any]:
        messages: List[AgentMessage] = []
        msg_id = 1

        def next_id():
            nonlocal msg_id
            v = msg_id
            msg_id += 1
            return v

        messages.append(AgentMessage(
            message_id=next_id(), sender="Supervisor", recipient="Researcher",
            content=f"Investigate foundational literature and historical facts for: '{research_topic}'.",
        ))
        messages.append(AgentMessage(
            message_id=next_id(), sender="Supervisor", recipient="Analyst",
            content=f"Evaluate statistical benchmarks, trade-offs, and failure rates for: '{research_topic}'. "
                    f"Cite a source tag for every number you state.",
        ))

        res_content = self.provider.generate(
            model=self.model, system=RESEARCHER_PROMPT, prompt=research_topic, temperature=0.3,
        )
        messages.append(AgentMessage(message_id=next_id(), sender="Researcher", recipient="Synthesizer", content=res_content))

        if inject_handoff_failure:
            messages.append(AgentMessage(
                message_id=next_id(), sender="Analyst", recipient="Synthesizer",
                content="[Communication error: Packet payload dropped during handoff].",
                handoff_status="failed",
            ))
            messages.append(AgentMessage(
                message_id=next_id(), sender="Supervisor", recipient="Analyst",
                content="Handoff timeout detected. Requesting immediate re-transmission.",
            ))

        analyst_content = self.provider.generate(
            model=self.model, system=ANALYST_PROMPT, prompt=research_topic, temperature=0.3,
        )
        messages.append(AgentMessage(message_id=next_id(), sender="Analyst", recipient="Synthesizer", content=analyst_content))

        synth_input = f"Researcher findings:\n{res_content}\n\nAnalyst findings:\n{analyst_content}"
        final_synthesis = self.provider.generate(
            model=self.model, system=SYNTHESIZER_PROMPT, prompt=synth_input, temperature=0.3,
        )
        messages.append(AgentMessage(message_id=next_id(), sender="Synthesizer", recipient="Supervisor", content=final_synthesis))

        return {
            "topic": research_topic,
            "messages": messages,
            "researcher_content": res_content,
            "analyst_content": analyst_content,
            "final_synthesis": final_synthesis,
            "completed": True,
        }
