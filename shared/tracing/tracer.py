"""Lightweight OpenTelemetry-compatible tracing tracer."""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Span(BaseModel):
    """Represents a single timed unit of work (e.g. tool call, LLM call, verifier)."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "ok"  # ok, error
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)

    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status


class Tracer:
    """Manages spans and trace hierarchy."""

    def __init__(self, service_name: str = "agentic-evals"):
        self.service_name = service_name
        self.spans: List[Span] = []
        self.current_trace_id: str = str(uuid.uuid4())[:12]

    def start_trace(self, task_name: str = "task") -> str:
        self.current_trace_id = str(uuid.uuid4())[:12]
        self.spans.clear()
        return self.current_trace_id

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        span = Span(
            trace_id=self.current_trace_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "trace_id": self.current_trace_id,
            "total_spans": len(self.spans),
            "spans": [s.model_dump() for s in self.spans],
        }
