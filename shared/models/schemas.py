"""Typed schemas and data models for agent evaluation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time
import uuid


class EvaluationCase(BaseModel):
    """Represents a single evaluation test case."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_prompt: str
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    category: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class MetricScore(BaseModel):
    """Represents a score for an individual evaluation metric."""
    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    passed: bool = True
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """Represents a single step in an agent trajectory."""
    step_index: int
    action: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    observation: Optional[str] = None
    result: str = "success"  # success, error, retry, aborted
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    """Complete execution trace of an agent run."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    task: str
    steps: List[AgentStep] = Field(default_factory=list)
    final_output: Optional[str] = None
    total_duration_ms: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_step(
        self,
        action: str,
        arguments: Dict[str, Any] = None,
        observation: Optional[str] = None,
        result: str = "success",
        duration_ms: float = 0.0,
        metadata: Dict[str, Any] = None,
    ) -> AgentStep:
        step = AgentStep(
            step_index=len(self.steps) + 1,
            action=action,
            arguments=arguments or {},
            observation=observation,
            result=result,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step


class JudgeRubric(BaseModel):
    """Rubric criterion for LLM-as-a-Judge evaluations."""
    dimension: str
    min_score: float = 0.0
    max_score: float = 5.0
    weight: float = 1.0
    description: str
    scoring_guide: Dict[int, str] = Field(default_factory=dict)


class JudgeEvaluation(BaseModel):
    """Structured evaluation output from an LLM Judge.

    `status` distinguishes a real verdict from an abstention: a judge
    whose output could not be parsed (even after one corrective retry)
    must abstain, not silently pass. `passed=None` is that abstain state
    -- it must never be treated as a pass by downstream code.
    """
    judge_model: str
    status: str = "scored"  # "scored" | "abstain"
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    passed: Optional[bool] = True
    reasoning: str = ""
    evidence: List[str] = Field(default_factory=list)
    raw: str = ""  # kept for audit and debugging


class EvaluationResult(BaseModel):
    """Evaluation result for a single test case."""
    case_id: str
    input_prompt: str
    actual_output: str
    expected_output: Optional[str] = None
    metrics: Dict[str, MetricScore] = Field(default_factory=dict)
    passed: bool = True
    latency_seconds: float = 0.0
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0
    trace: Optional[AgentTrace] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
