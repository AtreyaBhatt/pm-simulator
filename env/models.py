"""
Typed models for the PM Simulator OpenEnv environment.
All models are Pydantic v2 for full spec compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    P0 = "P0"  # Critical — drop everything
    P1 = "P1"  # High — this sprint
    P2 = "P2"  # Medium — next sprint
    P3 = "P3"  # Low — backlog / nice-to-have


class ItemType(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    TECH_DEBT = "tech_debt"
    RESEARCH = "research"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CustomerTier(str, Enum):
    ENTERPRISE = "enterprise"
    GROWTH = "growth"
    FREE = "free"


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class ProductItem(BaseModel):
    """A single item in the product backlog — bug, feature, tech debt, or research."""

    id: str
    title: str
    description: str
    type: ItemType
    reporter: str
    severity: Severity
    user_impact: int = Field(ge=0, description="Estimated number of affected users")
    effort_points: int = Field(ge=1, le=21, description="Story points (Fibonacci: 1,2,3,5,8,13,21)")
    strategic_score: float = Field(ge=0.0, le=1.0, description="Alignment with company OKRs (0=none, 1=core)")
    age_days: int = Field(ge=0, description="Days since item was created/reported")
    customer_tier: CustomerTier
    stakeholder_votes: dict[str, int] = Field(
        default_factory=dict,
        description="Votes/weight from each stakeholder (Sales, Eng, Support, Growth)"
    )
    tags: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list, description="IDs of items this depends on")
    revenue_impact_usd: Optional[int] = Field(None, description="Estimated ARR impact if addressed")
    okr_tags: list[str] = Field(default_factory=list, description="Which company OKRs this item contributes to")

    # Mutable state — set by agent actions
    assigned_priority: Optional[Priority] = None
    assigned_sprint: Optional[int] = None
    deferred: bool = False
    defer_reason: Optional[str] = None


class TeamMetrics(BaseModel):
    """Live product/team health metrics shown to the agent for context."""

    nps: float = Field(description="Net Promoter Score (-100 to 100)")
    churn_rate_pct: float = Field(description="Monthly churn rate as percentage")
    open_bug_count: int
    p0_bug_count: int
    avg_resolution_days: float
    sprint_velocity: int = Field(description="Avg story points completed per sprint")
    enterprise_at_risk_count: int = Field(description="Enterprise customers flagged as churn risk")


# ---------------------------------------------------------------------------
# OpenEnv interface types
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at every step."""

    task_id: str
    task_name: str
    task_description: str
    items: list[ProductItem]
    sprint_capacity: int = Field(description="Total story points available this sprint")
    current_sprint_load: int = Field(description="Story points already committed")
    metrics: TeamMetrics
    step_number: int
    max_steps: int
    context: dict[str, Any] = Field(default_factory=dict, description="Task-specific extra context")
    message: str = Field(default="", description="Environment feedback on last action")
    available_actions: list[str] = Field(default_factory=list)


class Action(BaseModel):
    """
    What the agent submits at each step.

    action_type options:
      - "prioritize"    : assign a P0–P3 priority to one item
      - "schedule"      : add a prioritized item to a sprint number
      - "defer"         : explicitly move item to future/backlog with a reason
      - "request_info"  : ask a clarifying question (costs 1 step, returns more context)
      - "submit"        : finalize all decisions and trigger the grader
    """

    action_type: Literal["prioritize", "schedule", "defer", "request_info", "submit"]
    item_id: Optional[str] = Field(None, description="Target item ID for the action")
    priority: Optional[Priority] = Field(None, description="For 'prioritize' actions")
    sprint: Optional[int] = Field(None, ge=1, le=3, description="Sprint number (1, 2, or 3)")
    reason: Optional[str] = Field(None, description="Rationale for defer or prioritize")
    question: Optional[str] = Field(None, description="Clarification question for request_info")
    thought: Optional[str] = Field(None, description="Agent chain-of-thought (logged, not graded)")


class Reward(BaseModel):
    """Reward signal returned after each step."""

    value: float = Field(ge=-1.0, le=1.0, description="Per-step reward")
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Named reward components for interpretability"
    )
    cumulative: float = Field(default=0.0, description="Total reward so far this episode")
    message: str = Field(default="", description="Human-readable explanation")


class StepResult(BaseModel):
    """Full return value of env.step()."""

    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Complete serialisable state for state() endpoint."""

    task_id: str
    task_name: str
    step: int
    max_steps: int
    items: list[ProductItem]
    sprint_capacity: int
    current_sprint_load: int
    actions_taken: list[dict[str, Any]]
    cumulative_reward: float
    done: bool
    final_score: Optional[float] = None


class TaskInfo(BaseModel):
    """Metadata about a task (returned by /tasks endpoint)."""

    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    sprint_capacity: int
    item_count: int
    grader_description: str
