"""
Pydantic models for the AI Customer Support Resolution Environment.
Defines the core data structures: Observation, Action, and Reward.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action Space — structured, no free-form text
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    # Classification actions
    CLASSIFY_BILLING    = "classify_billing"
    CLASSIFY_TECHNICAL  = "classify_technical"
    CLASSIFY_REFUND     = "classify_refund"
    CLASSIFY_GENERAL    = "classify_general"

    # Information gathering
    REQUEST_MORE_INFO   = "request_more_info"

    # Resolution actions
    ISSUE_REFUND        = "issue_refund"
    APPLY_BILLING_CREDIT = "apply_billing_credit"
    RESTART_SERVICE     = "restart_service"
    SEND_TECHNICAL_GUIDE = "send_technical_guide"
    RESET_PASSWORD      = "reset_password"

    # Escalation / closure
    ESCALATE_TO_HUMAN   = "escalate_to_human"
    CLOSE_TICKET        = "close_ticket"


class Action(BaseModel):
    """Agent action submitted to the environment."""
    action_type: ActionType = Field(..., description="The action the agent wants to perform.")
    reasoning: Optional[str] = Field(
        None,
        description="Optional agent reasoning (not used for grading, aids interpretability).",
    )

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Observation — what the agent sees each step
# ---------------------------------------------------------------------------

class SentimentLevel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"
    ANGRY    = "angry"


class TicketStatus(str, Enum):
    OPEN       = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED  = "escalated"
    RESOLVED   = "resolved"
    CLOSED     = "closed"


class Observation(BaseModel):
    """The full observable state returned to the agent."""
    ticket_id: str = Field(..., description="Unique identifier for the support ticket.")
    customer_message: str = Field(..., description="The customer's original support message.")
    customer_history: List[str] = Field(
        default_factory=list,
        description="List of prior interactions / notes on this customer's account.",
    )
    sentiment: SentimentLevel = Field(
        SentimentLevel.NEUTRAL,
        description="Detected sentiment of the customer message.",
    )
    current_status: TicketStatus = Field(
        TicketStatus.OPEN,
        description="Current lifecycle status of the ticket.",
    )
    steps_taken: List[str] = Field(
        default_factory=list,
        description="Ordered list of action_type strings already executed this episode.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types that are currently valid to call.",
    )
    sla_steps_remaining: int = Field(
        ...,
        description="Steps remaining before SLA breach (penalty kicks in).",
    )
    info_message: Optional[str] = Field(
        None,
        description="Contextual feedback from the environment after the last action.",
    )


# ---------------------------------------------------------------------------
# Reward — per-step and cumulative
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Reward signal returned after each step."""
    step_reward: float = Field(..., description="Reward earned in this single step.")
    cumulative_reward: float = Field(..., description="Total reward accumulated this episode.")
    done: bool = Field(..., description="Whether the episode has ended.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised task score in [0, 1] — only meaningful when done=True.",
    )
    feedback: str = Field(..., description="Human-readable explanation of the reward signal.")


# ---------------------------------------------------------------------------
# API-level wrappers
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward


class StateResponse(BaseModel):
    observation: Observation
    cumulative_reward: float
    episode_steps: int


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    task_description: str
