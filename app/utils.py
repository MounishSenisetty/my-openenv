"""
Utility helpers for the Customer Support Resolution Environment.
"""

from typing import List, Dict, Any
from app.models import ActionType, SentimentLevel, TicketStatus


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

SENTIMENT_REWARD_MODIFIER: Dict[str, float] = {
    SentimentLevel.POSITIVE: 0.05,   # Small bonus — customer is happy
    SentimentLevel.NEUTRAL: 0.0,
    SentimentLevel.NEGATIVE: -0.05,  # Slight pressure to resolve faster
    SentimentLevel.ANGRY: -0.10,     # Angry customers need precise handling
}


def sentiment_modifier(sentiment: str) -> float:
    """Return the per-step reward modifier based on customer sentiment."""
    return SENTIMENT_REWARD_MODIFIER.get(sentiment, 0.0)


# ---------------------------------------------------------------------------
# Valid action sets by ticket status
# ---------------------------------------------------------------------------

VALID_ACTIONS_BY_STATUS: Dict[str, List[str]] = {
    TicketStatus.OPEN: [
        ActionType.CLASSIFY_BILLING,
        ActionType.CLASSIFY_TECHNICAL,
        ActionType.CLASSIFY_REFUND,
        ActionType.CLASSIFY_GENERAL,
        ActionType.REQUEST_MORE_INFO,
        ActionType.FETCH_CUSTOMER_HISTORY,
        ActionType.CHECK_SERVICE_STATUS,
        ActionType.VERIFY_BILLING_LEDGER,
        ActionType.ESCALATE_TO_HUMAN,
    ],
    TicketStatus.IN_PROGRESS: [
        ActionType.ISSUE_REFUND,
        ActionType.APPLY_BILLING_CREDIT,
        ActionType.RESTART_SERVICE,
        ActionType.SEND_TECHNICAL_GUIDE,
        ActionType.RESET_PASSWORD,
        ActionType.REQUEST_MORE_INFO,
        ActionType.FETCH_CUSTOMER_HISTORY,
        ActionType.CHECK_SERVICE_STATUS,
        ActionType.VERIFY_BILLING_LEDGER,
        ActionType.ESCALATE_TO_HUMAN,
        ActionType.CLOSE_TICKET,
    ],
    TicketStatus.ESCALATED: [
        ActionType.ISSUE_REFUND,
        ActionType.APPLY_BILLING_CREDIT,
        ActionType.RESTART_SERVICE,
        ActionType.SEND_TECHNICAL_GUIDE,
        ActionType.RESET_PASSWORD,
        ActionType.CHECK_SERVICE_STATUS,
        ActionType.VERIFY_BILLING_LEDGER,
        ActionType.CLOSE_TICKET,
    ],
    TicketStatus.RESOLVED: [
        ActionType.CLOSE_TICKET,
    ],
    TicketStatus.CLOSED: [],
}


def get_valid_actions(status: str) -> List[str]:
    """Return the list of valid action type strings for the given status."""
    return [a.value if hasattr(a, "value") else a
            for a in VALID_ACTIONS_BY_STATUS.get(status, [])]


# ---------------------------------------------------------------------------
# Status transition logic
# ---------------------------------------------------------------------------

CLASSIFICATION_ACTIONS = {
    ActionType.CLASSIFY_BILLING,
    ActionType.CLASSIFY_TECHNICAL,
    ActionType.CLASSIFY_REFUND,
    ActionType.CLASSIFY_GENERAL,
}

RESOLUTION_ACTIONS = {
    ActionType.ISSUE_REFUND,
    ActionType.APPLY_BILLING_CREDIT,
    ActionType.RESTART_SERVICE,
    ActionType.SEND_TECHNICAL_GUIDE,
    ActionType.RESET_PASSWORD,
}


def next_status(current_status: str, action_type: str) -> str:
    """Deterministic status transition given current status and action."""
    at = action_type

    if at == ActionType.ESCALATE_TO_HUMAN:
        return TicketStatus.ESCALATED

    if at == ActionType.CLOSE_TICKET:
        return TicketStatus.CLOSED

    if at in {a.value for a in CLASSIFICATION_ACTIONS}:
        if current_status == TicketStatus.OPEN:
            return TicketStatus.IN_PROGRESS

    if at in {a.value for a in RESOLUTION_ACTIONS}:
        return TicketStatus.RESOLVED

    return current_status  # No transition


# ---------------------------------------------------------------------------
# Dense reward calculator
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_type: str,
    goal_actions: List[str],
    steps_taken: List[str],   # BEFORE this action
    current_status: str,
    sentiment: str,
    is_done: bool,
    final_score: float,
    repeated_count: int = 0,
    investigation_steps: int = 0,
    required_investigation_actions: List[str] | None = None,
) -> tuple[float, str]:
    """
    Dense reward function.

    Base rules:
      +0.2  correct step (action matches expected position in goal sequence)
      -0.1  incorrect action
      +0.3  progress toward resolution (moves status forward)
      -0.2  repeated / useless action
      +1.0  correct final resolution bonus (applied at done=True)

    Modifiers:
      ±sentiment_modifier  per-step adjustment based on customer mood
      -0.3  escalate_to_human when NOT in goal sequence (unnecessary escalation)
    """
    smod = sentiment_modifier(sentiment)
    step_index = len(steps_taken)  # 0-based index of the action about to be taken

    # Check for repeated action (anti-loop shaping)
    if action_type in steps_taken:
        reward = -0.2 + smod
        if repeated_count >= 2:
            reward -= 0.1
            return round(reward, 4), "Looping repeated action penalty."
        return round(reward, 4), "Repeated action — no value added."

    # Check if action matches the expected position in goal_actions
    if step_index < len(goal_actions) and action_type == goal_actions[step_index]:
        reward = 0.2 + smod
        feedback = f"Correct step {step_index + 1}/{len(goal_actions)}."
    elif action_type in goal_actions:
        # Right action, wrong position — partial credit
        reward = 0.05 + smod
        feedback = "Action is in the goal sequence but out of order."
    else:
        reward = -0.1 + smod
        feedback = "Incorrect action for this ticket."

    # Progress bonus: resolution-moving actions
    progress_actions = {
        ActionType.ISSUE_REFUND,
        ActionType.APPLY_BILLING_CREDIT,
        ActionType.RESTART_SERVICE,
        ActionType.SEND_TECHNICAL_GUIDE,
        ActionType.RESET_PASSWORD,
        ActionType.CLOSE_TICKET,
    }
    investigation_actions = {
        ActionType.REQUEST_MORE_INFO,
        ActionType.FETCH_CUSTOMER_HISTORY,
        ActionType.CHECK_SERVICE_STATUS,
        ActionType.VERIFY_BILLING_LEDGER,
    }
    if action_type in {a.value for a in progress_actions}:
        reward += 0.3
        feedback += " +Progress bonus."
    elif action_type in {a.value for a in investigation_actions}:
        reward += 0.08
        feedback += " +Information gain bonus."

    # Unnecessary escalation penalty
    if action_type == ActionType.ESCALATE_TO_HUMAN and ActionType.ESCALATE_TO_HUMAN not in goal_actions:
        reward -= 0.3
        feedback += " Unnecessary escalation penalty."

    required_investigation_actions = required_investigation_actions or []
    if (
        action_type == ActionType.ESCALATE_TO_HUMAN
        and required_investigation_actions
        and investigation_steps == 0
    ):
        reward -= 0.12
        feedback += " Blind escalation penalty."

    if (
        action_type == ActionType.CLOSE_TICKET
        and required_investigation_actions
        and investigation_steps == 0
    ):
        reward -= 0.15
        feedback += " Premature close penalty."

    # Terminal bonus
    if is_done and final_score >= 0.8:
        reward += 1.0
        feedback += " +Terminal resolution bonus."

    return round(reward, 4), feedback
