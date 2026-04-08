"""
AI Customer Support Resolution Environment — core logic.

Implements the OpenEnv interface:
  reset(task_id)  → ResetResponse
  step(action)    → StepResponse
  state()         → StateResponse
"""

from typing import Optional, List
import copy

from app.models import (
    Action, ActionType, Observation,
    SentimentLevel, TicketStatus,
    StepResponse, StateResponse, ResetResponse,
)
from app.tasks import TICKET_BY_ID, TASK_BY_ID, TASKS
from app.grader import grade
from app.utils import (
    get_valid_actions, next_status, compute_step_reward,
)

STRICT_SCORE_EPS = 0.001


def strict_score(score: float) -> float:
    return min(1.0 - STRICT_SCORE_EPS, max(STRICT_SCORE_EPS, score))


class CustomerSupportEnv:
    """
    Stateful OpenEnv-compatible environment for AI customer support resolution.

    Episode lifecycle
    -----------------
    1. reset(task_id) — loads a task and returns the initial observation.
    2. step(action)   — agent submits one action; env returns (obs, reward).
    3. state()        — read-only snapshot of current observation + stats.

    An episode ends when:
      • close_ticket is called, OR
      • max_steps is reached (SLA breach + forced termination).
    """

    def __init__(self) -> None:
        self._task: Optional[dict] = None
        self._ticket: Optional[dict] = None
        self._steps_taken: List[str] = []
        self._cumulative_reward: float = 0.0
        self._current_status: str = TicketStatus.OPEN
        self._done: bool = False
        self._history_revealed: bool = False
        self._known_facts: List[str] = []
        self._risk_flags: List[str] = []
        self._investigation_steps: int = 0
        self._decision_trace: List[str] = []
        self._action_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> ResetResponse:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            If None, the first task is used (useful for quick testing).
        """
        if task_id is None:
            task_id = TASKS[0]["task_id"]

        if task_id not in TASK_BY_ID:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid ids: {list(TASK_BY_ID.keys())}"
            )

        self._task = copy.deepcopy(TASK_BY_ID[task_id])
        self._ticket = copy.deepcopy(TICKET_BY_ID[self._task["ticket_id"]])
        self._steps_taken = []
        self._cumulative_reward = 0.0
        self._current_status = TicketStatus.OPEN
        self._done = False
        self._history_revealed = False
        self._known_facts = []
        self._risk_flags = []
        self._investigation_steps = 0
        self._decision_trace = []
        self._action_counts = {}

        obs = self._build_observation()
        return ResetResponse(
            observation=obs,
            task_id=task_id,
            task_description=self._task["description"],
        )

    def step(self, action: Action) -> StepResponse:
        """
        Execute one agent action and return the updated observation + reward.
        """
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        action_type = action.action_type  # already a string via use_enum_values
        goal_actions = self._task["goal_actions"]
        max_steps = self._task["max_steps"]
        sla_steps = self._task["sla_steps"]
        sentiment = self._ticket["sentiment"]
        repeated_count = self._action_counts.get(action_type, 0)

        # ------------------------------------------------------------------
        # Determine done condition BEFORE updating state
        # ------------------------------------------------------------------
        will_close = action_type == ActionType.CLOSE_TICKET
        at_max = len(self._steps_taken) + 1 >= max_steps

        # After this action, will the episode end?
        next_done = will_close or at_max

        # ------------------------------------------------------------------
        # Compute final score (needed for terminal reward)
        # ------------------------------------------------------------------
        future_steps = self._steps_taken + [action_type]
        final_score, grade_feedback = grade(
            difficulty=self._task["difficulty"],
            steps_taken=future_steps,
            goal_actions=goal_actions,
            max_steps=max_steps,
            sla_steps=sla_steps,
            valid_paths=self._task.get("valid_paths"),
            required_investigation_actions=self._task.get("required_investigation_actions"),
        ) if next_done else (0.0, "")

        # ------------------------------------------------------------------
        # Dense step reward
        # ------------------------------------------------------------------
        step_reward, reward_feedback = compute_step_reward(
            action_type=action_type,
            goal_actions=goal_actions,
            steps_taken=self._steps_taken,
            current_status=self._current_status,
            sentiment=sentiment,
            is_done=next_done,
            final_score=final_score,
            repeated_count=repeated_count,
            investigation_steps=self._investigation_steps,
            required_investigation_actions=self._task.get("required_investigation_actions", []),
        )
        bounded_step_reward = max(0.0, min(1.0, step_reward))

        investigation_feedback = self._apply_investigation(action_type)
        if investigation_feedback:
            reward_feedback = f"{reward_feedback} {investigation_feedback}".strip()

        # ------------------------------------------------------------------
        # Apply state transitions
        # ------------------------------------------------------------------
        self._steps_taken.append(action_type)
        self._action_counts[action_type] = repeated_count + 1
        self._decision_trace.append(f"{len(self._steps_taken)}:{action_type}")
        self._current_status = next_status(self._current_status, action_type)
        self._cumulative_reward += bounded_step_reward
        self._done = next_done

        # ------------------------------------------------------------------
        # Build outputs
        # ------------------------------------------------------------------
        obs = self._build_observation(info_message=reward_feedback)

        info_message = (
            f"{reward_feedback} | Grade: {grade_feedback}"
            if self._done
            else reward_feedback
        )

        return StepResponse(
            observation=obs,
            reward=round(bounded_step_reward, 4),
            done=self._done,
            info=info_message,
            score=strict_score(final_score) if self._done else 0.0,
            cumulative_reward=round(self._cumulative_reward, 4),
        )

    def state(self) -> StateResponse:
        """Return a read-only snapshot of the current episode state."""
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return StateResponse(
            observation=self._build_observation(),
            cumulative_reward=round(self._cumulative_reward, 4),
            episode_steps=len(self._steps_taken),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, info_message: Optional[str] = None) -> Observation:
        sla_remaining = max(
            0,
            self._task["sla_steps"] - len(self._steps_taken),
        )
        return Observation(
            ticket_id=self._ticket["ticket_id"],
            customer_message=self._ticket["customer_message"],
            customer_history=(
                self._ticket["customer_history"] if self._history_revealed else []
            ),
            sentiment=SentimentLevel(self._ticket["sentiment"]),
            current_status=TicketStatus(self._current_status),
            steps_taken=list(self._steps_taken),
            available_actions=get_valid_actions(self._current_status),
            sla_steps_remaining=sla_remaining,
            info_message=info_message,
            known_facts=list(self._known_facts),
            hidden_context_revealed=bool(self._known_facts),
            risk_flags=list(self._risk_flags),
            investigation_steps_used=self._investigation_steps,
            decision_trace=list(self._decision_trace),
        )

    def _apply_investigation(self, action_type: str) -> str:
        """Reveal hidden deterministic context through investigation actions."""
        hidden_context = self._ticket.get("hidden_context", {})
        feedback_parts: List[str] = []

        if action_type == ActionType.FETCH_CUSTOMER_HISTORY:
            self._history_revealed = True

        if action_type in {
            ActionType.REQUEST_MORE_INFO,
            ActionType.FETCH_CUSTOMER_HISTORY,
            ActionType.CHECK_SERVICE_STATUS,
            ActionType.VERIFY_BILLING_LEDGER,
        }:
            self._investigation_steps += 1

        if action_type in hidden_context:
            payload = hidden_context[action_type]
            for fact in payload.get("facts", []):
                if fact not in self._known_facts:
                    self._known_facts.append(fact)
            for flag in payload.get("risk_flags", []):
                if flag not in self._risk_flags:
                    self._risk_flags.append(flag)
            feedback_parts.append("Investigation revealed hidden context.")

        return " ".join(feedback_parts)
