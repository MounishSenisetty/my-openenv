"""
Deterministic graders for all three task difficulties.

All graders return a float score in [0.0, 1.0].

Grading philosophy
------------------
Easy   → binary: did the agent pick the correct classification action?
Medium → sequence prefix: how far along the correct path did the agent get?
Hard   → weighted composite:
           40 % correctness  (how many actions matched?)
           30 % sequence quality (were they in the right order?)
           30 % efficiency   (penalty for extra / redundant steps)
"""

from typing import List, Tuple, Optional

STRICT_SCORE_EPS = 0.001


def _strict_score(score: float) -> float:
    """Clamp scores to strict open interval (0, 1)."""
    return min(1.0 - STRICT_SCORE_EPS, max(STRICT_SCORE_EPS, score))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _longest_common_subsequence(seq_a: List[str], seq_b: List[str]) -> int:
    """Standard LCS length — used to measure sequence quality."""
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ---------------------------------------------------------------------------
# Task 1 — Easy: single classification check
# ---------------------------------------------------------------------------

def grade_easy(
    steps_taken: List[str],
    goal_actions: List[str],
) -> Tuple[float, str]:
    """
    Returns (score, feedback).

    A perfect score requires the FIRST action to match goal_actions[0].
    We award 0.5 if the agent eventually classifies correctly (just not first).
    """
    if not steps_taken:
        return 0.0, "No actions taken."

    target = goal_actions[0]

    if steps_taken[0] == target:
        return 1.0, f"Correct first action: {target}."

    if target in steps_taken:
        return 0.5, f"Correct classification found but not as the first action."

    return 0.0, f"Incorrect classification. Expected '{target}', got '{steps_taken[0]}'."


# ---------------------------------------------------------------------------
# Task 2 — Medium: correct resolution path
# ---------------------------------------------------------------------------

def grade_medium(
    steps_taken: List[str],
    goal_actions: List[str],
) -> Tuple[float, str]:
    """
    Scores based on how many steps in the expected sequence were completed
    AND in the correct order (prefix matching).

    Score = matched_prefix_length / len(goal_actions)
    Bonus +0.1 if sequence completed with no extra steps.
    """
    if not steps_taken:
        return 0.0, "No actions taken."

    # Count matching prefix
    matched = 0
    for taken, expected in zip(steps_taken, goal_actions):
        if taken == expected:
            matched += 1
        else:
            break  # prefix broken

    base_score = matched / len(goal_actions)

    # Small bonus for clean completion
    bonus = 0.0
    if steps_taken == goal_actions:
        bonus = 0.1

    score = min(1.0, base_score + bonus)
    feedback = (
        f"Matched {matched}/{len(goal_actions)} steps in correct order. "
        f"{'Perfect sequence!' if bonus else ''}"
    )
    return round(score, 4), feedback


# ---------------------------------------------------------------------------
# Task 3 — Hard: multi-step weighted grading
# ---------------------------------------------------------------------------

def grade_hard(
    steps_taken: List[str],
    goal_actions: List[str],
    max_steps: int,
    sla_steps: int,
    valid_paths: Optional[List[List[str]]] = None,
    required_investigation_actions: Optional[List[str]] = None,
) -> Tuple[float, str]:
    """
    Weighted composite score:
      40% correctness  — what fraction of expected actions appeared?
      30% sequence     — LCS / len(goal) normalised
      30% efficiency   — penalised for extra steps and SLA breach

    SLA breach (len(steps_taken) > sla_steps) applies a further -0.15 penalty.
    Escalation penalty: if 'escalate_to_human' used when NOT in goal_actions,
    deduct 0.1 (unnecessary escalation cost).
    """
    if not steps_taken:
        return 0.0, "No actions taken."

    taken_set = set(steps_taken)
    candidate_paths = valid_paths if valid_paths else [goal_actions]

    # Evaluate against all valid paths and keep best deterministic candidate.
    best_correctness = 0.0
    best_sequence = 0.0
    best_path_len = len(goal_actions)
    for candidate in candidate_paths:
        candidate_set = set(candidate)
        correct_count = len(candidate_set & taken_set)
        correctness = correct_count / max(len(candidate), 1)
        lcs_len = _longest_common_subsequence(steps_taken, candidate)
        sequence_score = lcs_len / max(len(candidate), 1)
        if correctness + sequence_score > best_correctness + best_sequence:
            best_correctness = correctness
            best_sequence = sequence_score
            best_path_len = len(candidate)

    # 3. Efficiency: penalise extra steps
    extra_steps = max(0, len(steps_taken) - best_path_len)
    efficiency = max(0.0, 1.0 - (extra_steps / max(max_steps, 1)))

    composite = 0.4 * best_correctness + 0.3 * best_sequence + 0.3 * efficiency

    # Investigation coverage bonus for tasks with partial observability.
    investigation_bonus = 0.0
    required_investigation_actions = required_investigation_actions or []
    if required_investigation_actions:
        covered = len(set(required_investigation_actions) & taken_set)
        coverage = covered / len(required_investigation_actions)
        investigation_bonus = 0.08 * coverage
        composite += investigation_bonus

    # SLA breach penalty
    sla_penalty = 0.0
    if len(steps_taken) > sla_steps:
        sla_penalty = 0.15

    # Unnecessary escalation penalty
    escalation_penalty = 0.0
    if "escalate_to_human" in taken_set and "escalate_to_human" not in goal_set:
        escalation_penalty = 0.1

    score = max(0.0, composite - sla_penalty - escalation_penalty)
    score = min(1.0, round(score, 4))

    feedback_parts = [
        f"Correctness: {best_correctness:.2f}",
        f"Sequence: {best_sequence:.2f}",
        f"Efficiency: {efficiency:.2f}",
    ]
    if required_investigation_actions:
        feedback_parts.append(f"Investigation bonus: +{investigation_bonus:.3f}")
    if sla_penalty:
        feedback_parts.append(f"SLA breach penalty: -{sla_penalty}")
    if escalation_penalty:
        feedback_parts.append(f"Unnecessary escalation penalty: -{escalation_penalty}")

    return score, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Unified grader dispatcher
# ---------------------------------------------------------------------------

def grade(
    difficulty: str,
    steps_taken: List[str],
    goal_actions: List[str],
    max_steps: int = 8,
    sla_steps: int = 6,
    valid_paths: Optional[List[List[str]]] = None,
    required_investigation_actions: Optional[List[str]] = None,
) -> Tuple[float, str]:
    """Route to the correct grader based on task difficulty."""
    if difficulty == "easy":
        score, feedback = grade_easy(steps_taken, goal_actions)
    elif difficulty == "medium":
        score, feedback = grade_medium(steps_taken, goal_actions)
    elif difficulty == "hard":
        score, feedback = grade_hard(
            steps_taken,
            goal_actions,
            max_steps,
            sla_steps,
            valid_paths=valid_paths,
            required_investigation_actions=required_investigation_actions,
        )
    else:
        raise ValueError(f"Unknown difficulty: {difficulty!r}")

    return round(_strict_score(score), 4), feedback
