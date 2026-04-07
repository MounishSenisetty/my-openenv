"""Baseline inference runner with strict OpenEnv evaluation log format."""

import json
import os
import time
from typing import List, Optional

import requests
from openai import OpenAI


API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
BENCHMARK = "ai-customer-support-resolution"
DEFAULT_TASK_IDS = [
    "task_1_classify",
    "task_2b_technical_path",
    "task_3c_ambiguous_ticket",
]
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 128


SYSTEM_PROMPT = (
    "You are an AI customer support agent. Return exactly one action_type string "
    "from available_actions and nothing else."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def env_get_tasks() -> List[dict]:
    r = requests.get(f"{ENV_BASE_URL}/tasks", timeout=30)
    r.raise_for_status()
    return r.json()["tasks"]


def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_type: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action_type": action_type, "reasoning": "baseline inference"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def build_user_prompt(observation: dict, task_description: str) -> str:
    return (
        f"Task: {task_description}\n"
        f"Ticket: {observation['customer_message']}\n"
        f"Sentiment: {observation['sentiment']}\n"
        f"Status: {observation['current_status']}\n"
        f"Steps taken: {observation['steps_taken']}\n"
        f"SLA remaining: {observation['sla_steps_remaining']}\n"
        f"Available actions: {observation['available_actions']}\n"
        "Return only the next action_type token."
    )


def choose_action(client: OpenAI, observation: dict, task_description: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation, task_description)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (response.choices[0].message.content or "").strip()
    cleaned = text.strip().strip('"\'').lower().replace(" ", "_")
    return cleaned


def choose_fallback_action(observation: dict) -> str:
    available_actions = observation.get("available_actions", []) or []
    if available_actions:
        return str(available_actions[0])
    return "close_ticket"


def run_task(client: OpenAI, task: dict) -> dict:
    task_id = task["task_id"]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    try:
        reset_data = env_reset(task_id)
        observation = reset_data["observation"]
        task_description = reset_data["task_description"]

        for step_num in range(1, MAX_STEPS + 1):
            action_error: Optional[str] = None
            try:
                action = choose_action(client, observation, task_description)
            except Exception as exc:
                action = choose_fallback_action(observation)
                action_error = str(exc)

            if action not in observation.get("available_actions", []):
                action = choose_fallback_action(observation)
                if not action_error:
                    action_error = "model_action_not_in_available_actions"

            try:
                step_data = env_step(action)
                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))
                observation = step_data["observation"]
                score = float(step_data.get("score", 0.0))
                last_error = action_error
            except Exception as exc:
                reward = 0.0
                done = True
                last_error = action_error or str(exc)

            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=action, reward=reward, done=done, error=last_error)

            if done:
                break

            time.sleep(0.2)

        success = score >= 0.8

    except Exception as exc:
        steps_taken = max(steps_taken, 1)
        rewards.append(0.0)
        log_step(step=steps_taken, action="close_ticket", reward=0.0, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
    }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-api-key")

    try:
        tasks = env_get_tasks()
    except Exception:
        tasks = [{"task_id": task_id} for task_id in DEFAULT_TASK_IDS]

    results: List[dict] = []
    for task in tasks:
        try:
            results.append(run_task(client, task))
        except Exception:
            results.append(
                {
                    "task_id": task.get("task_id", "unknown_task"),
                    "score": 0.0,
                    "steps": 0,
                    "success": False,
                }
            )

    try:
        with open("inference_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
