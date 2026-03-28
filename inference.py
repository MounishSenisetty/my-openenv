"""
inference.py — Baseline agent for the AI Customer Support Resolution Environment.

Uses an LLM (via OpenAI-compatible API) to act as the support agent.
Runs all tasks and reports per-task and aggregate scores.

Environment variables
---------------------
API_BASE_URL  : Base URL for the OpenAI-compatible endpoint
                Default: http://localhost:8000   (local FastAPI server)
MODEL_NAME    : Model identifier passed to the LLM
                Default: gpt-4o-mini
HF_TOKEN      : Bearer token for Hugging Face Inference Endpoints (optional)

Usage
-----
  python inference.py
  API_BASE_URL=https://my-hf-space.hf.space MODEL_NAME=mistral python inference.py
"""

import os
import json
import time
import requests
from typing import Optional

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

ENV_BASE_URL  = os.getenv("ENV_BASE_URL", "http://localhost:8000")   # FastAPI env server
LLM_BASE_URL  = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN      = os.getenv("HF_TOKEN", "")

HEADERS = {"Content-Type": "application/json"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

MAX_RETRIES = 3
STEP_DELAY  = 0.5   # seconds between steps (rate-limit courtesy)


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM client
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return the text response."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    llm_headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        llm_headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=llm_headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  [LLM retry {attempt+1}] {e}")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: Optional[str] = None) -> dict:
    payload = {"task_id": task_id} if task_id else {}
    r = requests.post(f"{ENV_BASE_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_type: str, reasoning: str = "") -> dict:
    payload = {"action_type": action_type, "reasoning": reasoning}
    r = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_get_tasks() -> list:
    r = requests.get(f"{ENV_BASE_URL}/tasks", timeout=15)
    r.raise_for_status()
    return r.json()["tasks"]


# ---------------------------------------------------------------------------
# System prompt for the agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI customer support agent operating inside an automated support system.

Your job: read the customer's ticket and choose the single best action from the
AVAILABLE ACTIONS list to progress toward resolving the issue.

IMPORTANT RULES:
1. Reply with ONLY the action name (e.g. classify_billing). No explanation, no quotes.
2. Choose from the available_actions list provided.
3. Follow this general strategy:
   - First, classify the ticket (classify_billing / classify_technical / classify_refund / classify_general)
   - Then, take the most appropriate resolution action
   - If the issue is complex or the customer is very angry and escalation is warranted, escalate
   - Finally, close_ticket when resolved

Action reference:
  classify_billing       → ticket is about billing/payments
  classify_technical     → ticket is about technical/product issue
  classify_refund        → ticket is about requesting a refund
  classify_general       → ticket type is ambiguous or general
  request_more_info      → need more details from the customer
  issue_refund           → process a monetary refund
  apply_billing_credit   → apply credit to the account
  restart_service        → restart or reset the customer's service
  send_technical_guide   → send troubleshooting documentation
  reset_password         → trigger a password reset
  escalate_to_human      → hand off to a human agent (use sparingly)
  close_ticket           → mark the ticket as resolved and close it
""".strip()


def build_user_prompt(obs: dict, task_description: str) -> str:
    return f"""
TASK: {task_description}

TICKET ID: {obs['ticket_id']}
CUSTOMER MESSAGE: {obs['customer_message']}
CUSTOMER HISTORY: {json.dumps(obs['customer_history'], indent=2)}
SENTIMENT: {obs['sentiment']}
STATUS: {obs['current_status']}
STEPS TAKEN: {obs['steps_taken']}
SLA STEPS REMAINING: {obs['sla_steps_remaining']}
AVAILABLE ACTIONS: {obs['available_actions']}
LAST FEEDBACK: {obs.get('info_message', 'None')}

Choose your next action:
""".strip()


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(task: dict) -> dict:
    task_id = task["task_id"]
    print(f"\n{'='*60}")
    print(f"Task: {task_id} | Difficulty: {task['difficulty']}")
    print(f"Description: {task['description'][:80]}...")

    # Reset environment
    reset_resp = env_reset(task_id)
    obs = reset_resp["observation"]
    task_description = reset_resp["task_description"]

    episode_done = False
    final_score = 0.0
    steps = 0

    while not episode_done:
        # Build prompt and call LLM
        user_prompt = build_user_prompt(obs, task_description)
        action_type = call_llm(SYSTEM_PROMPT, user_prompt)

        # Clean up model output (strip whitespace, quotes)
        action_type = action_type.strip().strip('"\'').lower().replace(" ", "_")
        print(f"  Step {steps+1}: {action_type}")

        # Execute action
        step_resp = env_step(action_type)
        obs = step_resp["observation"]
        reward = step_resp["reward"]

        print(f"    reward={reward['step_reward']:+.3f}  cumulative={reward['cumulative_reward']:+.3f}")

        episode_done = reward["done"]
        final_score = reward["score"]
        steps += 1

        time.sleep(STEP_DELAY)

    print(f"  FINAL SCORE: {final_score:.4f}  |  Steps: {steps}")
    return {
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "final_score": final_score,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AI Customer Support Resolution Environment — Inference")
    print(f"LLM endpoint : {LLM_BASE_URL}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Env server   : {ENV_BASE_URL}")
    print("=" * 60)

    tasks = env_get_tasks()
    print(f"Found {len(tasks)} tasks.\n")

    results = []
    for task in tasks:
        try:
            result = run_task(task)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on task {task['task_id']}: {e}")
            results.append({
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "final_score": 0.0,
                "steps": -1,
                "error": str(e),
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        status = f"score={r['final_score']:.4f}  steps={r['steps']}"
        if "error" in r:
            status = f"ERROR: {r['error']}"
        print(f"  {r['task_id']:<40} [{r['difficulty']:<6}]  {status}")

    scores = [r["final_score"] for r in results if "error" not in r]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average score across {len(scores)} tasks: {avg:.4f}")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to inference_results.json")


if __name__ == "__main__":
    main()
