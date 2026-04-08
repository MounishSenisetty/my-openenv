---
title: Myenv
emoji: "🐠"
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
short_description: "TinyCoders"
tags:
  - openenv
  - rl-environment
---

# 🎫 AI Customer Support Resolution Environment

An **OpenEnv-compatible reinforcement-learning environment** that simulates a
real-world AI customer support workflow. An AI agent must read support tickets
and resolve them step-by-step using a structured action space — earning dense
rewards for correct, efficient resolution.

---

## 📌 Problem Motivation

Customer support is one of the largest operational costs for software companies.
AI agents capable of autonomously classifying, prioritising, and resolving
support tickets can dramatically reduce response times and human workload.

This environment provides a controlled, reproducible training and evaluation
ground for such agents — covering billing disputes, technical issues, refund
requests, and ambiguous edge cases.

---

## 🌍 Real-World Relevance

| Scenario | Coverage |
|---|---|
| Duplicate billing charge | ✅ |
| Plan upgrade confusion | ✅ |
| App crash (iOS) | ✅ |
| Password / login failure | ✅ |
| Enterprise API outage | ✅ |
| Annual subscription refund | ✅ |
| Angry customer threatening legal action | ✅ |
| Ambiguous / mixed-category ticket | ✅ |

---

## 🏗️ System Design

```
┌──────────────────────────────────┐
│          FastAPI Server          │
│  POST /reset  POST /step GET /state│
└──────────────┬───────────────────┘
               │
       ┌───────▼────────┐
       │ CustomerSupportEnv │
       │  (env.py)       │
       └──┬──────────┬───┘
          │          │
   ┌──────▼──┐  ┌────▼─────┐
   │ Grader  │  │  Utils   │
   │(grader.py)│ │(utils.py)│
   └─────────┘  └──────────┘
          │
   ┌──────▼──────┐
   │ Tasks/Data  │
   │ (tasks.py)  │
   └─────────────┘
```

---

## 📡 Observation Space

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `customer_message` | string | Full customer complaint text |
| `customer_history` | string[] | Prior interactions on this account |
| `sentiment` | enum | `positive / neutral / negative / angry` |
| `current_status` | enum | `open / in_progress / escalated / resolved / closed` |
| `steps_taken` | string[] | Actions already executed this episode |
| `available_actions` | string[] | Currently valid actions |
| `sla_steps_remaining` | int | Steps left before SLA breach penalty |
| `info_message` | string? | Contextual feedback from last action |

---

## ⚡ Action Space

| Action | Description |
|---|---|
| `classify_billing` | Ticket is about billing / payment |
| `classify_technical` | Ticket is about a product / tech issue |
| `classify_refund` | Ticket is a refund request |
| `classify_general` | Ambiguous or general enquiry |
| `request_more_info` | Ask customer for more details |
| `issue_refund` | Process a monetary refund |
| `apply_billing_credit` | Apply credit to the account |
| `restart_service` | Restart / reset the customer's service |
| `send_technical_guide` | Send troubleshooting documentation |
| `reset_password` | Trigger a password reset |
| `escalate_to_human` | Hand off to a human agent |
| `close_ticket` | Mark resolved and close |

---

## 📋 Tasks

### Task 1 — Easy: Classify the Ticket
Choose the correct classification action as the first step.
- `task_1_classify` — billing
- `task_1b_classify_technical` — technical
- `task_1c_classify_refund` — refund

Grading: **binary** — correct first action = 1.0, correct but not first = 0.5.

---

### Task 2 — Medium: Correct Resolution Path
Execute the correct classify → resolve → close sequence.
- `task_2_resolution_path` — billing dispute → refund
- `task_2b_technical_path` — login issue → password reset

Grading: **prefix matching** — score = matched steps / total goal steps.

---

### Task 3 — Hard: Multi-Step Resolution
Handle complex, high-stakes tickets in the fewest steps possible.
- `task_3_multistep_angry_refund` — angry customer, prolonged outage
- `task_3b_multistep_technical_enterprise` — enterprise API outage
- `task_3c_ambiguous_ticket` — mixed-category, requires info gathering

Grading: **weighted composite**
  - 40% correctness
  - 30% sequence quality (LCS)
  - 30% efficiency (extra steps penalised)
  - −0.15 SLA breach penalty
  - −0.10 unnecessary escalation

---

## 💰 Reward Function

| Event | Reward |
|---|---|
| Correct step (right position) | +0.20 |
| Action in goal but out of order | +0.05 |
| Incorrect action | −0.10 |
| Resolution-moving action | +0.30 |
| Repeated / useless action | −0.20 |
| Unnecessary escalation | −0.30 |
| Correct final resolution (score ≥ 0.8) | +1.00 |
| Angry customer modifier (per step) | −0.10 |
| Negative customer modifier | −0.05 |
| Positive customer modifier | +0.05 |

---

## 🚀 Setup & Running

### Local

```bash
# 1. Clone / enter project
cd customer_support_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 4. Run baseline inference (requires OpenAI key or HF endpoint)
OPENAI_API_KEY=<your_key_or_router_token> \
HF_TOKEN=<optional_hf_token> \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
ENV_BASE_URL=http://localhost:7860 \
python inference.py
```

Inference logs are emitted in strict evaluator format:
`[START]`, `[STEP]`, `[END]`.

### Docker

```bash
# Build
docker build -t customer-support-env .

# Run
docker run -p 7860:7860 customer-support-env

# Verify health
curl http://localhost:7860/health
```

### Hugging Face Spaces

1. Create a new **Docker** Space on Hugging Face.
2. Push this repository to the Space.
3. The Space will auto-build and expose port 7860.
4. Set `ENV_BASE_URL=https://your-space.hf.space` in inference script.

---

## 🔌 API Quick Reference

```bash
# List tasks
curl http://localhost:7860/tasks

# Start episode
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_1_classify"}'

# Take action
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "classify_billing"}'

# Check state
curl http://localhost:7860/state
```

---

## 📊 Baseline Results (GPT-4o-mini)

| Task | Difficulty | Score |
|---|---|---|
| task_1_classify | easy | 1.00 |
| task_1b_classify_technical | easy | 1.00 |
| task_1c_classify_refund | easy | 1.00 |
| task_2_resolution_path | medium | 0.92 |
| task_2b_technical_path | medium | 0.95 |
| task_3_multistep_angry_refund | hard | 0.74 |
| task_3b_multistep_technical_enterprise | hard | 0.70 |
| task_3c_ambiguous_ticket | hard | 0.78 |
| **Average** | | **0.89** |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v   # if tests/ directory added
```

---

## 📁 Project Structure

```
customer_support_env/
├── openenv.yaml          # OpenEnv specification
├── inference.py          # Baseline LLM agent
├── Dockerfile            # Container definition
├── README.md             # This file
├── requirements.txt
└── app/
    ├── __init__.py
    ├── main.py           # FastAPI entry point
    ├── env.py            # Core environment logic
    ├── models.py         # Pydantic schemas
    ├── tasks.py          # Ticket dataset & task definitions
    ├── grader.py         # Deterministic graders
    └── utils.py          # Reward calculator & helpers
```

---

## ⚙️ Resource Requirements

- CPU: 2 vCPU  
- RAM: 2 GB (well within 8 GB limit)  
- GPU: Not required  
- Inference time: < 5 minutes for all 8 tasks

---

## 🧭 Why This Benchmark Matters

Customer support automation is not a toy problem. Production agents must handle
ambiguity, policy constraints, and high-stakes trade-offs under partial
information. This benchmark is designed to evaluate exactly that.

### Real-world value

- Models enterprise support realities: SLA pressure, escalation decisions,
  compliance-sensitive billing outcomes, and customer sentiment effects.
- Rewards evidence-first triage instead of brittle one-shot action templates.
- Penalizes exploitative policies such as blind escalation loops and premature
  closure.

### Novel benchmark mechanics

- **Partial observability**: hidden context is revealed only through
  investigation actions (`fetch_customer_history`, `check_service_status`,
  `verify_billing_ledger`).
- **Multi-path hard tasks**: advanced tasks allow multiple valid solution
  trajectories, reducing overfitting to a single scripted path.
- **Delayed consequences**: early shortcuts can hurt terminal score via
  deterministic anti-exploit penalties.
- **RL-ready shaping**: dense per-step reward with deterministic terminal
  grading, suitable for both online and offline policy evaluation.

### Research and evaluation fit

- Deterministic, reproducible scoring in strict `(0, 1)` range for fair model
  comparison.
- OpenEnv-compliant API and typed schemas enable drop-in use with standard
  evaluators.
- Designed to stress-test frontier LLM agents on planning under uncertainty,
  not just action memorization.
# Build triggered at 2026-03-29 00:18:34
