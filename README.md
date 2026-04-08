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

# AI Customer Support Resolution Environment

An OpenEnv-compatible reinforcement learning environment for enterprise-style
customer support workflows. The benchmark evaluates whether an agent can make
correct, safe, and efficient decisions across classification, investigation,
resolution, escalation, and closure.

## Problem Statement

Real-world support is not a single-step classification task. Teams must resolve
issues under uncertainty, SLA pressure, and policy constraints. Agents need to:

- identify the right issue type
- gather missing evidence before acting
- choose between automated resolution and escalation
- avoid premature closure and low-quality resolution paths

This environment models those requirements in a deterministic and reproducible
form suitable for both RL and LLM agent evaluation.

## Submission Highlights

- OpenEnv-style API with typed request/response models
- Deterministic task grading and bounded rewards
- Difficulty progression from basic classification to multi-signal hard tasks
- Partial observability and hidden-context reveal mechanics
- Dockerized deployment for local and Hugging Face Spaces runtime

## Live Deployment

- Endpoint: https://mounishmou-myenv.hf.space
- Health: `/health`
- Interactive docs: `/docs`

## Architecture

- API server: FastAPI app in `app/main.py`
- Environment core: episode state and transitions in `app/env.py`
- Reward shaping and transition rules: `app/utils.py`
- Task definitions and ticket corpus: `app/tasks.py`
- Deterministic grading logic: `app/grader.py`
- Typed schemas: `app/models.py`
- OpenEnv spec: `openenv.yaml`

## Environment Contract

### Observation fields

- `ticket_id`
- `customer_message`
- `customer_history`
- `sentiment`
- `current_status`
- `steps_taken`
- `available_actions`
- `sla_steps_remaining`
- `info_message`
- `known_facts`
- `hidden_context_revealed`
- `risk_flags`
- `investigation_steps_used`
- `decision_trace`

### Action space

Classification:

- `classify_billing`
- `classify_technical`
- `classify_refund`
- `classify_general`

Investigation:

- `request_more_info`
- `fetch_customer_history`
- `check_service_status`
- `verify_billing_ledger`

Resolution:

- `issue_refund`
- `apply_billing_credit`
- `restart_service`
- `send_technical_guide`
- `reset_password`

Escalation and closure:

- `escalate_to_human`
- `close_ticket`

### Step response

`POST /step` returns:

- `observation`
- `reward`
- `done`
- `info`
- `score`
- `cumulative_reward`

## Task Suite

Easy:

- `task_1_classify`
- `task_1b_classify_technical`
- `task_1c_classify_refund`

Medium:

- `task_2_resolution_path`
- `task_2b_technical_path`

Hard:

- `task_3_multistep_angry_refund`
- `task_3b_multistep_technical_enterprise`
- `task_3c_ambiguous_ticket`
- `task_4_enterprise_rca_tradeoff`
- `task_4b_billing_compliance_tradeoff`
- `task_4c_ambiguous_multisignal_resolution`

## Quick Start

### Local setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### API smoke test

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_1_classify"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"classify_billing"}'
curl http://localhost:7860/state
curl -X POST http://localhost:7860/state
```

### Baseline inference runner

```bash
OPENAI_API_KEY=<your_key_or_router_token> \
HF_TOKEN=<optional_hf_token> \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
ENV_BASE_URL=http://localhost:7860 \
python inference.py
```

Expected evaluator log tags:

- `[START]`
- `[STEP]`
- `[END]`

## Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

## Hugging Face Spaces Deployment

1. Create a Docker Space.
2. Push this repository to the Space remote.
3. Wait for successful build.
4. Validate `/health`, `/tasks`, `/reset`, and `/step`.

## Validation Assets

- `OpenEnv-Hackathon.postman_collection.json`
- `OpenEnv-Phase2-Validation.postman_collection.json`

Use these collections for repeatable endpoint and flow validation.

## Project Directory

```text
.
|-- Dockerfile
|-- README.md
|-- inference.py
|-- openenv.yaml
|-- requirements.txt
|-- OpenEnv-Hackathon.postman_collection.json
|-- OpenEnv-Phase2-Validation.postman_collection.json
`-- app/
    |-- env.py
    |-- grader.py
    |-- main.py
    |-- models.py
    |-- tasks.py
    `-- utils.py
```

## Compliance Notes

- Deterministic scoring and reproducible episode behavior
- Typed models for request/response consistency
- OpenEnv-compatible endpoints and schema
- Production-ready Docker packaging for deployment
