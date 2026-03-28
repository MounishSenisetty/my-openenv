"""
FastAPI application — exposes the CustomerSupportEnv as an HTTP API.

Endpoints
---------
POST /reset          → ResetResponse
POST /step           → StepResponse
GET  /state          → StateResponse
GET  /tasks          → list of available tasks
GET  /health         → liveness probe
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.env import CustomerSupportEnv
from app.models import Action, StepResponse, StateResponse, ResetResponse
from app.tasks import TASKS

# One shared environment instance (single-session server).
# For multi-user scenarios replace with session-keyed dict.
env = CustomerSupportEnv()

app = FastAPI(
    title="AI Customer Support Resolution Environment",
    description=(
        "An OpenEnv-compatible reinforcement-learning environment that simulates "
        "real-world customer support workflows."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with metadata."""
    return {"tasks": TASKS}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    """Start a new episode for the given task_id."""
    try:
        response = env.reset(task_id=req.task_id)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """Execute one agent action."""
    try:
        response = env.step(action)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state():
    """Read current episode state without side effects."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
