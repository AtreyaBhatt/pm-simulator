"""
PM Simulator — FastAPI server exposing the OpenEnv HTTP interface.

Endpoints:
  POST /reset          — start a new episode
  POST /step           — take one action
  GET  /state          — current environment state
  GET  /tasks          — list available tasks
  GET  /health         — liveness probe
  GET  /               — environment info
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import PMEnvironment
from env.models import Action, EnvironmentState, Observation, StepResult, TaskInfo

app = FastAPI(
    title="PM Simulator",
    description=(
        "An OpenEnv-compliant environment where an AI agent plays the role of a "
        "Product Manager: triaging bugs, planning sprints, and building quarterly roadmaps "
        "for a fictional B2B SaaS company (Nexus Analytics)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = PMEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "task1_bug_triage"


class HealthResponse(BaseModel):
    status: str
    version: str
    available_tasks: list[str]


class EnvInfoResponse(BaseModel):
    name: str
    description: str
    version: str
    tasks: list[str]
    action_space: dict
    observation_fields: list[str]
    openenv_compliant: bool


@app.get("/", response_model=EnvInfoResponse)
def env_info():
    return EnvInfoResponse(
        name="PM Simulator",
        description=(
            "A Product Manager simulation environment. The agent triages bugs, "
            "plans sprints, and builds quarterly roadmaps under real-world constraints."
        ),
        version="1.0.0",
        tasks=["task1_bug_triage", "task2_sprint_planning", "task3_quarterly_roadmap"],
        action_space={
            "action_type": ["prioritize", "schedule", "defer", "request_info", "submit"],
            "item_id": "string",
            "priority": ["P0", "P1", "P2", "P3"],
            "sprint": "integer 1-3",
            "reason": "string",
            "thought": "string (chain-of-thought, not graded)",
        },
        observation_fields=[
            "task_id", "task_name", "task_description", "items",
            "sprint_capacity", "current_sprint_load", "metrics",
            "step_number", "max_steps", "context", "message", "available_actions",
        ],
        openenv_compliant=True,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_tasks=["task1_bug_triage", "task2_sprint_planning", "task3_quarterly_roadmap"],
    )


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    try:
        return _env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    return _env.step(action)


@app.get("/state", response_model=EnvironmentState)
def state():
    return _env.state()


@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks():
    return _env.tasks()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
