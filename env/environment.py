"""
PM Simulator — Core Environment

Implements the OpenEnv interface:
  reset(task_id) → Observation
  step(action)   → StepResult
  state()        → EnvironmentState
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from env.models import (
    Action,
    CustomerTier,
    EnvironmentState,
    ItemType,
    Observation,
    Priority,
    ProductItem,
    Reward,
    Severity,
    StepResult,
    TaskInfo,
    TeamMetrics,
)
from env.tasks import task1_grader, task2_grader, task3_grader

DATA_DIR = Path(__file__).parent / "data"

TASK_DATA_FILES = {
    "task1_bug_triage": DATA_DIR / "task1_data.json",
    "task2_sprint_planning": DATA_DIR / "task2_data.json",
    "task3_quarterly_roadmap": DATA_DIR / "task3_data.json",
}

TASK_GRADERS = {
    "task1_bug_triage": task1_grader,
    "task2_sprint_planning": task2_grader,
    "task3_quarterly_roadmap": task3_grader,
}

DEFAULT_METRICS = TeamMetrics(
    nps=32.0,
    churn_rate_pct=3.8,
    open_bug_count=47,
    p0_bug_count=3,
    avg_resolution_days=8.2,
    sprint_velocity=34,
    enterprise_at_risk_count=4,
)


def _parse_items(raw_items: list[dict]) -> list[ProductItem]:
    """Parse raw JSON dicts into ProductItem objects."""
    items = []
    for raw in raw_items:
        raw = dict(raw)
        raw["type"] = ItemType(raw["type"])
        raw["severity"] = Severity(raw["severity"])
        raw["customer_tier"] = CustomerTier(raw["customer_tier"])
        if raw.get("assigned_priority"):
            raw["assigned_priority"] = Priority(raw["assigned_priority"])
        else:
            raw["assigned_priority"] = None
        raw.setdefault("assigned_sprint", None)
        raw.setdefault("deferred", False)
        raw.setdefault("defer_reason", None)
        items.append(ProductItem(**raw))
    return items


class PMEnvironment:
    """
    PM Simulator environment.

    Usage:
        env = PMEnvironment()
        obs = env.reset("task1_bug_triage")
        result = env.step(Action(action_type="prioritize", item_id="BUG-001", priority=Priority.P0))
        result = env.step(Action(action_type="submit"))
        print(result.reward.value, result.info["final_score"])
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._task_data: dict[str, Any] = {}
        self._items: list[ProductItem] = []
        self._step: int = 0
        self._max_steps: int = 20
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._actions_taken: list[dict] = []
        self._sprint_loads: dict[int, int] = {1: 0, 2: 0, 3: 0}
        self._scheduled_by_sprint: dict[int, list[str]] = {1: [], 2: [], 3: []}
        self._sprint_capacity: int = 30
        self._final_score: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1_bug_triage") -> Observation:
        """Start a new episode for the given task."""
        if task_id not in TASK_DATA_FILES:
            raise ValueError(
                f"Unknown task '{task_id}'. Available: {list(TASK_DATA_FILES.keys())}"
            )

        data = json.loads(TASK_DATA_FILES[task_id].read_text())
        self._task_id = task_id
        self._task_data = data
        self._items = _parse_items(data["items"])
        self._step = 0
        self._max_steps = data["max_steps"]
        self._cumulative_reward = 0.0
        self._done = False
        self._actions_taken = []
        self._sprint_capacity = data["sprint_capacity"]
        num_sprints = data.get("num_sprints", 1)
        self._sprint_loads = {sp: 0 for sp in range(1, num_sprints + 1)}
        self._scheduled_by_sprint = {sp: [] for sp in range(1, num_sprints + 1)}
        self._final_score = None

        return self._build_observation("Welcome! Review the items and start taking actions.")

    def step(self, action: Action) -> StepResult:
        """Process one action and return the result."""
        if self._done:
            return StepResult(
                observation=self._build_observation("Episode is done. Call reset() to start a new episode."),
                reward=Reward(value=0.0, cumulative=self._cumulative_reward, message="Done."),
                done=True,
                info={"final_score": self._final_score},
            )

        self._step += 1
        reward_value, message, info = self._process_action(action)

        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._actions_taken.append(
            {
                "step": self._step,
                "action": action.model_dump(),
                "reward": reward_value,
                "message": message,
            }
        )

        # Check termination
        done = info.get("submitted", False) or self._step >= self._max_steps
        self._done = done

        if done and not info.get("submitted"):
            # Max steps reached without submit — auto-grade
            final_score, grade_info = self._run_grader()
            self._final_score = final_score
            info["final_score"] = final_score
            info["grade_breakdown"] = grade_info
            info["reason"] = "max_steps_reached"
            message = f"⏰ Max steps reached. Auto-grading... Final score: {final_score:.3f}"

        reward = Reward(
            value=round(reward_value, 4),
            breakdown=info.get("reward_breakdown", {}),
            cumulative=self._cumulative_reward,
            message=message,
        )
        return StepResult(
            observation=self._build_observation(message),
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return current serialisable environment state."""
        return EnvironmentState(
            task_id=self._task_id or "",
            task_name=self._task_data.get("task_name", ""),
            step=self._step,
            max_steps=self._max_steps,
            items=deepcopy(self._items),
            sprint_capacity=self._sprint_capacity,
            current_sprint_load=self._sprint_loads.get(1, 0),
            actions_taken=list(self._actions_taken),
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            final_score=self._final_score,
        )

    def tasks(self) -> list[TaskInfo]:
        """List all available tasks with metadata."""
        result = []
        for task_id, fpath in TASK_DATA_FILES.items():
            data = json.loads(fpath.read_text())
            result.append(
                TaskInfo(
                    task_id=task_id,
                    name=data["task_name"],
                    description=data["task_description"],
                    difficulty=data["difficulty"],
                    max_steps=data["max_steps"],
                    sprint_capacity=data["sprint_capacity"],
                    item_count=len(data["items"]),
                    grader_description=TASK_GRADERS[task_id].__doc__.strip().split("\n")[0],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_item(self, item_id: str) -> Optional[ProductItem]:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def _build_observation(self, message: str = "") -> Observation:
        sprint_load = self._sprint_loads.get(1, 0)
        context = dict(self._task_data.get("context", {}))
        context["sprint_loads"] = dict(self._sprint_loads)
        context["actions_taken"] = len(self._actions_taken)

        return Observation(
            task_id=self._task_id or "",
            task_name=self._task_data.get("task_name", ""),
            task_description=self._task_data.get("task_description", ""),
            items=deepcopy(self._items),
            sprint_capacity=self._sprint_capacity,
            current_sprint_load=sprint_load,
            metrics=DEFAULT_METRICS,
            step_number=self._step,
            max_steps=self._max_steps,
            context=context,
            message=message,
            available_actions=["prioritize", "schedule", "defer", "request_info", "submit"],
        )

    def _process_action(self, action: Action) -> tuple[float, str, dict]:
        """
        Dispatch action to handler.
        Returns (reward_value, message, info_dict).
        """
        handlers = {
            "prioritize": self._handle_prioritize,
            "schedule": self._handle_schedule,
            "defer": self._handle_defer,
            "request_info": self._handle_request_info,
            "submit": self._handle_submit,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return -0.05, f"Unknown action type '{action.action_type}'", {}
        return handler(action)

    def _handle_prioritize(self, action: Action) -> tuple[float, str, dict]:
        item = self._get_item(action.item_id or "")
        if item is None:
            return -0.05, f"Item '{action.item_id}' not found.", {}
        if action.priority is None:
            return -0.05, "Priority must be specified for 'prioritize' action.", {}

        old_priority = item.assigned_priority
        item.assigned_priority = action.priority

        # Task-specific grading
        if self._task_id == "task1_bug_triage":
            gt = self._task_data.get("ground_truth_priorities", {})
            reward, msg = task1_grader.score_priority_assignment(
                item, action.priority.value, gt.get(item.id, "P3")
            )
            return reward, msg, {"reward_breakdown": {"priority_accuracy": reward}}

        # For other tasks, small positive reward for taking any action
        msg = f"Assigned {action.priority.value} to {item.id}"
        if old_priority and old_priority != action.priority:
            msg += f" (was {old_priority.value})"
        return 0.02, msg, {}

    def _handle_schedule(self, action: Action) -> tuple[float, str, dict]:
        item = self._get_item(action.item_id or "")
        if item is None:
            return -0.05, f"Item '{action.item_id}' not found.", {}
        sprint = action.sprint or 1
        if sprint not in self._sprint_loads:
            return -0.05, f"Invalid sprint number {sprint}. Valid: {list(self._sprint_loads.keys())}", {}

        if self._task_id == "task2_sprint_planning":
            reward, msg = task2_grader.score_schedule_action(
                item, sprint, self._sprint_loads[sprint], set(self._scheduled_by_sprint.get(sprint, []))
            )
        elif self._task_id == "task3_quarterly_roadmap":
            reward, msg = task3_grader.score_schedule_action(
                item, sprint, self._sprint_loads, self._scheduled_by_sprint
            )
        else:
            reward, msg = 0.02, f"Scheduled {item.id} to sprint {sprint}"

        if reward >= 0:
            # Only commit if valid
            item.assigned_sprint = sprint
            self._sprint_loads[sprint] = self._sprint_loads.get(sprint, 0) + item.effort_points
            if sprint not in self._scheduled_by_sprint:
                self._scheduled_by_sprint[sprint] = []
            self._scheduled_by_sprint[sprint].append(item.id)

        return reward, msg, {"reward_breakdown": {"schedule": reward}}

    def _handle_defer(self, action: Action) -> tuple[float, str, dict]:
        item = self._get_item(action.item_id or "")
        if item is None:
            return -0.05, f"Item '{action.item_id}' not found.", {}

        if self._task_id in ("task2_sprint_planning",):
            reward, msg = task2_grader.score_defer_action(item)
        elif self._task_id == "task3_quarterly_roadmap":
            reward, msg = task3_grader.score_defer_action(item)
        else:
            reward, msg = 0.01, f"Deferred {item.id}"

        item.deferred = True
        item.defer_reason = action.reason
        return reward, msg, {}

    def _handle_request_info(self, action: Action) -> tuple[float, str, dict]:
        """Costs a step but may return additional context."""
        item = self._get_item(action.item_id or "")
        if item is None:
            return -0.02, "No such item. Info request failed.", {}

        # Return richer context for the item
        extra = {
            "item_id": item.id,
            "stakeholder_votes_detail": item.stakeholder_votes,
            "revenue_impact_usd": item.revenue_impact_usd,
            "age_days": item.age_days,
            "dependencies": item.dependencies,
            "tags": item.tags,
            "hint": "Consider: severity, user_impact, revenue_impact, and age together.",
        }
        return -0.02, f"ℹ️ Additional context for {item.id} returned.", {"extra_context": extra}

    def _handle_submit(self, action: Action) -> tuple[float, str, dict]:
        """Run the grader and finalize the episode."""
        final_score, breakdown = self._run_grader()
        self._final_score = final_score

        bonus = final_score * 0.5  # Bonus reward proportional to quality
        msg = (
            f"📋 Submission graded. Final score: {final_score:.3f}/1.000. "
            f"Episode reward bonus: +{bonus:.3f}"
        )
        return round(bonus, 4), msg, {
            "submitted": True,
            "final_score": final_score,
            "grade_breakdown": breakdown,
        }

    def _run_grader(self) -> tuple[float, dict]:
        grader = TASK_GRADERS.get(self._task_id)
        if grader is None:
            return 0.0, {"error": "No grader for this task"}

        if self._task_id == "task1_bug_triage":
            gt = self._task_data.get("ground_truth_priorities", {})
            return task1_grader.compute_final_score(self._items, gt)
        elif self._task_id == "task2_sprint_planning":
            return task2_grader.compute_final_score(self._items, self._sprint_capacity)
        elif self._task_id == "task3_quarterly_roadmap":
            return task3_grader.compute_final_score(
                self._items,
                self._sprint_capacity,
                self._task_data.get("num_sprints", 3),
            )
        return 0.0, {}
