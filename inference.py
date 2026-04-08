"""
PM Simulator — Baseline Inference Script

Runs an LLM agent against all 3 tasks and reports baseline scores.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible API base URL
  MODEL_NAME    — Model identifier for inference
  HF_TOKEN      — Hugging Face / API token (used as the API key)

Optional:
  PM_ENV_URL    — URL of the running PM Simulator API (default: http://localhost:7860)

Usage:
  python inference.py
  python inference.py --task task_1   # run a single task
"""

import os
import sys
import json
import argparse
import requests
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PM_ENV_URL = os.environ.get("PM_ENV_URL", "http://localhost:7860")

MAX_STEPS = 5  # safety cap per episode
TEMPERATURE = 0.2

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Product Manager AI agent.

You will receive a JSON observation describing a PM scenario. Your job is to analyse it
carefully and respond with a valid JSON action.

## Action types

**rank_items** — for bug triage tasks:
{
  "action_type": "rank_items",
  "ranked_ids": ["BUG-008", "BUG-001", ...],   // ALL item IDs, most urgent first
  "reasoning": "Brief explanation"
}

**plan_sprint** — for sprint planning tasks:
{
  "action_type": "plan_sprint",
  "selected_ids": ["FEAT-101", "BUG-203", ...],  // items to include in sprint
  "reasoning": "Brief explanation"
}

**finalize_sprint** — for stakeholder gauntlet:
{
  "action_type": "finalize_sprint",
  "selected_ids": ["S1-REV-01", "S1-TECH-01", ...],  // items to deliver this sprint
  "reasoning": "Brief explanation"
}

## PM decision principles

1. **For bug triage**: Prioritise by (severity × impact) — fix critical/high-impact bugs first.
   Consider: payment failures > security issues > data loss > performance > UX polish.

2. **For sprint planning**: Maximise delivered value within the story point cap.
   Avoid expensive low-value items. Balance bugs, features, and tech debt.
   NEVER exceed sprint capacity.

3. **For stakeholder gauntlet**: Represent ALL stakeholder groups every sprint.
   Neglecting any group (users, engineering, revenue) causes metric decay that compounds.
   Address critical/legal items immediately.

Respond ONLY with a valid JSON object — no markdown, no preamble, no explanation outside the JSON.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{PM_ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(f"{PM_ENV_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_health() -> bool:
    try:
        resp = requests.get(f"{PM_ENV_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Agent logic
# ──────────────────────────────────────────────────────────────────────────────

def format_backlog_for_prompt(backlog: list) -> str:
    """Format backlog items as readable text for the LLM."""
    lines = []
    for item in backlog:
        line = (
            f"  [{item['id']}] {item['title']}\n"
            f"    Type: {item['type']} | Severity: {item['severity']} | "
            f"Effort: {item['effort']}pts | "
            f"Business value: {item['business_value']}/10 | "
            f"User impact: {item['user_impact']}/10 | "
            f"Stakeholder: {item['stakeholder']}\n"
            f"    Description: {item['description']}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_messages_for_prompt(messages: list) -> str:
    if not messages:
        return "  (none)"
    lines = []
    for msg in messages:
        lines.append(
            f"  [{msg['urgency'].upper()}] {msg['sender']} ({msg['role']}): {msg['message']}"
        )
    return "\n".join(lines)


def build_user_prompt(observation: dict) -> str:
    backlog_str = format_backlog_for_prompt(observation.get("backlog", []))
    messages_str = format_messages_for_prompt(observation.get("stakeholder_messages", []))

    metrics = observation.get("product_metrics", {})
    metrics_str = (
        f"user_satisfaction={metrics.get('user_satisfaction', '?'):.0f}, "
        f"revenue_health={metrics.get('revenue_health', '?'):.0f}, "
        f"tech_debt_score={metrics.get('tech_debt_score', '?'):.0f}, "
        f"team_velocity={metrics.get('team_velocity', '?'):.0f}"
    ) if metrics else "N/A"

    sprint_info = ""
    if observation.get("sprint_capacity", 0) > 0:
        sprint_info = (
            f"\nSprint capacity: {observation['sprint_capacity']} story points "
            f"(sprint {observation.get('current_sprint', 1)}/{observation.get('total_sprints', 1)})"
        )

    return f"""=== PM SIMULATOR — {observation.get('task_name', '').upper()} ===

Task: {observation.get('task_description', '')}

Context:
{observation.get('context', '')}
{sprint_info}

Product metrics: {metrics_str}

Stakeholder messages:
{messages_str}

Backlog items:
{backlog_str}

Valid actions: {observation.get('valid_actions', [])}

Respond with a valid JSON action object.
"""


def call_model(user_prompt: str, attempt: int = 0) -> str:
    """Call the LLM and return the raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=1000,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [!] Model call failed (attempt {attempt}): {exc}")
        return ""


def parse_action(response_text: str, valid_actions: list) -> dict:
    """Parse the model's response into a valid action dict."""
    # Strip markdown code blocks if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        action = json.loads(text)
        if "action_type" in action:
            return action
    except json.JSONDecodeError:
        pass

    # Fallback: construct minimal valid action
    print(f"  [!] Could not parse model response. Using fallback action.")
    print(f"  Raw response: {response_text[:200]}")

    fallback_type = valid_actions[0] if valid_actions else "rank_items"
    return {"action_type": fallback_type, "ranked_ids": [], "selected_ids": [], "reasoning": "fallback"}


# ──────────────────────────────────────────────────────────────────────────────
# Task runners
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    """Run the agent on a single task and return the final score."""
    print(f"\n{'='*60}")
    print(f"  Running {task_id.upper()}")
    print(f"{'='*60}")

    observation = env_reset(task_id)
    print(f"  Task: {observation.get('task_name')} ({observation.get('task_description', '')[:80]}...)")

    final_score = 0.0
    step_count = 0

    while step_count < MAX_STEPS:
        step_count += 1
        valid_actions = observation.get("valid_actions", [])

        if not valid_actions:
            print("  No valid actions — episode may be complete.")
            break

        print(f"\n  Step {step_count}: Generating action (valid: {valid_actions})")
        user_prompt = build_user_prompt(observation)
        response_text = call_model(user_prompt)

        if not response_text:
            print("  Empty model response — using fallback.")
            response_text = json.dumps({"action_type": valid_actions[0]})

        action = parse_action(response_text, valid_actions)
        print(f"  Action: {action.get('action_type')} | "
              f"items={len(action.get('selected_ids') or action.get('ranked_ids') or [])}")

        if action.get("reasoning"):
            print(f"  Reasoning: {action['reasoning'][:120]}")

        result = env_step(action)
        reward = result.get("reward", {})
        done = result.get("done", True)
        final_score = reward.get("value", 0.0)

        print(f"  Reward: {final_score:.4f} | Done: {done}")
        if reward.get("feedback"):
            # Print first 3 lines of feedback
            feedback_lines = reward["feedback"].split("\n")[:4]
            for line in feedback_lines:
                print(f"    {line}")

        observation = result.get("observation", observation)

        if done:
            print(f"\n  ✓ Episode complete. Final score: {final_score:.4f}")
            break

    return final_score


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PM Simulator baseline inference")
    parser.add_argument("--task", type=str, default=None, help="Run a single task (task_1/task_2/task_3)")
    parser.add_argument("--env-url", type=str, default=None, help="Override PM_ENV_URL")
    args = parser.parse_args()

    global PM_ENV_URL
    if args.env_url:
        PM_ENV_URL = args.env_url

    print(f"PM Simulator Baseline Inference")
    print(f"  API base:  {API_BASE_URL}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Env URL:   {PM_ENV_URL}")

    if not HF_TOKEN:
        print("  [WARNING] HF_TOKEN not set. API calls may fail.")

    # Check environment is up
    if not env_health():
        print(f"\n[ERROR] PM Simulator not reachable at {PM_ENV_URL}")
        print("  Start with: uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    print("  Environment: ✓ reachable\n")

    tasks = ["task_1", "task_2", "task_3"] if not args.task else [args.task]
    scores = {}

    for task_id in tasks:
        score = run_task(task_id)
        scores[task_id] = score

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE SCORES")
    print(f"{'='*60}")
    task_names = {"task_1": "Bug Triage (Easy)", "task_2": "Sprint Planning (Medium)", "task_3": "Stakeholder Gauntlet (Hard)"}
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_names.get(task_id, task_id):35s}  [{bar}]  {score:.4f}")

    if len(scores) == 3:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average score: {avg:.4f}")

    return scores


if __name__ == "__main__":
    main()
