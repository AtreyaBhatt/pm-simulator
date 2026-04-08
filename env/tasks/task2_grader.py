"""
Grader for Task 2: Sprint Planning.

Scoring dimensions:
  1. Constraint adherence (hard constraints — pass/fail weighted):
     - All P0 items must be scheduled          (weight: 0.30)
     - Sprint capacity not exceeded            (weight: 0.20)
     - No scheduling of items with unscheduled dependencies  (weight: 0.10)

  2. Value optimization:
     - Total value score of scheduled items / optimal value  (weight: 0.30)

  3. Strategic balance:
     - Coverage across OKR areas              (weight: 0.10)

Value formula per item:
  value = (user_impact / max_impact) * 0.35
        + (revenue_impact_norm) * 0.30
        + strategic_score * 0.20
        + (stakeholder_vote_sum / max_votes) * 0.15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import ProductItem

SPRINT_CAPACITY = 34

# These items MUST be scheduled (P0 bugs)
REQUIRED_ITEM_IDS = {"BUG-003", "BUG-008"}

# Dependency map: item -> must come after
DEPENDENCIES: dict[str, list[str]] = {
    "FEAT-104": ["TECH-202"],
    "FEAT-108": ["FEAT-103"],
}

# Step rewards
SCHEDULE_VALID_REWARD = 0.04
SCHEDULE_OVER_CAPACITY_PENALTY = -0.15
SCHEDULE_MISSING_DEP_PENALTY = -0.10
DEFER_P0_PENALTY = -0.20


def item_value(item: "ProductItem", max_impact: int = 4200, max_revenue: int = 920000) -> float:
    """Compute the value score (0–1) for a single item."""
    impact_norm = min(item.user_impact / max_impact, 1.0)
    revenue_norm = (
        min(item.revenue_impact_usd / max_revenue, 1.0)
        if item.revenue_impact_usd
        else 0.0
    )
    vote_sum = sum(item.stakeholder_votes.values())
    vote_norm = min(vote_sum / 20.0, 1.0)

    return round(
        impact_norm * 0.35
        + revenue_norm * 0.30
        + item.strategic_score * 0.20
        + vote_norm * 0.15,
        4,
    )


def score_schedule_action(
    item: "ProductItem",
    sprint: int,
    current_sprint_load: int,
    scheduled_ids: set[str],
) -> tuple[float, str]:
    """Per-step reward when agent schedules an item."""
    if current_sprint_load + item.effort_points > SPRINT_CAPACITY:
        return SCHEDULE_OVER_CAPACITY_PENALTY, (
            f"✗ Over capacity! Scheduling {item.id} ({item.effort_points}pt) would exceed "
            f"{SPRINT_CAPACITY}pt sprint limit. Current load: {current_sprint_load}pt."
        )

    # Check dependency satisfied
    deps = DEPENDENCIES.get(item.id, [])
    missing_deps = [d for d in deps if d not in scheduled_ids]
    if missing_deps:
        return SCHEDULE_MISSING_DEP_PENALTY, (
            f"✗ Dependency violation: {item.id} requires {missing_deps} to be scheduled first."
        )

    val = item_value(item)
    reward = SCHEDULE_VALID_REWARD + val * 0.05  # bonus proportional to item value
    return round(reward, 4), f"✓ Scheduled {item.id} ({item.effort_points}pt, value={val:.2f})"


def score_defer_action(item: "ProductItem") -> tuple[float, str]:
    """Per-step reward when agent defers an item."""
    if item.assigned_priority and item.assigned_priority.value == "P0":
        return DEFER_P0_PENALTY, (
            f"✗ Deferred a P0 item ({item.id})! P0 bugs must be fixed this sprint."
        )
    return 0.01, f"~ Deferred {item.id} to future sprint."


def compute_final_score(
    items: list["ProductItem"],
    sprint_capacity: int = SPRINT_CAPACITY,
) -> tuple[float, dict]:
    """Final grader score at episode end."""
    scheduled = [i for i in items if i.assigned_sprint == 1]
    deferred = [i for i in items if i.deferred or i.assigned_sprint is None]

    scheduled_ids = {i.id for i in scheduled}
    total_points = sum(i.effort_points for i in scheduled)

    # --- Hard constraint checks ---
    missing_required = REQUIRED_ITEM_IDS - scheduled_ids
    p0_coverage = 1.0 - (len(missing_required) / len(REQUIRED_ITEM_IDS))

    over_capacity = max(0, total_points - sprint_capacity)
    capacity_ok = 1.0 if over_capacity == 0 else max(0.0, 1.0 - over_capacity / sprint_capacity)

    # Dependency violations among scheduled items
    dep_violations = 0
    for item in scheduled:
        deps = DEPENDENCIES.get(item.id, [])
        for dep in deps:
            if dep not in scheduled_ids:
                dep_violations += 1
    dep_score = max(0.0, 1.0 - dep_violations * 0.25)

    # --- Value optimization ---
    max_impact = max((i.user_impact for i in items), default=1)
    max_revenue = max((i.revenue_impact_usd or 0 for i in items), default=1)

    def _val(item):
        return item_value(item, max_impact, max_revenue)

    scheduled_value = sum(_val(i) for i in scheduled)

    # Compute approximate optimal (greedy by value/point ratio, must include required)
    required_items = [i for i in items if i.id in REQUIRED_ITEM_IDS]
    optional_items = sorted(
        [i for i in items if i.id not in REQUIRED_ITEM_IDS],
        key=lambda x: _val(x) / x.effort_points,
        reverse=True,
    )
    opt_load = sum(i.effort_points for i in required_items)
    opt_value = sum(_val(i) for i in required_items)
    for opt_item in optional_items:
        if opt_load + opt_item.effort_points <= sprint_capacity:
            opt_load += opt_item.effort_points
            opt_value += _val(opt_item)

    value_ratio = min(scheduled_value / opt_value, 1.0) if opt_value > 0 else 0.0

    # --- OKR coverage ---
    okr_tags_covered: set[str] = set()
    for item in scheduled:
        for tag in getattr(item, "tags", []):
            if tag.startswith("OKR"):
                okr_tags_covered.add(tag)
    okr_coverage = min(len(okr_tags_covered) / 3.0, 1.0)

    # --- Final weighted score ---
    final_score = (
        p0_coverage * 0.30
        + capacity_ok * 0.20
        + dep_score * 0.10
        + value_ratio * 0.30
        + okr_coverage * 0.10
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    breakdown = {
        "scheduled_items": len(scheduled),
        "scheduled_points": total_points,
        "capacity": sprint_capacity,
        "over_capacity_points": over_capacity,
        "missing_required_items": list(missing_required),
        "p0_coverage": round(p0_coverage, 4),
        "capacity_score": round(capacity_ok, 4),
        "dependency_violations": dep_violations,
        "dep_score": round(dep_score, 4),
        "scheduled_value": round(scheduled_value, 4),
        "optimal_value": round(opt_value, 4),
        "value_ratio": round(value_ratio, 4),
        "okr_tags_covered": list(okr_tags_covered),
        "okr_coverage": round(okr_coverage, 4),
        "final_score": final_score,
    }
    return final_score, breakdown
