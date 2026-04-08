"""
Grader for Task 3: Quarterly Roadmap — Stakeholder Alignment.

This is the hard task. Scoring is multi-dimensional:

  1. OKR coverage (0.25): Are all 4 OKRs addressed by at least one scheduled item?
  2. Value delivered (0.25): Total item value scheduled / theoretical maximum
  3. Stakeholder balance (0.20): Did the agent give each stakeholder at least some wins?
  4. Constraint adherence (0.15): Capacity, dependencies, required P0s
  5. Tech debt ratio (0.10): At least 15% of capacity on tech debt
  6. Dependency ordering (0.05): Dependencies scheduled before dependents

The hard part: OKR weights conflict with stakeholder weights, and high-value
items have dependencies that force tough sprint sequencing decisions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import ProductItem

SPRINT_CAPACITY = 34
NUM_SPRINTS = 3
TOTAL_CAPACITY = SPRINT_CAPACITY * NUM_SPRINTS  # 102

REQUIRED_P0_IDS = {"BUG-003", "BUG-008"}

# Global dependency map (item_id -> list of prerequisite item_ids)
DEPENDENCIES: dict[str, list[str]] = {
    "FEAT-104": ["TECH-202"],
    "FEAT-108": ["FEAT-103"],
    "TECH-204": ["TECH-202"],
    "FEAT-106": ["FEAT-103"],
}

# OKR tag mapping
OKR_TAGS = {
    "OKR1": "Enterprise retention",
    "OKR2": "Self-serve growth",
    "OKR3": "Platform reliability",
    "OKR4": "Developer ecosystem",
}

OKR_WEIGHTS = {"OKR1": 0.35, "OKR2": 0.25, "OKR3": 0.25, "OKR4": 0.15}

STAKEHOLDER_WEIGHTS = {"Sales": 0.30, "Eng": 0.25, "Support": 0.25, "Growth": 0.20}

MIN_TECH_DEBT_FRACTION = 0.15

# Per-step rewards
VALID_SCHEDULE_REWARD = 0.02
OVER_CAPACITY_PENALTY = -0.12
DEP_VIOLATION_PENALTY = -0.08
DEFER_P0_PENALTY = -0.25


def item_value(
    item: "ProductItem",
    max_impact: int = 8000,
    max_revenue: int = 1800000,
    max_votes: int = 20,
) -> float:
    """Value score for an item (0–1)."""
    impact = min(item.user_impact / max_impact, 1.0)
    revenue = (
        min(item.revenue_impact_usd / max_revenue, 1.0) if item.revenue_impact_usd else 0.0
    )
    votes = min(sum(item.stakeholder_votes.values()) / max_votes, 1.0)
    return round(
        impact * 0.30 + revenue * 0.35 + item.strategic_score * 0.20 + votes * 0.15, 4
    )


def score_schedule_action(
    item: "ProductItem",
    sprint: int,
    sprint_loads: dict[int, int],
    scheduled_items_by_sprint: dict[int, list[str]],
) -> tuple[float, str]:
    """Reward for a single schedule action."""
    current_load = sprint_loads.get(sprint, 0)
    if current_load + item.effort_points > SPRINT_CAPACITY:
        return OVER_CAPACITY_PENALTY, (
            f"✗ Sprint {sprint} over capacity! {item.id} ({item.effort_points}pt) + "
            f"{current_load}pt existing > {SPRINT_CAPACITY}pt limit."
        )

    # Check dependency: dep must be in an earlier sprint
    deps = DEPENDENCIES.get(item.id, [])
    for dep_id in deps:
        dep_sprint = None
        for sp, sp_items in scheduled_items_by_sprint.items():
            if dep_id in sp_items:
                dep_sprint = sp
                break
        if dep_sprint is None or dep_sprint >= sprint:
            return DEP_VIOLATION_PENALTY, (
                f"✗ Dependency violation: {item.id} requires {dep_id} "
                f"to be in an earlier sprint (got dep_sprint={dep_sprint}, current={sprint})."
            )

    val = item_value(item)
    reward = VALID_SCHEDULE_REWARD + val * 0.03
    return round(reward, 4), f"✓ Scheduled {item.id} in Sprint {sprint} (value={val:.2f})"


def score_defer_action(item: "ProductItem") -> tuple[float, str]:
    """Reward for deferring an item."""
    if item.id in REQUIRED_P0_IDS:
        return DEFER_P0_PENALTY, f"✗ Cannot defer P0 item {item.id}!"
    return 0.005, f"~ Deferred {item.id} to Q4."


def compute_final_score(
    items: list["ProductItem"],
    sprint_capacity: int = SPRINT_CAPACITY,
    num_sprints: int = NUM_SPRINTS,
) -> tuple[float, dict]:
    """Compute final 0–1 score at episode end."""

    scheduled = [i for i in items if i.assigned_sprint is not None]
    scheduled_ids = {i.id for i in scheduled}

    # Sprint loads and per-sprint item maps
    sprint_loads: dict[int, int] = defaultdict(int)
    sprint_items: dict[int, list[str]] = defaultdict(list)
    for item in scheduled:
        sprint_loads[item.assigned_sprint] += item.effort_points
        sprint_items[item.assigned_sprint].append(item.id)

    # -----------------------------------------------------------------------
    # 1. OKR coverage
    # -----------------------------------------------------------------------
    okr_covered: dict[str, float] = {okr: 0.0 for okr in OKR_TAGS}
    for item in scheduled:
        for tag in getattr(item, "okr_tags", []) or []:
            if tag in okr_covered:
                okr_covered[tag] = min(1.0, okr_covered[tag] + item_value(item))

    okr_score = sum(
        OKR_WEIGHTS[okr] * min(okr_covered[okr] / 0.3, 1.0)  # normalise: 0.3 value = full credit
        for okr in OKR_TAGS
    )

    # -----------------------------------------------------------------------
    # 2. Value delivered
    # -----------------------------------------------------------------------
    max_impact = max((i.user_impact for i in items), default=1)
    max_revenue = max((i.revenue_impact_usd or 0 for i in items), default=1)

    def _val(i):
        return item_value(i, max_impact, max_revenue)

    scheduled_value = sum(_val(i) for i in scheduled)

    # Greedy theoretical max (respecting capacity, ignoring dependency order for upper bound)
    sorted_all = sorted(items, key=lambda x: _val(x) / x.effort_points, reverse=True)
    cap_remaining = sprint_capacity * num_sprints
    theoretical_max = 0.0
    for si in sorted_all:
        if si.effort_points <= cap_remaining:
            theoretical_max += _val(si)
            cap_remaining -= si.effort_points

    value_score = min(scheduled_value / theoretical_max, 1.0) if theoretical_max > 0 else 0.0

    # -----------------------------------------------------------------------
    # 3. Stakeholder balance
    # -----------------------------------------------------------------------
    stakeholder_satisfaction: dict[str, float] = {s: 0.0 for s in STAKEHOLDER_WEIGHTS}
    stakeholder_max: dict[str, float] = {s: 0.0 for s in STAKEHOLDER_WEIGHTS}

    for item in items:
        for stakeholder, votes in item.stakeholder_votes.items():
            if stakeholder not in STAKEHOLDER_WEIGHTS:
                continue
            stakeholder_max[stakeholder] += votes * _val(item)
            if item.id in scheduled_ids:
                stakeholder_satisfaction[stakeholder] += votes * _val(item)

    stakeholder_score = 0.0
    for sh, weight in STAKEHOLDER_WEIGHTS.items():
        sat = (
            min(stakeholder_satisfaction[sh] / stakeholder_max[sh], 1.0)
            if stakeholder_max[sh] > 0
            else 0.0
        )
        stakeholder_score += weight * sat

    # -----------------------------------------------------------------------
    # 4. Constraint adherence
    # -----------------------------------------------------------------------
    # P0 coverage
    missing_p0 = REQUIRED_P0_IDS - scheduled_ids
    p0_score = 1.0 - len(missing_p0) / len(REQUIRED_P0_IDS)

    # Capacity violations per sprint
    capacity_violations = sum(
        max(0, sprint_loads[sp] - sprint_capacity) for sp in range(1, num_sprints + 1)
    )
    capacity_score = max(0.0, 1.0 - capacity_violations / (sprint_capacity * num_sprints * 0.2))

    # Dependency ordering violations
    dep_violations = 0
    for item in scheduled:
        deps = DEPENDENCIES.get(item.id, [])
        for dep_id in deps:
            dep_sprint = next(
                (sp for sp, itms in sprint_items.items() if dep_id in itms), None
            )
            if dep_sprint is None or dep_sprint >= item.assigned_sprint:
                dep_violations += 1

    constraint_score = (
        p0_score * 0.50
        + capacity_score * 0.30
        + max(0.0, 1.0 - dep_violations * 0.2) * 0.20
    )

    # -----------------------------------------------------------------------
    # 5. Tech debt ratio
    # -----------------------------------------------------------------------
    tech_debt_points = sum(
        i.effort_points for i in scheduled if i.type.value == "tech_debt"
    )
    total_scheduled_points = sum(i.effort_points for i in scheduled)
    tech_debt_fraction = (
        tech_debt_points / total_scheduled_points if total_scheduled_points > 0 else 0.0
    )
    tech_debt_score = min(tech_debt_fraction / MIN_TECH_DEBT_FRACTION, 1.0)

    # -----------------------------------------------------------------------
    # 6. Dependency ordering bonus
    # -----------------------------------------------------------------------
    dep_order_score = max(0.0, 1.0 - dep_violations * 0.25)

    # -----------------------------------------------------------------------
    # Weighted final
    # -----------------------------------------------------------------------
    final_score = (
        okr_score * 0.25
        + value_score * 0.25
        + stakeholder_score * 0.20
        + constraint_score * 0.15
        + tech_debt_score * 0.10
        + dep_order_score * 0.05
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    breakdown = {
        "scheduled_items": len(scheduled),
        "total_items": len(items),
        "sprint_loads": dict(sprint_loads),
        "missing_p0_items": list(missing_p0),
        "dependency_violations": dep_violations,
        "okr_coverage": {k: round(v, 3) for k, v in okr_covered.items()},
        "okr_score": round(okr_score, 4),
        "scheduled_value": round(scheduled_value, 4),
        "theoretical_max_value": round(theoretical_max, 4),
        "value_score": round(value_score, 4),
        "stakeholder_satisfaction": {
            k: round(v / stakeholder_max[k], 3) if stakeholder_max[k] > 0 else 0.0
            for k, v in stakeholder_satisfaction.items()
        },
        "stakeholder_score": round(stakeholder_score, 4),
        "p0_coverage_score": round(p0_score, 4),
        "capacity_score": round(capacity_score, 4),
        "constraint_score": round(constraint_score, 4),
        "tech_debt_points": tech_debt_points,
        "tech_debt_fraction": round(tech_debt_fraction, 4),
        "tech_debt_score": round(tech_debt_score, 4),
        "dep_order_score": round(dep_order_score, 4),
        "final_score": final_score,
    }
    return final_score, breakdown
