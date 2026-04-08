"""
Grader for Task 1: Bug Triage.

Scoring logic:
  - Each item has a ground-truth priority computed from a deterministic formula.
  - The agent assigns P0–P3 to each item via 'prioritize' actions.
  - Score = weighted accuracy with partial credit for off-by-one errors.

Ground truth formula (encoded in task1_data.json):
  priority_score = (severity_weight * 0.4) + (user_impact_norm * 0.3)
                 + (revenue_norm * 0.2) + (age_norm * 0.1)
  where critical=4, high=3, medium=2, low=1 for severity_weight

  P0: score >= 0.75 AND severity == critical
  P1: score >= 0.55 OR (severity == high AND revenue_impact > 100k)
  P2: score >= 0.35
  P3: score < 0.35
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import ProductItem

# Priority ordering for off-by-one distance calculation
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

# Per-step reward for correct / near-correct prioritization
STEP_REWARDS = {
    0: 0.15,   # Exact match
    1: 0.06,   # Off by 1 (e.g. P0 → P1)
    2: -0.05,  # Off by 2
    3: -0.15,  # Completely wrong (P0 → P3 or vice versa)
}

# P0 miss carries extra penalty (missing a critical bug is worse than mislabeling a low one)
P0_MISS_PENALTY = -0.20


def score_priority_assignment(
    item: "ProductItem",
    assigned: str,
    ground_truth: str,
) -> tuple[float, str]:
    """
    Score a single priority assignment.
    Returns (reward_value, explanation_message).
    """
    if assigned not in PRIORITY_ORDER or ground_truth not in PRIORITY_ORDER:
        return -0.10, f"Invalid priority value '{assigned}'"

    assigned_rank = PRIORITY_ORDER[assigned]
    truth_rank = PRIORITY_ORDER[ground_truth]
    distance = abs(assigned_rank - truth_rank)

    reward = STEP_REWARDS.get(distance, -0.15)

    # Extra penalty for missing a P0
    if ground_truth == "P0" and assigned != "P0":
        reward += P0_MISS_PENALTY

    if distance == 0:
        msg = f"✓ Correct! {item.id} is {ground_truth}."
    elif distance == 1:
        msg = f"~ Close. {item.id} should be {ground_truth}, not {assigned}."
    else:
        msg = f"✗ Wrong. {item.id} should be {ground_truth}, not {assigned}."

    return round(reward, 4), msg


def compute_final_score(
    items: list["ProductItem"],
    ground_truth: dict[str, str],
) -> tuple[float, dict]:
    """
    Compute the final 0.0–1.0 grader score at episode end.

    Returns (score, breakdown_dict).
    """
    if not items:
        return 0.0, {"error": "No items to grade"}

    total_items = len(items)
    exact_matches = 0
    off_by_one = 0
    p0_misses = 0
    p0_total = sum(1 for v in ground_truth.values() if v == "P0")
    ungraded = 0

    for item in items:
        gt = ground_truth.get(item.id)
        if gt is None:
            continue

        assigned = item.assigned_priority.value if item.assigned_priority else None
        if assigned is None:
            ungraded += 1
            continue

        distance = abs(PRIORITY_ORDER[assigned] - PRIORITY_ORDER[gt])
        if distance == 0:
            exact_matches += 1
        elif distance == 1:
            off_by_one += 1

        if gt == "P0" and assigned != "P0":
            p0_misses += 1

    graded = total_items - ungraded
    if graded == 0:
        return 0.0, {"error": "No items were prioritized before submit"}

    # Base accuracy: exact + partial credit for off-by-one
    accuracy = (exact_matches + 0.4 * off_by_one) / total_items

    # P0 coverage bonus/penalty
    p0_coverage = (p0_total - p0_misses) / p0_total if p0_total > 0 else 1.0

    # Completeness: fraction of items that got any priority
    completeness = graded / total_items

    # Weighted final score
    final_score = (
        accuracy * 0.55
        + p0_coverage * 0.30
        + completeness * 0.15
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    breakdown = {
        "total_items": total_items,
        "graded": graded,
        "ungraded": ungraded,
        "exact_matches": exact_matches,
        "off_by_one": off_by_one,
        "p0_total": p0_total,
        "p0_misses": p0_misses,
        "p0_coverage": round(p0_coverage, 4),
        "accuracy": round(accuracy, 4),
        "completeness": round(completeness, 4),
        "final_score": final_score,
    }
    return final_score, breakdown
