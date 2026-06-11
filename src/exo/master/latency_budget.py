"""
End-to-end latency budget enforcer: assigns a total time budget to each request
based on its SLA tier. Tracks elapsed time at key phases (queue, prefill, decode).
Cancels the request if the budget is exceeded before completion.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any
from loguru import logger

# Budget seconds per SLA tier
_TIER_BUDGETS: dict[str, float] = {
    "platinum": 30.0,
    "gold": 60.0,
    "silver": 120.0,
    "bronze": 300.0,
    "default": 120.0,
}


@dataclass
class LatencyBudget:
    request_id: str
    tier: str
    budget_s: float
    started_at: float = field(default_factory=time.time)
    phases: dict[str, float] = field(default_factory=dict)  # phase_name → elapsed_ms

    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def remaining_s(self) -> float:
        return max(0.0, self.budget_s - self.elapsed_s())

    def is_exceeded(self) -> bool:
        return self.elapsed_s() >= self.budget_s

    def mark_phase(self, phase: str) -> None:
        self.phases[phase] = round(self.elapsed_s() * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier": self.tier,
            "budget_s": self.budget_s,
            "elapsed_s": round(self.elapsed_s(), 3),
            "remaining_s": round(self.remaining_s(), 3),
            "exceeded": self.is_exceeded(),
            "phases_ms": self.phases,
        }


class LatencyBudgetManager:
    """
    start(request_id, tier): allocates budget for a new request.
    check(request_id): returns True if budget exceeded (caller should cancel).
    finish(request_id): removes from tracking.
    """

    def __init__(self) -> None:
        self._budgets: dict[str, LatencyBudget] = {}

    def start(self, request_id: str, tier: str = "default") -> LatencyBudget:
        budget_s = _TIER_BUDGETS.get(tier, _TIER_BUDGETS["default"])
        lb = LatencyBudget(request_id=request_id, tier=tier, budget_s=budget_s)
        self._budgets[request_id] = lb
        return lb

    def check(self, request_id: str) -> bool:
        lb = self._budgets.get(request_id)
        if lb is None:
            return False
        if lb.is_exceeded():
            logger.warning(
                f"Latency budget exceeded request_id={request_id} "
                f"tier={lb.tier} budget={lb.budget_s}s elapsed={lb.elapsed_s():.1f}s"
            )
            return True
        return False

    def mark_phase(self, request_id: str, phase: str) -> None:
        if request_id in self._budgets:
            self._budgets[request_id].mark_phase(phase)

    def finish(self, request_id: str) -> dict[str, Any] | None:
        lb = self._budgets.pop(request_id, None)
        return lb.to_dict() if lb else None

    def get_active(self) -> list[dict[str, Any]]:
        return [lb.to_dict() for lb in self._budgets.values()]

    def get_stats(self) -> dict[str, Any]:
        active = self.get_active()
        exceeded = [a for a in active if a["exceeded"]]
        return {"active": len(active), "exceeded": len(exceeded), "budgets": active}


LATENCY_BUDGET_MANAGER = LatencyBudgetManager()
