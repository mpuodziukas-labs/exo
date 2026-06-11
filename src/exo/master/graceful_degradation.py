"""
Graceful degradation mode: when cluster is critically degraded (≥50% workers down,
CB open for all active workers, or health_score < 0.3), switch to DEGRADED mode
and serve only cached responses. Blocks new inference; returns 503 with Retry-After.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class DegradationLevel(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"     # >30% workers unhealthy
    CRITICAL = "critical"     # >60% workers down or health_score < 0.3
    EMERGENCY = "emergency"   # all workers down

@dataclass
class DegradationState:
    level: DegradationLevel = DegradationLevel.NORMAL
    reason: str = ""
    degraded_since: float = 0.0
    last_check: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "reason": self.reason,
            "degraded_since": self.degraded_since,
            "last_check": self.last_check,
            "serves_cache_only": self.level in (DegradationLevel.CRITICAL, DegradationLevel.EMERGENCY),
        }

class GracefulDegradationController:
    """
    Monitors cluster health signals and sets degradation level.
    In CRITICAL/EMERGENCY: new inference requests get 503 + Retry-After: 30.
    DEGRADED: requests allowed but warned.
    """
    def __init__(self) -> None:
        self._state = DegradationState()

    def evaluate(self, total_workers: int, healthy_workers: int, health_score: float) -> DegradationState:
        """Called periodically by health check loop. Updates internal state."""
        if total_workers == 0:
            level = DegradationLevel.EMERGENCY
            reason = "no workers registered"
        elif healthy_workers == 0:
            level = DegradationLevel.EMERGENCY
            reason = "all workers unhealthy"
        elif health_score < 0.3 or (healthy_workers / total_workers) < 0.4:
            level = DegradationLevel.CRITICAL
            reason = f"health_score={health_score:.2f} workers={healthy_workers}/{total_workers}"
        elif (healthy_workers / total_workers) < 0.7:
            level = DegradationLevel.DEGRADED
            reason = f"workers={healthy_workers}/{total_workers} below 70%"
        else:
            level = DegradationLevel.NORMAL
            reason = ""

        now = time.time()
        if level != self._state.level:
            if level in (DegradationLevel.CRITICAL, DegradationLevel.EMERGENCY):
                self._state.degraded_since = now
                logger.critical(f"Cluster entering {level.value}: {reason}")
            elif self._state.level in (DegradationLevel.CRITICAL, DegradationLevel.EMERGENCY):
                logger.info(f"Cluster recovering from {self._state.level.value} → {level.value}")
            self._state.level = level
            self._state.reason = reason
        self._state.last_check = now
        return self._state

    @property
    def blocks_inference(self) -> bool:
        return self._state.level in (DegradationLevel.CRITICAL, DegradationLevel.EMERGENCY)

    @property
    def state(self) -> DegradationState:
        return self._state

    def get_status(self) -> dict[str, Any]:
        return self._state.to_dict()

DEGRADATION_CONTROLLER = GracefulDegradationController()
