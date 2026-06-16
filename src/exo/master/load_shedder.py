from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class ShedLevel(str, Enum):
    NONE = "none"  # no shedding
    LIGHT = "light"  # shed BACKGROUND only
    MODERATE = "moderate"  # shed LOW + BACKGROUND
    HEAVY = "heavy"  # shed NORMAL + LOW + BACKGROUND
    EMERGENCY = "emergency"  # shed everything except CRITICAL


# Retry-after seconds per shed level
_RETRY_AFTER: dict[ShedLevel, int] = {
    ShedLevel.NONE: 0,
    ShedLevel.LIGHT: 5,
    ShedLevel.MODERATE: 15,
    ShedLevel.HEAVY: 30,
    ShedLevel.EMERGENCY: 60,
}

# Priority levels that are shed at each level (cumulative)
_SHED_PRIORITIES: dict[ShedLevel, set[str]] = {
    ShedLevel.NONE: set(),
    ShedLevel.LIGHT: {"background"},
    ShedLevel.MODERATE: {"background", "low"},
    ShedLevel.HEAVY: {"background", "low", "normal"},
    ShedLevel.EMERGENCY: {"background", "low", "normal", "high"},
}


@dataclass
class ShedEvent:
    level: ShedLevel
    reason: str
    timestamp: float = field(default_factory=time.time)


class LoadShedder:
    """
    Controls active load shedding under cluster pressure.
    Shed level is set by monitoring components (memory, queue depth).
    check() returns (should_shed: bool, retry_after: int) per request priority.
    """

    def __init__(self) -> None:
        self._level = ShedLevel.NONE
        self._history: list[ShedEvent] = []
        self._shed_count = 0
        self._total_checked = 0

    def set_level(self, level: ShedLevel, reason: str = "") -> None:
        if level != self._level:
            logger.warning(
                f"LoadShedder level change: {self._level.value} → {level.value} reason={reason}"
            )
            self._level = level
            event = ShedEvent(level=level, reason=reason)
            self._history.append(event)
            if len(self._history) > 100:
                self._history = self._history[-100:]

    def check(self, priority: str = "normal") -> tuple[bool, int]:
        """
        Returns (should_shed, retry_after_seconds).
        should_shed=True means reject this request.
        """
        self._total_checked += 1
        shed_priorities = _SHED_PRIORITIES[self._level]
        if priority.lower() in shed_priorities:
            self._shed_count += 1
            retry = _RETRY_AFTER[self._level]
            logger.debug(
                f"LoadShedder SHED priority={priority} level={self._level.value}"
            )
            return True, retry
        return False, 0

    @property
    def current_level(self) -> ShedLevel:
        return self._level

    def auto_set_from_metrics(
        self,
        queue_depth: int,
        memory_pressure: str,
    ) -> ShedLevel:
        """Auto-compute shed level from current metrics. Call periodically."""
        if memory_pressure == "fatal" or queue_depth > 100:
            level = ShedLevel.EMERGENCY
        elif memory_pressure == "critical" or queue_depth > 50:
            level = ShedLevel.HEAVY
        elif memory_pressure == "warning" or queue_depth > 20:
            level = ShedLevel.MODERATE
        elif queue_depth > 10:
            level = ShedLevel.LIGHT
        else:
            level = ShedLevel.NONE
        self.set_level(level, reason=f"auto: queue={queue_depth} mem={memory_pressure}")
        return level

    def stats(self) -> dict[str, Any]:
        return {
            "current_level": self._level.value,
            "retry_after_s": _RETRY_AFTER[self._level],
            "shed_count": self._shed_count,
            "total_checked": self._total_checked,
            "shed_rate": round(self._shed_count / max(self._total_checked, 1), 4),
            "recent_events": [
                {"level": e.level.value, "reason": e.reason, "timestamp": e.timestamp}
                for e in self._history[-10:]
            ],
        }


LOAD_SHEDDER = LoadShedder()
