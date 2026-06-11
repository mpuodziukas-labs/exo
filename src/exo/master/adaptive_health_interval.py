"""
Adaptive health check intervals: under high load increase polling frequency,
under low load reduce it to save overhead. Bounds: 1s min, 30s max.
Uses exponential smoothing on request rate to determine load tier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Load tier thresholds (requests per second)
_HIGH_LOAD_RPS = 10.0
_MEDIUM_LOAD_RPS = 2.0

# Poll interval seconds per tier
_INTERVAL_HIGH = 1.0      # high load: check every second
_INTERVAL_MEDIUM = 5.0    # medium load
_INTERVAL_IDLE = 30.0     # idle: back off

@dataclass
class HealthIntervalState:
    current_interval: float = 5.0
    smoothed_rps: float = 0.0
    last_updated: float = field(default_factory=time.time)
    requests_in_window: int = 0
    window_start: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_interval_seconds": self.current_interval,
            "smoothed_rps": round(self.smoothed_rps, 3),
            "load_tier": (
                "high" if self.smoothed_rps >= _HIGH_LOAD_RPS
                else "medium" if self.smoothed_rps >= _MEDIUM_LOAD_RPS
                else "idle"
            ),
        }

class AdaptiveHealthInterval:
    """
    Call record_request() on each incoming inference request.
    Call get_interval() to get current recommended poll interval.
    Interval updates every ~5 seconds based on smoothed RPS.
    """
    _ALPHA = 0.3  # EMA smoothing factor
    _UPDATE_WINDOW = 5.0  # seconds

    def __init__(self) -> None:
        self._state = HealthIntervalState()

    def record_request(self) -> None:
        self._state.requests_in_window += 1
        self._maybe_update()

    def _maybe_update(self) -> None:
        now = time.time()
        elapsed = now - self._state.window_start
        if elapsed < self._UPDATE_WINDOW:
            return
        raw_rps = self._state.requests_in_window / max(elapsed, 0.001)
        self._state.smoothed_rps = (
            self._ALPHA * raw_rps + (1 - self._ALPHA) * self._state.smoothed_rps
        )
        self._state.requests_in_window = 0
        self._state.window_start = now

        prev = self._state.current_interval
        if self._state.smoothed_rps >= _HIGH_LOAD_RPS:
            self._state.current_interval = _INTERVAL_HIGH
        elif self._state.smoothed_rps >= _MEDIUM_LOAD_RPS:
            self._state.current_interval = _INTERVAL_MEDIUM
        else:
            self._state.current_interval = _INTERVAL_IDLE

        if self._state.current_interval != prev:
            logger.debug(
                f"Health interval adjusted {prev}s→{self._state.current_interval}s "
                f"(rps={self._state.smoothed_rps:.2f})"
            )
        self._state.last_updated = now

    def get_interval(self) -> float:
        self._maybe_update()
        return self._state.current_interval

    def get_status(self) -> dict[str, Any]:
        return self._state.to_dict()

ADAPTIVE_HEALTH_INTERVAL = AdaptiveHealthInterval()
