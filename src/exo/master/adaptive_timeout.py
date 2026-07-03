from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

_WINDOW = 200
_SAFETY_MULTIPLIER = 1.5  # p99 * 1.5 = adaptive timeout
_MIN_TIMEOUT_S = 5.0
_MAX_TIMEOUT_S = 300.0
_FALLBACK_TIMEOUT_S = 60.0  # used when < 10 samples

# Priority tier multipliers applied to base adaptive timeout
_TIER_MULTIPLIERS: dict[str, float] = {
    "critical": 3.0,  # critical gets 3x base (never timeout a critical request prematurely)
    "high": 2.0,
    "normal": 1.5,
    "low": 1.0,
    "background": 0.8,  # background gets shorter timeout — deprioritize
}


@dataclass
class TimeoutSample:
    latency_s: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveTimeoutCalculator:
    """
    Computes per-model adaptive timeouts from observed latency distribution.
    Updates on every completed request. Thread-safe (GIL protected, single-threaded async).
    """

    def __init__(self) -> None:
        self._samples: dict[str, deque[TimeoutSample]] = {}

    def record(self, model_id: str, latency_s: float) -> None:
        if model_id not in self._samples:
            self._samples[model_id] = deque(maxlen=_WINDOW)
        self._samples[model_id].append(TimeoutSample(latency_s=latency_s))

    def _p99(self, model_id: str) -> float | None:
        samples = self._samples.get(model_id)
        if samples is None or len(samples) < 10:
            return None
        lats = sorted(s.latency_s for s in samples)
        return lats[min(int(len(lats) * 0.99), len(lats) - 1)]

    def base_timeout(self, model_id: str) -> float:
        p99 = self._p99(model_id)
        if p99 is None:
            return _FALLBACK_TIMEOUT_S
        raw = p99 * _SAFETY_MULTIPLIER
        return max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, raw))

    def timeout_for(self, model_id: str, priority: str = "normal") -> float:
        """Return timeout in seconds for model + priority combination."""
        base = self.base_timeout(model_id)
        multiplier = _TIER_MULTIPLIERS.get(priority, 1.5)
        return max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, base * multiplier))

    def all_timeouts(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for model_id in self._samples:
            p99 = self._p99(model_id)
            results.append(
                {
                    "model_id": model_id,
                    "sample_count": len(self._samples[model_id]),
                    "p99_latency_s": round(p99, 3) if p99 else None,
                    "base_timeout_s": round(self.base_timeout(model_id), 2),
                    "timeouts_by_priority": {
                        tier: round(self.timeout_for(model_id, tier), 2)
                        for tier in _TIER_MULTIPLIERS
                    },
                }
            )
        return results


ADAPTIVE_TIMEOUT = AdaptiveTimeoutCalculator()
