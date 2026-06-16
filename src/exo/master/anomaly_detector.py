"""
Inference anomaly detector: detects statistical anomalies in inference metrics
(TTFT, TPS, error rate) using Z-score against rolling baseline.
Fires SSE events and log warnings when anomalies are detected.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_WINDOW = 50  # rolling window size
_Z_THRESHOLD = 3.0  # Z-score threshold for anomaly
_MIN_SAMPLES = 10  # need at least this many samples before detecting


@dataclass
class AnomalyEvent:
    metric: str
    value: float
    z_score: float
    baseline_mean: float
    baseline_std: float
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": round(self.value, 3),
            "z_score": round(self.z_score, 3),
            "baseline_mean": round(self.baseline_mean, 3),
            "baseline_std": round(self.baseline_std, 3),
            "detected_at": self.detected_at,
        }


class MetricTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._window: deque[float] = deque(maxlen=_WINDOW)

    def add(self, value: float) -> AnomalyEvent | None:
        if len(self._window) >= _MIN_SAMPLES:
            vals = list(self._window)
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(variance)
            if std > 0:
                z = abs(value - mean) / std
                if z > _Z_THRESHOLD:
                    event = AnomalyEvent(
                        metric=self.name,
                        value=value,
                        z_score=z,
                        baseline_mean=mean,
                        baseline_std=std,
                    )
                    logger.warning(
                        f"Anomaly: metric={self.name} value={value:.3f} "
                        f"z={z:.2f} mean={mean:.3f}"
                    )
                    self._window.append(value)
                    return event
        self._window.append(value)
        return None

    def stats(self) -> dict[str, Any]:
        vals = list(self._window)
        if not vals:
            return {"name": self.name, "samples": 0}
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {
            "name": self.name,
            "samples": len(vals),
            "mean": round(mean, 3),
            "std": round(math.sqrt(variance), 3),
        }


class AnomalyDetector:
    def __init__(self) -> None:
        self._trackers: dict[str, MetricTracker] = {
            "ttft_ms": MetricTracker("ttft_ms"),
            "tps": MetricTracker("tps"),
            "total_latency_ms": MetricTracker("total_latency_ms"),
            "error_rate": MetricTracker("error_rate"),
        }
        self._events: deque[AnomalyEvent] = deque(maxlen=100)
        self._total_anomalies = 0

    def record(self, metric: str, value: float) -> AnomalyEvent | None:
        tracker = self._trackers.get(metric)
        if tracker is None:
            self._trackers[metric] = MetricTracker(metric)
            tracker = self._trackers[metric]
        event = tracker.add(value)
        if event:
            self._events.append(event)
            self._total_anomalies += 1
        return event

    def get_recent_anomalies(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_anomalies": self._total_anomalies,
            "tracker_stats": [t.stats() for t in self._trackers.values()],
            "recent_anomalies": self.get_recent_anomalies()[-10:],
        }


ANOMALY_DETECTOR = AnomalyDetector()
