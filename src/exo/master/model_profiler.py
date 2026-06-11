"""
Per-model inference profiler: tracks TTFT (time-to-first-token), TPS
(tokens-per-second), and total latency per model. Maintains rolling
p50/p99 statistics over the last 100 requests per model.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class InferenceSample:
    ttft_ms: float
    tps: float
    total_ms: float
    tokens: int
    timestamp: float = field(default_factory=time.time)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * pct), len(sorted_v) - 1)
    return sorted_v[idx]


class ModelProfile:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._samples: deque[InferenceSample] = deque(maxlen=100)

    def record(self, ttft_ms: float, tokens: int, total_ms: float) -> None:
        tps = (tokens / max(total_ms, 1)) * 1000
        self._samples.append(InferenceSample(
            ttft_ms=ttft_ms, tps=tps, total_ms=total_ms, tokens=tokens
        ))

    def to_dict(self) -> dict[str, Any]:
        if not self._samples:
            return {"model_id": self.model_id, "sample_count": 0}
        ttfts = [s.ttft_ms for s in self._samples]
        tps_vals = [s.tps for s in self._samples]
        latencies = [s.total_ms for s in self._samples]
        return {
            "model_id": self.model_id,
            "sample_count": len(self._samples),
            "ttft_p50_ms": round(_percentile(ttfts, 0.5), 2),
            "ttft_p99_ms": round(_percentile(ttfts, 0.99), 2),
            "tps_p50": round(_percentile(tps_vals, 0.5), 2),
            "tps_p99": round(_percentile(tps_vals, 0.99), 2),
            "latency_p50_ms": round(_percentile(latencies, 0.5), 2),
            "latency_p99_ms": round(_percentile(latencies, 0.99), 2),
        }


class ModelProfiler:
    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}

    def record(self, model_id: str, ttft_ms: float, tokens: int, total_ms: float) -> None:
        if model_id not in self._profiles:
            self._profiles[model_id] = ModelProfile(model_id)
            logger.debug(f"[model_profiler] new profile created for model={model_id}")
        self._profiles[model_id].record(ttft_ms, tokens, total_ms)

    def get_profile(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._profiles:
            return {"model_id": model_id, "sample_count": 0}
        return self._profiles[model_id].to_dict()

    def get_all(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._profiles.values()]

    def get_stats(self) -> dict[str, Any]:
        return {"models_profiled": len(self._profiles), "profiles": self.get_all()}


MODEL_PROFILER = ModelProfiler()
