from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from loguru import logger

_KEEP_PROFILES = 500
_KEEP_SAMPLES = 1000  # max flat latency samples per model


@dataclass
class PhaseSpan:
    name: str
    start: float
    end: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration_ms": round(self.duration_ms, 3),
            "metadata": self.metadata,
        }


@dataclass
class InferenceProfile:
    trace_id: str
    model_id: str
    spans: list[PhaseSpan] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def total_ms(self) -> float:
        if not self.spans:
            return 0.0
        return sum(s.duration_ms for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model_id": self.model_id,
            "total_ms": round(self.total_ms(), 3),
            "spans": [s.to_dict() for s in self.spans],
            "created_at": self.created_at,
        }

    def to_chrome_trace(self) -> list[dict[str, Any]]:
        """Chrome tracing format compatible with chrome://tracing and speedscope."""
        events = []
        for span in self.spans:
            events.append({
                "name": span.name,
                "cat": "inference",
                "ph": "X",
                "ts": span.start * 1e6,        # microseconds
                "dur": span.duration_ms * 1000,  # microseconds
                "pid": 1,
                "tid": 1,
                "args": span.metadata,
            })
        return events


@dataclass
class _LatencySample:
    """Flat per-request sample stored by record()."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    node_id: str
    ts: float = field(default_factory=time.time)


class InferenceProfiler:
    """
    Lightweight always-on inference profiler.
    Stores last _KEEP_PROFILES complete profiles in memory.
    Chrome trace export compatible.

    record() additionally maintains a flat per-model latency ring-buffer that
    powers per_model_stats() and hotspots() without touching the span machinery.
    """

    def __init__(self) -> None:
        self._profiles: deque[InferenceProfile] = deque(maxlen=_KEEP_PROFILES)
        self._active: dict[str, InferenceProfile] = {}
        # model_id -> ring-buffer of flat samples
        self._samples: dict[str, deque[_LatencySample]] = {}
        self._total_requests: int = 0

    def start_profile(self, trace_id: str, model_id: str) -> InferenceProfile:
        profile = InferenceProfile(trace_id=trace_id, model_id=model_id)
        self._active[trace_id] = profile
        return profile

    @contextmanager
    def span(self, trace_id: str, phase: str, **metadata: Any) -> Generator[PhaseSpan, None, None]:
        profile = self._active.get(trace_id)
        span = PhaseSpan(name=phase, start=time.monotonic(), metadata=dict(metadata))
        try:
            yield span
        finally:
            span.end = time.monotonic()
            if profile is not None:
                profile.spans.append(span)

    def finish_profile(self, trace_id: str) -> InferenceProfile | None:
        profile = self._active.pop(trace_id, None)
        if profile is not None:
            self._profiles.append(profile)
            logger.debug(
                f"InferenceProfiler completed trace_id={trace_id[:8]} "
                f"total_ms={profile.total_ms():.0f} spans={len(profile.spans)}"
            )
        return profile

    def get_profile(self, trace_id: str) -> InferenceProfile | None:
        for p in self._profiles:
            if p.trace_id == trace_id:
                return p
        return self._active.get(trace_id)

    def recent_profiles(self, limit: int = 20) -> list[dict[str, Any]]:
        profiles = list(self._profiles)[-limit:]
        return [p.to_dict() for p in reversed(profiles)]

    def chrome_trace_export(self, trace_id: str) -> list[dict[str, Any]]:
        profile = self.get_profile(trace_id)
        if profile is None:
            return []
        return profile.to_chrome_trace()

    def aggregate_stats(self) -> dict[str, Any]:
        if not self._profiles:
            return {"sample_count": 0}
        totals = [p.total_ms() for p in self._profiles]
        return {
            "sample_count": len(self._profiles),
            "avg_total_ms": round(sum(totals) / len(totals), 2),
            "min_total_ms": round(min(totals), 2),
            "max_total_ms": round(max(totals), 2),
            "active_profiles": len(self._active),
        }

    # ------------------------------------------------------------------
    # Flat per-request record API (wired from api.py finally block)
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        node_id: str,
    ) -> None:
        """Store a flat latency sample for the given model.

        Safe to call with 0 tokens — the sample is still recorded so that
        total_requests counts every call.
        """
        self._total_requests += 1
        if model not in self._samples:
            self._samples[model] = deque(maxlen=_KEEP_SAMPLES)
        sample = _LatencySample(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            node_id=node_id,
        )
        self._samples[model].append(sample)
        logger.debug(
            f"[inference_profiler] record model={model} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"latency_ms={latency_ms:.1f} node_id={node_id}"
        )

    @property
    def total_requests(self) -> int:
        return self._total_requests

    def per_model_stats(self) -> dict[str, Any]:
        """Return latency percentile stats broken down by model."""
        result: dict[str, Any] = {}
        for model, ring in self._samples.items():
            lats = [s.latency_ms for s in ring]
            if not lats:
                continue
            sorted_lats = sorted(lats)
            n = len(sorted_lats)

            def _pct(
                p: float, n: int = n, sorted_lats: list[float] = sorted_lats
            ) -> float:
                idx = max(0, int(n * p / 100) - 1)
                return round(sorted_lats[idx], 2)

            result[model] = {
                "count": n,
                "p50_ms": _pct(50),
                "p95_ms": _pct(95),
                "p99_ms": _pct(99),
                "avg_ms": round(statistics.mean(lats), 2),
                "min_ms": round(sorted_lats[0], 2),
                "max_ms": round(sorted_lats[-1], 2),
            }
        return result

    def hotspots(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the top-N slowest (model, token_bucket) combinations by p95 latency.

        Token count is bucketed into powers of two (128, 256, 512, 1024, 2048+)
        so that different prompt/completion sizes produce distinct hotspot entries.
        """
        def _bucket(tokens: int) -> str:
            for threshold in (128, 256, 512, 1024, 2048):
                if tokens <= threshold:
                    return str(threshold)
            return "2048+"

        # Aggregate latencies by (model, token_bucket)
        buckets: dict[tuple[str, str], list[float]] = {}
        for model, ring in self._samples.items():
            for s in ring:
                total_tokens = s.prompt_tokens + s.completion_tokens
                bkt = _bucket(total_tokens)
                key = (model, bkt)
                buckets.setdefault(key, []).append(s.latency_ms)

        entries: list[dict[str, Any]] = []
        for (model, bkt), lats in buckets.items():
            sorted_lats = sorted(lats)
            n = len(sorted_lats)
            p95_idx = max(0, int(n * 0.95) - 1)
            entries.append({
                "model": model,
                "token_bucket": bkt,
                "count": n,
                "p95_ms": round(sorted_lats[p95_idx], 2),
                "avg_ms": round(statistics.mean(lats), 2),
            })

        entries.sort(key=lambda e: e["p95_ms"], reverse=True)
        return entries[:top_n]


INFERENCE_PROFILER = InferenceProfiler()
