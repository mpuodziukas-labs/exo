from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

from loguru import logger


@dataclass
class AdmissionDecision:
    admitted: bool
    reason: str
    retry_after_seconds: float = 0.0


class AdmissionController:
    """
    Gate for incoming inference requests. Rejects with 429 when:
    - active_requests >= max_concurrent_requests
    - memory pressure > memory_pressure_threshold
    - kv_cache pressure > kv_cache_threshold
    - queue_depth > max_queue_depth (requests waiting, not being served)

    All thresholds configurable via env vars.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._rejected_total: int = 0
        self._admitted_total: int = 0
        self._max_concurrent_requests: int | None = None

    def configure(self, *, max_concurrent_requests: int) -> None:
        """Override max_concurrent (e.g. from hot-reloaded config)."""
        self._max_concurrent_requests = max_concurrent_requests

    @property
    def max_concurrent(self) -> int:
        if self._max_concurrent_requests is not None:
            return self._max_concurrent_requests
        return int(os.getenv("EXO_MAX_CONCURRENT_REQUESTS", "16"))

    @property
    def memory_pressure_threshold(self) -> float:
        return float(os.getenv("EXO_ADMISSION_MEMORY_THRESHOLD", "0.90"))

    @property
    def kv_cache_threshold(self) -> float:
        return float(os.getenv("EXO_ADMISSION_KV_THRESHOLD", "0.95"))

    @property
    def max_queue_depth(self) -> int:
        return int(os.getenv("EXO_MAX_QUEUE_DEPTH", "32"))

    def check(
        self,
        active_requests: int,
        memory_pressure: float = 0.0,
        kv_cache_pressure: float = 0.0,
        queue_depth: int = 0,
    ) -> AdmissionDecision:
        with self._lock:
            # Check concurrent request limit
            if active_requests >= self.max_concurrent:
                self._rejected_total += 1
                wait = max(0.5, active_requests / max(self.max_concurrent, 1))
                logger.warning(
                    f"Admission: REJECT active_requests={active_requests} "
                    f">= max={self.max_concurrent}"
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"Too many concurrent requests ({active_requests}/{self.max_concurrent})",
                    retry_after_seconds=wait,
                )

            # Check memory pressure
            if memory_pressure > self.memory_pressure_threshold:
                self._rejected_total += 1
                logger.warning(
                    f"Admission: REJECT memory_pressure={memory_pressure:.1%} "
                    f"> threshold={self.memory_pressure_threshold:.1%}"
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"Memory pressure too high ({memory_pressure:.1%})",
                    retry_after_seconds=5.0,
                )

            # Check KV cache pressure
            if kv_cache_pressure > self.kv_cache_threshold:
                self._rejected_total += 1
                logger.warning(
                    f"Admission: REJECT kv_cache_pressure={kv_cache_pressure:.1%} "
                    f"> threshold={self.kv_cache_threshold:.1%}"
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"KV cache full ({kv_cache_pressure:.1%})",
                    retry_after_seconds=2.0,
                )

            # Check queue depth
            if queue_depth > self.max_queue_depth:
                self._rejected_total += 1
                logger.warning(
                    f"Admission: REJECT queue_depth={queue_depth} > max={self.max_queue_depth}"
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"Queue full ({queue_depth}/{self.max_queue_depth})",
                    retry_after_seconds=1.0,
                )

            self._admitted_total += 1
            return AdmissionDecision(admitted=True, reason="ok")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._admitted_total + self._rejected_total
            return {
                "admitted_total": self._admitted_total,
                "rejected_total": self._rejected_total,
                "rejection_rate": round(self._rejected_total / max(total, 1), 4),
                "max_concurrent": self.max_concurrent,
                "max_queue_depth": self.max_queue_depth,
                "memory_pressure_threshold": self.memory_pressure_threshold,
                "kv_cache_threshold": self.kv_cache_threshold,
            }

    def prometheus_metrics(self) -> str:
        s = self.stats()
        return (
            "# HELP exo_admission_rejected_total Total requests rejected by admission control\n"
            "# TYPE exo_admission_rejected_total counter\n"
            f"exo_admission_rejected_total {s['rejected_total']}\n"
            "# HELP exo_admission_rejection_rate Current rejection rate\n"
            "# TYPE exo_admission_rejection_rate gauge\n"
            f"exo_admission_rejection_rate {s['rejection_rate']:.4f}\n"
        )


ADMISSION_CONTROLLER = AdmissionController()
