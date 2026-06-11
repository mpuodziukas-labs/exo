"""worker_circuit_breaker.py — time-windowed circuit breaker for inference workers.

Semantics (Wave Pi):
  - If a worker crashes 3+ times within 60 s  → circuit OPEN for 30 s
  - After 30 s cooldown                        → circuit HALF-OPEN (probe)
  - Probe succeeds                             → circuit CLOSED
  - Probe fails                                → back to OPEN for another 30 s

Unlike the general-purpose CircuitBreaker in circuit_breaker.py (which counts
consecutive failures), this one uses a rolling time-window so a burst of
failures is detected even when there are successes in between.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class WorkerCircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class WorkerCircuitBreaker:
    """Per-worker circuit breaker with a rolling failure-window.

    Parameters
    ----------
    worker_id:
        Unique identifier for the worker (node-id or address string).
    failure_window_seconds:
        Rolling window in which failures are counted (default 60 s).
    failure_threshold:
        Number of failures in the window to trip the circuit (default 3).
    open_duration_seconds:
        How long the circuit stays OPEN before transitioning to HALF_OPEN
        (default 30 s).
    """

    worker_id: str
    failure_window_seconds: float = 60.0
    failure_threshold: int = 3
    open_duration_seconds: float = 30.0

    state: WorkerCircuitState = field(default=WorkerCircuitState.CLOSED)
    _failure_timestamps: deque[float] = field(default_factory=deque)
    _opened_at: float = field(default=0.0)
    _lock: Lock = field(default_factory=Lock)

    # Observability counters
    total_requests: int = field(default=0)
    total_failures: int = field(default=0)
    total_successes: int = field(default=0)
    total_trips: int = field(default=0)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _prune_window(self, now: float) -> None:
        """Remove failure timestamps older than the rolling window."""
        cutoff = now - self.failure_window_seconds
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

    def _failures_in_window(self, now: float) -> int:
        self._prune_window(now)
        return len(self._failure_timestamps)

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def allow_request(self) -> bool:
        """Return True if a request may be forwarded to this worker."""
        with self._lock:
            now = time.monotonic()

            if self.state == WorkerCircuitState.CLOSED:
                return True

            if self.state == WorkerCircuitState.OPEN:
                if now - self._opened_at >= self.open_duration_seconds:
                    self.state = WorkerCircuitState.HALF_OPEN
                    return True  # one probe request
                return False

            # HALF_OPEN: allow exactly one probe at a time
            return True

    def record_success(self) -> None:
        """Record a successful response from the worker."""
        with self._lock:
            self.total_requests += 1
            self.total_successes += 1
            if self.state == WorkerCircuitState.HALF_OPEN:
                # Probe succeeded → close the circuit
                self.state = WorkerCircuitState.CLOSED
                self._failure_timestamps.clear()

    def record_failure(self) -> None:
        """Record a worker crash/error."""
        with self._lock:
            now = time.monotonic()
            self.total_requests += 1
            self.total_failures += 1
            self._failure_timestamps.append(now)
            self._prune_window(now)

            if self.state == WorkerCircuitState.HALF_OPEN:
                # Probe failed → reopen
                self.state = WorkerCircuitState.OPEN
                self._opened_at = now
                return

            if self.state == WorkerCircuitState.CLOSED:
                if len(self._failure_timestamps) >= self.failure_threshold:
                    self.state = WorkerCircuitState.OPEN
                    self._opened_at = now
                    self.total_trips += 1

    @property
    def is_open(self) -> bool:
        return self.state == WorkerCircuitState.OPEN

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._prune_window(now)
            return {
                "worker_id": self.worker_id,
                "state": self.state.value,
                "failures_in_window": len(self._failure_timestamps),
                "failure_threshold": self.failure_threshold,
                "failure_window_seconds": self.failure_window_seconds,
                "open_duration_seconds": self.open_duration_seconds,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "total_trips": self.total_trips,
            }


class WorkerCircuitBreakerRegistry:
    """Manages WorkerCircuitBreaker instances for all inference workers."""

    def __init__(
        self,
        failure_window_seconds: float = 60.0,
        failure_threshold: int = 3,
        open_duration_seconds: float = 30.0,
    ) -> None:
        self._cfg = dict(
            failure_window_seconds=failure_window_seconds,
            failure_threshold=failure_threshold,
            open_duration_seconds=open_duration_seconds,
        )
        self._breakers: dict[str, WorkerCircuitBreaker] = {}
        self._lock = Lock()

    def get(self, worker_id: str) -> WorkerCircuitBreaker:
        with self._lock:
            if worker_id not in self._breakers:
                self._breakers[worker_id] = WorkerCircuitBreaker(
                    worker_id=worker_id, **self._cfg  # type: ignore[arg-type]
                )
            return self._breakers[worker_id]

    def allow_request(self, worker_id: str) -> bool:
        return self.get(worker_id).allow_request()

    def record_success(self, worker_id: str) -> None:
        self.get(worker_id).record_success()

    def record_failure(self, worker_id: str) -> None:
        self.get(worker_id).record_failure()

    def all_states(self) -> list[dict[str, Any]]:
        with self._lock:
            return [b.to_dict() for b in self._breakers.values()]

    def any_open(self) -> bool:
        with self._lock:
            return any(b.is_open for b in self._breakers.values())


# Module-level singleton with Wave Pi defaults (3 failures / 60 s → open 30 s)
WORKER_CIRCUIT_BREAKERS = WorkerCircuitBreakerRegistry()
