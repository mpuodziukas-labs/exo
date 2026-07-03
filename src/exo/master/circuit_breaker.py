from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from loguru import logger


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation — requests flow through
    OPEN = "open"  # Failure threshold exceeded — fail fast
    HALF_OPEN = "half_open"  # Probe mode — allow limited traffic to test recovery


@dataclass
class CircuitBreaker:
    """
    Per-worker circuit breaker.

    State machine:
    CLOSED -> OPEN: when failure_count >= failure_threshold within window_seconds
    OPEN -> HALF_OPEN: after cooldown_seconds
    HALF_OPEN -> CLOSED: on success_threshold consecutive successes
    HALF_OPEN -> OPEN: on any failure

    Configuration:
    - failure_threshold: consecutive failures before opening (default 5)
    - success_threshold: consecutive successes to close from half-open (default 2)
    - cooldown_seconds: time in OPEN state before probing (default 30.0)
    - window_seconds: rolling window for failure counting (default 60.0)
    """

    worker_id: str
    failure_threshold: int = 5
    success_threshold: int = 2
    cooldown_seconds: float = 30.0
    window_seconds: float = 60.0

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_streak: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    last_state_change: float = field(default_factory=time.monotonic)
    total_requests: int = field(default=0)
    total_failures: int = field(default=0)
    total_successes: int = field(default=0)
    _lock: Lock = field(default_factory=Lock)

    def allow_request(self) -> bool:
        """Returns True if request should be forwarded to this worker."""
        with self._lock:
            now = time.monotonic()

            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.cooldown_seconds:
                    logger.info(
                        f"Circuit breaker [{self.worker_id}]: OPEN -> HALF_OPEN (probing)"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    self.success_streak = 0
                    return True  # Let one probe through
                return False  # Still open — fail fast

            # HALF_OPEN: allow one probe at a time
            return True

    def record_success(self) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_successes += 1
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_streak += 1
                if self.success_streak >= self.success_threshold:
                    logger.info(
                        f"Circuit breaker [{self.worker_id}]: "
                        f"HALF_OPEN -> CLOSED (recovered)"
                    )
                    self.state = CircuitState.CLOSED
                    self.last_state_change = time.monotonic()

    def record_failure(self) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_failures += 1
            self.failure_count += 1
            self.success_streak = 0
            self.last_failure_time = time.monotonic()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit breaker [{self.worker_id}]: HALF_OPEN -> OPEN (probe failed)"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = time.monotonic()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker [{self.worker_id}]: "
                        f"CLOSED -> OPEN ({self.failure_count} consecutive failures)"
                    )
                    self.state = CircuitState.OPEN
                    self.last_state_change = time.monotonic()

    @property
    def error_rate(self) -> float:
        return self.total_failures / max(self.total_requests, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_streak": self.success_streak,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "error_rate": round(self.error_rate, 4),
            "seconds_in_current_state": round(
                time.monotonic() - self.last_state_change, 1
            ),
        }


class CircuitBreakerRegistry:
    """Manages circuit breakers for all workers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()
        self._default_failure_threshold: int = 5
        self._default_cooldown_seconds: float = 30.0

    def configure(
        self,
        *,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Update failure_threshold/cooldown for all existing breakers and
        set the defaults used by breakers created afterwards (e.g. from
        hot-reloaded config)."""
        with self._lock:
            if failure_threshold is not None:
                self._default_failure_threshold = failure_threshold
            if cooldown_seconds is not None:
                self._default_cooldown_seconds = cooldown_seconds
            for breaker in self._breakers.values():
                if failure_threshold is not None:
                    breaker.failure_threshold = failure_threshold
                if cooldown_seconds is not None:
                    breaker.cooldown_seconds = cooldown_seconds

    def get(self, worker_id: str) -> CircuitBreaker:
        with self._lock:
            if worker_id not in self._breakers:
                self._breakers[worker_id] = CircuitBreaker(
                    worker_id=worker_id,
                    failure_threshold=self._default_failure_threshold,
                    cooldown_seconds=self._default_cooldown_seconds,
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
            return any(b.state == CircuitState.OPEN for b in self._breakers.values())

    def prometheus_metrics(self) -> str:
        lines = [
            "# HELP exo_circuit_breaker_state Circuit breaker state (0=closed, 1=half_open, 2=open)",
            "# TYPE exo_circuit_breaker_state gauge",
        ]
        state_map = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2,
        }
        with self._lock:
            for b in self._breakers.values():
                lines.append(
                    f'exo_circuit_breaker_state{{worker="{b.worker_id}"}} {state_map[b.state]}'
                )
        return "\n".join(lines) + "\n"


CIRCUIT_BREAKERS = CircuitBreakerRegistry()
