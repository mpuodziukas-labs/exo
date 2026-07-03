from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class CircuitState(str, Enum):
    CLOSED = "closed"  # normal: requests flow through
    OPEN = "open"  # tripped: requests rejected
    HALF_OPEN = "half_open"  # testing: one probe request allowed


_FAILURE_THRESHOLD = 5  # failures to trip
_FAILURE_WINDOW = 60.0  # seconds
_RECOVERY_TIMEOUT = 30.0  # seconds before HALF_OPEN
_SUCCESS_THRESHOLD = 2  # successes in HALF_OPEN to close


@dataclass
class ModelCircuit:
    model_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_timestamps: list[float] = field(default_factory=list)
    success_streak: int = 0
    tripped_at: float | None = None
    total_trips: int = 0
    total_rejected: int = 0

    def _prune_failures(self) -> None:
        cutoff = time.monotonic() - _FAILURE_WINDOW
        self.failure_timestamps = [t for t in self.failure_timestamps if t >= cutoff]

    def record_failure(self) -> None:
        self._prune_failures()
        self.failure_timestamps.append(time.monotonic())
        self.success_streak = 0
        if (
            self.state == CircuitState.HALF_OPEN
            or len(self.failure_timestamps) >= _FAILURE_THRESHOLD
        ):
            self._trip()

    def record_success(self) -> None:
        self.success_streak += 1
        if (
            self.state == CircuitState.HALF_OPEN
            and self.success_streak >= _SUCCESS_THRESHOLD
        ):
            self.close()

    def _trip(self) -> None:
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.tripped_at = time.monotonic()
            self.total_trips += 1
            logger.warning(
                f"ModelCircuitBreaker TRIP model={self.model_id} "
                f"failures={len(self.failure_timestamps)} in {_FAILURE_WINDOW}s"
            )

    def close(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_timestamps.clear()
        self.success_streak = 0
        logger.info(f"ModelCircuitBreaker CLOSE model={self.model_id} — recovered")

    def allows_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if (
                self.tripped_at is not None
                and time.monotonic() - self.tripped_at >= _RECOVERY_TIMEOUT
            ):
                self.state = CircuitState.HALF_OPEN
                logger.info(f"ModelCircuitBreaker HALF_OPEN model={self.model_id}")
                return True  # allow probe
            self.total_rejected += 1
            return False
        # HALF_OPEN: allow one probe at a time
        return True

    def to_dict(self) -> dict[str, Any]:
        self._prune_failures()
        return {
            "model_id": self.model_id,
            "state": self.state.value,
            "recent_failures": len(self.failure_timestamps),
            "success_streak": self.success_streak,
            "tripped_at": self.tripped_at,
            "total_trips": self.total_trips,
            "total_rejected": self.total_rejected,
        }


class ModelCircuitBreakerRegistry:
    """
    Per-model circuit breaker registry.
    Call allows(model_id) before routing; record_success/failure after.
    """

    def __init__(self) -> None:
        self._circuits: dict[str, ModelCircuit] = {}

    def _get(self, model_id: str) -> ModelCircuit:
        if model_id not in self._circuits:
            self._circuits[model_id] = ModelCircuit(model_id=model_id)
        return self._circuits[model_id]

    def allows(self, model_id: str) -> bool:
        return self._get(model_id).allows_request()

    def record_success(self, model_id: str) -> None:
        self._get(model_id).record_success()

    def record_failure(self, model_id: str) -> None:
        self._get(model_id).record_failure()

    def reset(self, model_id: str) -> bool:
        circuit = self._circuits.get(model_id)
        if circuit:
            circuit.close()
            return True
        return False

    def all_circuits(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._circuits.values()]

    def open_circuits(self) -> list[str]:
        return [
            model_id
            for model_id, c in self._circuits.items()
            if c.state == CircuitState.OPEN
        ]

    def stats(self) -> dict[str, Any]:
        circuits = list(self._circuits.values())
        return {
            "total_models": len(circuits),
            "open_count": sum(1 for c in circuits if c.state == CircuitState.OPEN),
            "half_open_count": sum(
                1 for c in circuits if c.state == CircuitState.HALF_OPEN
            ),
            "total_trips": sum(c.total_trips for c in circuits),
            "total_rejected": sum(c.total_rejected for c in circuits),
        }


MODEL_CIRCUIT_BREAKER = ModelCircuitBreakerRegistry()
