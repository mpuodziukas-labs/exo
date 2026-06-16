"""retry_policy.py — exponential-backoff retry manager for exo master API.

Handles transient worker failures (503 unavailable, 502 bad gateway, 429
rate-limit) transparently before surfacing an error to the calling client.

Usage
-----
    from exo.master.retry_policy import RETRY_MANAGER

    if RETRY_MANAGER.should_retry(trace_id, status_code=503, error_type=""):
        delay = RETRY_MANAGER.next_delay_ms(trace_id)
        RETRY_MANAGER.record_attempt(trace_id, status_code, error)
        await asyncio.sleep(delay / 1000.0)
        # … re-run inference …
    else:
        RETRY_MANAGER.clear(trace_id)
        raise original_exception
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

_RETRYABLE_STATUSES: frozenset[int] = frozenset({502, 503, 429})

_QUOTA_ERROR_DETAIL_KEYWORDS: tuple[str, ...] = (
    "quota",
    "quota_exceeded",
    "daily token quota",
)


@dataclass(frozen=True)
class RetryPolicy:
    """Immutable retry configuration.

    Reads defaults from environment variables so operators can tune without
    touching code; constructor kwargs override env for tests.
    """

    max_attempts: int = field(
        default_factory=lambda: int(os.getenv("EXO_RETRY_MAX_ATTEMPTS", "3"))
    )
    base_delay_ms: float = field(
        default_factory=lambda: float(os.getenv("EXO_RETRY_BASE_DELAY_MS", "100.0"))
    )
    max_delay_ms: float = field(
        default_factory=lambda: float(os.getenv("EXO_RETRY_MAX_DELAY_MS", "2000.0"))
    )
    backoff_factor: float = field(
        default_factory=lambda: float(os.getenv("EXO_RETRY_BACKOFF_FACTOR", "2.0"))
    )
    retryable_statuses: frozenset[int] = field(
        default_factory=lambda: _RETRYABLE_STATUSES,
    )


# ---------------------------------------------------------------------------
# Per-attempt record
# ---------------------------------------------------------------------------


@dataclass
class RetryRecord:
    """Single attempt metadata stored in ring-buffer history."""

    trace_id: str
    attempt: int
    status_code: int | None
    error: str | None
    delay_ms: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RetryManager:
    """Thread-safe manager for exponential-backoff retries.

    One singleton (``RETRY_MANAGER``) is shared across all requests.
    State is keyed on ``trace_id``; cleared on success or max-attempts.
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self._policy: RetryPolicy = policy or RetryPolicy()
        self._history: deque[RetryRecord] = deque(maxlen=500)
        self._active: dict[str, int] = {}  # trace_id → current attempt count
        self._successes: dict[str, bool] = {}  # trace_id → True if eventual success
        self._lock: Lock = Lock()

        # aggregate counters (never reset — suitable for /v1/retry-policy)
        self._total_retries: int = 0
        self._successful_retries: int = 0
        self._max_attempts_reached: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def should_retry(
        self,
        trace_id: str,
        status_code: int | None,
        error_type: str,
    ) -> bool:
        """Return True when this trace should be retried.

        Rules (evaluated in order):
        1. Timeout errors are never retried — the request is dead.
        2. If attempt count >= max_attempts, stop.
        3. For HTTP status responses: only retry whitelisted status codes.
        4. 429 from quota exhaustion is NOT retried (backing off won't help
           before the quota window resets hours later).
        5. None status_code (network/internal error) → retry unless timeout.
        """
        if "timeout" in error_type.lower():
            logger.debug(f"[retry] trace_id={trace_id} not retrying: timeout")
            return False

        with self._lock:
            attempt = self._active.get(trace_id, 0)

        if attempt >= self._policy.max_attempts:
            logger.debug(
                f"[retry] trace_id={trace_id} max_attempts={self._policy.max_attempts} reached"
            )
            with self._lock:
                self._max_attempts_reached += 1
            return False

        if status_code is None:
            # Unknown/network error — retry unless it looked like a timeout
            return True

        if status_code not in self._policy.retryable_statuses:
            logger.debug(
                f"[retry] trace_id={trace_id} status={status_code} not retryable"
            )
            return False

        if status_code == 429 and any(
            kw in error_type.lower() for kw in _QUOTA_ERROR_DETAIL_KEYWORDS
        ):
            logger.debug(
                f"[retry] trace_id={trace_id} 429 quota-exceeded — not retrying"
            )
            return False

        return True

    def next_delay_ms(self, trace_id: str) -> float:
        """Compute next sleep duration with full jitter.

        Formula: min(max_delay, base * factor^attempt) + uniform(0, 100) ms
        """
        with self._lock:
            attempt = self._active.get(trace_id, 0)

        raw = self._policy.base_delay_ms * (self._policy.backoff_factor**attempt)
        capped = min(raw, self._policy.max_delay_ms)
        jitter = random.uniform(0.0, 100.0)
        return round(capped + jitter, 2)

    def record_attempt(
        self,
        trace_id: str,
        status_code: int | None,
        error: str | None,
    ) -> None:
        """Increment attempt counter and append history record."""
        delay = self.next_delay_ms(trace_id)
        with self._lock:
            attempt = self._active.get(trace_id, 0) + 1
            self._active[trace_id] = attempt
            self._total_retries += 1
            rec = RetryRecord(
                trace_id=trace_id,
                attempt=attempt,
                status_code=status_code,
                error=error,
                delay_ms=delay,
            )
            self._history.append(rec)

        logger.info(
            f"[retry] trace_id={trace_id} attempt={attempt}"
            f"/{self._policy.max_attempts}"
            f" status={status_code} delay_ms={delay}"
        )

    def mark_success(self, trace_id: str) -> None:
        """Record that a retry eventually succeeded."""
        with self._lock:
            if trace_id in self._active and self._active[trace_id] > 0:
                self._successful_retries += 1

    def clear(self, trace_id: str) -> None:
        """Remove trace from active tracking (call after success or final failure)."""
        with self._lock:
            self._active.pop(trace_id, None)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return aggregate counters and policy config for /v1/retry-policy."""
        with self._lock:
            total = self._total_retries
            success = self._successful_retries
            max_hit = self._max_attempts_reached
            active_count = len(self._active)
            # Avg attempts: mean over history ring-buffer, or 0 if empty
            attempts_in_history = [r.attempt for r in self._history]

        avg_attempts = (
            round(sum(attempts_in_history) / len(attempts_in_history), 2)
            if attempts_in_history
            else 0.0
        )

        return {
            "total_retries": total,
            "successful_retries": success,
            "max_attempts_reached": max_hit,
            "avg_attempts": avg_attempts,
            "active_traces": active_count,
            "history_size": len(self._history),
            "policy": {
                "max_attempts": self._policy.max_attempts,
                "base_delay_ms": self._policy.base_delay_ms,
                "max_delay_ms": self._policy.max_delay_ms,
                "backoff_factor": self._policy.backoff_factor,
                "retryable_statuses": sorted(self._policy.retryable_statuses),
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

RETRY_MANAGER: RetryManager = RetryManager()
